"""TMax Terminal-RL on Olmo 3.5 with the OLMo-core trainer, run by Miles' own loop (no Open Instruct).

=====================

The TMax recipe of run_tmax_qwen3_5.py: the same dataset, sandboxes, agent loop, reward and DPPO-TV
objective, with every shared flag documented there. On top of it this launcher:

  * trains with OLMo-core (--train-backend olmo_core), configured through miles.backends.core_utils.standalone:
    the HF checkpoint is loaded directly, with no torch_dist conversion;
  * serves the checkpoint with SGLang's out-of-tree Olmo models (SGLANG_EXTERNAL_MODEL_PACKAGE=olmo_sglang.models),
    in the checkpoint's own chat template through --tito-model olmo35 (XML tool calls, qwen3_coder parser);
  * always disaggregates: train_gpus train and rollout_gpus serve, both in synchronous and fully async runs;
  * replays the rollout's expert routing in training (R3), on by default.

Run it in the OLMo-core image (docker/olmo/Dockerfile), with the checkpoint's weka bucket mounted. With one
optimizer step per rollout (global batch = rollout_batch_size x n_samples_per_prompt, as here), a weight version is
one rollout, so --max-weight-staleness means the same as in the Qwen runs.

=====================

Args (in addition to run_tmax_qwen3_5.py's; its Megatron-only flags are ignored):
  --model-name: a key of _OLMO_MODELS (default: olmo35-tiny-s-c-noag).
  --train-gpus / --rollout-gpus: GPUs of the OLMo-core trainer / of the SGLang engines (default: 2 / 2).
  --asynchronous: train_async.py --fully-async, generation never pauses for training (default: off).
  --expert-parallel-size: OLMo-core expert parallelism; must divide train_gpus (default: 2).
  --sequence-packing: pack samples into --qkv-format thd rows of max_seq_len tokens (default: on).
  --attention-backend: the trainer's attention, flash_4 on B300, flash_3 on H100, flash_2 on A100 (default: flash_4).
  --rollout-routing-replay: R3 (default: on).
  --sglang-cuda-graph-decode: decode CUDA graphs on the SGLang engines (default: on).
  --sglang-radix-cache: prefix caching across turns, KDA state snapshotted with extra_buffer; sync only (default: off).
  --olmo-core-config: extra CoreConfig fields as a JSON object, merged over this launcher's.

=====================

  python examples/tmax/run_tmax_olmo.py prepare examples/tmax/configs/smoke_olmo_tiny_sync.yaml
  python examples/tmax/run_tmax_olmo.py execute examples/tmax/configs/smoke_olmo_tiny_sync.yaml

On Beaker set ``launcher: examples/tmax/run_tmax_olmo.py`` and the OLMo-core image in the config's beaker: section.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import run_tmax_qwen3_5 as dense
import typer

from miles.utils.external_utils import command_utils

# HF checkpoints, loaded in place (no copy, no conversion).
_OLMO_MODELS: dict[str, str] = {
    # Olmo 3.5 Tiny (olmo3moe, 12.5B total / 0.8B active, KDA hybrid): ledger H196's s-c SFT (C12 mix) on the base
    # mid-trained without agentic data; the checkpoint the OLMo team's recent MILES RL runs use.
    "olmo35-tiny-s-c-noag": "/weka/olmo-3p5-post-checkpoints/abhishekr/s-c-gpu/s-c-noag-s34521/hf_step63494",
    # The same Tiny after the OLMo team's math/science RL (ledger H235, arm B2048-Noff-LR6e6: Dolci prompts with the
    # H196 verifiers, batch 2048, LR 6e-6), the run's final HF export after update 100 (= checkpoints/core/
    # rollout_0000099). The export carries the SFT checkpoint's chat template.
    "olmo35-tiny-h235-rl100": (
        "/weka/olmo-3p5-post-checkpoints/robertb/miles/h235-continue-204800-20261117/B2048-Noff-LR6e6/export-hf"
    ),
    # A Tiny SFT from the TPU post-training runs (olmo35-sft-h250-b2e-c12r-ota-b2-s0, step 7893): the same
    # architecture, EOS ids and chat template (sha256 f43922cd...) as olmo35-tiny-s-c-noag.
    "olmo35-tiny-h250-b2e": (
        "/weka/olmo-3p5-post-checkpoints/pradeepd/tpu-posttrain/hf/olmo35-sft-h250-b2e-c12r-ota-b2-s0-step7893"
    ),
}
_OLMO_MODEL_NAMES = Literal["olmo35-tiny-s-c-noag", "olmo35-tiny-h235-rl100", "olmo35-tiny-h250-b2e"]

app = typer.Typer()


@dataclass
class OlmoScriptArgs(dense.ScriptArgs):
    model_name: _OLMO_MODEL_NAMES = "olmo35-tiny-s-c-noag"
    num_gpus_per_node: int = 4
    train_gpus: int = 2
    rollout_gpus: int = 2
    asynchronous: bool = False
    expert_parallel_size: int = 2
    sequence_packing: bool = True
    # OLMo-core trainer attention; packing needs a flash backend. Open Instruct's per-cluster choice: flash_4 on
    # Holmes (B300), flash_3 on Jupiter (H100), flash_2 on Saturn (A100).
    attention_backend: str = "flash_4"
    rollout_routing_replay: bool = True
    max_seq_len: int = 32768
    rollout_max_response_len: int = 8192
    lr: float = 1e-6
    # SGLang engines (Olmo 3.5 hybrid): the measured B300 profile's Triton-attention choices, with smaller pools.
    sglang_mem_fraction_static: float = 0.7
    sglang_attention_backend: str = "triton"
    sglang_max_running_requests: int = 64
    sglang_max_mamba_cache_size: int = 256
    # Decode CUDA graphs, as in the measured B300 preset (prefill graphs stay off): about twice the decode speed in
    # the 64K smoke (01M7QTQNZ4), for about 110 s more engine start.
    sglang_cuda_graph_decode: bool = True
    # Prefix (radix) caching across an episode's turns. Off by default, as in the B300 preset. KDA's recurrent state
    # cannot be split per token, so olmo-sglang snapshots it at prefix boundaries with SGLang's extra_buffer strategy.
    # Sync only: each weight update flushes the cache (retract pause). In-place async updates need the
    # policy-versioned cache backend, which the trainer's publication checks require. Each running request can hold
    # up to 5 KDA state slots under extra_buffer, so the mamba pool must exceed 5 x sglang_max_running_requests
    # (open-instruct's validated capacity rule).
    sglang_radix_cache: bool = False
    olmo_core_config: str = ""
    megatron_path: str = "/src/Megatron-LM"

    def __post_init__(self):
        super().__post_init__()
        assert self.opd_teacher is None, "OPD is not wired for the OLMo-core trainer"
        assert self.hf_snapshot_every == 0, "the OLMo-core trainer writes native checkpoints only (--save-hf)"
        assert (
            self.train_gpus % self.expert_parallel_size == 0
        ), f"expert_parallel_size {self.expert_parallel_size} must divide train_gpus {self.train_gpus}"
        assert self.num_nodes == 1, "this launcher places trainer and engines on one node"
        assert not (
            self.sglang_radix_cache and self.asynchronous
        ), "sglang_radix_cache is sync-only: in-place async weight updates need the policy-versioned cache backend"
        assert (
            not self.sglang_radix_cache or self.sglang_max_mamba_cache_size > 5 * self.sglang_max_running_requests
        ), (
            f"sglang_radix_cache needs sglang_max_mamba_cache_size > 5 x sglang_max_running_requests "
            f"({self.sglang_max_mamba_cache_size} <= 5 x {self.sglang_max_running_requests})"
        )
        assert (
            self.train_gpus + self.rollout_gpus <= self.num_gpus_per_node
        ), f"train_gpus {self.train_gpus} + rollout_gpus {self.rollout_gpus} exceed {self.num_gpus_per_node} GPUs"

    def recipe_for(self, model_name: str):
        raise NotImplementedError("Olmo checkpoints are HF paths in _OLMO_MODELS, not Megatron recipes")

    @property
    def hf_checkpoint(self) -> str:
        return _OLMO_MODELS[self.model_name]

    @property
    def fully_async(self) -> bool:
        return self.asynchronous

    def core_config(self) -> dict:
        config = {
            "max_sequence_length": self.max_seq_len,
            "expert_parallel_size": self.expert_parallel_size,
            "sequence_packing": self.sequence_packing,
            "attention_backend": self.attention_backend,
            # The OLMo team's RL recipes train without the MoE router's auxiliary losses.
            "router_aux_loss_weight": 0.0,
            "router_z_loss_weight": 0.0,
        }
        if self.sequence_packing:
            config["packing_max_tokens"] = self.max_seq_len
        return config | (json.loads(self.olmo_core_config) if self.olmo_core_config else {})


@app.command()
@command_utils.dataclass_cli
def prepare(args: OlmoScriptArgs):
    """The dataset and prompt JSONL only: the trainer and SGLang read the HF checkpoint in place."""
    U = args.create_backend()
    U.hf_download_dataset(dense._DATASET, data_dir=args.data_dir)
    if not Path(args.task_data_dir).exists():
        U.exec_command_cpu(
            f"mkdir -p {args.task_data_dir}.tmp && "
            f"tar -xzf {args.dataset_dir}/task-data.tar.gz -C {args.task_data_dir}.tmp && "
            f"mv {args.task_data_dir}.tmp {args.task_data_dir}"
        )
    dense._write_prompt_data(args)
    assert Path(args.hf_checkpoint, "config.json").exists(), f"checkpoint not mounted: {args.hf_checkpoint}"


@app.command()
@command_utils.dataclass_cli
def execute(args: OlmoScriptArgs):
    run_execute(args)


def run_execute(args: OlmoScriptArgs) -> None:
    U = args.create_backend()
    assert (
        args.sandbox_backend != "gateway" or args.sandbox_gateway_url
    ), "--sandbox-backend gateway needs --gateway-url or TMAX_GATEWAY_URL"

    core_args = (
        "--train-backend olmo_core "
        f"--olmo-core-config '{json.dumps(args.core_config(), separators=(',', ':'))}' "
        f"--max-seq-len {args.max_seq_len} "
    )
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--load {args.checkpoint_dir} "
        f"--save {args.checkpoint_dir} "
        f"--save-interval {args.save_interval} "
    )

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 1.0 "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        # <|im_end|> ends a tool-calling turn, <|endoftext|> (the eos token) an answer.
        "--rollout-stop-token-ids 100265 100257 "
    )

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path tmax_agent_function.run "
        "--custom-rm-path generate.reward_func "
        f"--use-session-server {args.session_server} "
        "--tito-model olmo35 "
    )
    group_filter = "nonzero_std_filter" if args.filter_zero_std else "missing_reward_filter"
    agent_args += (
        f"--dynamic-sampling-filter-path tmax_metrics.{group_filter} "
        "--custom-rollout-log-function-path tmax_metrics.log_rollout_data "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--disable-grpo-std-normalization "
        "--use-rollout-logprobs "
        "--calculate-per-token-loss "
        "--entropy-coef 0.0 "
        "--observe-training-entropy "
    )
    if args.log_kl_to_base:
        # The reference is the same HF checkpoint; OLMo-core builds it eval-only.
        grpo_args += f"--use-kl-loss --kl-loss-coef 0.0 --kl-loss-type k3 --ref-load {args.hf_checkpoint} "
    if args.policy_loss == "dppo_tv":
        grpo_args += "--policy-loss-type dppo_tv --dppo-tv-threshold 0.1 "
    else:
        grpo_args += "--eps-clip 0.2 --eps-clip-high 0.28 "

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.0 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.999 "
        "--seed 42 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-reasoning-parser qwen3 "
        "--sglang-tool-call-parser qwen3_coder "
        f"--sglang-attention-backend {args.sglang_attention_backend} "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
        f"--sglang-max-mamba-cache-size {args.sglang_max_mamba_cache_size} "
    )
    if args.sglang_radix_cache:
        sglang_args += "--sglang-mamba-radix-cache-strategy extra_buffer "
    else:
        sglang_args += "--sglang-disable-radix-cache "
    sglang_args += "--sglang-cuda-graph-backend-prefill disabled "
    if args.sglang_cuda_graph_decode:
        sglang_args += (
            "--sglang-cuda-graph-backend-decode full "
            f"--sglang-cuda-graph-max-bs-decode {args.sglang_max_running_requests} "
        )
    else:
        sglang_args += "--sglang-cuda-graph-backend-decode disabled "

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.train_gpus} "
        f"--rollout-num-gpus {args.rollout_gpus} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    if args.rollout_routing_replay:
        misc_args += "--use-rollout-routing-replay "
    if args.fully_async:
        # Generation never stops: in-flight requests take new weights in place (radix cache off above), and a group
        # is dropped only when its oldest weight version is more than max_weight_staleness behind.
        misc_args += (
            "--fully-async "
            f"--max-weight-staleness {args.max_weight_staleness} "
            f"--async-data-buffer-capacity-factor {args.async_data_buffer_capacity_factor} "
            "--pause-generation-mode in_place "
        )
        if args.async_max_concurrent_samples:
            misc_args += f"--async-max-concurrent-samples {args.async_max_concurrent_samples} "

    if args.dump_training_data:
        assert args.trace_dir, "dump_training_data writes next to the traces: set trace_dir"
        misc_args += (
            f"--save-debug-rollout-data {args.trace_run_dir}/rollout_data/{{rollout_id}}.pt "
            f"--save-debug-train-data {args.trace_run_dir}/train_data/{{rollout_id}}_{{rank}}.pt "
        )

    train_args = (
        f"{core_args} {ckpt_args} {rollout_args} {agent_args} {optimizer_args} {grpo_args} "
        f"{dense._wandb_args(args)} {sglang_args} {misc_args} {args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
        megatron_path=args.megatron_path,
        train_script="train_async.py" if args.fully_async else "train.py",
        extra_env_vars={
            "PYTHONPATH": f"{dense._SCRIPT_DIR}:{command_utils.repo_base_dir}",
            "SGLANG_EXTERNAL_MODEL_PACKAGE": "olmo_sglang.models",
            "TMAX_TASK_DATA_DIR": args.task_data_dir,
            "TMAX_SANDBOX_BACKEND": args.sandbox_backend,
            "TMAX_GATEWAY_URL": args.sandbox_gateway_url,
            "TMAX_DOCKER_HOST": args.sandbox_docker_host,
            "TMAX_MAX_STEPS": str(args.max_steps),
            "TMAX_START_CONCURRENCY": str(args.sandbox_start_concurrency),
            "TMAX_EXEC_CONCURRENCY": str(args.sandbox_exec_concurrency),
            "TMAX_SANDBOXES_PER_NODE": str(args.sandboxes_per_node),
            "TMAX_START_FAILURE": args.start_failure,
            "TMAX_EPISODE_TIMEOUT_S": str(args.episode_timeout_s),
            "TMAX_TOOL_CALL_TIMEOUT_S": str(args.tool_call_timeout_s),
            "TMAX_TRACE_DIR": args.trace_run_dir,
        },
    )


if __name__ == "__main__":
    sys.argv = dense.apply_yaml_config(sys.argv, OlmoScriptArgs)
    app()
