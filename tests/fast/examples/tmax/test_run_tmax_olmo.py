"""The OLMo-core TMax launcher (run_tmax_olmo.py): trainer config, GPU layout checks and the configs it reads."""

import json
import os
import sys
from pathlib import Path

import pytest

TMAX = Path(__file__).resolve().parents[4] / "examples" / "tmax"
sys.path.insert(0, str(TMAX))

import run_tmax_olmo as launcher  # noqa: E402


def test_core_config_packs_to_the_context_and_drops_router_losses():
    config = launcher.OlmoScriptArgs(max_seq_len=16384).core_config()
    assert config == {
        "max_sequence_length": 16384,
        "expert_parallel_size": 2,
        "sequence_packing": True,
        "attention_backend": "flash_4",
        "router_aux_loss_weight": 0.0,
        "router_z_loss_weight": 0.0,
        "packing_max_tokens": 16384,
    }


def test_extra_core_fields_override_the_launcher_ones():
    args = launcher.OlmoScriptArgs(sequence_packing=False, olmo_core_config=json.dumps({"fp32_lm_head": True}))
    config = args.core_config()
    assert config["fp32_lm_head"] is True and config["sequence_packing"] is False
    assert "packing_max_tokens" not in config


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"train_gpus": 3}, "must divide train_gpus"),
        ({"train_gpus": 4, "rollout_gpus": 2}, "exceed 4 GPUs"),
        ({"hf_snapshot_every": 10, "save_interval": 20}, "multiple of save_interval"),
        ({"sglang_radix_cache": True, "sglang_max_mamba_cache_size": 320}, "5 x sglang_max_running_requests"),
    ],
)
def test_layouts_the_launcher_cannot_place_are_rejected(kwargs, message):
    with pytest.raises(AssertionError, match=message):
        launcher.OlmoScriptArgs(**kwargs)


def test_async_follows_the_flag_not_the_rollout_gpu_count():
    assert not launcher.OlmoScriptArgs().fully_async
    assert launcher.OlmoScriptArgs(asynchronous=True).fully_async


def test_every_olmo_config_sets_only_launcher_fields():
    configs = sorted((TMAX / "configs").glob("*olmo*.yaml"))
    assert configs
    for path in configs:
        # apply_yaml_config exports the config as MILES_SCRIPT_* defaults; keep them out of other tests.
        saved = dict(os.environ)
        try:
            argv = launcher.dense.apply_yaml_config(
                ["run_tmax_olmo.py", "execute", str(path)], launcher.OlmoScriptArgs
            )
        finally:
            os.environ.clear()
            os.environ.update(saved)
        assert argv == ["run_tmax_olmo.py", "execute"]


def test_skip_labeled_keeps_only_tasks_short_of_their_scored_episodes(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"prompt": [], "metadata": {"task_id": t}}) + "\n" for t in "abcd"))
    run_dir = tmp_path / "traces" / "run"
    run_dir.mkdir(parents=True)
    episodes = [("a", 1.0, "submitted"), ("a", 0.0, "max_steps"), ("b", 0.0, "length"), ("b", None, "sandbox_lost")]
    episodes += [("c", 1.0, "submitted"), ("c", 0.0, "no_tool_call"), ("d", 0.0, "start_failed")]
    (run_dir / "traces-host.jsonl").write_text(
        "".join(json.dumps({"task_id": t, "reward": r, "metrics": {"end_reason": e}}) + "\n" for t, r, e in episodes)
        + '{"task_id": "d", "rew'
    )
    args = launcher.OlmoScriptArgs(
        rollout_only=True,
        skip_labeled=True,
        prompt_file=str(prompts),
        trace_dir=str(tmp_path / "traces"),
        exp_name="run",
        n_samples_per_prompt=2,
    )
    # a and c are labeled; b has one scored episode (the other lost its sandbox); d's sandbox never started and its
    # other line was cut off.
    assert launcher._write_unlabeled_prompts(args) == 2
    remaining = [json.loads(line)["metadata"]["task_id"] for line in open(args.prompt_data)]
    assert remaining == ["b", "d"] and args.prompt_data == str(run_dir / "unlabeled_prompts.jsonl")


def test_skip_labeled_needs_a_rollout_only_run_with_traces():
    with pytest.raises(AssertionError, match="skip_labeled"):
        launcher.OlmoScriptArgs(skip_labeled=True, trace_dir="/traces")


@pytest.mark.parametrize("every, expected", [(0, None), (10, "--save-hf /out/run/checkpoints/hf/step_{rollout_id}")])
def test_hf_snapshots_export_at_saves_when_requested(monkeypatch, every, expected):
    captured = {}

    class Backend:
        @staticmethod
        def execute_train(train_args, **_kwargs):
            captured["train_args"] = train_args

    monkeypatch.setattr(launcher.OlmoScriptArgs, "create_backend", lambda self: Backend())
    args = launcher.OlmoScriptArgs(
        sandbox_backend="docker", save_interval=5, hf_snapshot_every=every, output_dir="/out", exp_name="run"
    )
    launcher.run_execute(args)
    found = [
        part for part in ("--save-hf /out/run/checkpoints/hf/step_{rollout_id}",) if part in captured["train_args"]
    ]
    assert found == ([expected] if expected else [])
    assert "--save /out/run/checkpoints " in captured["train_args"]


def test_hf_path_replaces_the_model_names_checkpoint():
    assert (
        launcher.OlmoScriptArgs(model_name="olmo35-tiny-h250-b2e").hf_checkpoint
        == launcher._OLMO_MODELS["olmo35-tiny-h250-b2e"]
    )
    args = launcher.OlmoScriptArgs(model_name="olmo35-tiny-h250-b2e", hf_path="/ckpt/hf/step_75")
    assert args.hf_checkpoint == "/ckpt/hf/step_75"


def _train_args(monkeypatch, **kwargs):
    captured = {}

    class Backend:
        @staticmethod
        def execute_train(train_args, **_kwargs):
            captured["train_args"] = train_args

    monkeypatch.setattr(launcher.OlmoScriptArgs, "create_backend", lambda self: Backend())
    launcher.run_execute(launcher.OlmoScriptArgs(sandbox_backend="docker", **kwargs))
    return captured["train_args"]


def test_one_node_keeps_trainer_and_engines_together(monkeypatch):
    args = _train_args(monkeypatch, num_gpus_per_node=8, train_gpus=4, rollout_gpus=4, expert_parallel_size=4)
    assert "--actor-num-nodes 1 --actor-num-gpus-per-node 4 --rollout-num-gpus 4 " in args
    assert "--pin-rollout-manager-to-head" not in args


def test_several_nodes_put_the_trainer_on_the_first_whole_nodes(monkeypatch):
    layout = dict(num_nodes=3, num_gpus_per_node=8, train_gpus=16, rollout_gpus=8, expert_parallel_size=16)
    assert launcher.OlmoScriptArgs(**layout).train_layout == (2, 8)
    args = _train_args(monkeypatch, rollout_num_gpus_per_engine=2, **layout)
    assert "--actor-num-nodes 2 --actor-num-gpus-per-node 8 --rollout-num-gpus 8 --num-gpus-per-node 8 " in args
    assert "--rollout-num-gpus-per-engine 2 " in args
    assert "--pin-rollout-manager-to-head" in args


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_nodes": 2, "train_gpus": 12, "rollout_gpus": 4, "expert_parallel_size": 4}, "whole nodes"),
        ({"num_nodes": 3, "train_gpus": 8, "rollout_gpus": 8, "expert_parallel_size": 8}, "must fill"),
        ({"rollout_gpus": 3, "rollout_num_gpus_per_engine": 2}, "whole engines"),
    ],
)
def test_layouts_that_leave_nodes_or_engines_partial_are_rejected(kwargs, message):
    with pytest.raises(AssertionError, match=message):
        launcher.OlmoScriptArgs(num_gpus_per_node=8, **kwargs)
