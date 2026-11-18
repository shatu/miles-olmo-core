"""Load OLMo-core trainer arguments without Open Instruct.

``integration.load_args`` uses this loader unless ``MILES_OLMO_CORE_LOADER`` is
``open_instruct``. The run then uses Miles' own pieces end to end:

- the training loop (``train.py``, or ``train_async.py`` for ``--fully-async``);
- the data source (``--prompt-data``) and reward function (``--custom-rm-path``,
  or reward set by the generate function);
- the fully async producer and its default buffer.

Generation keeps running across weight updates, so a multi-turn episode can
span several weight versions. The buffer drops a group only when its oldest
version is more than ``--max-weight-staleness`` behind. With the OLMo-core
trainer, a weight version is its count of completed optimizer steps, so
staleness is measured in optimizer steps, not in rollouts.

Trainer settings come from ``--olmo-core-config``: a JSON object of
``CoreConfig`` fields, or ``@path`` to a JSON file. Fields that only take effect
through Open Instruct's driver, data source, reward adapter or inference
recorder are rejected here, rather than silently ignored.
"""

import dataclasses
import json
import sys
from pathlib import Path

from miles.backends.core_utils.standalone_config import CoreConfig, InputError
from miles.backends.fsdp_utils.arguments import load_fsdp_args

LOADER = "standalone"

OPEN_INSTRUCT_ONLY_FIELDS = (
    # Open Instruct's reward adapter and the reward-group filters it installs.
    "reward_zero_truncated",
    "reward_final_answer_only",
    "reward_reasoning_mode",
    "exclude_ungraded",
    "filter_zero_std_groups",
    "reward_config",
    # Open Instruct's driver: run limits, final save and the startup compiler cache.
    "max_run_seconds",
    "max_run_final",
    "save_final_checkpoint",
    "compiler_cache",
    "compiler_cache_root",
    "compiler_cache_restore",
    "compiler_cache_diagnostics",
    "compiler_cache_tilelang",
    "compiler_cache_max_storage_bytes",
    "compiler_cache_publish_interval_seconds",
    # Open Instruct's inference recorder.
    "records_root",
    "records_responses",
    "records_response_sample_rate",
    # Open Instruct's prompt providers and data source.
    "prompt_provider",
    "prompt_mixture",
    "prompt_priority_field",
    "prompt_priority_source_field",
    "prompt_priority_source_weights",
    "remind_review_fraction",
    "remind_enqueue_probability",
    "prompt_replay_enabled",
    "prompt_replay_fraction",
    "prompt_replay_length_drop",
    "prompt_replay_window",
    "prompt_replay_max_successes",
    "prompt_replay_truncation",
    "prompt_replay_answer_probe",
    "selection_table",
    "selection_sha256",
)

# Trainer options the OLMo-core trainer does not implement (as in Open Instruct's checks).
_PARALLEL_OPTIONS = ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size")
_UNIMPLEMENTED_OPTIONS = ("use_critic", "multi_lora", "indep_dp", "use_opd", "use_routing_replay")
_OFFLOAD_OPTIONS = (
    "offload",
    "offload_train",
    "fsdp_cpu_offload",
    "optimizer_cpu_offload",
    "stream_optimizer_state_to_disk",
)


def read_core_config(value: str) -> dict:
    """Parse ``--olmo-core-config``: inline JSON, or ``@path`` to a JSON file."""
    text = Path(value[1:]).read_text() if value.startswith("@") else value
    try:
        fields = json.loads(text)
    except json.JSONDecodeError as error:
        raise InputError(f"--olmo-core-config must be a JSON object or @path to one: {error}") from error
    if not isinstance(fields, dict):
        raise InputError("--olmo-core-config must be a JSON object of CoreConfig fields")
    unknown = sorted(set(fields) - {field.name for field in dataclasses.fields(CoreConfig)})
    if unknown:
        raise InputError(f"--olmo-core-config has unknown CoreConfig fields: {', '.join(unknown)}")
    open_instruct_only = sorted(set(fields) & set(OPEN_INSTRUCT_ONLY_FIELDS))
    if open_instruct_only:
        raise InputError(
            f"--olmo-core-config fields {', '.join(open_instruct_only)} only take effect under Open Instruct's "
            "driver; use Miles' own options (for example --custom-rm-path, --dynamic-sampling-filter-path), or set "
            "MILES_OLMO_CORE_LOADER=open_instruct"
        )
    return fields


def _supplied_flags():
    return {token.partition("=")[0] for token in sys.argv[1:] if token.startswith("--")}


def resolve_core_config(args, fields: dict, supplied=frozenset()) -> CoreConfig:
    """Build and check the trainer config against the parsed Miles arguments.

    ``supplied`` holds the flags given on the command line; Miles defaults that the
    trainer cannot use are set to what it needs unless the user chose them explicitly.
    """
    # The startup compiler cache is prepared by Open Instruct's driver only.
    core = CoreConfig(**fields, compiler_cache=False)
    if core.publication_mode != "barrier":
        raise InputError(
            f"core.publication_mode={core.publication_mode} needs Open Instruct's driver and producers; "
            "the standalone loader supports barrier publication"
        )
    for name in _PARALLEL_OPTIONS:
        if (getattr(args, name, 1) or 1) != 1:
            raise InputError(f"The OLMo-core trainer does not support {name}>1")
    for name in (*_UNIMPLEMENTED_OPTIONS, *_OFFLOAD_OPTIONS):
        if getattr(args, name, False):
            raise InputError(f"The OLMo-core trainer has no implementation for --{name.replace('_', '-')}")
    if getattr(args, "micro_batch_size", 1) != 1 or getattr(args, "use_dynamic_batch_size", False):
        raise InputError(
            "The OLMo-core trainer packs its own batches: keep --micro-batch-size 1, no dynamic batch size"
        )
    layout = "thd" if core.sequence_packing else "bshd"
    if "--qkv-format" not in supplied:
        args.qkv_format = layout
    elif args.qkv_format != layout:
        raise InputError(f"core.sequence_packing={core.sequence_packing} requires --qkv-format {layout}")
    if getattr(args, "check_weight_update_selector", "all") != "all":
        raise InputError("The OLMo-core trainer checks weight updates with --check-weight-update-selector all only")
    if getattr(args, "ref_update_interval", None) is not None:
        raise InputError("The OLMo-core trainer does not implement --ref-update-interval")
    if core.replay_diagnostics and not getattr(args, "use_rollout_routing_replay", False):
        raise InputError("core.replay_diagnostics requires --use-rollout-routing-replay")

    world = args.actor_num_nodes * args.actor_num_gpus_per_node
    batch = args.global_batch_size
    if not batch or batch % world:
        raise InputError(f"--global-batch-size {batch} must be a positive multiple of the trainer world size {world}")
    collection = args.rollout_batch_size * args.n_samples_per_prompt
    if collection % batch:
        raise InputError(
            f"A rollout collects {collection} samples (--rollout-batch-size x --n-samples-per-prompt); "
            f"--global-batch-size {batch} must divide it into complete optimizer batches"
        )
    steps = collection // batch

    # The trainer raises when a sample's oldest weight version is more than max_policy_lag
    # optimizer steps behind. Within one collection the trainer advances steps - 1 more.
    if args.fully_async:
        if args.colocate or getattr(args, "offload_rollout", False):
            raise InputError("--fully-async needs disaggregated serving: no --colocate or --offload-rollout")
        if getattr(args, "update_weights_interval", 1) != 1:
            raise InputError("--fully-async publishes after every collection: keep --update-weights-interval 1")
        if args.max_weight_staleness is None:
            raise InputError("--fully-async with the OLMo-core trainer needs --max-weight-staleness (optimizer steps)")
        needed = args.max_weight_staleness + steps - 1
    else:
        needed = steps * getattr(args, "update_weights_interval", 1) - 1
    # agentic_tool_call caps each episode at args.max_seq_len (Megatron's flag, which the
    # FSDP-style parser here lacks); an uncapped episode longer than the trainer's context
    # would fail in training rather than end as a truncated episode.
    max_seq_len = getattr(args, "max_seq_len", None)
    if max_seq_len is None:
        args.max_seq_len = core.max_sequence_length
    elif max_seq_len > core.max_sequence_length:
        raise InputError(
            f"--max-seq-len {max_seq_len} exceeds core.max_sequence_length={core.max_sequence_length}: "
            "the trainer could not train an episode of that length"
        )
    if "max_policy_lag" not in fields:
        core = dataclasses.replace(core, max_policy_lag=needed)
    elif core.max_policy_lag < needed:
        raise InputError(
            f"core.max_policy_lag={core.max_policy_lag} is below the {needed} optimizer steps this run can lag "
            "(--max-weight-staleness plus optimizer steps per rollout, minus one); raise it or omit it"
        )
    return core


def load_args(*, extra_args_provider):
    def add_arguments(parser):
        parser = extra_args_provider(parser)
        # agentic_tool_call registers --max-seq-len itself when it is the generate function.
        if "--max-seq-len" not in parser._option_string_actions:
            parser.add_argument(
                "--max-seq-len",
                type=int,
                default=None,
                help="Token cap of a multi-turn episode (agentic_tool_call); defaults to core.max_sequence_length.",
            )
        parser.add_argument(
            "--olmo-core-config",
            default="{}",
            help="OLMo-core trainer settings: a JSON object of CoreConfig fields, or @path to a JSON file.",
        )
        return parser

    args = load_fsdp_args(extra_args_provider=add_arguments)
    args.olmo_core = resolve_core_config(args, read_core_config(args.olmo_core_config), _supplied_flags())
    args.olmo_core_loader = LOADER
    args.compress_ratios = None
    return args
