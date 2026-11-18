"""The OLMo-core trainer's settings, without Open Instruct.

``CoreConfig`` is a verbatim copy of Open Instruct's class of the same name
(``open_instruct/miles/configuration/config.py`` at allenai/open-instruct
``28c181666``, 2026-11-14), so standalone runs and Open Instruct runs read the
same fields with the same defaults and checks. Open Instruct's validation
helpers are inlined below. Resync this file when the OLMo team adds or changes
a field there; ``tests/fast/backends/core_utils/test_standalone_loader.py``
pins the field list it was copied with.

Some fields only take effect through Open Instruct's driver, data source and
reward adapter; ``standalone.OPEN_INSTRUCT_ONLY_FIELDS`` lists them, and the
standalone loader rejects them when supplied.
"""

import dataclasses
import math
import re
from pathlib import Path

RECORD_RESPONSE_MODES = ("off", "all", "sample")


class InputError(ValueError):
    """Invalid OLMo-core configuration that a user can fix before launching."""


def _error(message):
    return InputError(f"Invalid configuration: {message}")


def _boolean(value, name):
    if type(value) is not bool:
        raise InputError(f"{name} must be a boolean; use true or false without quotes.")
    return value


def _integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        requirement = "positive integer" if minimum == 1 else "nonnegative integer"
        raise InputError(f"{name} must be a {requirement}; got {value!r}.")
    return value


def _number(value, name, *, minimum=0, maximum=None, exclusive_min=False, exclusive_max=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    bounds = f"{'>' if exclusive_min else '>='} {minimum}"
    if maximum is not None:
        bounds += f" and {'<' if exclusive_max else '<='} {maximum}"
    if (
        not finite
        or value < minimum
        or (exclusive_min and value == minimum)
        or (maximum is not None and (value > maximum or (exclusive_max and value == maximum)))
    ):
        raise InputError(f"{name} must be a finite number {bounds}; got {value!r}.")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise InputError(f'{name} must be a nonempty string; use a quoted value, for example "value".')
    return value


def _choice(value, name, choices):
    if value not in choices:
        raise InputError(f"{name} must be one of {', '.join(map(repr, choices))}; got {value!r}.")
    return value


def _compiler_cache_root(shared):
    """Validate retention policy before creating reports or cache artifacts."""
    if not shared.is_absolute():
        raise ValueError("Compiler cache root must be an absolute path")
    resolved = shared.resolve()
    if resolved.is_relative_to("/weka") and not any(
        re.fullmatch(r"tmp-[1-9][0-9]*[hdwmy]", part) for part in resolved.parts
    ):
        raise ValueError("WEKA cache root requires an expiry component such as tmp-30d")


@dataclasses.dataclass(frozen=True)
class CoreConfig:
    reward_zero_truncated: bool = True
    # False: historical consumer dispatch on full text. True: shared selection
    # of the text after thinking; trusted per-verifier process exceptions apply.
    # Selection preserves answer whitespace and does not extract answer wrappers.
    reward_final_answer_only: bool = False
    # Only used by final-answer grading. Non-thinking runs must opt into none.
    reward_reasoning_mode: str = "required"
    exclude_ungraded: bool = True
    max_run_seconds: float | None = None
    max_run_final: bool = False
    filter_zero_std_groups: bool = True
    max_train_rollout_logprob_abs_diff: float | None = None
    diagnostic_interval: int = 0
    layer_profile_interval: int = 0
    pipeline_observation_interval: float = 0.0
    replay_diagnostics: bool = False
    stream_moe_export: bool = True
    weight_sync_mode: str = "flattened"
    publication_mode: str = "barrier"
    # One batch ahead for disaggregated refresh with one optimizer update per collection.
    prepared_batch_overlap: bool = True
    engine_drain_timeout: float = 180.0
    engine_update_timeout: float = 180.0
    refresh_request_timeout: float = 1800.0
    snapshot_capacity: int = 2
    row_specialization: str = "static"
    compiler_cache: bool = True
    compiler_cache_root: str | None = None
    compiler_cache_restore: bool = True
    compiler_cache_diagnostics: bool = False
    compiler_cache_tilelang: bool = False
    compiler_cache_max_storage_bytes: int = 8 * 1024**3
    compiler_cache_publish_interval_seconds: float = 600.0
    checkpoint_profile: bool = False
    checkpoint_thread_count: int | None = None
    checkpoint_process_count: int | None = None
    checkpoint_compact_storage: bool = True
    checkpoint_dedup_save_to_lowest_rank: bool = False
    checkpoint_constant_memory_planning: bool = True
    # Driver policy for epoch-derived budgets without a periodic save cadence.
    save_final_checkpoint: bool = False
    # Committed native checkpoints to retain after each commit: the newest
    # checkpoint_keep_last plus every checkpoint_keep_every-th completed update.
    # None keeps every checkpoint.
    checkpoint_keep_last: int | None = None
    checkpoint_keep_every: int | None = None
    model_config: str | None = None
    reward_config: str | None = None
    expert_parallel_size: int = 1
    attention_backend: str = "torch"
    fp32_lm_head: bool = False
    activation_checkpointing: bool = True
    compile_model: bool = False
    compile_optimizer: bool = False
    use_reduce_scatter: bool = False
    max_sequence_length: int = 8192
    sequence_packing: bool = False
    expert_balanced_packing: bool = False
    expert_balance_layer_stride: int = 1
    expert_balance_search_proposals: int = 1024
    expert_balance_search_seconds: float = 0.25
    packing_max_tokens: int | None = None
    max_policy_lag: int = 0
    router_aux_loss_grouping: str = "pack"
    router_aux_loss_reduction: str = "token"
    router_z_loss_reduction: str = "token"
    router_aux_count_source: str = "executed"
    router_aux_loss_weight: float = 0.01
    router_z_loss_weight: float = 1e-5
    # The standalone pre-update scoring pass is skipped when the recipe makes it
    # redundant (see scoring_pass); these control the override and the periodic
    # standalone-versus-training check that guards the skipped path.
    scoring_pass_required: bool = False
    scoring_check_interval: int = 50
    # Fail when the mean absolute standalone/training gap exceeds this tolerance.
    scoring_check_tolerance: float = 1e-3
    # Publication layout for routed experts: per-expert HF slices, or one stacked
    # tensor per layer and projection in the serving engine's fused layout.
    expert_publication: str = "per_expert"
    # Append every scored training group, kept or filtered, to this shared store.
    records_root: str | None = None
    records_responses: str = "off"
    records_response_sample_rate: float | None = None
    # Prompt selection; ReMind uses scored outcomes from inference records.
    prompt_provider: str = "shuffle"
    prompt_mixture: str | None = None
    prompt_priority_field: str | None = None
    prompt_priority_source_field: str | None = None
    prompt_priority_source_weights: dict[str, float] | None = None
    remind_review_fraction: float = 0.02
    remind_enqueue_probability: float = 0.25
    # Optional history/replay layer over the ordinary shuffle or mixture provider.
    prompt_replay_enabled: bool = False
    prompt_replay_fraction: float = 0.1
    prompt_replay_length_drop: float = 0.1
    prompt_replay_window: int = 256
    prompt_replay_max_successes: int = 1
    prompt_replay_truncation: str = "any"
    prompt_replay_answer_probe: bool = False
    # Frozen prompt-exclusion table from `records select`, pinned by SHA-256.
    selection_table: str | None = None
    selection_sha256: str | None = None

    def __post_init__(self):
        _choice(self.reward_reasoning_mode, "core.reward_reasoning_mode", ("required", "none"))
        _boolean(self.prompt_replay_enabled, "core.prompt_replay_enabled")
        _boolean(self.prompt_replay_answer_probe, "core.prompt_replay_answer_probe")
        _number(self.prompt_replay_fraction, "core.prompt_replay_fraction", maximum=1)
        _number(self.prompt_replay_length_drop, "core.prompt_replay_length_drop", maximum=1)
        if self.prompt_replay_length_drop == 1:
            raise _error("core.prompt_replay_length_drop must be below 1")
        _integer(self.prompt_replay_window, "core.prompt_replay_window", minimum=1)
        _integer(self.prompt_replay_max_successes, "core.prompt_replay_max_successes", minimum=0)
        _choice(self.prompt_replay_truncation, "core.prompt_replay_truncation", ("none", "any", "half", "all"))
        _choice(self.prompt_provider, "core.prompt_provider", ("shuffle", "remind", "mixture", "pools", "priority"))
        if self.prompt_priority_field is not None:
            _text(self.prompt_priority_field, "core.prompt_priority_field")
        if self.prompt_priority_source_field is not None:
            _text(self.prompt_priority_source_field, "core.prompt_priority_source_field")
        if self.prompt_priority_source_weights is not None:
            weights = self.prompt_priority_source_weights
            if not isinstance(weights, dict) or not weights:
                raise _error("core.prompt_priority_source_weights must be a nonempty source-to-weight table")
            for name, weight in weights.items():
                _text(name, "core.prompt_priority_source_weights source")
                _number(weight, "core.prompt_priority_source_weights entry", exclusive_min=True)
            _number(sum(weights.values()), "core.prompt_priority_source_weights total", exclusive_min=True)
        _number(self.remind_review_fraction, "core.remind_review_fraction", maximum=1, exclusive_min=True)
        _number(self.remind_enqueue_probability, "core.remind_enqueue_probability", maximum=1)
        if self.max_run_seconds is not None:
            _number(self.max_run_seconds, "core.max_run_seconds", exclusive_min=True)
        _boolean(self.max_run_final, "core.max_run_final")
        _boolean(self.prepared_batch_overlap, "core.prepared_batch_overlap")
        _choice(self.publication_mode, "core.publication_mode", ("barrier", "engine_drain", "refresh"))
        _integer(self.snapshot_capacity, "core.snapshot_capacity", minimum=1)
        for name in ("engine_drain_timeout", "engine_update_timeout", "refresh_request_timeout"):
            _number(getattr(self, name), f"core.{name}")
            if getattr(self, name) <= 0:
                raise _error(f"core.{name} must be positive")
        _integer(self.compiler_cache_max_storage_bytes, "core.compiler_cache_max_storage_bytes", minimum=0)
        _number(
            self.compiler_cache_publish_interval_seconds,
            "core.compiler_cache_publish_interval_seconds",
            exclusive_min=True,
        )
        if self.compiler_cache_root is not None:
            if not isinstance(self.compiler_cache_root, str) or not self.compiler_cache_root:
                raise _error("core.compiler_cache_root must be a nonempty absolute path or unset")
            try:
                _compiler_cache_root(Path(self.compiler_cache_root))
            except ValueError as error:
                raise _error(f"core.compiler_cache_root: {error}") from error
        if self.records_root is not None:
            _text(self.records_root, "core.records_root")
            if not Path(self.records_root).is_absolute():
                raise _error("core.records_root must be an absolute path or unset")
        _choice(self.records_responses, "core.records_responses", RECORD_RESPONSE_MODES)
        if self.selection_table is not None:
            _text(self.selection_table, "core.selection_table")
            if not Path(self.selection_table).is_absolute():
                raise _error("core.selection_table must be an absolute path or unset")
        if self.selection_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", str(self.selection_sha256)):
            raise _error("core.selection_sha256 must be a lowercase 64-character SHA-256 hex digest")
        if self.records_response_sample_rate is not None:
            _number(
                self.records_response_sample_rate, "core.records_response_sample_rate", maximum=1, exclusive_min=True
            )
        _choice(self.attention_backend, "core.attention_backend", ("torch", "flash_2", "flash_3", "flash_4", "te"))
        if self.row_specialization not in ("static", "dynamic"):
            raise _error("core.row_specialization must be static or dynamic")
        if self.expert_publication not in ("per_expert", "fused"):
            raise _error("core.expert_publication must be per_expert or fused")
        for name in ("diagnostic_interval", "scoring_check_interval", "layer_profile_interval"):
            _integer(getattr(self, name), f"core.{name}", minimum=0)
        _number(self.pipeline_observation_interval, "core.pipeline_observation_interval")
        if self.pipeline_observation_interval < 0:
            raise _error("core.pipeline_observation_interval must be nonnegative (0 disables observation)")
        limit = self.max_train_rollout_logprob_abs_diff
        if limit is not None:
            _number(limit, "core.max_train_rollout_logprob_abs_diff")
        _number(self.scoring_check_tolerance, "core.scoring_check_tolerance")
        for name in ("model_config", "reward_config"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), f"core.{name}")
        if self.weight_sync_mode not in ("flattened", "per_tensor"):
            raise _error("core.weight_sync_mode must be flattened or per_tensor")
        for name in (
            "reward_zero_truncated",
            "reward_final_answer_only",
            "exclude_ungraded",
            "filter_zero_std_groups",
            "save_final_checkpoint",
            "compiler_cache",
            "compiler_cache_restore",
            "compiler_cache_diagnostics",
            "compiler_cache_tilelang",
            "replay_diagnostics",
            "stream_moe_export",
            "activation_checkpointing",
            "fp32_lm_head",
            "compile_model",
            "compile_optimizer",
            "use_reduce_scatter",
            "checkpoint_profile",
            "checkpoint_compact_storage",
            "checkpoint_dedup_save_to_lowest_rank",
            "checkpoint_constant_memory_planning",
            "scoring_pass_required",
            "sequence_packing",
            "expert_balanced_packing",
        ):
            _boolean(getattr(self, name), f"core.{name}")
        for name in (
            "checkpoint_thread_count",
            "checkpoint_process_count",
            "checkpoint_keep_last",
            "checkpoint_keep_every",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise _error(f"core.{name} must be a positive integer or unset")
        for name in ("expert_parallel_size", "max_sequence_length", "expert_balance_layer_stride"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise _error(f"core.{name} must be a positive integer")
        _integer(self.expert_balance_search_proposals, "core.expert_balance_search_proposals", minimum=0)
        _number(self.expert_balance_search_seconds, "core.expert_balance_search_seconds")
        if self.packing_max_tokens is not None:
            _integer(self.packing_max_tokens, "core.packing_max_tokens", minimum=1)
        if type(self.max_policy_lag) is not int or self.max_policy_lag < 0:
            raise _error("core.max_policy_lag must be a nonnegative integer")
        _choice(self.router_aux_loss_grouping, "core.router_aux_loss_grouping", ("pack", "sequence"))
        _choice(self.router_aux_count_source, "core.router_aux_count_source", ("executed", "router_selected"))
        for name in ("router_aux_loss_reduction", "router_z_loss_reduction"):
            _choice(getattr(self, name), f"core.{name}", ("token", "response"))
        if self.compile_model and (
            self.router_aux_loss_grouping,
            self.router_aux_loss_reduction,
            self.router_z_loss_reduction,
        ) != ("pack", "token", "token"):
            raise _error("Document router objectives currently require core.compile_model=false")
        for name in ("router_aux_loss_weight", "router_z_loss_weight"):
            value = getattr(self, name)
            _number(value, f"core.{name}")

    def checkpoint_save_options(self):
        """Writer policy, kept separate from model geometry and resume compatibility."""
        return {
            field.name.removeprefix("checkpoint_"): getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name.startswith("checkpoint_")
            and field.name not in {"checkpoint_keep_last", "checkpoint_keep_every"}
            and getattr(self, field.name) is not None
        }
