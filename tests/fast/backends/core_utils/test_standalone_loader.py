"""The OLMo-core trainer's standalone loader: config copy, checks and producer choice."""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from miles.backends.core_utils import integration, standalone
from miles.backends.core_utils.standalone_config import CoreConfig, InputError

# Open Instruct's CoreConfig at 28c181666, in order. A mismatch means the copy was edited
# or needs a resync; update standalone_config.py and its header together with this list.
OPEN_INSTRUCT_FIELDS = (
    "reward_zero_truncated reward_final_answer_only reward_reasoning_mode exclude_ungraded max_run_seconds "
    "max_run_final filter_zero_std_groups max_train_rollout_logprob_abs_diff diagnostic_interval "
    "layer_profile_interval pipeline_observation_interval replay_diagnostics stream_moe_export weight_sync_mode "
    "publication_mode prepared_batch_overlap engine_drain_timeout engine_update_timeout refresh_request_timeout "
    "snapshot_capacity row_specialization compiler_cache compiler_cache_root compiler_cache_restore "
    "compiler_cache_diagnostics compiler_cache_tilelang compiler_cache_max_storage_bytes "
    "compiler_cache_publish_interval_seconds checkpoint_profile checkpoint_thread_count checkpoint_process_count "
    "checkpoint_compact_storage checkpoint_dedup_save_to_lowest_rank checkpoint_constant_memory_planning "
    "save_final_checkpoint checkpoint_keep_last checkpoint_keep_every model_config reward_config "
    "expert_parallel_size attention_backend fp32_lm_head activation_checkpointing compile_model compile_optimizer "
    "use_reduce_scatter max_sequence_length sequence_packing expert_balanced_packing expert_balance_layer_stride "
    "expert_balance_search_proposals expert_balance_search_seconds packing_max_tokens max_policy_lag "
    "router_aux_loss_grouping router_aux_loss_reduction router_z_loss_reduction router_aux_count_source "
    "router_aux_loss_weight router_z_loss_weight scoring_pass_required scoring_check_interval "
    "scoring_check_tolerance expert_publication records_root records_responses records_response_sample_rate "
    "prompt_provider prompt_mixture prompt_priority_field prompt_priority_source_field "
    "prompt_priority_source_weights remind_review_fraction remind_enqueue_probability prompt_replay_enabled "
    "prompt_replay_fraction prompt_replay_length_drop prompt_replay_window prompt_replay_max_successes "
    "prompt_replay_truncation prompt_replay_answer_probe selection_table selection_sha256"
).split()


def _args(**overrides):
    # Two trainer GPUs; a rollout collects 8 samples, two optimizer steps of 4.
    values = dict(
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        global_batch_size=4,
        rollout_batch_size=4,
        n_samples_per_prompt=2,
        fully_async=False,
        colocate=True,
        max_weight_staleness=None,
        update_weights_interval=1,
    )
    return SimpleNamespace(**(values | overrides))


def test_core_config_copy_keeps_open_instruct_fields_in_order():
    assert [field.name for field in dataclasses.fields(CoreConfig)] == OPEN_INSTRUCT_FIELDS


def test_checkpoint_writer_options_omit_retention_and_unset_values():
    options = CoreConfig(checkpoint_keep_last=2, checkpoint_thread_count=4).checkpoint_save_options()
    assert options == {
        "profile": False,
        "thread_count": 4,
        "compact_storage": True,
        "dedup_save_to_lowest_rank": False,
        "constant_memory_planning": True,
    }


def test_read_core_config_accepts_inline_json_and_a_file(tmp_path):
    assert standalone.read_core_config('{"max_sequence_length": 32768}') == {"max_sequence_length": 32768}
    path = tmp_path / "core.json"
    path.write_text(json.dumps({"fp32_lm_head": True}))
    assert standalone.read_core_config(f"@{path}") == {"fp32_lm_head": True}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("[1]", "JSON object of CoreConfig fields"),
        ("{", "JSON object or @path"),
        ('{"max_seq_len": 1}', "unknown CoreConfig fields: max_seq_len"),
        ('{"filter_zero_std_groups": false}', "only take effect under Open Instruct"),
        ('{"prompt_provider": "remind"}', "only take effect under Open Instruct"),
    ],
)
def test_read_core_config_rejects_bad_and_open_instruct_only_fields(value, message):
    with pytest.raises(InputError, match=message):
        standalone.read_core_config(value)


def test_sync_lag_covers_the_optimizer_steps_of_one_rollout():
    core = standalone.resolve_core_config(_args(), {})
    assert core.max_policy_lag == 1
    # The startup compiler cache needs Open Instruct's driver.
    assert core.compiler_cache is False


def test_async_lag_adds_staleness_to_the_steps_within_a_rollout():
    args = _args(fully_async=True, colocate=False, max_weight_staleness=4)
    assert standalone.resolve_core_config(args, {}).max_policy_lag == 5
    assert standalone.resolve_core_config(args, {"max_policy_lag": 8}).max_policy_lag == 8
    with pytest.raises(InputError, match="below the 5 optimizer steps"):
        standalone.resolve_core_config(args, {"max_policy_lag": 4})


@pytest.mark.parametrize(
    ("overrides", "fields", "message"),
    [
        ({"fully_async": True, "colocate": False}, {}, "needs --max-weight-staleness"),
        ({"fully_async": True, "colocate": True, "max_weight_staleness": 2}, {}, "disaggregated serving"),
        ({}, {"publication_mode": "refresh"}, "supports barrier publication"),
        ({"context_parallel_size": 2}, {}, "context_parallel_size>1"),
        ({"use_critic": True}, {}, "--use-critic"),
        ({"global_batch_size": 3}, {}, "multiple of the trainer world size 2"),
        ({"global_batch_size": 6}, {}, "complete optimizer batches"),
        ({"micro_batch_size": 2}, {}, "--micro-batch-size 1"),
        ({}, {"sequence_packing": True}, "packing-capable core.attention_backend"),
    ],
)
def test_resolve_rejects_what_the_trainer_cannot_run(overrides, fields, message):
    with pytest.raises(InputError, match=message):
        standalone.resolve_core_config(_args(**overrides), fields)


def test_qkv_layout_follows_sequence_packing_unless_chosen_explicitly():
    args = _args(qkv_format="thd")  # Miles' default
    standalone.resolve_core_config(args, {})
    assert args.qkv_format == "bshd"
    args = _args(qkv_format="bshd")
    standalone.resolve_core_config(args, {"sequence_packing": True, "attention_backend": "flash_4"})
    assert args.qkv_format == "thd"
    with pytest.raises(InputError, match="--qkv-format bshd"):
        standalone.resolve_core_config(_args(qkv_format="thd"), {}, {"--qkv-format"})


def test_loader_defaults_to_standalone_and_open_instruct_is_opt_in(monkeypatch):
    calls = []
    monkeypatch.setattr(integration, "load_function", lambda path: lambda **kwargs: calls.append(path))
    monkeypatch.delenv(integration.LOADER_ENV, raising=False)
    integration.load_args(extra_args_provider=None)
    monkeypatch.setenv(integration.LOADER_ENV, "open_instruct")
    integration.load_args(extra_args_provider=None)
    assert calls == [
        "miles.backends.core_utils.standalone.load_args",
        "open_instruct.miles.configuration.arguments.load_core_args",
    ]
    monkeypatch.setenv(integration.LOADER_ENV, "other")
    with pytest.raises(ValueError, match="standalone or open_instruct"):
        integration.load_args(extra_args_provider=None)


@pytest.mark.parametrize("loader", ["standalone", None])
def test_only_open_instruct_runs_swap_in_the_managed_async_producer(loader):
    path = "miles.rollout.fully_async_rollout.FullyAsyncRolloutFn"
    args = SimpleNamespace(
        fully_async=True,
        olmo_core=CoreConfig(),
        olmo_core_loader=loader,
        rollout_function_path=path,
        eval_function_path=path,
    )
    integration.configure_rollout(args, None)
    expected = (
        path
        if loader == "standalone"
        else "miles.backends.core_utils.rollout.async_rollout.ManagedFullyAsyncRolloutFn"
    )
    assert args.rollout_function_path == expected


def test_episode_cap_defaults_to_the_trainer_context_and_cannot_exceed_it():
    args = _args()
    standalone.resolve_core_config(args, {"max_sequence_length": 32768})
    assert args.max_seq_len == 32768
    args = _args(max_seq_len=16384)
    standalone.resolve_core_config(args, {"max_sequence_length": 32768})
    assert args.max_seq_len == 16384
    with pytest.raises(InputError, match="exceeds core.max_sequence_length=8192"):
        standalone.resolve_core_config(_args(max_seq_len=16384), {})


def test_load_without_a_committed_checkpoint_starts_fresh(tmp_path):
    args = SimpleNamespace(load=str(tmp_path), start_rollout_id=None)
    standalone.resolve_resume(args)
    assert args.load is None and args.start_rollout_id is None


def test_load_with_a_committed_checkpoint_resumes_at_its_next_rollout(tmp_path, monkeypatch):
    (tmp_path / "core-latest.json").write_text("{}")
    monkeypatch.setattr(standalone, "_resume_manifest", lambda root: {"clock": {"next_rollout_id": 7}})
    args = SimpleNamespace(load=str(tmp_path), start_rollout_id=None)
    standalone.resolve_resume(args)
    assert args.load == str(tmp_path) and args.start_rollout_id == 7
    args = SimpleNamespace(load=str(tmp_path), start_rollout_id=3)
    standalone.resolve_resume(args)
    assert args.start_rollout_id == 3
