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
        ({"hf_snapshot_every": 20, "save_interval": 20}, "native checkpoints only"),
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
