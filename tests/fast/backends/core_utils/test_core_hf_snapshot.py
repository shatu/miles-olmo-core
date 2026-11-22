"""--save-hf with the OLMo-core trainer: an HF export after each native save, finished once even across retries."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("olmo_core")

from miles.backends.core_utils import actor  # noqa: E402
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER  # noqa: E402


@pytest.fixture
def trainer(monkeypatch, tmp_path):
    saves, exports = [], []
    monkeypatch.setattr(actor.checkpoint, "save", lambda self, rollout_id: saves.append(rollout_id))
    monkeypatch.setattr(actor.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor.dist, "broadcast_object_list", lambda objects, src: None)

    def export_hf(rollout_id, path):
        target = Path(path)
        target.mkdir(parents=True, exist_ok=False)  # as the real export: an existing directory is an error
        (target / "model.safetensors").write_text("weights")
        (target / HF_EXPORT_COMPLETE_MARKER).touch()
        exports.append((rollout_id, path))

    instance = object.__new__(actor.OLMoCoreTrainRayActor)
    instance.export_hf = export_hf
    instance.args = SimpleNamespace(debug_rollout_only=False, save_hf=str(tmp_path / "hf" / "step_{rollout_id}"))
    return instance, saves, exports, tmp_path / "hf"


def test_each_save_writes_the_native_checkpoint_then_the_hf_snapshot(trainer):
    instance, saves, exports, hf = trainer
    instance.save_model(4)
    instance.save_model(9)
    assert saves == [4, 9]
    assert exports == [(4, str(hf / "step_4")), (9, str(hf / "step_9"))]


def test_a_retry_keeps_a_finished_snapshot_and_rewrites_an_unfinished_one(trainer):
    instance, saves, exports, hf = trainer
    instance.save_model(4)
    (hf / "step_9").mkdir(parents=True)
    (hf / "step_9" / "model.safetensors").write_text("cut off by a preemption")
    instance.save_model(4)
    instance.save_model(9)
    assert saves == [4, 4, 9]
    assert exports == [(4, str(hf / "step_4")), (9, str(hf / "step_9"))]
    assert (hf / "step_9" / "model.safetensors").read_text() == "weights"


def test_without_save_hf_only_the_native_checkpoint_is_written(trainer):
    instance, saves, exports, _ = trainer
    instance.args.save_hf = None
    instance.save_model(4)
    assert saves == [4] and exports == []
