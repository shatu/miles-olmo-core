"""--debug-rollout-only with the OLMo-core trainer: the trainer builds and publishes nothing, as Megatron's does."""

from types import SimpleNamespace

import pytest

pytest.importorskip("olmo_core")

from miles.backends.core_utils import actor  # noqa: E402
from miles.backends.training_utils.types import TrainStepOutcome  # noqa: E402


def _refuse(*_args, **_kwargs):
    raise AssertionError("a rollout-only trainer must not touch the model, data or checkpoints")


@pytest.fixture
def rollout_only(monkeypatch):
    calls = []
    monkeypatch.setattr(actor.TrainRayActor, "_init_common", lambda self, args, role, **kwargs: calls.append(role))
    monkeypatch.setattr(actor.dist, "get_world_size", lambda: 4)
    for target, name in (
        (actor.models, "build_train_module"),
        (actor.miles_data, "get_rollout_data"),
        (actor.checkpoint, "save"),
        (actor.checkpoint, "finalize"),
        (actor.checkpoint, "restore"),
        (actor.policy_refresh, "validate_pause_mode"),
    ):
        monkeypatch.setattr(target, name, _refuse)
    instance = object.__new__(actor.OLMoCoreTrainRayActor)
    instance._heartbeat = SimpleNamespace(bump=lambda: None)
    instance.args = SimpleNamespace(debug_rollout_only=True)
    return instance, calls


def test_rollout_only_trainer_builds_nothing(rollout_only):
    instance, calls = rollout_only
    assert instance.init(instance.args, "actor") == 0
    assert calls == ["actor"]
    assert instance.weight_updater is None
    assert instance.get_train_parallel_config() == {"dp_size": 4}


def test_rollout_only_trainer_trains_saves_and_publishes_nothing(rollout_only):
    instance, _ = rollout_only
    assert instance.train(0, object()).outcome == TrainStepOutcome.NORMAL
    assert instance.save_model(0) is None
    assert instance.finalize_checkpoint(0) is None
    assert instance.update_weights(object()) is None
    instance.close_weight_transport()
