"""Optional Core actor integration; protocol tests do not require OLMo-core."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("olmo_core")

from miles.backends.core_utils import actor


def test_actor_close_delegates_transport_cleanup():
    instance = object.__new__(actor.OLMoCoreTrainRayActor)
    instance.weight_updater = SimpleNamespace(close=Mock())
    instance.close_weight_transport()
    instance.weight_updater.close.assert_called_once_with()


def test_actor_dispose_retires_the_weight_transport_before_its_process_groups(monkeypatch):
    calls = []
    instance = object.__new__(actor.OLMoCoreTrainRayActor)
    instance.weight_updater = SimpleNamespace(close=lambda: calls.append("close_weight_transport"))
    monkeypatch.setattr(actor.TrainRayActor, "dispose", lambda self: calls.append("destroy_process_groups"))
    instance.dispose()
    assert calls == ["close_weight_transport", "destroy_process_groups"]
