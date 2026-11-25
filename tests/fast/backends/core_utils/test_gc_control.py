import gc
import logging

import pytest

from miles.backends.core_utils import gc_control


@pytest.fixture(autouse=True)
def restore_gc_state():
    was_enabled = gc.isenabled()
    callbacks = list(gc.callbacks)
    yield
    gc.callbacks[:] = callbacks
    if was_enabled:
        gc.enable()
    else:
        gc.disable()


def test_the_collector_is_paused_inside_and_restored_after():
    gc.enable()
    with gc_control.automatic_collection_paused():
        assert not gc.isenabled()
    assert gc.isenabled()


def test_the_collector_is_restored_when_the_block_raises():
    gc.enable()
    with pytest.raises(RuntimeError), gc_control.automatic_collection_paused():
        raise RuntimeError("publication failed")
    assert gc.isenabled()


def test_a_collector_disabled_before_stays_disabled():
    gc.disable()
    with gc_control.automatic_collection_paused():
        assert not gc.isenabled()
    assert not gc.isenabled()


def test_the_decorator_pauses_the_collector_for_the_call():
    gc.enable()
    seen = []

    @gc_control.paused_during
    def publish(value):
        seen.append(gc.isenabled())
        return value + 1

    assert publish(1) == 2
    assert seen == [False]
    assert gc.isenabled()
    assert publish.__name__ == "publish"


def test_collect_frees_cycles_and_reports_seconds():
    class Node:
        pass

    a, b = Node(), Node()
    a.other, b.other = b, a
    del a, b
    seconds = gc_control.collect("a test save")
    assert seconds >= 0


def test_a_slow_collection_is_logged(monkeypatch, caplog):
    clock = iter([10.0, 12.5])
    monkeypatch.setattr(gc_control.time, "perf_counter", lambda: next(clock))
    with caplog.at_level(logging.WARNING):
        gc_control._on_collection("start", {"generation": 2})
        gc_control._on_collection("stop", {"generation": 2, "collected": 7, "uncollectable": 0})
    assert "generation 2 took 2.50 s" in caplog.text


def test_a_fast_collection_is_not_logged(monkeypatch, caplog):
    clock = iter([10.0, 10.1])
    monkeypatch.setattr(gc_control.time, "perf_counter", lambda: next(clock))
    with caplog.at_level(logging.WARNING):
        gc_control._on_collection("start", {"generation": 0})
        gc_control._on_collection("stop", {"generation": 0, "collected": 0, "uncollectable": 0})
    assert "Slow garbage collection" not in caplog.text


def test_the_slow_collection_logger_is_installed_once():
    gc_control.log_slow_collections()
    gc_control.log_slow_collections()
    assert gc.callbacks.count(gc_control._on_collection) == 1
