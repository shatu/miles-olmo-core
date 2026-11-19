"""OLMo-core policy at the shared MILES orchestration boundaries."""

import asyncio
import os

from miles.backends.hooks import BackendHooks
from miles.utils.async_utils import run
from miles.utils.function_registry import load_function

# "standalone" (default): miles.backends.core_utils.standalone, Miles' own loop, data
# source and async producer. "open_instruct": Open Instruct's loader and policies.
LOADER_ENV = "MILES_OLMO_CORE_LOADER"


def load_args(*, extra_args_provider):
    loader = os.environ.get(LOADER_ENV, "standalone")
    if loader == "standalone":
        return load_function("miles.backends.core_utils.standalone.load_args")(extra_args_provider=extra_args_provider)
    if loader != "open_instruct":
        raise ValueError(f"{LOADER_ENV} must be standalone or open_instruct, got {loader!r}")
    # Open Instruct is optional for the other training backends.
    return load_function("open_instruct.miles.configuration.arguments.load_core_args")(
        extra_args_provider=extra_args_provider
    )


def configure_rollout(args, user_eval_path: str | None) -> None:
    # Standalone runs keep Miles' fully async producer: the managed producers below
    # drive Open Instruct's data source and cancel unfinished groups at publication.
    if not args.fully_async or getattr(args, "olmo_core_loader", None) == "standalone":
        return
    original_path = args.rollout_function_path
    mode = getattr(getattr(args, "olmo_core", None), "publication_mode", "barrier")
    draining = mode == "engine_drain"
    if mode in ("engine_drain", "refresh") and user_eval_path is not None:
        raise ValueError(f"{mode} requires managed shared-engine evaluation; omit --eval-function-path")
    args.rollout_function_path = (
        "miles.backends.core_utils.rollout.draining_rollout.DrainingRolloutFn"
        if draining
        else (
            "miles.backends.core_utils.rollout.refreshing_rollout.RefreshingRolloutFn"
            if mode == "refresh"
            else "miles.backends.core_utils.rollout.async_rollout.ManagedFullyAsyncRolloutFn"
        )
    )
    if args.eval_function_path == original_path:
        args.eval_function_path = args.rollout_function_path


async def publication_boundary(producer, begin: bool, *, refresh: bool = False, abort: bool = False):
    if abort:
        if not begin or refresh:
            raise ValueError("Abort is a terminal lifecycle boundary, not a refresh or resume")
        return await asyncio.to_thread(run, producer.shutdown(abort=True))
    if refresh:
        method = "begin_refresh" if begin else "end_refresh"
        if not hasattr(producer, method):
            raise RuntimeError("Policy refresh requires the managed refreshing producer")
    else:
        method = "prepare_publication" if begin else "finish_publication"
    if hasattr(producer, method):
        return await asyncio.to_thread(run, getattr(producer, method)())


async def engine_drain(producer, operation: str, *, set_weight_version, **kwargs):
    """Mutate ownership on the producer loop, then update the executor on its loop."""
    if not hasattr(producer, "engine_drain_control"):
        raise RuntimeError("engine drain requires the Open Instruct draining producer")
    result = await asyncio.to_thread(run, producer.engine_drain_control(operation, **kwargs))
    if operation in ("initialize", "step", "publish"):
        # Dequeue lag uses the consuming trainer step; serving versions are per engine.
        set_weight_version(result["optimizer_step"])
    return result


def shutdown_rollout(producer) -> None:
    if hasattr(producer, "shutdown"):
        run(producer.shutdown())


def diagnostic_writer(args):
    return load_function("miles.backends.core_utils.rollout.diagnostic_writer.create_writer")(args)


async def finalize_checkpoint(actor, rollout_id: int) -> None:
    await actor.finalize_checkpoint(rollout_id)


def transform_worker_specs(args, specs):
    # Open Instruct's driver launches its workers itself, with its own serving entry.
    return load_function("miles.backends.core_utils.serving.configure_specs")(args, specs)


_HOOKS = BackendHooks(
    configure_rollout=configure_rollout,
    diagnostic_writer=diagnostic_writer,
    shutdown_rollout=shutdown_rollout,
    finalize_checkpoint=finalize_checkpoint,
    transform_worker_specs=transform_worker_specs,
)


def get_hooks() -> BackendHooks:
    return _HOOKS
