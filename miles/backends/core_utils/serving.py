"""Register olmo-sglang's KDA backend before SGLang resolves its server arguments.

SGLang imports an external model package (``SGLANG_EXTERNAL_MODEL_PACKAGE``) with its model registry, which
``python -m sglang.launch_server`` loads only after argument resolution has run the radix capability check. Until
olmo-sglang registers its KDA linear-attention spec, an Olmo hybrid checkpoint is not a recurrent-state model to
SGLang: the branch-capable ``extra_buffer`` radix strategy is refused ("needs mamba state"), and the radix cache
cannot snapshot KDA state. Open Instruct's driver launches engines through its own entry for this
(open_instruct/miles/execution/serving.py); this is the standalone loader's.
"""

import importlib
import runpy
import shlex
from functools import partial

MODULE = "miles.backends.core_utils.serving"


def configure_specs(args, specs):
    """Launch every inference engine through this module instead of ``sglang.launch_server``."""
    return [
        (
            spec.model_copy(update={"launch_command": partial(_serving_command, spec)})
            if spec.name.startswith("inference-engine-")
            else spec
        )
        for spec in specs
    ]


def _serving_command(spec, context):
    command = shlex.split(spec.launch_command(context))
    module_index = command.index("-m") + 1
    if command[module_index] != "sglang.launch_server":
        raise ValueError(f"Expected a SGLang module launch for the KDA serving hook, got {command[module_index]}")
    command[module_index] = MODULE
    return shlex.join(command)


def main():
    importlib.import_module("olmo_sglang.kda.backend").register_olmo_kda_backend()
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
