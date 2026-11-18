# syntax=docker/dockerfile:1
# The OLMo-core runtime for this fork, without Open Instruct.
#
# It mirrors Open Instruct's runtime/miles/Dockerfile (allenai/open-instruct 28c181666):
# - the same pinned binary base (torch 2.13, FA2/FA3/FA4, sglang-kernel 0.4.7);
# - OLMo-core, olmo-sglang, SGLang and Megatron installed editable from source at
#   pinned commits;
# - this repository's Miles from the build context.
#
# Build it with docker/olmo/build_image.sh.
ARG BASE_IMAGE=open-instruct-miles-binary-base:20261027
FROM ${BASE_IMAGE}

ARG OLMO_CORE_REPO=https://github.com/allenai/Olmo-core.git
ARG OLMO_CORE_COMMIT=903e57d46cc2c62cb5611e49c4d0e2052eb798d2
ARG OLMO_SGLANG_REPO=https://github.com/allenai/olmo-sglang.git
ARG OLMO_SGLANG_COMMIT=069dc0a3cf0e28bd9f03d035aa98fc2bd5e8896a
# Open Instruct pins b84bec57. 7b6993a3 is that commit plus 9 sglang-miles commits;
# among them, --enable-prefill-weight-versions (SGLang #44164), which the upstream
# Miles weight-version tracking in this fork reads for every sample. torch,
# flashinfer and sglang-kernel are unchanged, so the binary base still fits.
ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG SGLANG_COMMIT=7b6993a310d0dbb67929b51582d1729a6295e417
ARG MEGATRON_LM_REPO=https://github.com/radixark/Megatron-LM.git
ARG MEGATRON_LM_COMMIT=235952df607b3820716e5e67728a5ab470ca33ae
ARG MEGATRON_BRIDGE_REPO=https://github.com/radixark/Megatron-Bridge.git
ARG MEGATRON_BRIDGE_COMMIT=8cd3466d14d2337c8492827b3712482c2b3e4866

ENV MILES_SOURCES=/opt/miles-olmo/sources
RUN set -eu; \
    fetch() { \
        git init -q "$MILES_SOURCES/$1"; \
        git -C "$MILES_SOURCES/$1" fetch -q --depth=1 "$2" "$3"; \
        git -C "$MILES_SOURCES/$1" checkout -q --detach FETCH_HEAD; \
        test "$(git -C "$MILES_SOURCES/$1" rev-parse HEAD)" = "$3"; \
    }; \
    fetch olmo-core "$OLMO_CORE_REPO" "$OLMO_CORE_COMMIT"; \
    fetch olmo-sglang "$OLMO_SGLANG_REPO" "$OLMO_SGLANG_COMMIT"; \
    fetch sglang "$SGLANG_REPO" "$SGLANG_COMMIT"; \
    fetch megatron-lm "$MEGATRON_LM_REPO" "$MEGATRON_LM_COMMIT"; \
    fetch megatron-bridge "$MEGATRON_BRIDGE_REPO" "$MEGATRON_BRIDGE_COMMIT"

COPY . ${MILES_SOURCES}/miles

# Install metadata and entry points only; the base image already has every binary
# dependency, and the source overlay does not load SGLang's optional Rust extensions.
RUN SGLANG_BUILD_RUST_EXTS=none UV_LINK_MODE=copy uv pip install --system --no-deps \
    -e ${MILES_SOURCES}/olmo-core -e ${MILES_SOURCES}/miles \
    -e ${MILES_SOURCES}/olmo-sglang -e ${MILES_SOURCES}/sglang/python \
    -e ${MILES_SOURCES}/megatron-lm -e ${MILES_SOURCES}/megatron-bridge \
    && rm -rf /root/.cache/uv
# Native launchers that prepend the historical Megatron location still find it.
RUN mkdir -p /src && ln -sT ${MILES_SOURCES}/megatron-lm /src/Megatron-LM

# Job entry scripts pip-install small runtime clients (e.g. the TMax sandbox client) into the system Python,
# which Ubuntu marks externally managed (PEP 668); this image is a single-purpose environment.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

ENV PYTHONPATH=${MILES_SOURCES}/sglang/python:${MILES_SOURCES}/olmo-sglang/src:${MILES_SOURCES}/olmo-core/src:${MILES_SOURCES}/miles:${MILES_SOURCES}/megatron-lm:${MILES_SOURCES}/megatron-bridge/src
WORKDIR ${MILES_SOURCES}/miles
RUN python -m compileall -q ${MILES_SOURCES}/olmo-core/src/olmo_core ${MILES_SOURCES}/miles/miles

# The serving stack this fork expects: the OLMo SGLang plugin, the binary kernel the
# base ships, and the prefill-weight-version flag from the bumped SGLang.
RUN python -c "import olmo_sglang; from olmo_sglang.attention import PerHeadRMSNorm; assert olmo_sglang.__file__.startswith('${MILES_SOURCES}/olmo-sglang/src/')"
RUN python -c "import sglang, sglang.srt.entrypoints.engine; from importlib.metadata import version; assert sglang.__file__.startswith('${MILES_SOURCES}/sglang/python/'); assert version('sglang-kernel') == '0.4.7'"
# Miles registers SGLang's server flags through ServerArgs.add_cli_args; check the flag the same way.
RUN python -c "import argparse; from sglang.srt.server_args import ServerArgs; p = argparse.ArgumentParser(); ServerArgs.add_cli_args(p); assert 'enable_prefill_weight_versions' in {a.dest for a in p._actions}"
# The OLMo-core trainer loads without Open Instruct.
RUN python -c "import importlib.util, miles.backends.core_utils.standalone, olmo_core; assert importlib.util.find_spec('open_instruct') is None"
