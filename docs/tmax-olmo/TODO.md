# TMax on OLMo: port tracker

This fork runs the TMax terminal-RL pipeline (from `shatu/miles`, branches
`shashank/tmax-terminal-rl` and `shashank/tmax-qwen35-moe`) on OLMo MoE models
with the OLMo-core trainer backend (`--train-backend olmo_core`), driven by
Miles' own training loop rather than open-instruct.

## Branch layout

Three stacked branches on `allenai/miles-olmo-core` `main` at `22d2fb4`:

| Branch | Contains |
|---|---|
| `shashank/upstream-backports` | Upstream Miles commits only, nothing else |
| `shashank/olmo-standalone` | Running the OLMo-core trainer without open-instruct |
| `shashank/tmax-olmo` | TMax: DPPO-TV loss, agent/sandbox code, launchers, this tracker |

- `git log 22d2fb4..shashank/upstream-backports` lists only upstream Miles commits.
- `git log shashank/upstream-backports..shashank/olmo-standalone` lists the open-instruct replacement.
- `git log shashank/olmo-standalone..shashank/tmax-olmo` lists TMax.

### How upstream commits are carried

Each upstream commit is a `git cherry-pick -x`, tagged with miles-olmo-core's
own trailer, `Original-Upstream-Commit: <upstream sha>`. Find the local commit
with `git log --grep "Original-Upstream-Commit: <sha>"`.

- **Exclusions:** paths excluded by `upstream-exclusions.json` are dropped and
  listed in an `Upstream-Exclusions-Dropped:` trailer.
- **Conflicts:** conflict resolutions, and adaptations of miles-olmo-core code
  to upstream API changes, stay inside that commit. Its message explains them.
- **Why not the documented process:** `docs/upstream-updates.md` merges a
  filtered upstream lineage built by `tools/filter_upstream_history.py`. That
  tool refuses any target that does not descend from upstream's v0.1.2 release
  commit `a22fd824`. That commit only exists on `release/v0.1.2`; upstream
  `main` (our base, `e30d9a8ec6`) and the later releases (`release/v0.1.3`,
  `release/v0.2.0`) do not contain it.
- **When miles-olmo-core next merges upstream:** drop each cherry-pick whose
  `Original-Upstream-Commit` the merge already brings.

## Upstream Miles commits ported

Our TMax fork is based on upstream `main` at `e30d9a8ec6` (2026-10-15).
miles-olmo-core is based on upstream v0.1.2 (2026-10-08). Of the 127 upstream
commits in between, these 16 are ported, in upstream order:

| Upstream PR | Upstream SHA | What | Notes |
|---|---|---|---|
| #3987 | `fc5dc4ac8b` | SGLang server keep-alive beyond client expiry | |
| #3366 | `079375ea9e` | Track last fetched rollout id per policy | Resolved against the fork's prepared-batch path |
| #3367 | `c1792d3e58` | `WorkerType` enum | Adapted fork string comparisons in `core_utils/publication` and Rust-router membership |
| #3368 | `a87843494f` | Track prefill KV weight versions on samples | Needs the SGLang bump below |
| #3369 | `a3d09aa7ba` | Pin weight version metrics | |
| #3370 | `0aa44b6523` | Report prompt KV staleness | |
| #3371 | `fdff1264c2` | Read old rollout dumps | |
| #3372 | `6bb808d657` | Namespaced radix cache by rollout call | Excluded paths dropped; change to the relocated `miles/rollout/external_eval_fn.py` applied |
| #3373 | `67f8e56998` | Namespaced radix cache for fully-async and session servers | |
| #3374 | `6473a2e93c` | Assert prompt KV freshness in in-place modes | Resolved; excluded paths dropped |
| #3375 | `c0fe8f3faa` | Pause/continue/flush fan-outs on the inference controller | |
| #3376 | `f919d9610b` | Driver-owned generation pause | Fork trainer actor now calls `maybe_pause_engines` |
| #3963 | `fb96e2b46f` | Session: turn args by reference | |
| #3967 | `0d7e2abd5c` | Session: less per-turn CPU | Resolved (docstring, requirements) |
| #3972 | `2000d7a978` | Session: aiohttp proxy | |
| #3973 | `ef8de80272` | Session: less repeated work | Base64 speedup applied in the fork's `decode_int32_buffer` |

Fork tests that used APIs these commits changed were updated inside the
commit that changed the API:

- #3366: prepared-batch fake executor;
- #3367: `WorkerType` in router-readiness tests;
- #3368: `WeightVersionsPerCall.from_dict`/`to_dict`;
- #3373: reservation-test producer fake.

**Behavior change for miles-olmo-core's own async producers:** #3373 defaults
`--namespaced-radix-cache` on for `--fully-async` runs with
`--pause-generation-mode in_place`. That now includes the fork's managed and
refresh producers. Their SGLang requests carry an `extra_key` per rollout,
which partitions the prefix cache by rollout. The pinned SGLang supports
`extra_key`. Pass `--no-namespaced-radix-cache` to keep the fork's previous
caching.

## Running the OLMo-core trainer without open-instruct

Three commits on `shashank/olmo-standalone`:

1. **`miles/backends/core_utils/standalone_config.py`:** a verbatim copy of
   open-instruct's `CoreConfig` (`open_instruct/miles/configuration/config.py`
   at `28c181666`), with its validation helpers inlined. The 83 fields, their
   order, types, defaults and checks match the source.
   `tests/fast/backends/core_utils/test_standalone_loader.py` pins the field
   list, so resync both together when the OLMo team changes a field.
2. **`miles/backends/core_utils/standalone.py`**, selected by
   `integration.load_args` unless `MILES_OLMO_CORE_LOADER=open_instruct`.
   - **Miles' own pieces:** the loop (`train.py`, or `train_async.py` with
     `--fully-async`), data source, reward function, `FullyAsyncRolloutFn` and
     default async buffer. Groups keep generating across weight updates, with
     nothing cancelled at publication. A group is dropped only when its oldest
     weight version is more than `--max-weight-staleness` behind.
   - **Settings:** `--olmo-core-config` takes CoreConfig fields as JSON or
     `@path`. Fields that only take effect through open-instruct's driver, data
     source, reward adapter or inference recorder are rejected.
   - **Checks:** it rejects what the trainer cannot run: TP/PP/CP > 1, critic,
     offload, micro-batching, refresh and engine-drain publication.
   - **Derived settings:** `max_policy_lag` is derived (staleness plus optimizer
     steps per rollout, minus 1). `--qkv-format` follows `sequence_packing`
     (`bshd` unless packing), because Miles' default `thd` does not fit.
   - **Verified:** Miles' real `parse_args` with `--train-backend olmo_core`
     works for both sync and fully-async runs without importing open-instruct.
3. **`docker/olmo/Dockerfile` and `build_image.sh`:** open-instruct's pinned
   binary base (Beaker `01M5ZYQZVF7DDJ1VR619KRSS42`), with OLMo-core
   `903e57d`, olmo-sglang `069dc0a3`, SGLang `7b6993a3` and Megatron at the
   open-instruct pins, plus this repository's Miles.
   - **Build-time checks:** the olmo-sglang plugin loads on the bumped SGLang;
     sglang-kernel is 0.4.7; `--enable-prefill-weight-versions` is registered;
     open-instruct is absent.
   - **Built locally:** `localhost/miles-olmo:a258dbbc56`; not uploaded yet.

**Staleness units:** with the OLMo-core trainer, a weight version is a count of
completed optimizer steps, so `--max-weight-staleness` is in optimizer steps. In
the Megatron TMax runs it counted weight updates, one per rollout. TMax's
staleness of 4 becomes 4 × (optimizer steps per rollout) here.

## TMax commits

On `shashank/tmax-olmo`, after this tracker:

- **Miles core, cherry-picked one-to-one with `-x`:**
  - DPPO-TV policy loss (`aecf2f49`);
  - Ray head lookup through the GCS (`ef3dcd48`);
  - `MILES_SESSION_REQUEST_TIMEOUT_S` (`9fcfa58e`).
- **Build:** the Docker image pins (`16429689`, `d0831a47`).
- **Squashed:**
  - the `examples/tmax` tree and its tests;
  - the TMax and Deep Dive docs.

  Both are squashed from `shashank/tmax-terminal-rl` (`60a0111841`) plus
  `shashank/tmax-qwen35-moe` (`a51bd0a27b`), and carry `Ported-From` trailers.
- **Ray boundary test:** the TMax example commit also lists the two TMax
  modules that import `ray` in `tests/fast/utils/workers/test_ray_communication_boundary.py`.

## Test status (CPU, `tests/fast`)

Run with `/root/venvs/miles-olmo-core-test`: SGLang `b84bec57` plus Miles
requirements, on local disk. Install with `UV_CACHE_DIR` on local disk and
`SGLANG_BUILD_RUST_EXTS=none`. The suite was run on `shashank/tmax-olmo` and on
unmodified `22d2fb4`.

Failing on `shashank/tmax-olmo` but not on `22d2fb4`, in a serial rerun: only
`tests/fast/maintenance/test_upstream_exclusions.py::test_manifest_covers_removed_files_but_not_itself`.
It asserts that `examples/`, `docs/examples/` and `tests/fast/examples/` do not
exist, and TMax lives there by decision (see below).

The remaining failures are the same on both trees: Megatron is absent from the
test venv. Under `-n 24` a few worker e2e and timing tests flake on either tree.

## Blocking any run

- [x] **Bump SGLang** from `b84bec57` to `7b6993a3` (our TMax pin; 9 commits;
  `b84bec57` is its ancestor), in `docker/olmo/Dockerfile`. Code ported from
  #3368 onward reads `args.sglang_enable_prefill_weight_versions` for every
  sample, and that server flag only exists from SGLang #44164 (`f312781649`).
  The image build confirms the olmo-sglang plugin still loads.
- [ ] **Upload the OLMo runtime image** to Beaker
  (`build_image.sh --upload`); it is built locally only.
- [ ] **TMax launcher for the OLMo-core trainer:** OLMo MoE checkpoint,
  `--olmo-core-config`, staleness in optimizer steps, episodes of at most
  65,536 tokens, gateway sandboxes.
- [x] **CPU unit tests** on the ported commits (see Test status).
- [x] **Decided (2026-11-18): TMax stays under `examples/tmax`.** It keeps the
  same import paths, config strings and doc links as `shatu/miles`. The
  exclusion-manifest test therefore stays red in this fork, by decision. Moving
  TMax would only be needed before proposing it to allenai/miles-olmo-core,
  whose upstream filter excludes `examples/`.

## Deferred: needed for async

We will need async. Each item below blocks fully-async TMax on this fork.

- [ ] **Upstream fully-async buffer, checkpoint and sample-ownership rework,
  #3379–#3425 (43 commits).**
  - Our TMax async runs used it, e.g. #3380 retry ownership, #3409/#3412
    whole-batch takes, #3416 async state checkpoint/restore.
  - Expect conflicts: miles-olmo-core's `core_utils/rollout/async_buffer.py`
    subclasses v0.1.2 buffer internals (`_capacity`, `_buffer`, `_cond`,
    `on_dequeue`). `async_rollout.py` and checkpoint code also overlap.
- [ ] **Upstream rollout-worker event loops, #4082–#4087 (5 commits).** These
  overlap the fork's `miles/ray/rollout` changes.
- [x] **Keep Miles' `FullyAsyncRolloutFn`** for `olmo_core`
  (`olmo-standalone`; the standalone loader's default). Before this, `core_utils/integration.configure_rollout` swaps in
  the OLMo team's managed producer. With open-instruct's driver, that producer
  cancels every in-flight group at each weight publication.
- [x] **Use Miles' default async buffer** (`olmo-standalone`). It drops a group
  only when its oldest weight version is too old. open-instruct's
  `load_core_args` installs `HomogeneousPolicyDataBuffer`, which drops any
  group spanning two weight versions.
- [x] **Set the trainer's lag limit.** It raises rather than drops. The
  standalone loader derives `max_policy_lag` as `max_weight_staleness` plus
  optimizer steps per rollout minus 1, and rejects a smaller explicit value.
- [ ] **Decide how in-place weight updates are cached.**
  `core_utils/publication/policy_refresh.py` allows in-place pauses only with
  `disable_radix_cache=true` or the `olmo_policy_versioned_v1` plugin (one GPU
  per engine). Decide whether the now-ported namespaced radix cache (#3372,
  #3373) is safe for KDA recurrent state in olmo-sglang, or use `retract`.
- [ ] **Test on GPUs:** in-flight updates with KDA recurrent state across
  multi-turn episodes, routing replay (R3) under in-place updates, and staleness
  metrics. The OLMo team's in-place tests were single-turn.
- [ ] **Ask the OLMo team** whether they would take a multi-turn in-flight mode
  upstream, or prefer extending their refresh mode. Refresh today requires gap-free
  per-token version spans, so tool-output tokens break it.

## Deferred: other

- [ ] **#3974, #3975** (`/samples` reply into one buffer, decoded as views):
  speedup only. They replace `post_bytes_no_retry`, which the fork's reworked
  `miles/utils/http_utils.py` builds on (observers, keep-alive limits, route
  decoding).
- [ ] **#4099** (divide per-token loss by the step's DP-total token count): do
  not port as is. The OLMo-core trainer's loss call does not pass the step token
  count, so the new assertion would fail.
- [ ] **#3377** (`--fully-async` with `--colocate`): only if wanted.
- [ ] **Context above 65,536 tokens:** the OLMo-core trainer has no context
  parallelism. `kevinfarhat/a4b-sft-speed` in open-instruct adds Ulysses CP to
  the same train module for SFT. The Olmo 3.5 12.5@0.8 model's limit is 65,536.
- [ ] **OLMo TITO mode and tool-call parser:** depend on the target checkpoint:
  A4B? single-token tags from open-instruct's `tooltag-atomic` branch?
  `<function_calls>` or `<tool_call>`?

Not ported, because the OLMo-core trainer does not use them:
- peer-to-peer weight transfer (10 commits);
- Megatron/FSDP/MTP/LoRA/FP8/DeepSeek-V4 (9);
- docs/CI-only (35);
- #4021, #3965, #3920, #3921;
- #4031: miles-olmo-core enables the FP32 serving head itself.
