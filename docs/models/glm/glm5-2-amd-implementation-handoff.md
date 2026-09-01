---
title: GLM-5.2 AMD implementation handoff
description: Engineering status, design decisions, verification, and first-GPU procedure for GLM-5.2 on MI350X and MI355X.
---

## Purpose and repository state

This document hands off the DeepSeek-V4-Flash-to-GLM-5.2 AMD implementation and
hardening work started on 2026-08-31. The goal is a correct and reasonably efficient
full-parameter GLM-5.2 GRPO run on MI350X or MI355X, with the shortest possible debugging
cycle when a 64-GPU allocation becomes available.

The work is **not a support claim**. No physical MI350X or MI355X was available for this
implementation pass. Static checks and CPU tests can remove orchestration and shape bugs,
but they cannot establish ROCm kernel numerics, memory fit, RCCL stability, SGLang graph
capture, or end-to-end convergence.

At this checkpoint:

- repository: `radixark/miles`;
- branch: `main`;
- base commit: `542213b96264e818f2dca79356bf531aa1cea1f0`;
- all changes are uncommitted in the shared worktree;
- the full AMD acceptance target is 8 nodes x 8 GPUs, separately qualified on MI350X and
  MI355X;
- the detailed first-allocation procedure is in the
  [AMD bring-up runbook](glm5-2-amd-bringup.md).

Do not discard or bulk-reset the worktree. It contains changes across the model path,
launcher, kernels, validation infrastructure, image, tests, snapshots, and documentation.

## What was learned from DeepSeek V4 Flash

DeepSeek V4 Flash is the strongest existing AMD training reference in this repository. Its
AMD path has a reported 4-node x 8-GPU MI355X validation, whereas full GLM-5.2 has no AMD
run. The implementation was traced through the launcher, checkpoint conversion, model
specification, forward pass, custom sparse-attention backward, MoE routing replay, weight
update, and SGLang rollout path.

The relevant files are:

- [`run_deepseek_v4.py`](../../../scripts/amd/run_deepseek_v4.py): AMD topology and runtime
  recipe;
- [`deepseek_v4`](../../../miles_plugins/models/deepseek_v4): compressor, indexer, mHC,
  sparse attention, MoE, and QAT model path;
- [`tilelang_sparse_mla_fwd.py`](../../../miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_fwd.py)
  and
  [`tilelang_sparse_mla_bwd.py`](../../../miles_plugins/models/deepseek_v4/ops/kernel/tilelang_sparse_mla_bwd.py):
  custom sparse-MLA forward and backward;
- [`test_v4_tilelang_sparse_mla.py`](../../../tests/manual/models/deepseek_v4/test_v4_tilelang_sparse_mla.py):
  GPU kernel validation.

The forward/backward boundary is:

1. Project and compress Q/KV.
2. Compute or replay sparse index selections.
3. Run the TileLang sparse MLA forward and save Q, KV, selected indices, output, and LSE.
4. Apply the value/output projections, mHC residual path, and MoE blocks.
5. In backward, derive softmax delta from output and upstream gradient, then compute dQ and
   atomically accumulate dKV in the custom TileLang kernel.
6. Let autograd and Transformer Engine propagate those gradients through the absorbed
   projections and delayed weight-gradient path.

Two latent DeepSeek issues were fixed while using it as the reference:

- padded sparse indices use `-1`; the old kernels could read or atomically write before the
  tensor. The forward and backward now clamp the address and substitute exact zeros for
  invalid slots;
- full-model single-node checkpoint conversion requested PP1 but the conversion utility
  could auto-promote it. The launcher now explicitly preserves PP1 for that exact profile.

These fixes have CPU/static and snapshot coverage, but their GPU poison-memory test still
needs to run on ROCm.

## GLM-5.2 architecture map

The authoritative model definition is
[`glm5.2-744B-A40B.py`](../../../scripts/models/glm5.2-744B-A40B.py). The pinned full
checkpoint describes:

| Property | Value |
|---|---:|
| Transformer layers | 78 |
| Dense / MoE layers | 3 / 75 |
| Hidden size | 6144 |
| Dense FFN size | 12288 |
| Routed / active experts | 256 / 8 |
| Expert FFN size | 2048 |
| Shared experts | 1 |
| Attention heads | 64 |
| Q LoRA / KV LoRA rank | 2048 / 512 |
| QK no-PE / RoPE dimensions | 192 / 64 |
| Value dimension | 256 |
| Index heads / dimension / top-k | 32 / 128 / 2048 |
| RoPE | plain, theta 8,000,000, factor 1 |

The five-layer checkpoint contains the first three dense layers and two MoE layers. It is
not a uniformly reduced model: it intentionally exercises one index-computing MoE layer
and one index-reuse MoE layer.

GLM-5.2 computes DSA indices on Megatron layers 1, 2, 3, 7, 11, ..., 75. Other layers reuse
the latest computation. A pipeline stage must therefore begin on an index-computing layer,
because the current implementation does not send selected indices over a PP boundary.

### GLM forward and backward

The model implementation is
[`glm5.py`](../../../miles_plugins/models/glm5/glm5.py), with the sparse kernels in
[`miles_plugins/models/glm5/ops`](../../../miles_plugins/models/glm5/ops).

The forward path is:

1. Down-project hidden states into Q-LoRA and KV-LoRA spaces and normalize them.
2. Up-project Q, split non-positional and RoPE dimensions, and absorb the KV projection
   matrix into Q so sparse attention operates in the compressed KV space.
3. Gather the compressed KV and positional key across the relevant sequence/context
   groups, then apply packed-THD RoPE.
4. On an index-computing layer, form detached index queries, keys, and per-head weights,
   run the lighting indexer in blocks, and keep the top-2048 indices. On a reuse layer,
   obtain those indices from the per-microbatch `packed_seq_params` holder.
5. Run sparse MLA over the selected KV rows. The custom autograd function saves Q, KV,
   indices, output, and LSE.
6. Apply the absorbed value-up matrix, reshape, and run the row-parallel output projection.
7. Continue through the dense or MoE MLP, with the normal Megatron residual and pipeline
   machinery.

The backward path is:

1. Precompute `delta = sum(output * dOutput)` for every token/head.
2. Recreate sparse attention probabilities from Q, KV, indices, and LSE.
3. Compute dP, then dQ and dKV. dKV is accumulated atomically because multiple sparse
   selections can reference the same KV row.
4. Propagate dQ/dKV through the absorbed Q/KV/value/output projections; Transformer
   Engine's delayed weight-gradient hooks handle their weight gradients.
5. Continue through MoE/dense blocks and distributed optimizer state as normal.

The index selection is deliberately nondifferentiable. The current forward detaches the
indexer inputs and discards selected scores after top-k, so merely unfreezing the indexer
does not train it. The AMD launcher rejects `--no-freeze-indexer`. Indexer training would
need a defined differentiable consumer or auxiliary objective and a separate validation
effort.

## Implemented GLM-5.2 AMD recipe

The guarded launcher is
[`run_glm5_2_744b_a40b.py`](../../../scripts/amd/run_glm5_2_744b_a40b.py).

The full initial topology is locked to:

| Axis | Value |
|---|---:|
| Hosts and GPUs | 8 x 8 = 64 |
| TP / PP / CP / DP | 8 / 4 / 1 / 2 |
| EP / expert TP | 16 / 1 |
| PP layer counts | 18, 20, 20, 20 |
| PP stage starts | 1, 19, 39, 59 |
| Local attention heads | H8 |

The five-layer profile remains 1 node x 4 GPUs, TP4/PP1/CP1/EP4, which is the shape of
the existing manual MI355X proof of concept. Its defaults keep R3 off and leave the router
trainable. The full profile enables routing replay and freezes the router and e-score
correction bias.

Conservative full-profile defaults are:

- BF16 actor training with Transformer Engine;
- BF16 rollout weights initially, with an FP8 E4M3 KV cache;
- all-to-all Megatron MoE dispatch, no DeepEP/Mori initially;
- prompt limit 4096, response limit 8192, and packing target 16384 tokens/GPU;
- SGLang memory fraction 0.70, graph batch 1, and 32 running requests;
- full recomputation and a 4 GiB trainer memory margin;
- no MTP and no indexer replay;
- CPU trainer rematerialization by default, with explicit optimizer and NVMe offload
  options for memory recovery.

FP8 rollout weights, Mori/DeepEP, MTP, indexer replay, and a trainable full-model router
are behind `--allow-unvalidated-features`. FP8 refers only to SGLang rollout weights; the
Megatron actor remains BF16.

## Correctness and reliability hardening completed

### Model and kernel path

- GLM-5.2 now explicitly emits `--rope-type rope --rotary-scaling-factor 1`, matching the
  pinned checkpoint. The old implicit YaRN-factor-1 path was mathematically equivalent,
  but fragile and not the declared architecture.
- The GLM attention path accepts both the current Megatron plain-RoPE tensor return and the
  legacy tuple return.
- The ROCm sparse-MLA forward selects a single software-pipeline stage based on the actual
  HIP runtime as well as the Miles platform environment variable.
- H8 backward now masks the eight padded head lanes before scalar LSE/delta reads and dKV
  reduction.
- Padded `-1` indices use in-range zero gathers in forward and backward.
- An all-invalid sparse row produces finite zero output and a backward-safe finite LSE.
- A new H8/H16 GPU reference test covers output, LSE, dQ, dKV, padded indices, poisoned
  memory, and an all-invalid row.

### SGLang precision

The ROCm image applies
[`sglang_router_fp32.patch`](../../../docker/amd_patch/latest/sglang_router_fp32.patch).
It keeps AITER router GEMM output and the no-aux expert-correction bias in FP32, matching
the precision expected by the trainer. The Docker build either applies the exact patch or
verifies that it is already present; it does not silently ignore a conflicting source
tree.

This is temporary. Remove it only after the image pins an upstream SGLang commit with the
equivalent fix. FP8 weight-check canonicalization is a separate open upstream issue and
remains a blocker for an FP8 support claim.

### Checkpoint and artifact lifecycle

The launcher pins:

- full checkpoint revision `b4734de4facf877f85769a911abafc5283eab3d9`;
- five-layer revision `1c749139f70e158e4420ba67f342bef1de2e650d`;
- DAPO data revision `2e65612930298bde4c5d58fd97b3f23a483aaff9`.

It validates the complete critical config and the audited index layouts:

| Checkpoint | Weight entries | Shards | Metadata total size |
|---|---:|---:|---:|
| Full | 59,585 | 282 | 1,506,659,919,872 |
| Five-layer | 1,618 | 14 | 45,683,868,160 |

Each prepared source, FP8, and torch-distributed checkpoint carries an atomic
`.miles-artifact.json` provenance manifest with a path-to-byte-size inventory. Reuse is
rejected if the recipe or inventory differs. Torch-distributed validation checks the exact
`release` tracker, backend metadata, readable DCP metadata, every referenced `.distcp`
file, and every referenced byte range.

`prepare-cp` validates shared artifacts before copying and validates node-local artifacts
on every host afterward. `train` repeats node-local validation immediately before job
submission. The inventory intentionally does not hash 1.5 TB of payloads, so same-size bit
corruption remains outside this fast guard; the bring-up runbook includes an expensive
hash gate for qualification.

### Multi-node orchestration

Full preparation, conversion, copying, and training require an already joined dedicated
Ray cluster with `MILES_SCRIPT_EXTERNAL_RAY=1` and an explicit HTTP(S) `RAY_ADDRESS`.
Before work begins, the launcher verifies:

- exactly eight alive Ray nodes and 64 currently available GPU resources;
- exactly eight GPU resources per node;
- eight `gfx950` agents per node from `rocminfo`;
- eight MI350X or eight MI355X product names per node, with `amd-smi` fallback;
- one homogeneous requested SKU across all nodes;
- a dashboard address that resolves into the inspected cluster.

Hardware probes and artifact validation use hard node affinity. Multi-node copy never
trusts a sentinel observed only on the head. HTTP proxy variables are removed from the
driver, Ray submission, and worker runtime while validated node addresses are added to
`NO_PROXY`. Existing external-Ray processes are not killed by generic launcher cleanup.

### Weight synchronization evidence

Audit/CI modes on the default Ray path now query every allocated SGLang engine immediately
after each trainer-to-engine update. The validation requires:

- every engine to report the rollout manager's expected weight version;
- one checksum response per version response;
- identical full checksum maps across all engines.

The result is stored in the inference-engine checksum event schema. This proves that all
engines agree with each other and advanced to the expected version. It does **not** yet
prove byte-for-byte actor-versus-engine equality after every live update; the handoff must
not overstate that boundary.

## Verification completed without GPUs

The following focused results were green during implementation:

- AMD GLM launcher/artifact tests: 74 passed;
- command utility tests: 79 passed;
- default-Ray all-engine validation tests: 28 passed, 1 skipped;
- Megatron RoPE return compatibility test: 1 passed;
- GLM model-argument RoPE test and direct full/five-layer model snapshots: passed;
- relevant Ruff checks, `py_compile`, and `git diff --check`: passed at their respective
  checkpoints.

The manual H8/H16 and DeepSeek poison-memory suites were collected but cannot execute on
this Mac without TileLang and a ROCm GPU.

Snapshot regeneration was still in progress at the time of handoff. The manual launcher
fixture has been updated to represent the five-layer checkpoint's exact 1,618-entry,
14-shard index, but all affected launcher snapshots must be regenerated and reviewed after
the final refactor. Adding the explicit RoPE flags also changes generic GLM-5.2 launcher
snapshots, not only the AMD snapshots.

## Remaining work before requesting GPUs

Finish these in order:

1. Complete or abandon cleanly the private-helper extraction that brings the AMD launcher
   below the repository's approximate 1,000-line limit. Preserve launcher test access to
   private validation helpers.
2. Regenerate all affected model/launcher snapshots and inspect the command diffs. Expected
   semantic additions are explicit RoPE flags and artifact validation commands; investigate
   anything else.
3. Run the complete launcher suites required by the repository rule:

   ```bash
   pytest tests/fast/launch_scripts tests/manual/launch_scripts
   ```

4. Run focused command-utils, all-engine validation, model-args, and RoPE compatibility
   tests together, then run Ruff, `py_compile`, and `git diff --check` again.
5. Review the five-layer E2E wrapper on both branches. AMD needs the new local checkpoint
   copy; do not accidentally force the existing NVIDIA single-node path through a Ray copy
   before a Ray cluster exists.
6. Perform the final documentation consistency pass for the ROCm Docker sentinel and
   `docs/ci/02-docker-build.md`.
7. Read every changed snapshot and the final full worktree diff before committing.

## First GPU allocation

Use the [bring-up runbook](glm5-2-amd-bringup.md) as the command authority. Do not start
with the full E2E job. Preserve all logs and manifests under one shared bring-up root.

The gate order is:

1. Pin and record the exact image, Miles commit, Megatron commit, SGLang commit, ROCm
   packages, firmware, and kernel.
2. Establish clean eight-node Ray and RCCL inventory. Run pairwise and full-ring collective
   tests before loading the model.
3. Run the H8 and H16 sparse-MLA GPU reference tests. Stop on any non-finite output,
   padded-head contamination, or gradient error above the declared tolerances.
4. Re-establish the four-GPU five-layer BF16 boundary: trainer-only replay, SGLang-only,
   then two-step E2E with checkpoint/save and weight-sync auditing.
5. Prepare and validate the pinned full BF16 artifacts on all eight nodes.
6. Run one full trainer-only forward/backward/optimizer step with short sequences and no
   rollout engines. Record per-rank peak memory and first/steady-step time.
7. Run SGLang H8 loading and minimal eager prefill/decode, then graph capture. Compare a
   fixed prompt against the trainer/reference path before raising concurrency.
8. Run one full joined E2E step, then three steps with all-engine version/checksum auditing,
   finite metrics, leak checks, and explicit stop criteria.
9. Run uninterrupted versus split save/resume validation.
10. Only after BF16 passes, qualify FP8 rollout weights separately. Keep Megatron BF16 and
    start without Mori/DeepEP.

MI350X and MI355X results are separate qualification records even though both report
`gfx950`. Do not infer one from the other.

## Confidence and known risk

Confidence is high that the intended topology, DSA stage boundaries, checkpoint identity,
node inventory, artifact copying, and static forward/backward shapes are now represented
correctly. Confidence is medium that the H8 TileLang fixes are numerically correct because
they have a strong FP32 test oracle but have not compiled or run on gfx950. Confidence is
lower for full-system fit and efficiency because those depend on actual ROCm/TileLang
compilation, RCCL fabric behavior, SGLang AITER kernels, graph capture, and host/NVMe
capacity.

The largest unresolved risks are:

- H8 TileLang compile/runtime behavior on MI350X and MI355X;
- full 744B actor/optimizer/activation peak memory with colocation;
- SGLang H8 NSA prefill/decode parity and graph capture;
- multi-node RCCL stability under TP, PP, EP, and weight synchronization together;
- actor-versus-engine equality after live updates;
- save/resume determinism;
- FP8 checker canonicalization and any later Mori/DeepEP enablement.

The implementation is designed to make each failure attributable to one gate instead of
discovering all of these variables in the first long E2E run.
