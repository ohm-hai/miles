---
title: GLM-5.2 AMD GPU bring-up runbook
description: Evidence, topology, gates, and stop criteria for bringing up full GLM-5.2 training on 64 MI350X or MI355X GPUs.
---

This is an acceptance runbook for the first full-model AMD run, not a statement that the
configuration is supported today. The
[`scripts/amd/run_glm5_2_744b_a40b.py`](../../../scripts/amd/run_glm5_2_744b_a40b.py)
full profile is deliberately guarded to the topology below. That guard makes an accidental
shape fail fast; it does not mean the accepted shape has run successfully.

The primary target is **8 hosts x 8 MI355X GPUs = 64 gfx950 GPUs**. MI350X uses the same
image architecture, but must pass this entire runbook independently; MI355X results are not
MI350X validation.

## What is evidence and what is not

| Claim | Evidence | Status |
|---|---|---|
| Five-layer BF16 actor and rollout can complete two steps on AMD | [Miles PR #2268](https://github.com/radixark/miles/pull/2268) reports a manual 4x MI355X run covering both rollouts, training, checkpointing, and weight synchronization | Manual PoC; PR remains open |
| The ROCm path selects TileLang NSA and limits sparse-MLA staging for MI355X LDS | Merged [Miles PR #2347](https://github.com/radixark/miles/pull/2347), the [AMD launcher](../../../scripts/amd/run_glm5_2_744b_a40b.py), and the [TileLang sparse-MLA kernel](../../../miles_plugins/models/glm5/ops/tilelang_sparse_mla_fwd.py) | Implemented |
| SGLang keeps ROCm router logits and expert-correction bias in FP32 | The image applies [`sglang_router_fp32.patch`](../../../docker/amd_patch/latest/sglang_router_fp32.patch), tracking open [SGLang issue #34857](https://github.com/sgl-project/sglang/issues/34857) and [PR #35055](https://github.com/sgl-project/sglang/pull/35055) | Temporary image patch; upstream remains open |
| Five-layer GLM-5.2 is continuously tested on ROCm | The [E2E registration](../../../tests/e2e/megatron/model_scripts/test_glm5_2_744b_a40b_5layer_ci.py) is disabled for MI350 | **Not validated in CI** |
| H8 and H16 trainer sparse-MLA forward/backward have a common parity gate | [`test_tilelang_sparse_mla.py`](../../../tests/manual/models/glm5/test_tilelang_sparse_mla.py) covers output, LSE, dQ, dKV, an all-invalid row, and poisoned storage before padded `-1` indices | Test exists and statically passes; GPU cases remain unrun |
| Full 78-layer GLM-5.2 trains on AMD | No single-node or multi-node AMD run was found | **Unvalidated** |
| The 64-GPU topology below fits and is numerically correct | The same parallel shape has GB300 evidence and satisfies the model's DSA pipeline constraints; it has not run on AMD | **Unvalidated AMD acceptance target** |
| FP8 rollout weight checking is reliable on the current SGLang stack | The required AITER canonicalization is still open in [SGLang PR #34330](https://github.com/sgl-project/sglang/pull/34330) | **Blocked for a support claim** |

The full architecture is defined in
[`scripts/models/glm5.2-744B-A40B.py`](../../../scripts/models/glm5.2-744B-A40B.py):
78 layers, 64 attention heads, three dense layers, 75 MoE layers, and 256 routed experts.
The five-layer slice is defined separately in
[`scripts/models/glm5.2-744B-A40B_5layer.py`](../../../scripts/models/glm5.2-744B-A40B_5layer.py).
The generic [NVIDIA launcher](../../../scripts/run_glm5_2_744b_a40b.py) is useful design
evidence, but its FlashMLA backends and hardware results are not AMD validation.

## Locked 64-GPU topology

Do not tune topology during the first E2E attempt. Use this profile and change one axis only
after it passes:

| Axis | Value | Invariant |
|---|---:|---|
| Physical world | 8 nodes x 8 GPUs = 64 | Homogeneous MI355X/gfx950 |
| Dense tensor parallel | TP8 | 64 / 8 = **H8 local attention heads** |
| Pipeline parallel | PP4 | Layer counts `18,20,20,20` |
| Context parallel | CP1 | No context sharding in the first run |
| Data parallel | DP2 | `TP8 x PP4 x CP1 x DP2 = 64` |
| Expert tensor parallel | ETP1 | No expert tensor sharding |
| Expert parallel | EP16 | `ETP1 x PP4 x EP16 = 64` for MoE layers |
| Prompt / final response limit | 4096 / 8192 tokens | Start with a 1024-token response, then ramp to the launcher default |
| Trainer packing target | 16384 tokens/GPU | Covers the largest accepted prompt plus response; it is not a response limit |
| Initial SGLang limits | memory 0.70, graph batch 1, 32 requests | Conservative full-profile defaults; measure, do not assume fit |

The PP stage starts are Megatron layers 1, 19, 39, and 59. All are DSA
computing layers, which is required by cross-layer index sharing. A profile that changes PP
must prove this invariant again.

For the first BF16 rollout-only and E2E gates, use eight SGLang engines, one eight-GPU
TP8/EP8 engine per host. The trainer and rollout therefore both exercise H8, while the
five-layer PoC exercises H16.

### H8 and H16 are head tiles, not model variants

`H` means the model's 64 attention heads divided by TP:

- TP4 gives H16. This is the shape exercised by the 4-GPU MI355X PoC.
- TP8 gives H8. This is the full 64-GPU profile. The TileLang forward and backward kernels
  pad H8 to a 16-head tile; the padding does not create eight additional model heads.

The padding is visible in both
[`tilelang_sparse_mla_fwd.py`](../../../miles_plugins/models/glm5/ops/tilelang_sparse_mla_fwd.py)
and
[`tilelang_sparse_mla_bwd.py`](../../../miles_plugins/models/glm5/ops/tilelang_sparse_mla_bwd.py).
Only H16 has AMD PoC evidence. H8 trainer-kernel reference tests for forward, backward,
padded `-1` indices, an all-invalid row, finite `dQ`/`dKV`, and gradient agreement are
therefore a hard prerequisite for the full profile. SGLang prefill and decode are separate
Gate 3 checks. A forward-only H8 pass is not permission to train.

The full-profile qualification intentionally keeps the launcher's other conservative
defaults:

- Megatron actor weights are BF16. “BF16 rollout” below means BF16 rollout **weights**;
  SGLang's KV cache is still `fp8_e4m3`.
- R3 routing replay is enabled; router gates and e-score correction bias are frozen.
- The DSA indexer is frozen and indexer replay is off.
- MTP is off.

The five-layer manual-PoC profile deliberately preserves its older boundary instead: R3
is off and the router is not frozen. That keeps Gate 1 close to PR #2268; it does not make
trainable routing a qualified full-model feature.

Consequently, a pass qualifies that recipe, not indexer backward, MTP, FP8 actor training,
or fully trainable router gates. The launcher intentionally rejects
`--no-freeze-indexer`: the current GLM forward keeps the selected indices but discards the
indexer's selected scores, so there is no differentiable consumer that can train the
indexer. The sparse-MLA H8/H16 parity test below validates attention dQ/dKV; it does not
change that model-level gradient boundary.

Supporting indexer training requires a separate implementation: preserve and consume the
selected scores in a defined differentiable objective, specify its interaction with top-k
and indexer replay, validate
[`tilelang_indexer_bwd.py`](../../../miles_plugins/models/glm5/ops/tilelang_indexer_bwd.py),
and only then repeat trainer and E2E gates. Merely removing `--freeze-indexer` is neither
accepted by the launcher nor evidence of indexer training.

Run the checked-in H8/H16 parity gate before Gate 2:

```bash
pytest -q tests/manual/models/glm5/test_tilelang_sparse_mla.py
```

It checks TileLang output and base-2 LSE plus backward dQ/dKV against one FP32 reference,
and puts infinities immediately before KV storage so an unsafe padded `-1` gather fails
deterministically. Its predeclared limits are output relative error `< 1e-3` and maximum
error `< 0.1`, LSE `rtol=2e-2, atol=5e-2`, and dQ/dKV relative error `< 5e-2`, with no
NaN/Inf. Static collection and lint are not GPU evidence; until both H8 and H16 cases pass
on the pinned MI355X image, Gate 2 is blocked.

That test is deliberately narrow: sequence 16, KV length 80, top-k 64, and one process. It
does not cover SGLang prefill/decode, the production top-k 2048 or long sequences, indexer
kernels, or distributed TP. Those remain separate Gate 2/3 work rather than being implied
by a green unit test.

## Freeze the environment before allocating GPUs

The gfx950 image is built by
[`docker/Dockerfile.rocm`](../../../docker/Dockerfile.rocm) through the
[`rocm720-mi35x` variant](../../../docker/build.py). Its defaults include mutable Miles,
Megatron, and SGLang branch inputs, so a tag such as `latest` is insufficient. The
Dockerfile also applies `sglang_router_fp32.patch`; that patch is independent from the FP8
weight-checker work in SGLang PR #34330.

Record an immutable image reference and do not continue if nodes disagree:

```bash
export BRINGUP_ROOT=/shared/glm52-amd-bringup
export AMD_IMAGE='registry.example/miles@sha256:REPLACE_WITH_DIGEST'

mkdir -p "${BRINGUP_ROOT}/image"
docker pull "${AMD_IMAGE}"
docker image inspect "${AMD_IMAGE}" > "${BRINGUP_ROOT}/image/image-inspect.json"
docker image inspect --format '{{.Id}} {{json .RepoDigests}}' "${AMD_IMAGE}" \
  | tee "${BRINGUP_ROOT}/image/image-id.txt"
```

Inside the container on every node, write a dependency manifest. Keep the dependency block
separate from host diagnostics so its checksum can be compared byte-for-byte:

```bash
export BRINGUP_ROOT=/shared/glm52-amd-bringup
export NODE_RECORD="${BRINGUP_ROOT}/manifests/${HOSTNAME}"
mkdir -p "${NODE_RECORD}"

{
  git -C /root/miles rev-parse HEAD
  git -C /root/Megatron-LM rev-parse HEAD
  git -C /sgl-workspace/sglang rev-parse HEAD
  git -C /sgl-workspace/sglang diff --binary | sha256sum
  sha256sum /root/miles/docker/amd_patch/latest/sglang_router_fp32.patch
  python -m pip freeze | LC_ALL=C sort
} > "${NODE_RECORD}/dependencies.txt"

{
  date --iso-8601=seconds
  hostname
  uname -a
  free -h
  swapon --show
  numactl --hardware
  lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
  df -h /local_nvme /scratch 2>&1
  rocminfo
  rocm-smi --showproductname --showuniqueid --showmeminfo vram --showtopo
} > "${NODE_RECORD}/host.txt" 2>&1

sha256sum "${NODE_RECORD}/dependencies.txt" \
  | tee "${NODE_RECORD}/dependencies.sha256"
sha256sum "$(readlink -f /opt/rocm/lib/libhsa-runtime64.so.1)" \
  | tee "${NODE_RECORD}/rocr.sha256"
```

**Pass:** all eight nodes report the same image digest, dependency checksum, Miles,
Megatron, SGLang, TransformerEngine/TileLang/AITER versions, and patched ROCr checksum;
each node sees exactly eight gfx950 devices with the expected HBM capacity.

The SGLang source must satisfy two independent FP32 invariants: AITER router GEMM returns
`torch.float32`, and `e_score_correction_bias` remains a `torch.float32` parameter even for
FP8/compressed checkpoints. The image is acceptable if the patch applies exactly, or if a
pinned upstream SGLang commit already contains both invariants and the obsolete patch has
been removed.

**Stop:** any mutable tag without a digest, unrecorded source diff, mixed package/version,
failed/mutated router patch, either missing FP32 invariant, missing ROCr VMM fix on ROCm
7.2, missing GPU, or unexpected GPU architecture. The expected SGLang patch makes that
checkout intentionally dirty; its recorded diff hash must match on all nodes.

### Pin and size the checkpoint artifacts

The launcher pins the full HF checkpoint to
`b4734de4facf877f85769a911abafc5283eab3d9`, the five-layer checkpoint to
`1c749139f70e158e4420ba67f342bef1de2e650d`, and `dapo-math-17k` to
`2e65612930298bde4c5d58fd97b3f23a483aaff9`. Do not override these SHAs during
qualification. It validates every architecture-critical config field and the pinned HF
index layout before conversion:

| Artifact | Weight entries | Shards | `metadata.total_size` |
|---|---:|---:|---:|
| GLM-5.2 | 59,585 | 282 | 1,506,659,919,872 |
| GLM-5.2_5layer | 1,618 | 14 | 45,683,868,160 |

Each prepared directory also carries `.miles-artifact.json`, binding it to its source
repository, revision, conversion topology, and (for rollout FP8) 128x128 block recipe.
`prepare-cp` and `train` refuse a stale or untracked directory; multi-node copy validates
the node-local result on every host rather than trusting a sentinel visible on the head.
These checks detect missing files and recipe drift, but intentionally do not hash 1.5 TB
of tensor payload on every launch.

Before allocating the cluster, measure free space and sustained read/write bandwidth for
all simultaneously retained copies: the 1.51 TB HF source, the converted `torch_dist`
checkpoint, an optional FP8 rollout copy, a node-local trainer copy per host, a node-local
rollout copy per host, checkpoints, and offload files. Archive a one-time SHA-256 manifest
for the source index/config and every shard, then compare that manifest after transfer:

```bash
export GLM_MODEL_DIR=/shared/models
export GLM_LOCAL_DIR=/local_nvme/models
cd "${GLM_MODEL_DIR}/GLM-5.2"
sha256sum config.json model.safetensors.index.json model-*.safetensors \
  > "${BRINGUP_ROOT}/glm52-hf.sha256"
sha256sum --check "${BRINGUP_ROOT}/glm52-hf.sha256"
du -sb "${GLM_MODEL_DIR}/GLM-5.2"
df -B1 "${GLM_MODEL_DIR}" "${GLM_LOCAL_DIR}" "${BRINGUP_ROOT}"
```

Run the payload hash once while staging, not inside the short GPU window. A matching
`.miles-artifact.json` is provenance evidence; it is not a substitute for that transfer
integrity record.

### Pin and preflight the GRPO data

The training input is the transformed
[`zhuzilin/dapo-math-17k`](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)
JSONL, not the source DAPO parquet schema. Download it once, before reserving GPUs, at the
same absolute path that will be passed as `--data-dir`:

```bash
export GLM_DATA_DIR=/shared/datasets
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --revision 2e65612930298bde4c5d58fd97b3f23a483aaff9 \
  --local-dir "${GLM_DATA_DIR}/dapo-math-17k"

python tools/validate_grpo_dataset.py \
  "${GLM_DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl" \
  --profile dapo-math-17k \
  | tee "${BRINGUP_ROOT}/dapo-math-17k-validation.json"
```

The immutable profile requires all of the following, and parses every line instead of
silently skipping malformed JSON:

| Property | Required value |
|---|---:|
| Revision | `2e65612930298bde4c5d58fd97b3f23a483aaff9` |
| JSONL SHA-256 | `cc9c39c2aa19177abe9464741e121cf4cac90fd25484ef3cdf86535101e3a5b6` |
| File size | 10,490,834 bytes |
| Rows / unique prompts | 17,398 / 17,398 |
| Row keys | exactly `prompt`, `label` |
| `prompt` | one `{role: "user", content: <non-empty string>}` message |
| `label` | non-empty string |

Every prompt also contains both the `Answer:` and `\boxed` instructions expected by this
recipe. The launcher passes `--input-key prompt --label-key label --apply-chat-template`
and sends the label directly to `--rm-type deepscaler`; there is no `reward_model` or
metadata field to remap. DeepScaler assigns zero reward if a response does not close
thinking with `</think>` (or use its legacy `###Response` separator), and then requires a
boxed answer in the remaining text. Verify that boundary inside the frozen image:

```bash
python - <<'PY'
import json
import os

from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward

dataset_path = os.path.join(os.environ["GLM_DATA_DIR"], "dapo-math-17k", "dapo-math-17k.jsonl")
with open(dataset_path, encoding="utf-8") as dataset:
    first = json.loads(next(dataset))
label = first["label"]
assert get_deepscaler_rule_based_reward(rf"</think>Answer: \boxed{{{label}}}", label) == 1
assert get_deepscaler_rule_based_reward(rf"Answer: \boxed{{{label}}}", label) == 0
PY
```

After the full HF checkpoint is staged, run the tokenizer-aware preflight in the same
container that will train. It uses only local files, applies the checkpoint's chat template
with an assistant generation prompt, and rejects any prompt over the full recipe's limit:

```bash
python tools/validate_grpo_dataset.py \
  "${GLM_DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl" \
  --profile dapo-math-17k \
  --tokenizer "${GLM_MODEL_DIR}/GLM-5.2" \
  --max-prompt-tokens 4096 \
  --expected-min-prompt-tokens 73 \
  --expected-max-prompt-tokens 1521 \
  | tee "${BRINGUP_ROOT}/dapo-math-17k-tokenizer-validation.json"
```

At the pinned model and data revisions the rendered lengths are 73-1,521 tokens, so all
17,398 rows pass the full-model 4,096-token cap. The five-layer smoke cap is 1,024 and
intentionally filters seven long prompts, leaving 17,391; do not mistake that PoC behavior
for the full recipe. The training batch consumes eight prompts with eight generations each
(64 samples) per rollout and shuffles on epoch rollover. The default 3,000 rollouts reuse
the dataset after the first 2,174 complete prompt batches; qualification gates use only the
small rollout counts stated below.

This acceptance path deliberately uses the same pinned DAPO payload for the five-layer
smoke, full-model liveness gates, resume test, and eventual GRPO run; it does not create an
unrecorded train split. The AMD launcher does not currently download or schedule AIME
evaluation. Consequently these gates establish training and reward correctness, not a
model-quality gain. After BF16 E2E and resume pass, add a separately pinned AIME-2024
evaluation recipe and baseline before making a quality claim; do not mix evaluation
problems into the GRPO prompt stream.

`RolloutManager` is not pinned to the Ray head by this recipe and is allowed to start on any
node. Therefore `${GLM_DATA_DIR}` must be a genuinely shared mount at the same path on all
eight nodes, or the 10.5 MB dataset directory must be copied to that path on every node.
The launcher's pre-train validation probes every node and stops if any node sees a missing
or different payload. Do not rely on a head-only download into node-local storage.

### Host-memory and local-NVMe budget

Colocation moves a frontier-scale allocation off HBM during each half of the cycle. A
single unquantized 744B-parameter rollout replica is about 1.49 TB before allocator,
graph, and runtime overhead (`744e9 * 2` bytes), distributed across one eight-GPU host.
The paused trainer also carries sharded parameters and optimizer state. Rematerialization
removes one BF16 parameter backup; it does not make the rest free. Do not infer host fit
from 288 GB HBM capacity.

Before Gate 4, write a per-node budget for both alternating phases and reserve at least
25% above the larger estimate. If no measured MI35x baseline exists, treat 2.25 TiB of
usable RAM per host as the provisional floor for the CPU-offload recipe, not as a fit
guarantee. Disable swap for qualification; swap activity turns a slow run into an
unbounded hang and invalidates throughput evidence. Record `vmstat 1`, NUMA-local memory,
major faults, pinned memory, and process RSS throughout every load/offload transition.

If the trainer backup does not fit host RAM, the launcher can use node-local NVMe:

```text
--offload-train-target disk
--offload-train-disk-dir /local_nvme/glm52-offload
```

Add `--stream-optimizer-state-to-disk` only if optimizer state does not fit HBM during the
training step; it is incompatible with `--enable-optimizer-offload`. Disk trainer offload
does not move the SGLang weight backup to disk, so BF16 rollout still needs its own host
budget. Require real local NVMe (not tmpfs), enough free capacity for every rank, and a
measured sequential read/write rate before selecting this slower recipe. A disk-offload
pass qualifies a separate recipe from the CPU default.

## RCCL and host preflight

The image includes `rccl-tests`. Create a hostfile with eight slots per node, set the actual
fabric interface names, and run local-node tests before the 64-rank test:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,COLL
export NCCL_SOCKET_IFNAME=REPLACE_WITH_DATA_INTERFACE
export NCCL_IB_HCA=REPLACE_WITH_RDMA_DEVICES
mkdir -p "${BRINGUP_ROOT}/rccl"

# Repeat on each host first with -np 8 and a one-host hostfile.
mpirun --allow-run-as-root -np 64 --hostfile /shared/glm52-amd-hosts \
  --bind-to numa /usr/local/bin/all_reduce_perf \
  -b 8 -e 8G -f 2 -g 1 -n 50 \
  2>&1 | tee "${BRINGUP_ROOT}/rccl/all-reduce-64.log"
```

Also run `all_gather_perf` and `alltoall_perf` with the same placement; the trainer and
weight paths use gather collectives and MoE uses all-to-all. Record algorithm bandwidth,
bus bandwidth, latency at small and large messages, rank-to-device mapping, retransmits,
XGMI/IB errors, and three-run variation.

**Pass:** no mismatch, timeout, retry storm, or RCCL error; every rank maps to a unique GPU;
the intended RDMA transport is selected; large-message bandwidth is within the cluster's
accepted baseline and repeated-run variation is at most 10%.

**Stop:** socket fallback when RDMA is required, any disabled/degraded link, any hang, or a
node more than 10% below the accepted cluster baseline. Do not debug model code on a fabric
that has not passed this gate.

## Staged model gates

Run only one new boundary per gate. Archive stdout, per-rank stderr, Ray logs, ROCm metrics,
W&B/TensorBoard exports, and the dependency manifest with every result.

The examples use:

```bash
export BRINGUP_ROOT=/shared/glm52-amd-bringup
export GLM_AMD_SCRIPT=scripts/amd/run_glm5_2_744b_a40b.py
export GLM_MODEL_DIR=/shared/models
export GLM_LOCAL_DIR=/local_nvme/models
export GLM_DATA_DIR=/shared/datasets
```

Gates 2-5 require a fresh, dedicated eight-node Ray cluster that is already joined. This
includes the full `prepare` conversion and `prepare-cp` fan-out: both use Ray to run work on
all eight nodes, not just the later `train` command. The launcher deliberately does no
broad `pkill`, Redis cleanup, or `ray stop` in external-cluster mode, so stale application
actors must be removed before qualification.

Run all full-profile commands from a node joined to that cluster and export:

```bash
export HEAD_IP=REPLACE_WITH_RAY_HEAD_IP
export MASTER_ADDR="${HEAD_IP}"
export RAY_ADDRESS="http://${HEAD_IP}:8265"
export MILES_SCRIPT_EXTERNAL_RAY=1
```

The full launcher refuses anything except exactly eight alive nodes, eight advertised and
currently available GPUs per node, eight `gfx950` agents per node, and a homogeneous match
for the requested MI350X or MI355X SKU. It hard-affinity probes every node, verifies that
the dashboard host belongs to the inspected cluster, clears HTTP proxy variables, adds all
node addresses to `no_proxy`, and prints the accepted inventory. Save that JSON and the
rank/host/GPU mapping. Exported NCCL/RCCL socket, IB, and debug variables are forwarded into
the Ray job. Do not set external-Ray mode for standalone Gate 1.

### Single-node qualification driver

Use the checked-in
[`test_glm5_2_amd_single_node_stages.py`](../../../tests/e2e/megatron/model_scripts/test_glm5_2_amd_single_node_stages.py)
as the first executable entrypoint during a short MI350X or MI355X allocation. It orders
the one-node boundaries and makes the generated-rollout-to-trainer replay explicit; it
does not turn an unrun GPU path into a validation claim.

Configure immutable storage locations once, then run the ladder in this order:

```bash
unset MILES_SCRIPT_EXTERNAL_RAY RAY_ADDRESS MASTER_ADDR
export MILES_GLM52_AMD_HARDWARE=MI355X
export MILES_GLM52_TEST_ROOT=/shared/glm52-amd-single-node
export MILES_GLM52_MODEL_DIR=/shared/models
export MILES_GLM52_MODEL_LOCAL_DIR=/local_nvme/models
export MILES_GLM52_DATA_DIR=/shared/datasets

# Evidence-backed 4-GPU TP4/EP4 shape: H16 sparse-attention tiles.
export HIP_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
python tests/e2e/megatron/model_scripts/test_glm5_2_amd_single_node_stages.py \
  --profile poc-h16 --stage five-layer-all
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

# Guarded 8-GPU TP8/EP8 shape: H8 tiles matching the full recipe's local shape.
python tests/e2e/megatron/model_scripts/test_glm5_2_amd_single_node_stages.py \
  --profile full-shape-h8 --stage five-layer-all

# Full 744B checkpoint: separate eager and graph SGLang load/decode probes.
python tests/e2e/megatron/model_scripts/test_glm5_2_amd_single_node_stages.py \
  --stage full-rollout
```

If `MILES_GLM52_AMD_HARDWARE` is unset, the driver detects the SKU. If it is set, the
driver checks it against every visible GPU and aborts on a mixed SKU, a wrong count, or an
MI350X/MI355X mismatch; this prevents a MI350 run from being archived as MI355 evidence.
The H16 run intentionally requires exactly four visible GPUs, so retain the visibility
restriction on an eight-GPU host only for that command and remove it before H8 or full
rollout. The driver never silently treats an eight-GPU view as the four-GPU PoC.

The five-layer `*-all` path runs the H8/H16 TileLang forward/backward parity test, prepares
the pinned checkpoint and DAPO data, captures one deterministic eager rollout, strictly
replays that capture through trainer forward/backward/optimizer/save, and finally runs two
live GRPO rollout-to-weight-sync cycles. It rejects an empty/stale capture, a sample without
a completed or truncated response, non-finite rollout log probabilities or reward, and a
trainer run without a checkpoint tracker. H16 is the PR #2268 evidence-backed *shape path*,
but this driver still needs a new physical pass on the pinned image. H8 is a guarded shape
qualification with no prior hardware pass.

`full-rollout` is deliberately narrower. It stages only the pinned 1.51 TB HF source, then
runs `full-rollout-eager` before `full-rollout-graph`. Each substage loads one TP8/EP8
SGLang engine across all eight GPUs and captures exactly one deterministic completion with
a response limit of 128 tokens. The driver requires a fresh capture with one completed or
truncated response, finite per-token rollout log probabilities, and a finite DAPO reward.
Keeping eager and graph captures separate means graph compilation cannot obscure basic
checkpoint load and NSA prefill/decode. The combined alias then requires identical prompt
and response token IDs and a maximum per-token eager/graph log-probability difference of
`0.03`, the declared BF16 tolerance; the direct substages validate their own captures but
do not perform this cross-mode comparison. Both skip Megatron checkpoint conversion, trainer
parameter/optimizer initialization, backward, optimizer update, checkpoint save, and
weight synchronization. `--debug-rollout-only` still starts lightweight Megatron control
actors, torch.distributed, and tokenizer plumbing; standalone SGLang isolation is not
implemented. A pass is useful full-checkpoint rollout-path load/decode evidence; it is
**not single-node full-model training** and is **not multi-node validation**.

Each stage can also be resumed independently with `--stage kernel`,
`five-layer-prepare`, `five-layer-rollout`, `five-layer-trainer`, `five-layer-grpo`,
`full-rollout-eager`, or `full-rollout-graph`.
The trainer stage requires the rollout capture from the same profile, and every execution
stage requires prepared shared and node-local artifacts. Run the entire ladder separately
for MI350X and MI355X; both are gfx950, but a result on one SKU does not qualify the other.
The driver fails closed if `MILES_SCRIPT_EXTERNAL_RAY` or `RAY_ADDRESS` is still set, or
if `MASTER_ADDR` is non-loopback; use a clean shell and unset all three explicitly. It does
not silently switch modes, because doing so could make the generic local launcher cleanup
interfere with an existing cluster head.
After both five-layer shapes pass, continue with Gates 2-5 below for the locked 8x8 scale
plan rather than attempting to fit the full actor and optimizer on one node.

### Gate 1: re-establish the five-layer boundary

Start with BF16 actor and BF16 rollout weights (the KV cache remains FP8 E4M3). Preserve
the rollout dump for the trainer-only gate. The five-layer defaults keep R3 off, leave the
router trainable, and freeze the indexer, matching the original PoC boundary. Pinned source
revisions, node-local copy validation, the current Megatron API, and the SGLang router patch
still make this a revalidation rather than a claim that PR #2268 reproduced byte-for-byte:

```bash
# Required when this gate runs on an eight-GPU host; harmless on a four-GPU allocation.
unset MILES_SCRIPT_EXTERNAL_RAY RAY_ADDRESS MASTER_ADDR
export HIP_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
python "${GLM_AMD_SCRIPT}" full-train \
  --hardware MI355X \
  --model-org Pinaster --model-name GLM-5.2_5layer \
  --num-nodes 1 --num-gpus-per-node 4 --num-rollout 2 \
  --enable-optimizer-offload \
  --run-id glm52-amd-5layer-bf16 \
  --model-dir "${GLM_MODEL_DIR}" --model-local-dir "${GLM_LOCAL_DIR}" \
  --data-dir "${GLM_DATA_DIR}" --output-dir "${BRINGUP_ROOT}" \
  --extra-args "--ci-test --ci-disable-logprobs-checker --save-interval 1 \
    --save-debug-rollout-data ${BRINGUP_ROOT}/5layer/rollout_data/{rollout_id}.pt"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
```

`--ci-disable-logprobs-checker` reproduces the current PoC condition; it is not allowed in
the final BF16 E2E gate.

**Pass:** two complete rollout -> forward/backward -> optimizer -> weight-sync cycles;
finite nonzero gradient norm and finite loss/reward; no TileLang compile/LDS failure; the
startup weight checker reports no mismatch; a checkpoint and rollout files `0.pt` and `1.pt`
exist.

**Stop:** any NaN/Inf, second-rollout hang, missing DSA provider/consumer tensor, weight
mismatch, GPU reset, or checkpoint failure. Do not dismiss an AITER/rotary-cache mismatch
by disabling the weight checker; pin the checker fix instead.

### Gate 2: full model, trainer only

First, prepare the full checkpoint and copy it to node-local storage. The command must
print the locked topology above. Conversion itself uses all 64 ranks with TP1/PP4/EP16,
including the 18/20 pipeline edges; that storage layout is distinct from the TP8 runtime
layout and is expected:

```bash
# Gate 1 used a clean local shell; reconnect this process to the inspected 8x8 cluster.
export MASTER_ADDR="${HEAD_IP}"
export RAY_ADDRESS="http://${HEAD_IP}:8265"
export MILES_SCRIPT_EXTERNAL_RAY=1

python "${GLM_AMD_SCRIPT}" prepare \
  --hardware MI355X \
  --model-org zai-org --model-name GLM-5.2 \
  --num-nodes 8 --num-gpus-per-node 8 \
  --model-dir "${GLM_MODEL_DIR}" --model-local-dir "${GLM_LOCAL_DIR}" \
  --data-dir "${GLM_DATA_DIR}"

python "${GLM_AMD_SCRIPT}" prepare-cp \
  --hardware MI355X \
  --model-org zai-org --model-name GLM-5.2 \
  --num-nodes 8 --num-gpus-per-node 8 \
  --model-dir "${GLM_MODEL_DIR}" --model-local-dir "${GLM_LOCAL_DIR}" \
  --data-dir "${GLM_DATA_DIR}"
```

Then replay one five-layer rollout fixture through the full trainer. The tokenizer and
sample schema are shared; the rollout log probabilities are intentionally from the pruned
model, so this gate tests liveness and gradients, not KL or policy alignment:

```bash
python "${GLM_AMD_SCRIPT}" train \
  --hardware MI355X \
  --model-org zai-org --model-name GLM-5.2 \
  --num-nodes 8 --num-gpus-per-node 8 --num-rollout 1 \
  --no-enable-r3 \
  --run-id glm52-amd-full-trainer-only \
  --model-dir "${GLM_MODEL_DIR}" --model-local-dir "${GLM_LOCAL_DIR}" \
  --data-dir "${GLM_DATA_DIR}" --output-dir "${BRINGUP_ROOT}" \
  --extra-args "--load-debug-rollout-data \
    ${BRINGUP_ROOT}/5layer/rollout_data/{rollout_id}.pt \
    --no-offload-train --debug-exit-after-rollout 1 --save-interval 1"
```

The five-layer dump contains routing streams for five layers, not 78. R3 must therefore be
off for this synthetic trainer-only gate; otherwise the replay manager indexes beyond the
fixture before the full forward begins. The full profile still freezes the router by
default, independently of R3. `--no-offload-train` avoids a useless sleep/wake cycle when
no SGLang process exists. This gate does not qualify R3; the live full-model Gate 4 does.

**Pass:** all 78 layers and all checkpoint shards load with no missing/unexpected tensor;
the H8 reference suite and PP stage-start check are green; one full forward, backward,
optimizer step, and save complete; all losses and gradients are finite; every rank retains
at least 10% HBM headroom and rank peak-memory imbalance is below 10%.

**Stop:** conversion mismatch, stage starting on a DSA skip layer, OOM, CPU-offload swap,
zero/NaN/Inf gradient, RCCL timeout, or less than 10% HBM headroom. If optimizer offload is
needed, first prove that host RAM and NUMA bandwidth can hold it, then repeat this entire
gate with `--enable-optimizer-offload`; do not change it mid-run.

### Gate 3: full model, rollout path only

The eight-GPU engine uses TP8/H8, but the checked-in H8 unit test exercises the trainer
sparse-MLA kernel, not SGLang's prefill/decode path. Treat engine loading and live H8
attention as two subgates. First disable graph capture and request only a minimal decode;
this isolates checkpoint loading, FP32 router invariants, and eager kernel compilation.
The rollout-only implementation retains lightweight Megatron control ranks,
torch.distributed, and tokenizer setup while skipping model parameters, optimizer, and
backward; it is not a standalone SGLang process-isolation test. Run:

```bash
mkdir -p "${BRINGUP_ROOT}/gate3" || exit 1
GATE3_EVIDENCE_ROOT="$(
  mktemp -d "${BRINGUP_ROOT}/gate3/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXXXX"
)" || exit 1
GATE3_RUN_ID="${GATE3_EVIDENCE_ROOT##*/}"
GATE3_EAGER_DETAILS="${GATE3_EVIDENCE_ROOT}/eager-details"
GATE3_GRAPH_DETAILS="${GATE3_EVIDENCE_ROOT}/graph-details"

python "${GLM_AMD_SCRIPT}" train \
  --hardware MI355X \
  --model-org zai-org --model-name GLM-5.2 \
  --num-nodes 8 --num-gpus-per-node 8 --num-rollout 1 \
  --rollout-only \
  --rollout-probe-mode eager \
  --rollout-probe-capture \
    "${GATE3_EAGER_DETAILS}/rollout_data/{rollout_id}.pt" \
  --rollout-max-response-len 32 \
  --sglang-mem-fraction-static 0.65 --sglang-max-running-requests 8 \
  --run-id "glm52-amd-full-sglang-eager-${GATE3_RUN_ID}" \
  --model-dir "${GLM_MODEL_DIR}" --model-local-dir "${GLM_LOCAL_DIR}" \
  --data-dir "${GLM_DATA_DIR}" --output-dir "${BRINGUP_ROOT}" \
  --extra-args "--debug-exit-after-rollout 1 --use-miles-dashboard \
    --dump-details ${GATE3_EAGER_DETAILS}"
```

Keep the four `GATE3_*` variables in the same shell for all Gate 3 blocks and archive
`GATE3_EVIDENCE_ROOT` with the result. If either launch fails, generate a new run ID and
rerun both modes; never append a retry to the old root. Dashboard streams are append-only,
so reusing a directory would let counters from an earlier attempt masquerade as fresh
eight-engine evidence.

Then run the same boundary with the launcher's initial graph/concurrency defaults: memory
fraction 0.70, graph batch 1, 32 running requests, and a 1024-token response. Save prompts,
tokens, and per-token log probabilities and compare an accepted sample against eager mode
and an independent BF16 actor-forward computation before allowing E2E:

```bash
python "${GLM_AMD_SCRIPT}" train \
  --hardware MI355X \
  --model-org zai-org --model-name GLM-5.2 \
  --num-nodes 8 --num-gpus-per-node 8 --num-rollout 1 \
  --rollout-only \
  --rollout-probe-capture \
    "${GATE3_GRAPH_DETAILS}/rollout_data/{rollout_id}.pt" \
  --rollout-max-response-len 1024 \
  --run-id "glm52-amd-full-sglang-h8-${GATE3_RUN_ID}" \
  --model-dir "${GLM_MODEL_DIR}" --model-local-dir "${GLM_LOCAL_DIR}" \
  --data-dir "${GLM_DATA_DIR}" --output-dir "${BRINGUP_ROOT}" \
  --extra-args "--debug-exit-after-rollout 1 --use-miles-dashboard \
    --dump-details ${GATE3_GRAPH_DETAILS}"
```

Prove that the eight-request workload reached all eight node-local engines in both modes;
job success or a router aggregate alone is insufficient:

```bash
python - \
  "${GATE3_EAGER_DETAILS}" \
  "${GATE3_GRAPH_DETAILS}" <<'PY'
import json
import sys
from pathlib import Path

for details_arg in sys.argv[1:]:
    details = Path(details_arg)
    files = sorted((details / "dashboard" / "engine_series").glob("*.jsonl"))
    requests_by_addr = {}
    for path in files:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["metric"] == "sglang_num_requests_total":
                requests_by_addr[record["addr"]] = max(
                    float(record["value"]), requests_by_addr.get(record["addr"], 0.0)
                )
    if len(requests_by_addr) != 8 or any(value <= 0 for value in requests_by_addr.values()):
        raise SystemExit(f"{details}: expected 8 positive per-engine request counters, got {requests_by_addr}")
    print(details, requests_by_addr)
PY
```

The single-node driver checks eager/graph H8 parity for one 1x8 engine, but there is not yet
a checked-in 8x8 eager/graph/actor-forward parity test. That cluster comparison remains a
manual acceptance artifact. Do not let a green trainer-kernel unit test, one-node result,
or mere engine liveness stand in for it.

**Pass:** all eight engines load, compile TileLang NSA prefill/decode, capture only graph
batch 1, and return valid samples; no engine restarts; eager/graph and actor-forward
log-probability comparisons meet the declared BF16 tolerance; each rank retains at least
10% HBM after graph capture. Record load time, compile time, TTFT, decode tokens/s, KV
capacity, and peak HBM per rank.

**Stop:** H8 mismatch, padded-head contamination, graph-capture failure, watchdog timeout,
engine restart, invalid token/log-probability, or less than 10% HBM headroom.

### Gate 4: two-step BF16-weight E2E

This is the first gate that joins the full trainer, H8 rollout engines, offload/onload, and
live weight synchronization:

```bash
python "${GLM_AMD_SCRIPT}" train \
  --hardware MI355X \
  --model-org zai-org --model-name GLM-5.2 \
  --num-nodes 8 --num-gpus-per-node 8 --num-rollout 2 \
  --rollout-max-response-len 1024 \
  --run-id glm52-amd-full-bf16-e2e \
  --model-dir "${GLM_MODEL_DIR}" --model-local-dir "${GLM_LOCAL_DIR}" \
  --data-dir "${GLM_DATA_DIR}" --output-dir "${BRINGUP_ROOT}" \
  --extra-args "--ci-test --enable-event-analyzer \
    --debug-exit-after-rollout 2 --save-interval 1 \
    --dump-details ${BRINGUP_ROOT}/full_bf16_e2e/details"
```

**Pass:** both complete cycles finish; the startup CI checker finds zero missing,
unexpected, or unequal tensors when the initial actor is pushed into the original HF
engines; rollout 0 satisfies the CI KL invariant; BF16
`train/train_rollout_logprob_abs_diff <= 0.03`; gradient norm is finite and nonzero; no
metric moves by more than 3x between the two steps without an explained data change; every
update operation succeeds, every logical engine reports the rollout manager's current
weight version, and all eight engines have identical post-update weight checksums for each
logged update.

The existing exact-value checker is a **startup** check: it snapshots the original engine
weights before actor initialization, resets them, then verifies the initial actor push. On
later optimizer steps the default non-FT/v1 path now queries every engine's version and
checksum immediately after synchronization and fails on a stale version or replica
divergence. It still does not byte-compare the updated actor against every engine. Engine
checksum agreement proves replicas agree with each other, not that they equal the actor.
Keep this limitation in the result; do not report “exact live weight parity” unless a
post-update actor-versus-engine checker is added and run. Requalify this invariant
separately before enabling the fault-tolerant train-group path.

**Stop:** a disabled startup checker, startup weight mismatch, engine checksum divergence,
log-probability difference above 0.03, non-finite metric, failed engine-version probe,
resource leak or swap activity across wake/sleep, or a second step more than 25% slower
without a logged compile/checkpoint cause.

### Gate 5: save and resume

Do not accept “it loaded and ran” as resume correctness. Compare an uninterrupted
four-rollout control with a separate two-plus-two split run under deterministic inference,
training kernels, and collectives. The checksum mode below hashes the full actor and
optimizer state and is intentionally expensive; use it for this gate, not throughput runs.

Set deterministic runtime knobs once, then use the same image, data order, seed, topology,
and SGLang limits for all three invocations:

```bash
export DETERMINISTIC_ENV='{"NCCL_ALGO":"Ring","NVTE_ALLOW_NONDETERMINISTIC_ALGO":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8"}'

# Uninterrupted control: rollouts 0 through 3.
python "${GLM_AMD_SCRIPT}" train \
  --hardware MI355X \
  --model-org zai-org --model-name GLM-5.2 \
  --num-nodes 8 --num-gpus-per-node 8 --num-rollout 4 \
  --rollout-max-response-len 1024 \
  --run-id glm52-amd-full-control \
  --extra-env-vars "${DETERMINISTIC_ENV}" \
  --model-dir "${GLM_MODEL_DIR}" --model-local-dir "${GLM_LOCAL_DIR}" \
  --data-dir "${GLM_DATA_DIR}" --output-dir "${BRINGUP_ROOT}" \
  --extra-args "--ci-test --enable-event-analyzer --save-local-weight-checksum \
    --sglang-enable-deterministic-inference --deterministic-mode \
    --debug-deterministic-collective --seed 1234 \
    --debug-exit-after-rollout 4 --save-interval 1 \
    --dump-details ${BRINGUP_ROOT}/full_control/details"

# Split phase A: rollouts 0 and 1, saving each step.
python "${GLM_AMD_SCRIPT}" train \
  --hardware MI355X \
  --model-org zai-org --model-name GLM-5.2 \
  --num-nodes 8 --num-gpus-per-node 8 --num-rollout 4 \
  --rollout-max-response-len 1024 \
  --run-id glm52-amd-full-resume \
  --extra-env-vars "${DETERMINISTIC_ENV}" \
  --model-dir "${GLM_MODEL_DIR}" --model-local-dir "${GLM_LOCAL_DIR}" \
  --data-dir "${GLM_DATA_DIR}" --output-dir "${BRINGUP_ROOT}" \
  --extra-args "--ci-test --enable-event-analyzer --save-local-weight-checksum \
    --sglang-enable-deterministic-inference --deterministic-mode \
    --debug-deterministic-collective --seed 1234 \
    --debug-exit-after-rollout 2 --save-interval 1 \
    --dump-details ${BRINGUP_ROOT}/full_resume/details"

# Split phase B: identical state knobs; resume at rollout 2 and finish rollout 3.
python "${GLM_AMD_SCRIPT}" train \
  --hardware MI355X \
  --model-org zai-org --model-name GLM-5.2 \
  --num-nodes 8 --num-gpus-per-node 8 --num-rollout 4 \
  --rollout-max-response-len 1024 \
  --run-id glm52-amd-full-resume \
  --extra-env-vars "${DETERMINISTIC_ENV}" \
  --model-dir "${GLM_MODEL_DIR}" --model-local-dir "${GLM_LOCAL_DIR}" \
  --data-dir "${GLM_DATA_DIR}" --output-dir "${BRINGUP_ROOT}" \
  --extra-args "--ci-test --ci-disable-weight-update-checker \
    --enable-event-analyzer --save-local-weight-checksum \
    --sglang-enable-deterministic-inference --deterministic-mode \
    --debug-deterministic-collective --seed 1234 \
    --debug-exit-after-rollout 2 --save-interval 1 \
    --dump-details ${BRINGUP_ROOT}/full_resume/details"
```

`--debug-exit-after-rollout` counts work in the current process, while `--num-rollout` is
the global exclusive end. Split phase B intentionally disables only the startup
weight-equality checker. That checker
snapshots the original HF engine before the resumed actor loads; a correct resumed actor
has already changed and would fail comparison with that old snapshot. Keep the other CI
checks enabled, and require the resumed push and CI version probe to succeed with identical
engine checksums. This still does not provide actor-versus-engine byte parity after resume;
report that limitation with the result.

**Pass:** split phase B loads the latest complete checkpoint and begins at rollout 2;
control and split runs produce identical rollout token IDs and declared-tolerance
log probabilities for rollouts 2 and 3; the final actor parameter and optimizer checksum
events match bit-for-bit; scheduler/iteration/consumed-sample fields match; all-engine
version and cross-engine checksum probes succeed. Save/load logs contain no missing or
unexpected keys, and the final checkpoint advances rather than overwriting an older
iteration. Record checkpoint RNG metadata explicitly; do not claim RNG preservation from
liveness alone.

**Stop:** rollout ID or scheduler reset, partial-rank checkpoint, changed topology,
reconversion of the source checkpoint, metric discontinuity caused by lost state, or a
resume-only weight-sync failure.

After resume passes, repeat one BF16 E2E step at response lengths 4096 and 8192, in that
order, without changing topology. The 10-step stability run must use the final 4096-token
prompt limit, 8192-token response limit, and 16384-token/GPU packing target. This runbook
does not qualify 32K responses; that requires another memory and numerical ramp.

### Efficiency ramp after correctness

Do not spend the first allocation tuning three coupled SGLang limits. After Gates 1-5 and
the 10-step BF16 stability run pass, change one axis at a time and repeat at least two E2E
steps with the same numerical and weight-sync checks:

1. Static memory fraction: `0.70 -> 0.75 -> 0.80`.
2. CUDA-graph batch ceiling: `1 -> 8 -> 32`.
3. Maximum running requests: `32 -> 128 -> 256`.

Keep the highest setting that preserves at least 10% measured HBM headroom, stable
load/offload transitions, and the accepted log-probability tolerance. Record tokens/s,
TTFT, graph memory, KV capacity, training step time, all-to-all time, and weight-sync time
at every point. A faster setting that changes numerical acceptance or intermittently OOMs
is not the production recipe.

## FP8 rollout is a separate qualification

`--fp8-rollout` block-quantizes the HF checkpoint at 128x128 while the Megatron actor stays
BF16. It does **not** validate FP8 actor training. Run it only after all BF16 gates pass.

FP8, Mori/DeepEP, MTP, indexer replay, and a trainable full-model router are guarded as
unvalidated features. Every FP8 lifecycle command must opt in explicitly, and `prepare-cp`
must use the same flag so it copies the FP8 rather than BF16 rollout directory:

```bash
python "${GLM_AMD_SCRIPT}" prepare \
  --hardware MI355X --model-org zai-org --model-name GLM-5.2 \
  --num-nodes 8 --num-gpus-per-node 8 \
  --fp8-rollout --no-use-deepep --allow-unvalidated-features \
  --model-dir "${GLM_MODEL_DIR}" --model-local-dir "${GLM_LOCAL_DIR}" \
  --data-dir "${GLM_DATA_DIR}"

python "${GLM_AMD_SCRIPT}" prepare-cp \
  --hardware MI355X --model-org zai-org --model-name GLM-5.2 \
  --num-nodes 8 --num-gpus-per-node 8 \
  --fp8-rollout --no-use-deepep --allow-unvalidated-features \
  --model-dir "${GLM_MODEL_DIR}" --model-local-dir "${GLM_LOCAL_DIR}" \
  --data-dir "${GLM_DATA_DIR}"
```

Archive the FP8 `.miles-artifact.json`, conversion log, index, tensor-dtype inventory, and
scale-shape inventory. Verify that each block-quantized tensor has its expected 128x128
scale and that all referenced shards are present on every node.

First repeat both Gate 3 subgates with the following launcher flags; this isolates
quantization and TileLang from Mori/DeepEP communication. Then run at least three E2E steps
with the same flags plus:

```text
--fp8-rollout --no-use-deepep --allow-unvalidated-features
--ci-test --ci-disable-logprobs-checker
--check-weight-update-allow-quant-error
--enable-event-analyzer
--dump-details /shared/glm52-amd-bringup/full_fp8_e2e/details
```

Do not call this gate valid until the image pins SGLang PR
[#34330](https://github.com/sgl-project/sglang/pull/34330) or an equivalent merged commit.
Without canonicalizing AITER-shuffled block-FP8 tensors and scales, weight-check failures
can be checker artifacts. Conversely, do not turn the checker off and interpret liveness as
correctness. The quantization-aware exact-value check applies to the **initial** actor push
into the HF engine snapshot. Subsequent live updates currently provide success/version
checks and cross-engine checksums, not actor-versus-engine `num_exceed`. Do not describe
the initial checker as an every-step proof.

FP8-specific hazards and gates:

- Verify every quantized tensor has the expected 128x128 scale tensor and that conversion
  reports no missing or silently BF16 expert weights.
- Require startup `num_exceed=0` with the one-ULP quantization allowance, every later
  update and CI version probe to succeed, and identical post-update checksums across engines.
- Use provisional, explicitly unvalidated limits of
  `train/train_rollout_logprob_abs_diff <= 0.05` and
  `abs(train/train_rollout_kl) <= 0.005`, with no upward trend across three steps. Establish
  model-specific baselines before tightening them.
- The default eight-GPU TP8 engine exercises padded H8. A four-GPU TP4/H16 FP8 engine may
  fit because weights are smaller, but memory fit, graph capture, EP communication, and two
  engines per host are all unvalidated and the guarded launcher does not emit that shape.
  Treat it as a new topology, not a transparent optimization.
- Enable Mori/DeepEP only after the plain RCCL/all-to-all FP8 run passes, then repeat the
  SGLang-only and E2E gates. A communication change invalidates prior throughput and
  deadlock evidence.

Stop FP8 qualification on any missing scale, startup one-ULP checker exceedance,
cross-engine checksum divergence, a failed update/version probe, H8 reference mismatch,
log-probability difference above 0.05, KL above 0.005, monotonic precision drift, or engine
restart. A run that passes these checks still carries the explicit post-step
actor-versus-engine parity gap described in Gate 4.

## Minimum evidence before saying “supported”

Archive one row per rollout containing:

- image digest and dependency-manifest checksum;
- RCCL all-reduce/all-gather/all-to-all bandwidth and error counters;
- H8/H16 kernel shape, compile time, and reference error;
- peak HBM and host RAM per rank;
- rollout load/TTFT/decode time, trainer forward/backward/optimizer time, and weight-sync
  time;
- loss, gradient norm, `ppo_kl`, train/rollout log-probability difference and KL, raw
  reward, startup FP8 `num_exceed`, and per-rollout engine version/checksums;
- checkpoint save/load time, iteration, rollout ID, and resume result.

The minimum support bar is: five-layer boundary, one full trainer-only step, one full
SGLang-only rollout, two BF16 E2E steps, a four-step save/resume run, and then a separate
10-step BF16 stability run with no disabled checker. FP8 rollout requires its own three-step
qualification followed by a 10-step stability run. A single successful process launch or
forward pass is bring-up evidence, not training support.
