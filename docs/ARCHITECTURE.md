# VDN-H3 runtime architecture

This document describes the current runtime as a product architecture. It focuses on
mathematical invariants, tensor lifetime, memory placement and backend contracts.

## 1. Integration boundary

The project patches ComfyUI's native MiniMax-H3 model rather than loading an independent
Diffusers transformer.

Reused from ComfyUI:

- H3 base checkpoint and quantized loaders;
- fused native QKV projection;
- Q/K normalization and RoPE;
- output projection;
- text/audio/video packed layout;
- sampler and conditioning;
- VAE/audio pipeline;
- model load/offload lifecycle.

Added by this project:

- VDN local-window attention semantics;
- bidirectional linear recurrent branch;
- softmax/output gates;
- VDN branch checkpoint storage;
- Diffusers-to-native LoRA target conversion;
- pruned AdaLN curve injection;
- memory/attention/kernel policies.

No ComfyUI core file is modified.

---

## 2. Mathematical contract

For a frame `t`, after the checkpoint's feature transforms:

```text
A_t = K_t^T diag(beta_t) K_t
B_t = V_t^T diag(beta_t) K_t
```

The VDN solve produces a per-frame affine recurrence pair. The implementation retains
the released `vdn_solve` Cholesky formulation and executes forward and reverse scans.

The branch readout for a frame gathers recurrent state representing context outside the
frame's local softmax window.

The following values remain FP32 even when the H3 compute path is BF16:

- `alpha.A_log`;
- `alpha.dt_bias`;
- frame-statistic matrix A;
- Cholesky factorization / inverse construction;
- recurrence state banks.

The local softmax path remains an exact attention operation unless an explicitly chosen
compatibility/experimental backend says otherwise.

### Anchor handling

The trained first/last dense anchors are represented in two ways:

- local attention gives them the configured dense row/column behavior;
- the linear branch skips their output rows when the trained mode is `both`.

The optimized path removes those rows before the large output projection rather than
projecting zeros.

---

## 3. Packed layout

ComfyUI's `PackedLayout` is adapted into an immutable `VDNLayout` published through a
`ContextVar` for one diffusion-model forward.

The layout records:

- packed sequence length;
- video start/end rows;
- latent frame count;
- tokens per frame;
- spatial token grid;
- text segment;
- exact frame window bounds;
- full-cover flag;
- anchor mode.

The request-local `ContextVar` prevents layout state from leaking between nested or
concurrent model executions.

---

## 4. Attention dataflow

### Inference path

```text
x
│
├─ native fused qkv_proj
│      │
│      ├─ raw Q/K/V views ─────────────────────────────┐
│      │                                                │
│      │                         VDN linear branch      │
│      │                         ├─ stats / scan        │
│      │                         ├─ gather / gate       │
│      │                         └─ output projection   │
│      │                                                │
│      └─ native Q/K norm + RoPE                        │
│                │                                      │
│                └─ local softmax ── softmax gate       │
│                                   └─ H3 out_proj       │
│                                                          
└─────────────────────────────────────────────── add linear delta
```

The important ordering is that the linear branch consumes raw Q/K/V **before** the
native in-place inference RMSNorm+RoPE mutates Q/K storage.

Once the projected linear delta has been produced, only that H3-hidden-size tensor must
survive through softmax. Complete raw video Q/K/V copies are not needed.

### Reference/autograd path

The reference path deliberately remains conservative:

- compact raw video/text Q/K/V copies are made;
- native attention runs;
- the autograd-safe list/stack recurrence is used;
- no inference-only in-place scan buffers are required.

This separates correctness/reference semantics from memory optimization.

---

## 5. Linear branch execution

### Untiled inference

1. Build Q/K/V features.
2. Form alpha, A and B for all frames.
3. Run shared preallocated forward/reverse scans.
4. Gather complementary state for all frames.
5. Apply output gate and RMSNorm.
6. Project only non-anchor rows.

### Tiled inference

Tiling changes feature materialization but not the recurrence.

#### Statistics pass

For each frame tile:

```text
K/V raw tile + temporal halo
 -> optional spatial depthwise conv
 -> 5-tap temporal conv
 -> SiLU / L2 normalization
 -> beta
 -> A/B statistics
```

Temporal halo is two frames on each side for the 5-tap kernel. The halo is feature
context only; only the tile's center-frame statistics are retained.

#### Scan

All per-frame A/B/alpha statistics are now compact. The exact bidirectional scan runs
once across the complete frame sequence.

#### Readout pass

For each Q/x tile:

```text
Q feature
 -> gather prefix/suffix state for the tile's frame range
 -> matrix readout
 -> output gate / RMSNorm
 -> to_out_linear
 -> write final delta slice
```

Tiling therefore removes complete prepared Q/K/V, gate and readout tensors from the
long-lived working set.

---

## 6. Shared runtime caches

One `SharedBranchRuntime` belongs to one `VDNState` and is reused by all transformed
DiT blocks.

It owns:

- gather-index cache;
- static compiled-operation cache;
- temporal kernel cache;
- delta-backend cache.

This prevents 50 blocks from independently building equivalent compiler wrappers and
shape caches.

### Compile policy

`SharedCompilerCache` uses static `torch.compile(dynamic=False)` callables and latches
compile/runtime failures per operation/shape.

Policies:

- `off`;
- `shared`;
- `reduce_overhead`;
- `max_autotune`.

The scan compile path is an optimization only. Failure returns to the same preallocated
`baddbmm(..., out=...)` recurrence.

---

## 7. Temporal/spatial features

Spatial short convolution remains PyTorch depthwise Conv2d so cuDNN can choose its
native implementation.

Temporal short convolution dispatches through:

1. Triton fused 5-tap conv + SiLU + optional L2Norm;
2. compile spelling when requested/available;
3. grouped depthwise Conv1d;
4. eager shift reference.

The optimized kernels can write directly as:

```text
[F, H, S, D]
```

for frame-statistic consumption, avoiding a complete post-kernel repack.

---

## 8. Branch weight memory

Branch storage is represented by real non-trainable `nn.Parameter` trees and attached
to the base ModelPatcher through ComfyUI additional models.

### Resident

All branch weights load/offload through ComfyUI.

### Stream

All branch weights remain CPU masters. Two CUDA buffers are reused.

### Hybrid

Each block is partitioned into:

- resident small branch tensors;
- one streamable dominant projection representation.

The streamable representation may differ by block. This matters for the FP8 profile:

- edge blocks stream canonical BF16 `to_out_linear.weight`;
- interior blocks stream `to_out_linear.weight_fp8`;
- FP32 scale tensors are small and resident.

### Streaming correctness

A stream slot tracks independently:

- allocated GPU buffers;
- current block identity;
- `valid_keys` for that block;
- reusable `ready` event;
- reusable `consumed` event.

Switching blocks clears validity even if buffers with the same names/shapes are still
allocated. This prevents a partial gate-only load from making stale tensors appear valid
for a later full-block request.

### Prefetch schedule

- current block can be scheduled before native QKV;
- N+1 is scheduled while N computes;
- block 0 can be scheduled after block 49 for the next denoising step.

Pinned memory respects ComfyUI's policy unless `pin_strategy=all` is explicitly chosen.

---

## 9. Adapter runtime

### Target mapping

Upstream Diffusers names are mapped to native H3 names, including:

- transformer block prefix conversion;
- fused Q/K/V slices;
- text refiner blocks;
- SwiGLU half ordering;
- final/pruned AdaLN targets.

### Factorized bypass

For each native module, all LoRA A/down matrices are concatenated into one down
projection.

Terms are then grouped by output slice. Within a group, scaled B/up matrices are
concatenated column-wise, so:

```text
[B1*s1 | B2*s2] @ [A1(x); A2(x)]
== B1 A1(x) s1 + B2 A2(x) s2
```

This produces one up GEMM per slice with the exact same low-rank model.

### Fused fc2 exception

Native H3 can consume `mlp.fc2` through fused `linear_input_act` without invoking
`fc2.forward`. Those factors cannot be implemented as an activation bypass and are
therefore sent through native factorized merge.

---

## 10. AdaLN curve factors

Pruned curve factors are stored as another Comfy auxiliary model.

The e-grid search order is:

1. explicit path;
2. checkpoint root;
3. checkpoint `linear_branch` directory;
4. checkpoint parent;
5. configured VDN model roots;
6. legacy sibling-node location.

This allows new checkpoints to be self-contained while preserving old installations.

---

## 11. Attention implementations

### Grouped SDPA

Frames with identical allowed-key sets are represented as contiguous query runs. Each
run performs one exact dense SDPA against the selected global/video keys.

### FlexAttention

The exact mask is represented as a cached block mask. The compiled Flex call is model-
owned and failures are latched.

### Decomposed varlen attention

A cached plan partitions query rows into:

- dense globals / dense anchor rows;
- local window groups.

Dense rows use exact SDPA. Local groups are gathered into varlen batches and dispatched
to:

- FA2 (`flash2`), or
- CuTe/FA4 (`decomposed`).

The final output is scattered back to packed-row order.

### Calibration

`auto` checks a persistent calibration key before generic heuristics. The key contains
both hardware/software identity and exact packed video geometry.

No calibration benchmark runs during ordinary inference.

---

## 12. Experimental FP8 projection

Only the large linear-branch output projection is quantized.

### Storage

Interior block CPU master:

```text
weight_fp8 + weight_scale_fp32
```

There is no hidden BF16 copy of that interior projection in the branch store.

### Edge blocks

By default first/last 4 transformed blocks retain BF16 projection weights.

### Scaling

- Blackwell-class: per-tensor activation and weight scale;
- earlier supported path: rowwise activation scale + per-output-channel weight scale.

### Dispatch

1. actual-device `_scaled_mm` capability probe before model construction;
2. FP8 activation quantization;
3. `_scaled_mm`;
4. if a particular shape/kernel fails, dequantize the already-quantized weight and use
   ordinary F.linear rather than aborting the render.

Under grad-enabled execution the dequantized F.linear spelling is used directly.

FP8 is never selected silently by ordinary profiles.

---

## 13. Diagnostics

CUDA stage timing uses recorded start/end events rather than synchronizing around every
scope. Events are resolved together when a report is requested.

Branch streaming maintains counters for H2D bytes, copies, prefetches, served requests,
wait events, pinned CPU memory and allocated staging buffers.

Diagnostics are observational only; they do not change backend selection except when an
explicit calibration node is run and its result is persisted.

---

## 14. Single-GPU lifecycle

Auxiliary branch/LoRA/curve models use ComfyUI load/offload accounting.

A patched VDN model currently represents one compute device. `deepclone_multigpu` is
rejected by the auxiliary factory because shallowly copied runtime closures/caches would
not constitute a valid distributed VDN model.

A true Ulysses/distributed implementation needs explicit sharding/communication logic
and is intentionally outside this runtime contract.
