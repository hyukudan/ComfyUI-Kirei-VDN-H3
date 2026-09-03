# VDN-H3 optimization implementation map

Date: 2026-09-03  
Branch: `audit-optimization-20260903`

This document maps the performance/correctness audit to concrete code.

## 1. Factorized LoRA runtime

Files: `vdn_h3_private/bypass.py`, `nodes.py`, existing `adapters.py` / `apply.py`.

- `FrugalLoRABypassAdapter` evaluates `B(A(x))` directly.
- `CompositeQKVBypassAdapter` groups independent Q/K/V factors around one native
  fused-QKV forward and writes only their output slices.
- `LoRABypassRuntime` is an auxiliary Comfy model so factor memory participates in
  lifecycle/accounting.
- `PatcherInjection` installs/ejects reversible forward hooks with the MODEL lifecycle.
- `mlp.fc2` is excluded from bypass and stays on ComfyUI's native factorized merge path
  because MiniMax-H3 consumes it via fused `linear_input_act` rather than `fc2.forward`.
- Pruned AdaLN curve factors remain runtime low-rank and are stored in a Comfy-managed
  auxiliary model.

## 2. Full-sequence memory lifetime

File: `vdn_h3_private/hybrid.py`.

- Raw video Q/K/V and optional text K/V are copied into compact tensors before RoPE.
- In inference, fused RMSNorm+RoPE can mutate the full Q/K storage in place because the
  linear branch no longer depends on it.
- Full query/key/value/QKV views are deleted immediately after softmax.
- Softmax output and gate are deleted immediately after `out_proj`.
- The recurrent branch therefore overlaps only with compact video/text raw features,
  not the full packed QKV backing buffer.

## 3. Branch-weight lifecycle and streaming

File: `vdn_h3_private/weights.py`.

- Branch tensors are non-trainable parameters in `BranchWeightsModel`.
- The storage is registered through `set_additional_models("vdn_h3_branch", ...)` when
  supported by the active ComfyUI `ModelPatcher`.
- `resident` follows ComfyUI load/offload lifecycle.
- `stream` keeps CPU masters pinned and uses two reusable GPU slots.
- A dedicated CUDA stream fills slot N+1 while block N computes.
- `ready` and `consumed` events enforce producer/consumer ordering.
- Partial gate-only requests cannot be mistaken for a subsequently required full block.
- Stream release synchronizes before dropping reusable GPU buffers.
- `alpha.A_log` and `alpha.dt_bias` remain FP32 independent of compute dtype.

## 4. Separate reference and inference recurrence

File: `vdn_h3_private/branch.py`.

- `run_scans_reference` retains list/stack recurrence and remains autograd-safe.
- `run_scans_inference` preallocates prefix/suffix state banks and writes with
  `torch.baddbmm(..., out=...)` under no-grad/inference execution.
- `LinearBranch.readout(..., inference=...)` selects the complete algorithm body rather
  than sprinkling unsafe in-place operations through the reference path.

## 5. Linear kernels

Files: `vdn_h3_private/kernels.py`, `branch.py`.

Temporal K/V/Q short-convolution dispatch:

1. Triton five-tap convolution + SiLU + optional L2Norm;
2. static `torch.compile(dynamic=False)` fused shift chain;
3. grouped depthwise `conv1d`;
4. eager reference implementation.

The 5x5 spatial depthwise convolution remains `torch.nn.functional.conv2d`, allowing
cuDNN to select its optimized implementation.

Frame-stat preparation, gather and epilogue have model-owned static-shape compile
caches. Compile/runtime failures are latched per shape and fall back instead of being
retried every block.

The A-statistic FP32 matmul enables TF32 only around that inference matmul. Cholesky and
state recurrence remain outside autocast/low-precision math.

## 6. Attention backend dispatch

File: `vdn_h3_private/window.py`.

Available backends:

- `auto`
- `grouped`
- `flex`
- `decomposed`
- `reference`

`auto` prefers decomposed attention on Blackwell when `flash_attn.cute` is importable,
otherwise grouped SDPA for a small number of window groups, Flex for sufficiently long
CUDA sequences, then grouped as the final fallback.

The decomposed path builds one cached plan that partitions query rows into:

- dense globals / dense anchor rows; and
- varlen local-window groups.

Dense rows use dense SDPA (cuDNN requested when available). Local groups use
`flash_attn.cute.interface.flash_attn_varlen_func`. Plan construction is verified on CPU
against the complete attention-mask oracle for all four anchor modes.

## 7. Final ComfyUI node

File: `vdn_h3_private/nodes.py`.

Primary node id: `KireiApplyVDNH3`  
Display name: `Kirei Apply VDN-H3`

Profiles:

- `auto`
- `max_speed`
- `balanced`
- `low_vram`
- `reference`

Advanced overrides expose branch placement, LoRA mode, attention backend, linear
kernels, strict validation and diagnostics.

`KireiApplyVDNH3Alpha` remains registered with its original required-input schema and
maps onto the new implementation to preserve saved workflows.

## 8. Diagnostics

Files: `vdn_h3_private/diagnostics.py`, `benchmark.py`.

Diagnostics are model-owned and disabled by default. When enabled they measure major
attention/linear stages and capture CUDA allocated/reserved/peak memory. Timing scopes
synchronize CUDA intentionally, so diagnostics are for benchmarking rather than normal
production runs.

`runtime_snapshot(model)` returns the resolved branch/attention/kernel policy plus the
latest metrics without mutating the model.

## 9. Validation performed during implementation

CPU implementation harness:

- all rewritten/new Python modules compile;
- optimization-focused pytest suite: 6 passed;
- reference/inference scan parity;
- reference branch autograd;
- grouped window attention versus dense-mask oracle;
- decomposed plan versus mask oracle for `none`, `columns`, `rows`, `both`;
- FP32 branch-state policy;
- QKV bypass slice isolation;
- full-cover softmax gate parity.

The pre-existing repository tests were reviewed for API compatibility, including
`run_scans`, gather cache ownership, grouped SDPA dispatch, `ManagedBranchWeights`,
full-cover behavior and `apply_vdn` reversible patching.

## 10. CUDA validation still required

The implementation environment for this commit is CPU-only. Before merging into the
production ComfyUI installation, run on the target GPUs:

1. the complete repository pytest suite in the real Comfy environment;
2. Triton temporal-kernel parity against eager/compiled fallbacks;
3. stream-mode correctness and transfer overlap;
4. resident-mode Comfy load/offload lifecycle;
5. factorized bypass parity against native merged LoRA on BF16 and INT8/ConvRot bases;
6. Flex/grouped parity on representative H3 geometries;
7. FA4 decomposed/grouped parity when `flash_attn.cute` is available;
8. one-step real-checkpoint smoke, then 41/121/full-frame runs;
9. peak VRAM and sampler-only timing by profile.

Accelerated paths are optional and have portable fallbacks, so lack of Triton/FA4 does
not block the default `auto` profile.
