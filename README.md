# ComfyUI Kirei VDN-H3

> **EXPERIMENTAL SOFTWARE — OPTIMIZED INFERENCE BRANCH**
>
> `ComfyUI-Kirei-VDN-H3` is a native ComfyUI integration of VideoDeltaNet for
> MiniMax-H3. It patches ComfyUI's existing H3 model instead of replacing the model,
> sampler, VAE, audio path or conditioning stack.

## Optimization status

The optimization pass described in the 2026-09-03 audit is implemented on the
`audit-optimization-20260903` branch.

Implemented:

- one production-facing `Kirei Apply VDN-H3` node with runtime profiles;
- legacy `KireiApplyVDNH3Alpha` node id/schema retained for saved workflows;
- low-rank LoRA bypass without materializing dense `B @ A` deltas;
- independent Q/K/V low-rank terms applied to slices of native fused `qkv_proj`;
- selective native factorized merge for `mlp.fc2`, whose H3 fused SwiGLU path does
  not call `fc2.forward`;
- ComfyUI-accounted VDN branch weights as an auxiliary `ModelPatcher`;
- resident and pinned-CPU streaming modes;
- double-buffered CUDA branch prefetch with ready/consumed events;
- FP32 retention for `alpha.A_log` and `alpha.dt_bias`;
- early release of full-sequence Q/K/V/softmax intermediates;
- compact raw video/text Q/K/V copies for the recurrent branch;
- separate autograd-safe reference and inference-only linear branch implementations;
- preallocated prefix/suffix scan using `torch.baddbmm(..., out=...)` in inference;
- optional Triton 5-tap temporal-conv + SiLU + L2Norm kernel;
- fallback chain `Triton -> torch.compile -> depthwise Conv1d -> eager`;
- static-shape compiled frame-stat preparation, gather and epilogue with failure
  latching and model-owned caches;
- grouped SDPA, FlexAttention and Blackwell decomposed FA4 backends;
- automatic attention backend dispatch and graceful fallback;
- model-owned diagnostics for synchronized stage timings and CUDA memory snapshots;
- auxiliary Comfy-managed storage for pruned AdaLN curve adapters.

No ComfyUI core file is modified.

## Main node

Use **Kirei Apply VDN-H3**.

Required inputs:

- `model`
- `vdn_checkpoint`
- `profile`
- `apply_turbo_adapter`
- `strength`

Profiles:

| Profile | Intended use | Default behavior |
| --- | --- | --- |
| `auto` | Recommended | Chooses branch residency from available VRAM; low-rank bypass; automatic attention/kernels |
| `max_speed` | Large-VRAM GPU | Resident branch weights; low-rank bypass; fastest available kernels/backends |
| `balanced` | General use | Automatic branch placement with optimized inference paths |
| `low_vram` | 24 GB-class cards | Stream branch weights from pinned CPU memory with double buffering |
| `reference` | Numerical/debug comparison | Stream weights, native factorized merge, reference attention and eager/autograd-safe branch |

Advanced overrides:

- `branch_mode`: `auto | resident | stream`
- `lora_mode`: `auto | bypass | merge`
- `attention_backend`: `auto | grouped | flex | decomposed | reference`
- `linear_kernels`: `auto | triton | compile | conv1d | eager`
- `strict_validation`
- `diagnostics`

The old **Kirei Apply VDN-H3 (Legacy Alpha)** node remains registered so existing
workflows continue to open.

## Runtime design

### LoRA

Ordinary adapter terms stay factorized. The optimized path evaluates:

```text
base_linear(x) + scale * B(A(x))
```

rather than constructing a full dense `B @ A` weight delta. For fused QKV, Q, K and V
remain separate factor pairs and each result is added only to its native output slice.
This preserves independent PEFT alpha/rank scaling without the large block-diagonal
zero matrix used by some fused-QKV ports.

`mlp.fc2` is deliberately merged through ComfyUI's native factorized patch mechanism:
MiniMax-H3 invokes it through the fused `linear_input_act` SwiGLU path, so a normal
`forward` bypass hook would never execute.

### Branch weights

VDN branch tensors are represented as non-trainable parameters in an auxiliary model,
allowing ComfyUI to account for their real size and lifecycle.

- `resident`: ComfyUI can load/offload the auxiliary model with the patched H3 model.
- `stream`: CPU masters remain pinned and two reusable GPU slots overlap transfer of
  block N+1 with computation of block N. CUDA events prevent a slot from being
  overwritten while its current weights are still in use.

### Linear branch

Training/reference execution keeps the original list/stack recurrence for autograd.
Inference preallocates prefix/suffix state banks and writes them with
`torch.baddbmm(..., out=...)`, reducing transient scan storage.

The temporal short-convolution path prefers the OpenVDN-style five-tap Triton kernel
that fuses temporal convolution, SiLU and optional L2 normalization. Portable fallbacks
remain available.

### Attention

`auto` resolves per runtime geometry:

1. Blackwell + `flash_attn.cute`: decomposed backend;
2. small number of grouped windows: grouped SDPA;
3. sufficiently long CUDA sequence with Flex available: FlexAttention;
4. otherwise grouped SDPA.

The decomposed Blackwell backend sends dense global/anchor rows through dense SDPA and
window groups through FA4 varlen attention. A backend failure is latched in the
model-owned cache and falls back to grouped attention instead of repeatedly failing.

## Diagnostics

Enable `diagnostics` only while profiling. It intentionally synchronizes CUDA around
measured stages, so it is not a zero-overhead production option. The runtime records:

- QKV projection;
- norm/RoPE;
- softmax;
- branch-weight transfers;
- output projection;
- linear branch stages (features, frame statistics, scan, gather, epilogue);
- total forward time;
- allocated/reserved/peak CUDA memory.

`vdn_h3_private.benchmark.runtime_snapshot(model)` exposes the resolved runtime state
and the latest diagnostic snapshot for repeatable experiments.

## Validation

Before this optimization pass, the repository's synthetic suite had 55 passing tests.
The rewritten paths add optimization-specific CPU tests covering:

- reference/inference scan parity;
- reference-branch autograd;
- decomposed-attention plan equivalence for every anchor mode;
- FP32 branch-state policy;
- fused-QKV bypass slice isolation;
- full-cover gate parity.

Those focused tests pass in the implementation harness. Python compilation also passes
for every new/rewritten module.

A real post-rewrite CUDA integration run is still required before merging this branch
to a production installation. In particular, the following cannot be validated in the
CPU-only implementation container:

- Triton kernel execution;
- pinned-memory overlap and CUDA event timing;
- `flash_attn.cute` FA4 execution;
- end-to-end ComfyUI lifecycle with the released VDN checkpoint;
- visual parity and final generation speed on RTX PRO 6000 / RTX 4090.

The code is written so all three accelerated areas have safe portable fallbacks.

## Dependencies

The normal grouped path requires only the dependencies already used by ComfyUI plus the
checkpoint loader requirements of this node.

Optional acceleration:

- Triton for the fused temporal kernel;
- PyTorch FlexAttention for `flex`;
- `flash_attn.cute` / FA4 for the Blackwell `decomposed` backend.

Missing optional acceleration does not prevent the `auto` profile from falling back.

## Safety and licensing

The implementation references Apache-2.0 projects recorded in `THIRD_PARTY.md`. The
inference scan, fused temporal-convolution algorithm and decomposed attention design are
adapted from OpenVDN and must retain applicable attribution/license notices.

Model weights are not bundled and remain governed by their own licenses. Verify that
you are authorized to use the relevant MiniMax/VDN weights in your territory.

## Engineering notes

- [`docs/PORT_REPORT.md`](docs/PORT_REPORT.md) — original feasibility and upstream audit.
- [`docs/OPTIMIZATION_IMPLEMENTATION.md`](docs/OPTIMIZATION_IMPLEMENTATION.md) — implementation map for this optimization pass.
