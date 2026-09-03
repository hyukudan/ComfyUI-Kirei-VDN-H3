# Architecture

Kirei VDN-H3 integrates VideoDeltaNet with ComfyUI's native MiniMax-H3 implementation.
It keeps ComfyUI's model loader, sampler, VAE, audio path and conditioning stack, and
adds the VDN attention behaviour as reversible `ModelPatcher` extensions.

## Model integration

The node patches the MiniMax-H3 DiT attention blocks while leaving the rest of the
native model intact. Checkpoints are loaded from `models/vdn` with strict safetensors
inventory and shape validation.

The runtime consists of:

- local/windowed softmax attention;
- the bidirectional VDN linear branch;
- softmax/output gates;
- Stage-B and optional turbo LoRA adapters;
- optional AdaLN curve reconstruction for compatible pruned H3 bases.

## Adapter path

LoRA factors remain low-rank. Several terms targeting the same native module share one
concatenated down projection, while each up projection writes only to its own output
slice. This avoids dense `B @ A` materialization and avoids block-diagonal fused-QKV
weights.

MiniMax-H3 `mlp.fc2` is handled through ComfyUI's native factorized weight patch because
the fused SwiGLU path does not call `fc2.forward`.

## Branch weights

VDN branch weights are represented by an auxiliary `ModelPatcher`, so ComfyUI accounts
for their size and lifecycle.

- `resident`: branch weights follow normal ComfyUI load/offload behaviour.
- `stream`: CPU masters are pinned and two reusable GPU slots overlap transfer of the
  next block with computation of the current block. CUDA events prevent slot reuse
  until the current consumer has finished.

`alpha.A_log` and `alpha.dt_bias` remain FP32 independently of the main compute dtype.

## Linear branch

The reference path is autograd-safe. The inference path uses preallocated prefix/suffix
state banks and `torch.baddbmm(..., out=...)` to reduce transient memory.

Temporal short convolution dispatches through:

1. Triton fused 5-tap convolution + SiLU + optional L2Norm;
2. static `torch.compile`;
3. depthwise `conv1d`;
4. eager reference implementation.

The spatial 5x5 depthwise convolution remains on PyTorch/cuDNN.

## Attention backends

Available backends are `auto`, `grouped`, `flex`, `decomposed`, and `reference`.

`auto` prefers Blackwell decomposed FA4 when `flash_attn.cute` is available, otherwise
uses grouped SDPA for small window-group counts, FlexAttention for sufficiently long
CUDA sequences, and grouped SDPA as the portable fallback.

Backend failures are cached per patched model so an unavailable accelerated path is not
retried on every block.

## Memory lifetime

Raw video Q/K/V needed by the recurrent branch are copied into compact buffers before
RoPE. Full packed Q/K/V and softmax intermediates are released as soon as their branch
has finished, reducing overlap between the dense and recurrent working sets.

## Runtime profiles

- `auto`: recommended; selects placement and accelerated paths at runtime.
- `max_speed`: resident branch weights and fastest available kernels.
- `balanced`: optimized inference with automatic branch placement.
- `low_vram`: pinned-CPU branch streaming.
- `reference`: eager/reference paths for numerical comparison.

The current integration uses one compute device per patched H3 model. ComfyUI deep-clone
MultiGPU is rejected explicitly until a distributed VDN implementation is provided.
