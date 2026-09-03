# ComfyUI Kirei VDN-H3

Native ComfyUI integration of VideoDeltaNet for MiniMax-H3. The node patches ComfyUI's
existing H3 model and keeps the native loader, sampler, VAE, audio path and conditioning
stack intact.

## Main node

Use **Kirei Apply VDN-H3**.

Required inputs:

- `model`
- `vdn_checkpoint`
- `profile`
- `apply_turbo_adapter`
- `strength`

Profiles:

| Profile | Intended use | Behaviour |
| --- | --- | --- |
| `auto` | Recommended | Chooses branch placement from available VRAM and selects accelerated backends automatically |
| `max_speed` | Large-VRAM GPU | Resident VDN branch weights and fastest available paths |
| `balanced` | General use | Optimized inference with automatic branch placement |
| `low_vram` | 24 GB-class GPUs | Pinned-CPU branch streaming with double buffering |
| `reference` | Numerical/debug comparison | Reference attention, factorized merge and eager/autograd-safe linear branch |

Advanced overrides expose branch placement, LoRA mode, attention backend, linear
kernels, strict validation and diagnostics. Workflows saved with earlier versions remain
compatible.

## Runtime highlights

- low-rank LoRA execution without dense `B @ A` materialization;
- shared down projections for multiple LoRA terms targeting the same native module;
- selective native factorized merge for the H3 fused `mlp.fc2` path;
- ComfyUI-accounted VDN branch/adapter storage;
- resident and pinned-CPU double-buffered streaming modes;
- FP32 recurrent state parameters where required;
- early release of large Q/K/V and softmax intermediates;
- separate reference and optimized inference recurrences;
- Triton, `torch.compile`, Conv1d and eager kernel fallbacks;
- grouped SDPA, FlexAttention and Blackwell decomposed FA4 attention;
- automatic backend fallback and per-model caches;
- runtime diagnostics with stage timings and CUDA memory reporting.

No ComfyUI core files are modified.

## Runtime report

**Kirei VDN-H3 Runtime Report** returns a JSON snapshot containing the selected branch
mode, requested/actual attention backend, fallback reasons, adapter memory/dtypes,
forward count, stage timings and CUDA memory.

To capture the report after generation, connect the sampler LATENT (or another
downstream output) to the optional `after` input. Enable `diagnostics` on the Apply node
only while profiling because diagnostics synchronize CUDA.

## CUDA validation

A standalone synthetic probe is included:

```bash
python tests/probe_optimized_cuda.py --device cuda:0 --json vdn-gpu0.json
python tests/probe_optimized_cuda.py --device cuda:1 --json vdn-gpu1.json
```

Use `--quick` for a small first pass. The probe checks numerical parity and performance
of temporal kernels, recurrent scans, attention backends and streamed branch transfers.

## Dependencies

The portable grouped path uses the normal ComfyUI environment. Optional acceleration:

- Triton for fused temporal convolution;
- PyTorch FlexAttention;
- `flash_attn.cute` / FA4 for Blackwell decomposed attention.

Missing optional acceleration does not prevent `auto` from falling back to a supported
path.

## MultiGPU

The current integration runs a patched H3 model on one compute device. ComfyUI
`deepclone_multigpu` is rejected explicitly rather than sharing runtime closures across
independent GPU clones. Either GPU may still be selected as the model device.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — runtime architecture and memory model.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — tests, CUDA probes and profiling workflow.

## License and attribution

Source is Apache-2.0. Adapted upstream work and notices are recorded in `NOTICE` and
`THIRD_PARTY.md`. Model weights are not bundled and are governed by their own licenses.
