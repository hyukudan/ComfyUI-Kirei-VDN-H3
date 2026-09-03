# Validation and profiling

## Automated CPU tests

The repository includes mathematical, checkpoint, lifecycle and policy tests under
`tests/`. `.github/workflows/ci.yml` runs a dependency-light suite and a second suite
against a fresh checkout of current ComfyUI when GitHub-hosted runners are available.

## CUDA probe

Run the synthetic CUDA probe in the same Python environment as ComfyUI:

```bash
python tests/probe_optimized_cuda.py --device cuda:0 --json vdn-gpu0.json
python tests/probe_optimized_cuda.py --device cuda:1 --json vdn-gpu1.json
```

Use `--quick` for a smaller first pass.

The probe validates and times:

- eager / Conv1d / compiled / Triton temporal paths;
- reference and preallocated recurrent scans;
- grouped and Flex attention;
- decomposed FA4 attention when available;
- automatic attention backend resolution;
- streamed branch transfers and FP32 state handling.

It exits non-zero when an available accelerated path exceeds its numerical tolerance.

## ComfyUI runtime report

Enable `diagnostics` on **Kirei Apply VDN-H3** and add **Kirei VDN-H3 Runtime Report**.
For post-sampler metrics, connect a sampler or downstream output to the report node's
optional `after` input.

The JSON report contains:

- requested and resolved attention backend;
- backend fallback reasons;
- branch placement and memory footprint;
- LoRA/curve factor dtype and device inventory;
- stage timings;
- current and run-local peak CUDA memory.

Disable diagnostics for normal generation because timing scopes synchronize CUDA.

## Real-model qualification

For a new ComfyUI/PyTorch/GPU combination, verify in this order:

1. repository tests;
2. synthetic CUDA probe;
3. one short real-checkpoint generation;
4. reference-versus-optimized output comparison;
5. representative frame counts/resolutions;
6. sampler-only time and peak VRAM for the intended runtime profile.
