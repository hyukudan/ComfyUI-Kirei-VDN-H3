#!/usr/bin/env python3
"""CPU smoke test for Kirei bypass ownership on an installed ComfyUI runtime.

This imports and exercises ComfyUI's real ModelPatcher/PatcherInjection lifecycle on
a tiny torch module. It does not load MiniMax-H3 weights, start ComfyUI, or use a GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comfy-root",
        type=Path,
        required=True,
        help="ComfyUI checkout containing comfy/model_patcher.py",
    )
    return parser.parse_args()


def _prepare_imports(comfy_root: Path) -> Path:
    comfy_root = comfy_root.expanduser().resolve()
    model_patcher_file = comfy_root / "comfy" / "model_patcher.py"
    if not model_patcher_file.is_file():
        raise SystemExit(f"not a ComfyUI checkout: missing {model_patcher_file}")

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(comfy_root))

    # Force this process down ComfyUI's CPU path before model_management imports.
    # Preserve only the script name because ComfyUI owns a separate CLI parser.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.argv[:] = [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    return comfy_root


def _factor(torch, FactorPatch, WeightedFactor, strength: float):
    patch = FactorPatch(
        key="diffusion_model.proj.weight",
        up=torch.ones(2, 1),
        down=torch.ones(1, 2),
        offset=None,
        source="native-smoke",
        scale=1.0,
    )
    return WeightedFactor(patch, strength)


def _check(torch, label: str, actual, expected_value: float):
    expected = torch.full_like(actual, expected_value)
    try:
        torch.testing.assert_close(actual, expected)
    except AssertionError as exc:
        raise AssertionError(f"{label}: expected {expected}, got {actual}") from exc


def main() -> int:
    requested_root = _arguments().comfy_root
    comfy_root = _prepare_imports(requested_root)

    import torch
    import torch.nn as nn
    from comfy.model_patcher import ModelPatcher
    import comfy.model_patcher as native_model_patcher
    from comfy.patcher_extension import PatcherInjection

    from vdn_h3.adapters import FactorPatch
    from vdn_h3.bypass import WeightedFactor, install_bypass

    imported_from = Path(native_model_patcher.__file__).resolve()
    if not imported_from.is_relative_to(comfy_root):
        raise RuntimeError(
            f"expected ModelPatcher under {comfy_root}, imported {imported_from}"
        )

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = nn.Module()
            self.diffusion_model.proj = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.diffusion_model.proj.weight.copy_(torch.eye(2))

        def get_dtype(self):
            return torch.float32

    device = torch.device("cpu")
    base = ModelPatcher(TinyModel(), load_device=device, offload_device=device, size=16)
    clone_a = base.clone()
    clone_b = base.clone()
    merge_clone = base.clone()
    if not (base.model is clone_a.model is clone_b.model is merge_clone.model):
        raise AssertionError("native ModelPatcher clones did not share the tiny inner model")

    install_bypass(clone_a, [_factor(torch, FactorPatch, WeightedFactor, 1.0)])
    install_bypass(clone_b, [_factor(torch, FactorPatch, WeightedFactor, 2.0)])
    count, runtime = install_bypass(merge_clone, [])
    if count != 0 or runtime is not None:
        raise AssertionError("empty bypass installation returned an unexpected runtime")
    for name, patcher in (("A", clone_a), ("B", clone_b), ("merge", merge_clone)):
        injections = patcher.get_injections("vdn_h3_lora_bypass")
        if not injections or not all(isinstance(item, PatcherInjection) for item in injections):
            raise AssertionError(f"{name} did not register a native PatcherInjection")

    x = torch.ones(1, 2, device=device)
    project = base.model.diffusion_model.proj
    clones = (merge_clone, clone_b, clone_a)
    try:
        clone_a.inject_model()
        _check(torch, "A", project(x), 3.0)
        clone_a.inject_model()
        _check(torch, "A repeated", project(x), 3.0)

        clone_b.inject_model()
        _check(torch, "A -> B", project(x), 5.0)
        clone_a.eject_model()
        _check(torch, "stale A eject", project(x), 5.0)

        clone_a.inject_model()
        _check(torch, "B -> A", project(x), 3.0)
        clone_b.eject_model()
        _check(torch, "stale B eject", project(x), 3.0)

        merge_clone.inject_model()
        _check(torch, "A -> merge/empty", project(x), 1.0)
        clone_a.eject_model()
        _check(torch, "stale A eject under merge", project(x), 1.0)
        merge_clone.eject_model()
        _check(torch, "merge eject", project(x), 1.0)

        clone_b.inject_model()
        _check(torch, "merge -> B", project(x), 5.0)
        clone_b.eject_model()
        _check(torch, "final eject", project(x), 1.0)
    finally:
        for patcher in clones:
            try:
                patcher.eject_model()
            except Exception:
                pass

    print("PASS: native ComfyUI ModelPatcher bypass lifecycle (CPU tiny module)")
    print(f"ModelPatcher: {imported_from}")
    print("Covered: repeat inject, A -> B -> A, stale ejects, merge/empty ownership")
    print("Not covered: MiniMax-H3 weights, quantized kernels, CUDA, or image quality")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
