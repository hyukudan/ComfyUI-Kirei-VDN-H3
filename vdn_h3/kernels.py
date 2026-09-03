"""Inference kernels for the VDN-H3 linear branch.

The five-tap Triton temporal kernel follows the Apache-2.0 OpenVDN algorithm recorded
in THIRD_PARTY.md. Kernel choice and torch.compile policy are deliberately independent:
a user can request Triton without paying Inductor warm-up, or enable shared compilation
for the remaining pointwise/gather/scan bodies.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - environment dependent
    triton = None
    tl = None


BLOCK_T = 16
KERNEL_BACKENDS = ("auto", "triton", "conv1d", "eager")
COMPILE_POLICIES = ("off", "shared", "reduce_overhead", "max_autotune")


def _device_key(device: torch.device | str) -> tuple[str, int | None]:
    d = torch.device(device)
    return d.type, d.index


def _compile_kwargs(policy: str) -> dict[str, Any]:
    if policy not in COMPILE_POLICIES:
        raise ValueError(f"unknown compile policy {policy!r}")
    if policy == "off":
        return {}
    if policy == "reduce_overhead":
        return {"mode": "reduce-overhead"}
    if policy == "max_autotune":
        return {"mode": "max-autotune"}
    return {}


def activate(tokens: torch.Tensor, l2norm: bool) -> torch.Tensor:
    out = F.silu(tokens)
    if l2norm:
        return F.normalize(out, dim=-1, eps=1e-6).to(out.dtype)
    return out


def _to_fhsd(out: torch.Tensor, num_frames: int, heads: int, head_dim: int) -> torch.Tensor:
    per_frame = out.shape[1]
    return out.reshape(num_frames, per_frame, heads, head_dim).permute(0, 2, 1, 3).contiguous()


def temporal_shift_eager(x: torch.Tensor, w: torch.Tensor, kernel: int, pad: int) -> torch.Tensor:
    if kernel <= 0 or kernel % 2 == 0 or pad != kernel // 2 or w.shape[-1] != kernel:
        raise ValueError("temporal kernel must be positive, odd and symmetrically padded")
    xp = F.pad(x, (0, 0, 0, 0, pad, pad))
    out = None
    for dt in range(kernel):
        part = xp[dt : dt + x.shape[0]] * w[:, dt].view(1, 1, -1)
        out = part if out is None else out + part
    assert out is not None
    return out


def temporal_conv1d(x: torch.Tensor, w: torch.Tensor, kernel: int, pad: int) -> torch.Tensor:
    if w.shape != (x.shape[-1], kernel):
        raise ValueError(f"temporal weight {tuple(w.shape)} incompatible with x={tuple(x.shape)}")
    xs = x.permute(1, 2, 0).contiguous()
    ys = F.conv1d(xs, w[:, None, :].to(xs.dtype), padding=pad, groups=x.shape[-1])
    return ys.permute(2, 0, 1).contiguous()


def _compiled_tconv_body(
    x, w, kernel: int, pad: int, heads: int, head_dim: int, l2norm: bool, fhsd: bool
):
    raw = temporal_shift_eager(x, w, kernel, pad)
    y = activate(raw.reshape(-1, heads, head_dim), l2norm)
    if fhsd:
        return y.reshape(x.shape[0], x.shape[1], heads, head_dim).permute(0, 2, 1, 3).contiguous()
    return y


def _compiled_activate_body(tokens, l2norm: bool, num_frames: int, per_frame: int, fhsd: bool):
    out = activate(tokens, l2norm)
    if fhsd:
        heads, dim = out.shape[-2:]
        return out.reshape(num_frames, per_frame, heads, dim).permute(0, 2, 1, 3).contiguous()
    return out


if triton is not None:  # pragma: no cover - compiled only on supported CUDA hosts

    @triton.jit
    def _tconv_act_kernel(
        X, W, OUT, T, S_, C_, BLOCK_T_: tl.constexpr, H_: tl.constexpr,
        D_: tl.constexpr, L2: tl.constexpr, FHSD: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_h = tl.program_id(2)
        d = tl.arange(0, D_)
        chan = pid_h * D_ + d
        rows = pid_t * BLOCK_T_ + tl.arange(0, BLOCK_T_)
        valid = rows < T
        acc = tl.zeros((BLOCK_T_, D_), dtype=tl.float32)
        for dt in tl.static_range(5):
            r = rows + dt - 2
            ok = valid & (r >= 0) & (r < T)
            v = tl.load(
                X + (r[:, None] * S_ + pid_s) * C_ + chan[None, :],
                mask=ok[:, None], other=0.0,
            ).to(tl.float32)
            wd = tl.load(W + chan * 5 + dt).to(tl.float32)
            acc += v * wd[None, :]
        y = acc * tl.sigmoid(acc)
        if L2:
            inv = 1.0 / tl.sqrt(tl.maximum(tl.sum(y * y, axis=1), 1e-12))
            y = y * inv[:, None]
        if FHSD:
            offs = ((rows[:, None] * H_ + pid_h) * S_ + pid_s) * D_ + d[None, :]
        else:
            offs = (rows[:, None] * S_ + pid_s) * C_ + chan[None, :]
        tl.store(OUT + offs, y.to(OUT.dtype.element_ty), mask=valid[:, None])


def triton_temporal_conv_activate(
    x: torch.Tensor,
    w: torch.Tensor,
    kernel: int,
    pad: int,
    heads: int,
    head_dim: int,
    l2norm: bool,
    *,
    fhsd: bool = False,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not x.is_cuda:
        raise RuntimeError("Triton temporal convolution requires CUDA")
    if kernel != 5 or pad != 2:
        raise ValueError(f"Triton path supports exactly kernel=5,pad=2; got {kernel},{pad}")
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise ValueError(f"Triton path needs power-of-two head_dim >=16; got {head_dim}")
    if not x.is_contiguous() or not w.is_contiguous():
        raise ValueError("Triton temporal convolution requires contiguous x/w")
    T, S_, C_ = x.shape
    if C_ != heads * head_dim or w.shape != (C_, 5):
        raise ValueError("Triton temporal geometry mismatch")
    shape = (T, heads, S_, head_dim) if fhsd else tuple(x.shape)
    out = torch.empty(shape, device=x.device, dtype=x.dtype)
    _tconv_act_kernel[(triton.cdiv(T, BLOCK_T), S_, heads)](
        x, w, out, T, S_, C_, BLOCK_T_=BLOCK_T, H_=heads, D_=head_dim,
        L2=l2norm, FHSD=fhsd, num_warps=4, num_stages=2,
    )
    return out if fhsd else out.reshape(-1, heads, head_dim)


class LinearKernelCache:
    """Shared compiled-kernel cache with permanent per-shape failure latches."""

    def __init__(self, limit: int = 32, *, compile_policy: str = "shared"):
        if compile_policy not in COMPILE_POLICIES:
            raise ValueError(f"unknown compile policy {compile_policy!r}")
        self.limit = int(limit)
        self.compile_policy = compile_policy
        self._compiled: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._broken: set[tuple[Any, ...]] = set()

    def set_compile_policy(self, policy: str):
        if policy not in COMPILE_POLICIES:
            raise ValueError(f"unknown compile policy {policy!r}")
        self.compile_policy = policy

    def _get_compiled(self, key: tuple[Any, ...], fn, *, policy: str | None = None):
        policy = self.compile_policy if policy is None else policy
        if policy == "off" or key in self._broken:
            return None
        full_key = (policy, *key)
        compiled = self._compiled.get(full_key)
        if compiled is not None:
            self._compiled.move_to_end(full_key)
            return compiled
        try:
            compiled = torch.compile(fn, dynamic=False, **_compile_kwargs(policy))
        except Exception:
            self._broken.add(key)
            return None
        self._compiled[full_key] = compiled
        self._compiled.move_to_end(full_key)
        while len(self._compiled) > self.limit:
            self._compiled.popitem(last=False)
        return compiled

    def activate(
        self,
        tokens: torch.Tensor,
        l2norm: bool,
        *,
        num_frames: int | None = None,
        per_frame: int | None = None,
        fhsd: bool = False,
        compile_policy: str | None = None,
    ) -> torch.Tensor:
        policy = self.compile_policy if compile_policy is None else compile_policy
        if policy == "off" or not tokens.is_cuda:
            out = activate(tokens, l2norm)
            if fhsd:
                if num_frames is None or per_frame is None:
                    raise ValueError("fhsd activation requires frame geometry")
                h, d = out.shape[-2:]
                out = out.reshape(num_frames, per_frame, h, d).permute(0, 2, 1, 3).contiguous()
            return out
        key = (
            "activate", _device_key(tokens.device), tokens.dtype, tuple(tokens.shape),
            bool(l2norm), bool(fhsd), num_frames, per_frame,
        )
        compiled = self._get_compiled(key, _compiled_activate_body, policy=policy)
        if compiled is None:
            return self.activate(
                tokens, l2norm, num_frames=num_frames, per_frame=per_frame,
                fhsd=fhsd, compile_policy="off",
            )
        try:
            return compiled(tokens, l2norm, int(num_frames or 0), int(per_frame or 0), fhsd)
        except Exception:
            self._broken.add(key)
            self._compiled.pop((policy, *key), None)
            return self.activate(
                tokens, l2norm, num_frames=num_frames, per_frame=per_frame,
                fhsd=fhsd, compile_policy="off",
            )

    def temporal(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        kernel: int,
        pad: int,
        heads: int,
        head_dim: int,
        l2norm: bool,
        *,
        mode: str = "auto",
        compile_policy: str | None = None,
        fhsd: bool = False,
    ) -> torch.Tensor:
        # ``compile`` is accepted as a compatibility alias from earlier workflows.
        if mode == "compile":
            mode = "auto"
            compile_policy = "shared" if compile_policy in {None, "off"} else compile_policy
        if mode not in KERNEL_BACKENDS:
            raise ValueError(f"unknown linear kernel backend {mode!r}")
        policy = self.compile_policy if compile_policy is None else compile_policy
        if mode in {"auto", "triton"} and x.is_cuda and triton is not None:
            try:
                return triton_temporal_conv_activate(
                    x, w, kernel, pad, heads, head_dim, l2norm, fhsd=fhsd
                )
            except Exception:
                pass
        if policy != "off" and x.is_cuda:
            key = (
                "tconv", _device_key(x.device), x.dtype, tuple(x.shape), tuple(w.shape),
                kernel, pad, heads, head_dim, bool(l2norm), bool(fhsd),
            )
            compiled = self._get_compiled(key, _compiled_tconv_body, policy=policy)
            if compiled is not None:
                try:
                    return compiled(x, w, kernel, pad, heads, head_dim, l2norm, fhsd)
                except Exception:
                    self._broken.add(key)
                    self._compiled.pop((policy, *key), None)
        if mode in {"auto", "triton", "conv1d"}:
            try:
                out = temporal_conv1d(x, w, kernel, pad)
                activated = activate(out.reshape(-1, heads, head_dim), l2norm)
                if fhsd:
                    return activated.reshape(x.shape[0], x.shape[1], heads, head_dim).permute(0, 2, 1, 3).contiguous()
                return activated
            except Exception:
                pass
        out = temporal_shift_eager(x, w, kernel, pad)
        activated = activate(out.reshape(-1, heads, head_dim), l2norm)
        if fhsd:
            return activated.reshape(x.shape[0], x.shape[1], heads, head_dim).permute(0, 2, 1, 3).contiguous()
        return activated

    def release(self) -> None:
        self._compiled.clear()
        self._broken.clear()


__all__ = [
    "BLOCK_T",
    "COMPILE_POLICIES",
    "KERNEL_BACKENDS",
    "LinearKernelCache",
    "activate",
    "temporal_conv1d",
    "temporal_shift_eager",
    "triton_temporal_conv_activate",
]
