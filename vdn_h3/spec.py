"""Strict, safetensors-only loading for exploded VDN-H3 checkpoints.

The loader deliberately treats the checkpoint directory as an inventory rather than
as a bag of files.  Every adapter declared by the ModelSpec must have one directory,
every tensor in every safetensors file must be understood, and every resolved path
must remain below one of ComfyUI's ``models/vdn`` roots.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


SPEC_FORMAT_VERSION = 2
HYBRID_TRANSFORM_VERSION = 2
SUPPORTED_DELTA_RULES = ("vdn_solve", "sana_scaled")
SUPPORTED_ANCHORS = ("none", "columns", "rows", "both")
SUPPORTED_BRIDGES = ("alpha", "none")
SHORT_CONV_TARGETS = ("q", "k", "v")

_RUNTIME_KEYS = {
    "softmax_backend", "rmsnorm_backend", "fp8", "compile",
    "inference_kernels", "optimized_paths", "window_decomp", "warmup_steps",
    "w_o_far_scale",
}
_UNSAFE_WEIGHT_SUFFIXES = {".pt", ".pth", ".bin", ".ckpt", ".pickle", ".pkl"}
_BRANCH_KEY = re.compile(r"^transformer_blocks\.(\d+)\.attn\.(.+)$")


@dataclass(frozen=True)
class AdapterFiles:
    """One fully accounted-for adapter directory."""

    name: str
    directory: str
    config_file: str
    weights_file: str
    spec: Mapping[str, Any]


@dataclass(frozen=True)
class CheckpointInventory:
    """Validated paths for one exploded checkpoint."""

    root: str
    spec_file: str
    branch_weights_file: str
    branch_config_file: str | None
    adapters: tuple[AdapterFiles, ...]


@dataclass(frozen=True)
class LoadedAdapter:
    name: str
    state: Mapping[str, Any]
    spec: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedCheckpoint:
    """In-memory checkpoint, with branch weights grouped by transformer block."""

    config: Mapping[str, Any]
    branches: tuple[Mapping[str, Any], ...]
    adapters: Mapping[str, LoadedAdapter]
    spec: Mapping[str, Any]
    root: str


def _real(path: os.PathLike[str] | str) -> str:
    # Preserve the spelling for callers/logs; containment normalises case separately
    # because Windows paths are case-insensitive.
    return os.path.realpath(os.fspath(path))


def _contained(path: os.PathLike[str] | str, root: os.PathLike[str] | str) -> bool:
    path_real = os.path.normcase(_real(path))
    root_real = os.path.normcase(_real(root))
    try:
        return os.path.commonpath((path_real, root_real)) == root_real
    except ValueError:  # different Windows drives
        return False


def register_folder() -> None:
    """Register sibling ``models/vdn`` roots without importing ComfyUI at import time."""
    import folder_paths

    if "vdn" in folder_paths.folder_names_and_paths:
        return
    roots = {os.path.dirname(path) for path in folder_paths.get_folder_paths("loras")}
    for base in sorted(roots):
        folder_paths.add_model_folder_path("vdn", os.path.join(base, "vdn"))


def vdn_folders() -> tuple[str, ...]:
    import folder_paths

    if "vdn" not in folder_paths.folder_names_and_paths:
        register_folder()
    return tuple(_real(path) for path in folder_paths.get_folder_paths("vdn"))


def _roots(roots: Iterable[os.PathLike[str] | str] | None) -> tuple[str, ...]:
    resolved = tuple(_real(root) for root in (vdn_folders() if roots is None else roots))
    if not resolved:
        raise FileNotFoundError("no ComfyUI models/vdn roots are registered")
    return resolved


def _safe_candidate(root: str, name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("VDN checkpoint name must be a non-empty relative path")
    if os.path.isabs(name) or Path(name).drive:
        raise ValueError(f"VDN checkpoint name must be relative, got {name!r}")
    candidate = _real(os.path.join(root, *name.replace("\\", "/").split("/")))
    if not _contained(candidate, root):
        raise ValueError(f"VDN checkpoint {name!r} escapes models/vdn root {root!r}")
    return candidate


def list_vdn_checkpoints(
    roots: Iterable[os.PathLike[str] | str] | None = None,
) -> list[str]:
    """Return checkpoint directory names while refusing symlink traversal escapes."""
    found: set[str] = set()
    for root in _roots(roots):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, files in os.walk(root, followlinks=False):
            dirnames[:] = [
                name for name in dirnames
                if _contained(os.path.join(dirpath, name), root)
            ]
            if "model_spec.json" not in files:
                continue
            spec_file = os.path.join(dirpath, "model_spec.json")
            weights_file = os.path.join(dirpath, "linear_branch", "model.safetensors")
            if (os.path.isfile(weights_file) and _contained(spec_file, root)
                    and _contained(weights_file, root)):
                rel = os.path.relpath(_real(dirpath), root).replace("\\", "/")
                found.add(rel)
                dirnames[:] = []
    return sorted(found)


def resolve_vdn_checkpoint(
    name: str,
    roots: Iterable[os.PathLike[str] | str] | None = None,
) -> str:
    """Resolve a UI-relative name and prove realpath containment under models/vdn."""
    checked = _roots(roots)
    for root in checked:
        candidate = _safe_candidate(root, name)
        weights_file = os.path.join(candidate, "linear_branch", "model.safetensors")
        spec_file = os.path.join(candidate, "model_spec.json")
        if os.path.isfile(weights_file) and os.path.isfile(spec_file):
            if not _contained(weights_file, root) or not _contained(spec_file, root):
                raise ValueError(f"VDN checkpoint {name!r} contains a file escaping models/vdn")
            return candidate
    raise FileNotFoundError(f"VDN checkpoint {name!r} not found under {list(checked)!r}")


def _read_json(path: str) -> dict[str, Any]:
    if os.path.getsize(path) > 4 * 1024 * 1024:
        raise ValueError(f"JSON metadata is unexpectedly large: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _checkpoint_recipe_metadata(root: str) -> dict[str, Any]:
    """Load the small inference-recipe fields shipped beside released weights."""
    path = os.path.join(root, "metadata.json")
    if not os.path.isfile(path):
        return {}
    if not _contained(path, root):
        raise ValueError("metadata.json realpath escapes the checkpoint")
    payload = _read_json(path)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json metadata must be an object")
    result = {}
    if "turbo_num_steps" in metadata:
        result["turbo_num_steps"] = _require_int(
            metadata["turbo_num_steps"], "metadata.turbo_num_steps", minimum=1
        )
    return result


def _require_bool(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{where} must be a resolved bool, got {value!r}")
    return value


def _require_int(value: Any, where: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where} must be an int >= {minimum}, got {value!r}")
    return value


def _walk_config(value: Any, where: str) -> Iterable[tuple[str, Any, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_where = f"{where}.{key}"
            yield key, child, child_where
            yield from _walk_config(child, child_where)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_config(child, f"{where}[{index}]")


def _validate_adapter_spec(entry: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    if set(entry) != {"type", "version", "config"}:
        raise ValueError(f"{where}: expected exactly type/version/config, got {sorted(entry)}")
    if entry["type"] != "lora" or entry["version"] != 1:
        raise ValueError(f"{where}: only lora adapter version 1 is supported")
    cfg = entry["config"]
    if not isinstance(cfg, dict):
        raise ValueError(f"{where}.config must be an object")
    rank = _require_int(cfg.get("rank"), f"{where}.config.rank", minimum=1)
    alpha = cfg.get("alpha", rank)
    if type(alpha) not in (int, float) or alpha <= 0:
        raise ValueError(f"{where}.config.alpha must be positive, got {alpha!r}")
    targets = cfg.get("targets")
    if not isinstance(targets, list) or not targets or any(not isinstance(x, str) or not x for x in targets):
        raise ValueError(f"{where}.config.targets must be a non-empty string list")
    if len(targets) != len(set(targets)):
        raise ValueError(f"{where}.config.targets contains duplicates")
    for field in ("rank_pattern", "alpha_pattern"):
        pattern = cfg.get(field, {})
        if not isinstance(pattern, dict):
            raise ValueError(f"{where}.config.{field} must be an object")
        for module, value in pattern.items():
            if not isinstance(module, str) or not module:
                raise ValueError(f"{where}.config.{field} has an invalid module name")
            if field == "rank_pattern":
                _require_int(value, f"{where}.config.{field}[{module!r}]", minimum=1)
            elif type(value) not in (int, float) or value <= 0:
                raise ValueError(f"{where}.config.{field}[{module!r}] must be positive")
    if "exact_targets" in cfg:
        _require_bool(cfg["exact_targets"], f"{where}.config.exact_targets")
    return entry


def validate_model_spec(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the released ModelSpec v2, including its base-config identity."""
    if not isinstance(spec, dict):
        raise ValueError("model_spec.json must contain an object")
    if spec.get("format_version") != SPEC_FORMAT_VERSION:
        raise ValueError(f"spec format_version must be {SPEC_FORMAT_VERSION}")
    base = spec.get("base")
    if not isinstance(base, dict):
        raise ValueError("model_spec.base must be an object")
    required_base = {"library", "class_name", "source", "subfolder", "revision", "resolved_config"}
    missing = sorted(required_base - set(base))
    if missing:
        raise ValueError(f"model_spec.base is missing {missing}")
    if not isinstance(base["resolved_config"], dict):
        raise ValueError("model_spec.base.resolved_config must be an object")
    canonical = json.dumps(base["resolved_config"], sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if base.get("config_hash") not in (None, "", digest):
        raise ValueError("model_spec.base.config_hash does not match resolved_config")

    transforms = spec.get("transforms")
    if not isinstance(transforms, list):
        raise ValueError("model_spec.transforms must be a list")
    hybrids = [entry for entry in transforms if isinstance(entry, dict) and entry.get("type") == "hybrid_attention"]
    if len(hybrids) != 1 or len(transforms) != 1:
        raise ValueError("model_spec must contain exactly one hybrid_attention transform")
    transform_config(spec)

    adapters = spec.get("adapters", [])
    if not isinstance(adapters, list):
        raise ValueError("model_spec.adapters must be a list")
    names: set[str] = set()
    for index, entry in enumerate(adapters):
        validated = _validate_adapter_spec(entry, f"adapters[{index}]")
        name = validated["config"].get("name", "default" if index == 0 else None)
        if name is not None:
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f"adapters[{index}].config.name is invalid or duplicated")
            names.add(name)
    return spec


def transform_config(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated, flattened runtime view of hybrid transform v2."""
    transforms = spec.get("transforms", [])
    matches = [item for item in transforms if isinstance(item, dict) and item.get("type") == "hybrid_attention"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one hybrid_attention transform, got {len(matches)}")
    transform = matches[0]
    if transform.get("version") != HYBRID_TRANSFORM_VERSION:
        raise ValueError(f"hybrid_attention transform version must be {HYBRID_TRANSFORM_VERSION}")
    cfg = transform.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("hybrid_attention.config must be an object")
    for key, value, where in _walk_config(cfg, "hybrid_attention.config"):
        if value is None:
            raise ValueError(f"{where} is unresolved (null)")
        if key in _RUNTIME_KEYS:
            raise ValueError(f"{where} is a runtime/deleted key and cannot enter a ModelSpec")

    allowed_top = {"enable_softmax_gate", "anchor_frames", "softmax_attention", "linear_attention"}
    if set(cfg) != allowed_top:
        raise ValueError(f"hybrid_attention.config keys must be {sorted(allowed_top)}, got {sorted(cfg)}")
    linear, softmax = cfg["linear_attention"], cfg["softmax_attention"]
    if not isinstance(linear, dict) or not isinstance(softmax, dict):
        raise ValueError("linear_attention and softmax_attention must be objects")
    allowed_linear = {"delta_rule", "bridge", "a_fp32", "linear_head_dim", "short_conv", "enable_text_state"}
    allowed_softmax = {"radius", "chunk"}
    if set(linear) != allowed_linear:
        raise ValueError(f"linear_attention keys must be {sorted(allowed_linear)}, got {sorted(linear)}")
    if not {"radius"}.issubset(softmax) or not set(softmax).issubset(allowed_softmax):
        raise ValueError(f"softmax_attention keys must be radius and optional chunk, got {sorted(softmax)}")
    if linear["delta_rule"] not in SUPPORTED_DELTA_RULES:
        raise ValueError(f"unsupported delta_rule {linear['delta_rule']!r}")
    if linear["bridge"] not in SUPPORTED_BRIDGES:
        raise ValueError(f"unsupported bridge {linear['bridge']!r}")
    anchor = cfg["anchor_frames"]
    if anchor not in SUPPORTED_ANCHORS:
        raise ValueError(f"unsupported anchor_frames {anchor!r}")
    short = linear["short_conv"]
    targets = short.get("targets") if isinstance(short, dict) else None
    if (not isinstance(targets, list) or len(targets) != len(set(targets))
            or any(target not in SHORT_CONV_TARGETS for target in targets)):
        raise ValueError("short_conv must contain a distinct q/k/v target subset")
    radius = _require_int(softmax["radius"], "softmax_attention.radius")
    chunk = _require_int(softmax.get("chunk", 0), "softmax_attention.chunk")
    head_dim = _require_int(linear["linear_head_dim"], "linear_attention.linear_head_dim", minimum=1)
    return {
        "enable_softmax_gate": _require_bool(cfg["enable_softmax_gate"], "enable_softmax_gate"),
        "anchor_frames": anchor,
        "radius": radius,
        "chunk": chunk,
        "delta_rule": linear["delta_rule"],
        "bridge": linear["bridge"],
        "a_fp32": _require_bool(linear["a_fp32"], "linear_attention.a_fp32"),
        "linear_head_dim": head_dim,
        "short_conv": tuple(targets),
        "enable_text_state": _require_bool(linear["enable_text_state"], "linear_attention.enable_text_state"),
    }


def _same_json(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(right, sort_keys=True, separators=(",", ":"))


def inventory_checkpoint(path: os.PathLike[str] | str, *, roots: Iterable[os.PathLike[str] | str] | None = None) -> CheckpointInventory:
    """Validate directory containment and account for every checkpoint component."""
    root = _real(path)
    permitted = _roots(roots)
    if not any(_contained(root, allowed) for allowed in permitted):
        raise ValueError(f"checkpoint realpath {root!r} is outside models/vdn roots {list(permitted)!r}")
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    spec_file = os.path.join(root, "model_spec.json")
    branch_weights = os.path.join(root, "linear_branch", "model.safetensors")
    if not os.path.isfile(spec_file) or not os.path.isfile(branch_weights):
        raise FileNotFoundError(f"{root}: expected model_spec.json and linear_branch/model.safetensors")
    for required in (spec_file, branch_weights):
        if not _contained(required, root):
            raise ValueError(f"checkpoint component realpath escapes its root: {required}")
    spec = validate_model_spec(_read_json(spec_file))
    branch_config = os.path.join(root, "linear_branch", "config.json")
    if os.path.isfile(branch_config):
        if not _contained(branch_config, root):
            raise ValueError("linear_branch/config.json realpath escapes the checkpoint")
        recorded = _read_json(branch_config)
        transform = spec["transforms"][0]
        if not (_same_json(recorded, transform) or _same_json(recorded, transform["config"])):
            raise ValueError("linear_branch/config.json disagrees with model_spec.json")
    else:
        branch_config = None

    adapter_root = os.path.join(root, "adapters")
    adapters: list[AdapterFiles] = []
    if os.path.isdir(adapter_root):
        for name in sorted(os.listdir(adapter_root)):
            directory = _real(os.path.join(adapter_root, name))
            if not _contained(directory, adapter_root) or not os.path.isdir(directory):
                raise ValueError(f"unsafe or non-directory adapter entry {name!r}")
            cfg_file = os.path.join(directory, "adapter_config.json")
            weights_file = os.path.join(directory, "adapter_model.safetensors")
            if not os.path.isfile(cfg_file) or not os.path.isfile(weights_file):
                raise ValueError(f"adapter {name!r} is incomplete; no adapter directories are skipped")
            if not _contained(cfg_file, root) or not _contained(weights_file, root):
                raise ValueError(f"adapter {name!r} contains a file escaping the checkpoint")
            entry = _validate_adapter_spec(_read_json(cfg_file), f"adapter {name!r}")
            adapters.append(AdapterFiles(name, directory, cfg_file, weights_file, entry))

    declared = list(spec.get("adapters", []))
    unmatched = list(range(len(declared)))
    for adapter in adapters:
        candidates = [
            index for index in unmatched
            if declared[index]["config"].get("name", "default" if index == 0 else None) == adapter.name
        ]
        if not candidates:
            candidates = [index for index in unmatched if _same_json(declared[index], adapter.spec)]
        if len(candidates) != 1 or not _same_json(declared[candidates[0]], adapter.spec):
            raise ValueError(f"adapter directory {adapter.name!r} does not exactly match one ModelSpec entry")
        unmatched.remove(candidates[0])
    if unmatched:
        raise ValueError(f"ModelSpec adapters have no on-disk weights: {unmatched}")

    recognised = {_real(spec_file), _real(branch_weights)}
    if branch_config:
        recognised.add(_real(branch_config))
    for adapter in adapters:
        recognised.update({_real(adapter.config_file), _real(adapter.weights_file)})
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for dirname in dirnames:
            if not _contained(os.path.join(dirpath, dirname), root):
                raise ValueError(f"symlink directory escapes checkpoint: {os.path.join(dirpath, dirname)}")
        for filename in filenames:
            full = _real(os.path.join(dirpath, filename))
            suffix = Path(filename).suffix.lower()
            if suffix in _UNSAFE_WEIGHT_SUFFIXES:
                raise ValueError(f"unsafe non-safetensors weight file in checkpoint: {full}")
            if suffix == ".safetensors" and full not in recognised:
                raise ValueError(f"unaccounted safetensors file in checkpoint: {full}")
    return CheckpointInventory(root, spec_file, branch_weights, branch_config, tuple(adapters))


def _shape(tensor: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in tensor.shape)
    except Exception as exc:
        raise ValueError("checkpoint state contains a non-tensor value") from exc


def _validate_branch_block(weights: Mapping[str, Any], cfg: Mapping[str, Any], base: Mapping[str, Any], index: int) -> None:
    d = int(cfg["linear_head_dim"])
    required = {
        "to_out_linear.weight", "beta_proj.weight", "norm.weight",
        "alpha.A_log", "alpha.dt_bias", "alpha.down.weight", "alpha.up.weight",
        "output_gate.down.weight", "output_gate.up.weight", "output_gate.up.bias",
    }
    if cfg["enable_softmax_gate"]:
        required.update({"softmax_gate.up.weight", "softmax_gate.up.bias"})
    for target in cfg["short_conv"]:
        required.update({f"short_conv.{target}_sp.weight", f"short_conv.{target}_tm.weight"})
    actual = set(weights)
    missing, extra = sorted(required - actual), sorted(actual - required)
    if missing or extra:
        raise ValueError(f"block {index} branch inventory mismatch: missing={missing}, extra={extra}")
    out_shape = _shape(weights["to_out_linear.weight"])
    beta_shape = _shape(weights["beta_proj.weight"])
    if len(out_shape) != 2 or len(beta_shape) != 2:
        raise ValueError(f"block {index}: to_out_linear and beta_proj must be matrices")
    hidden, channels = out_shape
    heads = beta_shape[0]
    if not heads or channels != heads * d:
        raise ValueError(f"block {index}: branch channels {channels} != heads {heads} * linear_head_dim {d}")
    if base.get("hidden_size") is not None and hidden != int(base["hidden_size"]):
        raise ValueError(f"block {index}: hidden size {hidden} != base hidden_size {base['hidden_size']}")
    if base.get("num_attention_heads") is not None and heads != int(base["num_attention_heads"]):
        raise ValueError(f"block {index}: heads {heads} != base num_attention_heads {base['num_attention_heads']}")
    expected = {
        "to_out_linear.weight": (hidden, channels),
        "beta_proj.weight": (heads, hidden),
        "norm.weight": (d,),
        "alpha.A_log": (heads,),
        "alpha.dt_bias": (channels,),
        "alpha.down.weight": (d, hidden),
        "alpha.up.weight": (channels, d),
        "output_gate.down.weight": (d, hidden),
        "output_gate.up.weight": (channels, d),
        "output_gate.up.bias": (channels,),
    }
    if cfg["enable_softmax_gate"]:
        expected.update({"softmax_gate.up.weight": (heads, hidden), "softmax_gate.up.bias": (heads,)})
    for target in cfg["short_conv"]:
        expected.update({f"short_conv.{target}_sp.weight": (channels, 1, 5, 5), f"short_conv.{target}_tm.weight": (channels, 1, 5)})
    for name, want in expected.items():
        got = _shape(weights[name])
        if got != want:
            raise ValueError(f"block {index} {name}: shape {got}, expected {want}")


def load_vdn_checkpoint(
    path: os.PathLike[str] | str,
    *,
    roots: Iterable[os.PathLike[str] | str] | None = None,
    tensor_loader: Callable[[str], Mapping[str, Any]] | None = None,
) -> LoadedCheckpoint:
    """Load and strictly validate an exploded VDN checkpoint without a tensor cache."""
    inventory = inventory_checkpoint(path, roots=roots)
    spec = validate_model_spec(_read_json(inventory.spec_file))
    cfg = {**transform_config(spec), **_checkpoint_recipe_metadata(inventory.root)}
    if tensor_loader is None:
        from safetensors.torch import load_file
        tensor_loader = load_file
    branch_state = tensor_loader(inventory.branch_weights_file)
    if not isinstance(branch_state, Mapping):
        raise ValueError("safetensors loader returned a non-mapping branch state")
    grouped: dict[int, dict[str, Any]] = {}
    for key, tensor in branch_state.items():
        match = _BRANCH_KEY.fullmatch(key)
        if match is None or ".lora_" in key:
            raise ValueError(f"unrecognised linear_branch tensor {key!r}; no tensor is skipped")
        index, name = int(match.group(1)), match.group(2)
        if name.startswith("linear_attention."):
            name = name[len("linear_attention."):]
        if name in grouped.setdefault(index, {}):
            raise ValueError(f"branch key collision after mapping at block {index}: {name}")
        grouped[index][name] = tensor
    indices = sorted(grouped)
    if indices != list(range(len(indices))):
        raise ValueError(f"branch block indices must be contiguous from zero, got {indices}")
    base = spec["base"]["resolved_config"]
    if base.get("num_layers") is not None and len(indices) != int(base["num_layers"]):
        raise ValueError(f"branch has {len(indices)} blocks, base spec declares {base['num_layers']}")
    for index in indices:
        _validate_branch_block(grouped[index], cfg, base, index)

    loaded_adapters: dict[str, LoadedAdapter] = {}
    for adapter in inventory.adapters:
        state = tensor_loader(adapter.weights_file)
        if not isinstance(state, Mapping):
            raise ValueError(f"adapter {adapter.name!r} loader returned a non-mapping state")
        # Parsing is the inventory check: it rejects every unknown/missing pair.
        from .adapters import parse_adapter_state
        parse_adapter_state(state, adapter.spec)
        loaded_adapters[adapter.name] = LoadedAdapter(adapter.name, state, adapter.spec)
    return LoadedCheckpoint(cfg, tuple(grouped[index] for index in indices), loaded_adapters, spec, inventory.root)


__all__ = [
    "AdapterFiles", "CheckpointInventory", "LoadedAdapter", "LoadedCheckpoint",
    "inventory_checkpoint", "list_vdn_checkpoints", "load_vdn_checkpoint",
    "register_folder", "resolve_vdn_checkpoint", "transform_config",
    "validate_model_spec", "vdn_folders",
]
