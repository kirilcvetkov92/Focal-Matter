from __future__ import annotations

import argparse
import json
import math
import re
import types
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

cv2 = None
np = None
torch = None


def require_runtime_dependencies() -> None:
    global cv2, np, torch
    if cv2 is not None and np is not None and torch is not None:
        return

    missing = []
    try:
        import cv2 as cv2_module
    except ImportError:
        cv2_module = None
        missing.append("opencv-python")

    try:
        import numpy as numpy_module
    except ImportError:
        numpy_module = None
        missing.append("numpy")

    try:
        import torch as torch_module
    except ImportError:
        torch_module = None
        missing.append("torch")

    if missing:
        raise RuntimeError(
            "Missing Python package(s): {}. Install project dependencies with "
            "`python -m pip install -r requirements.txt`.".format(", ".join(missing))
        )

    cv2 = cv2_module
    np = numpy_module
    torch = torch_module


DEFAULT_FOCAL_TARGETS = (
    "encoder.layers.0.blocks.0.modulation",
    "encoder.layers.1.blocks.0.modulation",
    "encoder.layers.2.blocks.0.modulation",
    "encoder.layers.2.blocks.5.modulation",
    "encoder.layers.3.blocks.0.modulation",
    "refine4.block.blocks.0.modulation",
    "refine3.block.blocks.0.modulation",
    "refine2.block.blocks.0.modulation",
    "refine1.block.blocks.0.modulation",
)

DEFAULT_LAYER_HOOKS = (
    ("stem_pre", "stem_pre"),
    ("stem_down", "stem_down"),
    ("encoder", "encoder"),
    ("top_reduce", "top_reduce"),
    ("top_ctx", "top_ctx fused"),
    ("top_fuse", "top_fuse"),
    ("refine4", "refine l4"),
    ("up4to3", "up4to3"),
    ("skip3", "skip3 gated"),
    ("fuse3", "fuse3"),
    ("refine3", "refine l3"),
    ("up3to2", "up3to2"),
    ("skip2", "skip2 gated"),
    ("fuse2", "fuse2"),
    ("refine2", "refine l2"),
    ("up2to1", "up2to1"),
    ("skip1", "skip1 gated"),
    ("fuse1", "fuse1"),
    ("refine1", "refine l1"),
    ("neck", "neck"),
    ("alpha_head", "alpha_head"),
)


@dataclass
class PadInfo:
    top: int
    bottom: int
    left: int
    right: int
    original_height: int
    original_width: int


@dataclass
class PreparedSample:
    image_tensor: torch.Tensor
    trimap_tensor: torch.Tensor
    raw_bgr: np.ndarray
    trimap: np.ndarray
    pad: PadInfo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FocalMatter visualisations for focal-modulation context, "
            "focal gates, decoder layers, and context aggregation features."
        )
    )
    parser.add_argument("--image", required=False, help="Path to the RGB/BGR input image.", default="G:/Master/master_validation/merged/16452523375_08591714cf_o_0.png")
    parser.add_argument("--trimap", required=False, help="Path to the trimap image with values 0/128/255.", default="G:/Master/master_validation/trimaps/16452523375_08591714cf_o_0.png")
    parser.add_argument("--output-dir", default="visualisations", help="Directory for generated PNGs.")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path. Defaults to utils.config.")
    parser.add_argument("--encoder-weights", default=None, help="Encoder weights path. Defaults to utils.config.")
    parser.add_argument("--no-checkpoint", action="store_true", help="Skip loading the full model checkpoint.")
    parser.add_argument("--no-encoder-weights", action="store_true", help="Skip loading encoder weights.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--pad-to-multiple", type=int, default=32, help="Pad input size to this multiple.")
    parser.add_argument(
        "--module-filter",
        action="append",
        default=None,
        help=(
            "Capture focal-modulation modules whose names contain this text. "
            "Can be supplied multiple times. If omitted, representative encoder and decoder blocks are used."
        ),
    )
    parser.add_argument(
        "--all-focal-modules",
        action="store_true",
        help="Capture all focal-modulation modules up to --max-focal-modules.",
    )
    parser.add_argument("--max-focal-modules", type=int, default=9, help="Maximum detailed focal modules to save.")
    parser.add_argument("--feature-mode", choices=("rgb", "energy"), default="rgb", help="Feature map rendering mode.")
    parser.add_argument("--cell-size", type=int, default=224, help="Panel size in pixels.")
    parser.add_argument("--cols", type=int, default=4, help="Columns in contact sheets.")
    parser.add_argument("--channel-grid", type=int, default=24, help="Top varying channels to show per layer.")
    parser.add_argument("--no-channel-grids", action="store_true", help="Do not save per-channel feature grids.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = value.strip().replace(".", "_").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "unnamed"


def optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value.lower() in {"none", "null", "false", "0"}:
        return None
    return value


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model(device: torch.device, args: argparse.Namespace):
    from train import create_model
    from utils.config import clone_config

    config = clone_config()
    config.runtime.device = device.type

    if args.checkpoint is not None:
        config.model.checkpoint_path = optional_path(args.checkpoint)
    if args.encoder_weights is not None:
        config.model.encoder_weights_path = optional_path(args.encoder_weights)
    if args.no_checkpoint:
        config.model.checkpoint_path = None
    if args.no_encoder_weights:
        config.model.encoder_weights_path = None

    if config.model.encoder_weights_path and not Path(config.model.encoder_weights_path).is_file():
        print(f"Warning: encoder weights not found, using current encoder weights: {config.model.encoder_weights_path}")
    if (not args.no_checkpoint) and config.model.checkpoint_path and not Path(config.model.checkpoint_path).is_file():
        print(f"Warning: checkpoint not found, using current model weights: {config.model.checkpoint_path}")

    model = create_model(
        config=config,
        device=device,
        logger=None,
        load_checkpoint=not args.no_checkpoint,
        require_checkpoint=False,
    )
    model.eval()
    return model, config


def read_sample(image_path: str, trimap_path: str, pad_to_multiple: int, device: torch.device) -> PreparedSample:
    raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    trimap = cv2.imread(str(trimap_path), cv2.IMREAD_GRAYSCALE)
    if trimap is None:
        raise FileNotFoundError(f"Could not read trimap: {trimap_path}")
    if raw_bgr.shape[:2] != trimap.shape[:2]:
        raise ValueError(
            "Image and trimap must have the same height/width. "
            f"Got image={raw_bgr.shape[:2]} trimap={trimap.shape[:2]}."
        )

    height, width = trimap.shape
    if pad_to_multiple <= 1:
        new_height, new_width = height, width
    else:
        new_height = ((height - 1) // pad_to_multiple + 1) * pad_to_multiple
        new_width = ((width - 1) // pad_to_multiple + 1) * pad_to_multiple
    pad_height = new_height - height
    pad_width = new_width - width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    raw_pad = cv2.copyMakeBorder(raw_bgr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    trimap_pad = cv2.copyMakeBorder(trimap, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

    trimap_tensor = np.zeros((*trimap_pad.shape, 3), dtype=np.float32)
    trimap_tensor[:, :, 0] = trimap_pad == 0
    trimap_tensor[:, :, 1] = (trimap_pad > 0) & (trimap_pad < 255)
    trimap_tensor[:, :, 2] = trimap_pad == 255
    trimap_tensor = np.transpose(trimap_tensor, (2, 0, 1))[None, :, :, :]

    image_tensor = np.transpose(raw_pad, (2, 0, 1))[None, ::-1, :, :]
    image_tensor = np.asarray(image_tensor, dtype=np.float32) / 255.0

    return PreparedSample(
        image_tensor=torch.from_numpy(image_tensor).to(device),
        trimap_tensor=torch.from_numpy(trimap_tensor).to(device),
        raw_bgr=raw_bgr,
        trimap=trimap,
        pad=PadInfo(
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            original_height=height,
            original_width=width,
        ),
    )


def first_bchw(tensor: torch.Tensor, layout: str = "bchw") -> torch.Tensor:
    tensor = tensor.detach().float()
    if layout == "bhwc":
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return tensor[:1].cpu()[0]


def tensor_stats(tensor: torch.Tensor) -> Dict[str, object]:
    tensor = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()) if tensor.numel() > 1 else 0.0,
    }


class FocalModulationRecorder:
    def __init__(
        self,
        model: torch.nn.Module,
        filters: Optional[Sequence[str]],
        capture_all: bool,
        max_modules: int,
    ) -> None:
        self.model = model
        self.filters = tuple(filters or ())
        self.capture_all = capture_all
        self.max_modules = max_modules
        self.records: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
        self._originals: List[Tuple[torch.nn.Module, object]] = []
        self.selected_names: List[str] = []

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()

    def install(self) -> None:
        from models.focal import FocalModulation

        candidates = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, FocalModulation)
        ]

        if self.capture_all:
            selected = candidates
        elif self.filters:
            selected = [
                (name, module)
                for name, module in candidates
                if any(filter_text in name for filter_text in self.filters)
            ]
        else:
            selected = [(name, module) for name, module in candidates if name in DEFAULT_FOCAL_TARGETS]
            if not selected:
                selected = candidates

        if self.max_modules > 0:
            selected = selected[: self.max_modules]

        self.selected_names = [name for name, _ in selected]
        for name, module in selected:
            original_forward = module.forward
            module.forward = types.MethodType(self._make_forward(name), module)
            self._originals.append((module, original_forward))

    def restore(self) -> None:
        for module, original_forward in self._originals:
            module.forward = original_forward
        self._originals.clear()

    def _make_forward(self, name: str):
        def instrumented_forward(module, x):
            batch, height, width, channels = x.shape
            x_projected = module.f(x)
            x_projected = x_projected.permute(0, 3, 1, 2).contiguous()
            q, ctx, gates = torch.split(x_projected, (channels, channels, module.focal_level + 1), 1)

            ctx_seed = ctx
            ctx_all = 0
            local_contexts = []
            for level in range(module.focal_level):
                ctx = module.focal_layers[level](ctx)
                local_contexts.append(ctx)
                ctx_all = ctx_all + ctx * gates[:, level : level + 1]

            ctx_global = module.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
            ctx_all = ctx_all + ctx_global * gates[:, module.focal_level :]
            modulator = module.h(ctx_all)

            x_modulated = q * modulator
            x_out = x_modulated.permute(0, 2, 3, 1).contiguous()
            if module.use_postln:
                x_out = module.ln(x_out)
            x_out = module.proj(x_out)
            x_out = module.proj_drop(x_out)

            self.records[name] = {
                "input": first_bchw(x, "bhwc"),
                "query": first_bchw(q),
                "context_seed": first_bchw(ctx_seed),
                "local_contexts": [first_bchw(value) for value in local_contexts],
                "gates": [first_bchw(gates[:, idx : idx + 1]) for idx in range(module.focal_level + 1)],
                "global_context": first_bchw(ctx_global),
                "aggregated_context": first_bchw(ctx_all),
                "modulator": first_bchw(modulator),
                "modulated": first_bchw(x_modulated),
                "projected_output": first_bchw(x_out, "bhwc"),
                "height": height,
                "width": width,
                "channels": channels,
            }
            return x_out

        return instrumented_forward


class LayerRecorder:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.records: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._handles = []

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()

    def install(self) -> None:
        modules = dict(self.model.named_modules())

        for module_name, record_name in DEFAULT_LAYER_HOOKS:
            if module_name in modules:
                self._handles.append(modules[module_name].register_forward_hook(self._make_hook(record_name)))

        if hasattr(self.model, "top_ctx") and hasattr(self.model.top_ctx, "branches"):
            rates = getattr(self.model.top_ctx, "rates", None) or (1, 2, 4, 8)
            for idx, _branch in enumerate(self.model.top_ctx.branches):
                module_name = f"top_ctx.branches.{idx}"
                if module_name in modules:
                    rate = rates[idx] if idx < len(rates) else idx
                    self._handles.append(
                        modules[module_name].register_forward_hook(self._make_hook(f"context rate {rate}"))
                    )

    def restore(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self._record_output(name, output)

        return hook

    def _record_output(self, name: str, output) -> None:
        if isinstance(output, torch.Tensor):
            self.records[name] = first_bchw(output)
            return

        if isinstance(output, (tuple, list)):
            tensor_index = 1
            for item in output:
                if isinstance(item, torch.Tensor):
                    self.records[f"{name} l{tensor_index}"] = first_bchw(item)
                    tensor_index += 1


def normalize01(array: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    low = float(np.percentile(array, lower_percentile))
    high = float(np.percentile(array, upper_percentile))
    if high <= low:
        low = float(array.min())
        high = float(array.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return (normalize01(image) * 255.0).astype(np.uint8)


def feature_to_bgr(feature: torch.Tensor | np.ndarray, mode: str = "rgb") -> np.ndarray:
    if isinstance(feature, torch.Tensor):
        feature = feature.detach().float().cpu().numpy()
    feature = np.asarray(feature, dtype=np.float32)

    if feature.ndim == 2:
        return scalar_to_heatmap(feature)

    if feature.ndim != 3:
        raise ValueError(f"Expected feature shape CxHxW or HxW, got {feature.shape}")

    channels, height, width = feature.shape
    if channels == 1:
        return scalar_to_heatmap(feature[0])

    if mode == "rgb" and channels >= 3 and height > 1 and width > 1:
        flat = feature.reshape(channels, -1)
        scores = np.nanstd(flat, axis=1)
        indices = np.argsort(scores)[-3:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        rgb = np.stack([normalize01(feature[idx]) for idx in indices], axis=-1)
        rgb = (rgb * 255.0).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    energy = np.mean(np.abs(feature), axis=0)
    return scalar_to_heatmap(energy)


def scalar_to_heatmap(array: np.ndarray) -> np.ndarray:
    gray = (normalize01(array) * 255.0).astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_VIRIDIS)
    return cv2.applyColorMap(gray, colormap)


def trimap_to_bgr(trimap: np.ndarray) -> np.ndarray:
    out = np.zeros((*trimap.shape, 3), dtype=np.uint8)
    out[trimap == 0] = (0, 0, 0)
    out[(trimap > 0) & (trimap < 255)] = (128, 128, 128)
    out[trimap == 255] = (255, 255, 255)
    return out


def alpha_to_bgr(alpha: np.ndarray) -> np.ndarray:
    alpha = to_uint8_image(alpha)
    return cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)


def fit_to_cell(image_bgr: np.ndarray, cell_size: int, background: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    image_bgr = to_uint8_image(image_bgr)
    height, width = image_bgr.shape[:2]
    scale = min(cell_size / max(width, 1), cell_size / max(height, 1))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=interpolation)
    canvas = np.full((cell_size, cell_size, 3), background, dtype=np.uint8)
    y = (cell_size - new_height) // 2
    x = (cell_size - new_width) // 2
    canvas[y : y + new_height, x : x + new_width] = resized
    return canvas


def draw_label(canvas: np.ndarray, label: str, x: int, y: int, max_width: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    label = label.strip()
    while label:
        (text_width, _text_height), _baseline = cv2.getTextSize(label, font, scale, thickness)
        if text_width <= max_width:
            break
        label = label[:-4].rstrip() + "..."
        if len(label) <= 4:
            break
    cv2.putText(canvas, label, (x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def make_panel(image_bgr: np.ndarray, label: str, cell_size: int) -> np.ndarray:
    title_height = 30
    panel = np.full((cell_size + title_height, cell_size, 3), 255, dtype=np.uint8)
    panel[title_height:, :] = fit_to_cell(image_bgr, cell_size)
    draw_label(panel, label, 6, 20, cell_size - 12)
    return panel


def make_grid(panels: Sequence[np.ndarray], cols: int, gap: int = 12) -> np.ndarray:
    if not panels:
        raise ValueError("No panels to render.")

    cols = max(1, min(cols, len(panels)))
    rows = int(math.ceil(len(panels) / cols))
    panel_height, panel_width = panels[0].shape[:2]
    grid_height = rows * panel_height + (rows + 1) * gap
    grid_width = cols * panel_width + (cols + 1) * gap
    grid = np.full((grid_height, grid_width, 3), 255, dtype=np.uint8)

    for idx, panel in enumerate(panels):
        row = idx // cols
        col = idx % cols
        y = gap + row * (panel_height + gap)
        x = gap + col * (panel_width + gap)
        grid[y : y + panel_height, x : x + panel_width] = panel

    return grid


def make_arrow_row(panels: Sequence[np.ndarray], gap: int = 48) -> np.ndarray:
    if not panels:
        raise ValueError("No panels to render.")

    panel_height, panel_width = panels[0].shape[:2]
    row_width = len(panels) * panel_width + (len(panels) + 1) * gap
    row = np.full((panel_height, row_width, 3), 255, dtype=np.uint8)

    positions = []
    for idx, panel in enumerate(panels):
        x = gap + idx * (panel_width + gap)
        positions.append(x)
        row[:, x : x + panel_width] = panel

    arrow_color = (190, 120, 40)
    y = panel_height // 2 + 10
    for idx in range(len(positions) - 1):
        start_x = positions[idx] + panel_width + 8
        end_x = positions[idx + 1] - 8
        if end_x > start_x:
            cv2.arrowedLine(row, (start_x, y), (end_x, y), arrow_color, 3, cv2.LINE_AA, tipLength=0.25)

    return row


def stack_rows(rows: Sequence[np.ndarray], gap: int = 18) -> np.ndarray:
    if not rows:
        raise ValueError("No rows to render.")

    width = max(row.shape[1] for row in rows)
    height = sum(row.shape[0] for row in rows) + (len(rows) + 1) * gap
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    y = gap
    for row in rows:
        x = (width - row.shape[1]) // 2
        canvas[y : y + row.shape[0], x : x + row.shape[1]] = row
        y += row.shape[0] + gap

    return canvas


def write_image(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"Failed to write image: {path}")


def save_architecture_cells(
    items: Sequence[Tuple[str, np.ndarray]],
    output_dir: Path,
    cell_size: int,
) -> None:
    architecture_dir = output_dir / "architecture"
    architecture_dir.mkdir(parents=True, exist_ok=True)

    for index, (label, image_bgr) in enumerate(items, start=1):
        filename = f"{index:03d}_{safe_name(label)}.png"
        image_bgr = to_uint8_image(image_bgr)
        height, width = image_bgr.shape[:2]
        scale = cell_size / max(width, 1)
        resized_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image_bgr, (cell_size, resized_height), interpolation=interpolation)
        write_image(architecture_dir / filename, resized)


def save_layer_overview(
    records: "OrderedDict[str, torch.Tensor]",
    sample: PreparedSample,
    alpha: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    items = [
        ("input image", sample.raw_bgr),
        ("trimap", trimap_to_bgr(sample.trimap)),
        ("pred alpha", alpha_to_bgr(alpha)),
    ]

    for name, tensor in records.items():
        items.append((name, feature_to_bgr(tensor, args.feature_mode)))

    panels = [make_panel(image, label, args.cell_size) for label, image in items]
    write_image(output_dir / "layer_context_overview.png", make_grid(panels, args.cols))
    save_architecture_cells(items, output_dir, args.cell_size)


def save_context_aggregation_overview(
    records: "OrderedDict[str, torch.Tensor]",
    sample: PreparedSample,
    alpha: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    context_keys = [
        "top_ctx rate 1",
        "top_ctx rate 2",
        "top_ctx rate 4",
        "top_ctx rate 8",
        "top_ctx fused",
        "top_reduce",
        "top_fuse",
        "refine4",
    ]
    panels = [
        make_panel(sample.raw_bgr, "input image", args.cell_size),
        make_panel(trimap_to_bgr(sample.trimap), "trimap", args.cell_size),
    ]
    for key in context_keys:
        if key in records:
            panels.append(make_panel(feature_to_bgr(records[key], args.feature_mode), key, args.cell_size))
    panels.append(make_panel(alpha_to_bgr(alpha), "pred alpha", args.cell_size))
    write_image(output_dir / "context_aggregation_overview.png", make_grid(panels, args.cols))


def save_context_pipeline_diagram(
    records: "OrderedDict[str, torch.Tensor]",
    sample: PreparedSample,
    alpha: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    cell_size = min(args.cell_size, 190)

    encoder_key = "encoder l4" if "encoder l4" in records else "encoder l3"
    decoder_key = "refine1" if "refine1" in records else "neck"
    main_specs = [
        ("image", sample.raw_bgr),
        ("trimap", trimap_to_bgr(sample.trimap)),
    ]
    if encoder_key in records:
        main_specs.append(("encoder context", feature_to_bgr(records[encoder_key], args.feature_mode)))
    if "top_ctx fused" in records:
        main_specs.append(("context agg", feature_to_bgr(records["top_ctx fused"], args.feature_mode)))
    if decoder_key in records:
        main_specs.append(("decoder feature", feature_to_bgr(records[decoder_key], args.feature_mode)))
    main_specs.append(("alpha matte", alpha_to_bgr(alpha)))

    main_row = make_arrow_row([make_panel(image, label, cell_size) for label, image in main_specs])

    branch_specs = []
    for key in ("top_ctx rate 1", "top_ctx rate 2", "top_ctx rate 4", "top_ctx rate 8", "top_ctx fused", "top_fuse"):
        if key in records:
            branch_specs.append((key, feature_to_bgr(records[key], args.feature_mode)))

    rows = [main_row]
    if len(branch_specs) >= 2:
        rows.append(make_arrow_row([make_panel(image, label, cell_size) for label, image in branch_specs]))

    write_image(output_dir / "context_pipeline_diagram.png", stack_rows(rows))


def save_channel_grid(feature: torch.Tensor, label: str, path: Path, cell_size: int, max_channels: int) -> None:
    feature_np = feature.detach().float().cpu().numpy()
    if feature_np.ndim != 3:
        return

    channels = feature_np.shape[0]
    if channels == 0 or max_channels <= 0:
        return

    flat = feature_np.reshape(channels, -1)
    scores = np.nanstd(flat, axis=1)
    indices = np.argsort(scores)[-min(max_channels, channels) :][::-1]

    panels = []
    for idx in indices:
        panels.append(
            make_panel(
                scalar_to_heatmap(feature_np[idx]),
                f"{label} ch{int(idx)}",
                cell_size,
            )
        )
    write_image(path, make_grid(panels, cols=min(6, len(panels))))


def save_common_channel_grids(
    records: "OrderedDict[str, torch.Tensor]",
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    if args.no_channel_grids or args.channel_grid <= 0:
        return

    candidates = (
        "encoder l3",
        "encoder l4",
        "top_ctx fused",
        "top_fuse",
        "refine4",
        "refine3",
    )
    for key in candidates:
        if key in records:
            save_channel_grid(
                records[key],
                key,
                output_dir / f"channel_grid_{safe_name(key)}.png",
                max(96, args.cell_size // 2),
                args.channel_grid,
            )


def save_focal_module_visualisations(
    focal_records: "OrderedDict[str, Dict[str, object]]",
    sample: PreparedSample,
    alpha: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    focal_dir = output_dir / "focal_modules"
    overview_panels = [
        make_panel(sample.raw_bgr, "input image", args.cell_size),
        make_panel(alpha_to_bgr(alpha), "pred alpha", args.cell_size),
    ]

    for module_name, record in focal_records.items():
        overview_panels.append(
            make_panel(
                feature_to_bgr(record["aggregated_context"], args.feature_mode),
                f"{module_name} ctx",
                args.cell_size,
            )
        )
        save_single_focal_module(module_name, record, focal_dir, args)

    write_image(output_dir / "focal_modulation_overview.png", make_grid(overview_panels, args.cols))


def save_single_focal_module(
    module_name: str,
    record: Dict[str, object],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    panels = [
        make_panel(feature_to_bgr(record["input"], args.feature_mode), "block input", args.cell_size),
        make_panel(feature_to_bgr(record["query"], args.feature_mode), "query q", args.cell_size),
        make_panel(feature_to_bgr(record["context_seed"], args.feature_mode), "context seed", args.cell_size),
    ]

    local_contexts: Sequence[torch.Tensor] = record["local_contexts"]  # type: ignore[assignment]
    gates: Sequence[torch.Tensor] = record["gates"]  # type: ignore[assignment]
    for idx, context in enumerate(local_contexts):
        panels.append(make_panel(feature_to_bgr(context, args.feature_mode), f"local ctx L{idx}", args.cell_size))
        if idx < len(gates):
            panels.append(make_panel(feature_to_bgr(gates[idx], "energy"), f"gate L{idx}", args.cell_size))

    panels.append(make_panel(feature_to_bgr(record["global_context"], args.feature_mode), "global ctx", args.cell_size))
    if gates:
        panels.append(make_panel(feature_to_bgr(gates[-1], "energy"), "global gate", args.cell_size))
    panels.extend(
        [
            make_panel(feature_to_bgr(record["aggregated_context"], args.feature_mode), "aggregated ctx", args.cell_size),
            make_panel(feature_to_bgr(record["modulator"], args.feature_mode), "modulator h(ctx)", args.cell_size),
            make_panel(feature_to_bgr(record["modulated"], args.feature_mode), "q * modulator", args.cell_size),
            make_panel(feature_to_bgr(record["projected_output"], args.feature_mode), "projected output", args.cell_size),
        ]
    )

    write_image(output_dir / f"{safe_name(module_name)}.png", make_grid(panels, args.cols))

    if not args.no_channel_grids and args.channel_grid > 0:
        save_channel_grid(
            record["aggregated_context"],  # type: ignore[arg-type]
            f"{module_name} ctx",
            output_dir / f"{safe_name(module_name)}_ctx_channels.png",
            max(96, args.cell_size // 2),
            args.channel_grid,
        )


def crop_alpha(prediction: torch.Tensor, pad: PadInfo) -> np.ndarray:
    alpha = prediction.detach().float().cpu().numpy()[0, 0]
    height = pad.original_height
    width = pad.original_width
    alpha = alpha[pad.top : pad.top + height, pad.left : pad.left + width]
    return np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)


def save_manifest(
    path: Path,
    args: argparse.Namespace,
    config,
    device: torch.device,
    layer_records: "OrderedDict[str, torch.Tensor]",
    focal_records: "OrderedDict[str, Dict[str, object]]",
) -> None:
    manifest = {
        "image": args.image,
        "trimap": args.trimap,
        "device": str(device),
        "checkpoint": config.model.checkpoint_path,
        "encoder_weights": config.model.encoder_weights_path,
        "layer_records": {name: tensor_stats(tensor) for name, tensor in layer_records.items()},
        "focal_records": {
            name: {
                "height": record["height"],
                "width": record["width"],
                "channels": record["channels"],
                "aggregated_context": tensor_stats(record["aggregated_context"]),  # type: ignore[arg-type]
                "modulator": tensor_stats(record["modulator"]),  # type: ignore[arg-type]
            }
            for name, record in focal_records.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        require_runtime_dependencies()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    model, config = build_model(device, args)
    sample = read_sample(args.image, args.trimap, args.pad_to_multiple, device)

    with LayerRecorder(model) as layer_recorder, FocalModulationRecorder(
        model=model,
        filters=args.module_filter,
        capture_all=args.all_focal_modules,
        max_modules=args.max_focal_modules,
    ) as focal_recorder:
        if focal_recorder.selected_names:
            print("Capturing focal modules:")
            for name in focal_recorder.selected_names:
                print(f"  {name}")
        else:
            print("Warning: no focal-modulation modules matched the requested filters.")

        with torch.no_grad():
            prediction = model(sample.image_tensor, sample.trimap_tensor)

    alpha = crop_alpha(prediction, sample.pad)
    write_image(output_dir / "pred_alpha.png", alpha_to_bgr(alpha))
    write_image(output_dir / "input_image.png", sample.raw_bgr)
    write_image(output_dir / "trimap.png", trimap_to_bgr(sample.trimap))

    save_layer_overview(layer_recorder.records, sample, alpha, output_dir, args)
    save_context_aggregation_overview(layer_recorder.records, sample, alpha, output_dir, args)
    save_context_pipeline_diagram(layer_recorder.records, sample, alpha, output_dir, args)
    save_common_channel_grids(layer_recorder.records, output_dir, args)
    if focal_recorder.records:
        save_focal_module_visualisations(focal_recorder.records, sample, alpha, output_dir, args)

    save_manifest(
        output_dir / "visualisation_manifest.json",
        args,
        config,
        device,
        layer_recorder.records,
        focal_recorder.records,
    )
    print(f"Wrote visualisations to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
