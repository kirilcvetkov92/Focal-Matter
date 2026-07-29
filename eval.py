from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from utils.config import ProjectConfig, clone_config
from utils.experiments import create_experiment, setup_logger


def apply_runtime_environment(config: ProjectConfig) -> None:
    if config.runtime.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.runtime.cuda_visible_devices)
    if config.runtime.kmp_duplicate_lib_ok:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def resolve_device(config: ProjectConfig):
    import torch

    if config.runtime.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(
    config: ProjectConfig,
    device=None,
    logger: Optional[logging.Logger] = None,
    load_checkpoint: bool = True,
    require_checkpoint: bool = True,
):
    import torch
    from models.focal import FocalNet
    from models.model import FocalMatter

    encoder = FocalNet(
        in_chans=config.encoder.in_chans,
        embed_dim=config.encoder.embed_dim,
        depths=list(config.encoder.depths),
        focal_levels=list(config.encoder.focal_levels),
        focal_windows=list(config.encoder.focal_windows),
    )

    encoder_weights_path = existing_path_or_none(config.model.encoder_weights_path, logger)
    model = FocalMatter(
        encoder=encoder,
        encoder_weights_path=encoder_weights_path,
    )

    if load_checkpoint:
        checkpoint_path = config.model.checkpoint_path
        if checkpoint_path and Path(checkpoint_path).is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get(config.model.checkpoint_key, checkpoint)
            load_result = model.load_state_dict(
                state_dict,
                strict=config.model.strict_checkpoint,
            )
            if logger:
                logger.info("Loaded checkpoint: %s", checkpoint_path)
                if load_result.missing_keys:
                    logger.warning("Missing checkpoint keys: %s", load_result.missing_keys)
                if load_result.unexpected_keys:
                    logger.warning("Unexpected checkpoint keys: %s", load_result.unexpected_keys)
        elif require_checkpoint:
            raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
        elif logger:
            logger.warning("Checkpoint not found, starting from current weights: %s", checkpoint_path)

    if device is not None:
        model = model.to(device)

    return model


def existing_path_or_none(path: Optional[str], logger: Optional[logging.Logger]) -> Optional[str]:
    if not path:
        return None
    if Path(path).is_file():
        return path
    if logger:
        logger.warning("Weights file not found: %s", path)
    return None


def resolve_output_dir(config: ProjectConfig, paths, epoch: Optional[int]) -> Path:
    if config.eval.output_dir:
        output_dir = Path(config.eval.output_dir)
    else:
        output_dir = paths.output_dir

    if epoch is not None:
        output_dir = output_dir / "epoch_{:04d}".format(epoch)
    return output_dir


def predict_alpha(model, raw_img, trimap, device, pad_to_multiple: int):
    import cv2
    import numpy as np
    import torch

    trimap_original = trimap.copy()
    height, width = trimap.shape
    new_height = ((height - 1) // pad_to_multiple + 1) * pad_to_multiple
    new_width = ((width - 1) // pad_to_multiple + 1) * pad_to_multiple
    pad_height = new_height - height
    pad_width = new_width - width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    raw_img_pad = cv2.copyMakeBorder(
        raw_img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_REFLECT,
    )
    trimap_pad = cv2.copyMakeBorder(
        trimap,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_REFLECT,
    )

    trimap_tensor = np.zeros([*trimap_pad.shape, 3], np.float32)
    trimap_tensor[:, :, 0] = trimap_pad == 0
    trimap_tensor[:, :, 1] = trimap_pad == 128
    trimap_tensor[:, :, 2] = trimap_pad == 255
    trimap_tensor = np.transpose(trimap_tensor, (2, 0, 1))[np.newaxis, :, :, :]

    image_tensor = np.transpose(raw_img_pad, (2, 0, 1))[np.newaxis, ::-1, :, :]
    image_tensor = np.asarray(image_tensor, np.float32) / 255.0

    image_tensor = torch.from_numpy(image_tensor).to(device)
    trimap_tensor = torch.from_numpy(trimap_tensor).to(device)

    with torch.no_grad():
        pred = model(image_tensor, trimap_tensor)
        pred = pred.detach().cpu().numpy()[0]
        pred = pred[:, pad_top:pad_top + height, pad_left:pad_left + width]
        pred_alpha = np.transpose(pred[0:1] * 255.0, (1, 2, 0))
        pred_alpha = (
            pred_alpha * (trimap_original[:, :, None] == 128)
            + (trimap_original[:, :, None] == 255) * 255
        )

    return pred_alpha.squeeze(-1)


def compute_mse_loss(pred, target, trimap):
    import numpy as np

    error_map = (pred - target) / 255.0
    return np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)


def compute_sad_loss(pred, target, trimap):
    import numpy as np

    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128))
    return loss / 1000, np.sum(trimap == 128) / 1000


def mat2gray(img):
    import numpy as np

    img = np.asarray(img, dtype=np.float32)
    min_value = np.min(img)
    max_value = np.max(img)
    if max_value <= min_value:
        return np.zeros_like(img, dtype=np.float32)
    return (img - min_value) / (max_value - min_value)


def gaussgradient(img, sigma):
    import cv2
    import numpy as np

    epsilon = 1e-2
    half_size = int(np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))))
    axis = np.arange(-half_size, half_size + 1, dtype=np.float32)
    gaussian = np.exp(-(axis ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    dgaussian = -axis * gaussian / (sigma ** 2)

    hx = np.outer(gaussian, dgaussian).astype(np.float32)
    hx_norm = np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    if hx_norm > 0:
        hx = hx / hx_norm
    hy = hx.T

    gx = cv2.filter2D(img.astype(np.float32), cv2.CV_32F, hx, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(img.astype(np.float32), cv2.CV_32F, hy, borderType=cv2.BORDER_REPLICATE)
    return gx, gy


def compute_gradient_loss(pred, target, trimap):
    import numpy as np

    pred = mat2gray(pred)
    target = mat2gray(target)
    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)
    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)
    error_map = (pred_amp - target_amp) ** 2
    return np.sum(error_map * (trimap == 128))


def compute_connectivity_error(pred, target, trimap, step: float = 0.1):
    import cv2
    import numpy as np

    pred = np.asarray(pred, dtype=np.float32) / 255.0
    target = np.asarray(target, dtype=np.float32) / 255.0
    dimy, dimx = pred.shape

    thresh_steps = np.linspace(0, 1, int(round(1 / step)) + 1, dtype=np.float32)
    l_map = np.ones((dimy, dimx), dtype=np.float32) * -1

    for ii in range(1, len(thresh_steps)):
        pred_alpha_thresh = pred >= thresh_steps[ii]
        target_alpha_thresh = target >= thresh_steps[ii]
        intersection = (pred_alpha_thresh & target_alpha_thresh).astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(intersection, connectivity=4)
        omega = np.zeros((dimy, dimx), dtype=bool)
        if num_labels > 1:
            areas = np.bincount(labels.reshape(-1))[1:]
            max_label = int(np.argmax(areas) + 1)
            omega = labels == max_label

        flag = (l_map == -1) & (~omega)
        l_map[flag] = thresh_steps[ii - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15)
    target_phi = 1 - target_d * (target_d >= 0.15)

    return np.sum(np.abs(pred_phi - target_phi) * (trimap == 128))


def run_evaluation(
    config: Optional[ProjectConfig] = None,
    model=None,
    epoch: Optional[int] = None,
    experiment_paths=None,
    logger: Optional[logging.Logger] = None,
    full_metrics: Optional[bool] = None,
) -> Dict[str, float]:
    config = config or clone_config()
    apply_runtime_environment(config)

    import cv2
    import numpy as np
    import torch

    created_experiment = experiment_paths is None
    if experiment_paths is None:
        experiment_paths = create_experiment(config, phase="eval", project_root=Path(__file__).resolve().parent)
    if logger is None:
        logger = setup_logger(config, experiment_paths, phase="eval")

    device = resolve_device(config)
    if model is None:
        model = build_model(
            config,
            device=device,
            logger=logger,
            load_checkpoint=True,
            require_checkpoint=True,
        )
    else:
        model = model.to(device)

    if device.type == "cuda":
        for param in model.parameters():
            param.grad = None
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    was_training = model.training
    model.eval()

    full_metrics = config.eval.full_metrics if full_metrics is None else full_metrics
    output_dir = resolve_output_dir(config, experiment_paths, epoch)
    if config.eval.write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    trimap_dir = Path(config.eval.trimap_dir)
    image_dir = Path(config.eval.image_dir)
    alpha_dir = Path(config.eval.alpha_dir)

    mse_values = []
    sad_values = []
    grad_values = []
    conn_values = []

    files = sorted(os.listdir(trimap_dir))
    if config.eval.limit is not None:
        files = files[: config.eval.limit]

    logger.info("Evaluating %s files", len(files))

    for idx, filename in enumerate(files):
        raw_img = cv2.imread(str(image_dir / filename), cv2.IMREAD_COLOR)
        trimap = cv2.imread(str(trimap_dir / filename), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(alpha_dir / filename), cv2.IMREAD_GRAYSCALE)

        if raw_img is None or trimap is None or label is None:
            logger.warning("Skipping unreadable sample: %s", filename)
            continue

        label = label.astype(np.float32)
        pred_alpha = predict_alpha(
            model=model,
            raw_img=raw_img,
            trimap=trimap,
            device=device,
            pad_to_multiple=config.eval.pad_to_multiple,
        )

        mse_value = compute_mse_loss(pred_alpha, label, trimap)
        sad_value, _ = compute_sad_loss(pred_alpha, label, trimap)
        mse_values.append(mse_value)
        sad_values.append(sad_value)

        message = (
            "{:04d} | MSE: {:.5f}, SAD: {:.5f} | Mean MSE: {:.5f}, Mean SAD: {:.5f}"
            .format(idx, mse_value, sad_value, float(np.mean(mse_values)), float(np.mean(sad_values)))
        )

        if full_metrics:
            grad_value = compute_gradient_loss(pred_alpha, label, trimap) / 1000
            conn_value = compute_connectivity_error(pred_alpha, label, trimap) / 1000
            grad_values.append(grad_value)
            conn_values.append(conn_value)
            message += ", Grad: {:.5f}, Conn: {:.5f} | Mean Grad: {:.5f}, Mean Conn: {:.5f}".format(
                grad_value,
                conn_value,
                float(np.mean(grad_values)),
                float(np.mean(conn_values)),
            )

        logger.info(message)

        if config.eval.write_outputs:
            cv2.imwrite(str(output_dir / filename), pred_alpha.astype(np.uint8))

    metrics = {
        "mse": float(np.mean(mse_values)) if mse_values else float("nan"),
        "sad": float(np.mean(sad_values)) if sad_values else float("nan"),
    }
    if full_metrics:
        metrics["grad"] = float(np.mean(grad_values)) if grad_values else float("nan")
        metrics["conn"] = float(np.mean(conn_values)) if conn_values else float("nan")

    logger.info("Final metrics: %s", metrics)
    if created_experiment:
        logger.info("Evaluation outputs: %s", output_dir)

    if was_training:
        model.train()
    return metrics


if __name__ == "__main__":
    run_evaluation(clone_config())
