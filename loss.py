from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel2d(kernel_size: Tuple[int, int], sigma: Tuple[float, float]) -> torch.Tensor:
    kernels = []
    for size, std in zip(kernel_size, sigma):
        if size <= 0 or size % 2 == 0:
            raise ValueError("kernel_size values must be odd positive integers")
        x = torch.arange(size).float() - size // 2
        kernel = torch.exp(-x.square() / (2 * std ** 2))
        kernels.append(kernel / kernel.sum())
    return torch.outer(kernels[0], kernels[1]).unsqueeze(0)


def same_padding(kernel: torch.Tensor):
    height, width = kernel.shape[-2:]
    return [
        width // 2 - int(height % 2 == 0),
        width // 2,
        height // 2 - int(width % 2 == 0),
        height // 2,
    ]


class LaplacianLoss(nn.Module):
    def __init__(
        self,
        levels: int = 5,
        k_size: Tuple[int, int] = (5, 5),
        sigma: Tuple[float, float] = (1.5, 1.5),
        border_type: str = "reflect",
        loss_type: str = "L1",
        loss_multiplier: float = 2.0,
        clip: bool = True,
        clipmin: float = 0.0,
        clipmax: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        losses = {"L1": nn.L1Loss, "L2": nn.MSELoss}
        if loss_type not in losses:
            raise ValueError("loss_type must be 'L1' or 'L2'")

        self.levels = levels
        self.border_type = border_type
        self.loss_multiplier = loss_multiplier
        self.clip = clip
        self.clipmin = clipmin
        self.clipmax = clipmax
        self.loss = losses[loss_type](reduction=reduction)
        self.register_buffer("_gauss_kernel", gaussian_kernel2d(k_size, sigma))

    def blur(self, image: torch.Tensor) -> torch.Tensor:
        channels = image.shape[1]
        kernel = self._gauss_kernel.to(device=image.device, dtype=image.dtype)
        image = F.pad(image, same_padding(kernel), mode=self.border_type)
        return F.conv2d(image, kernel.unsqueeze(0).expand(channels, -1, -1, -1), groups=channels)

    def pyramid(self, image: torch.Tensor):
        levels = []
        for _ in range(self.levels):
            blurred = self.blur(image)
            levels.append(image - blurred)
            image = F.avg_pool2d(blurred, 2)
        return levels

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.clip:
            input = torch.clamp(input, self.clipmin, self.clipmax)

        weight = 1
        loss = 0
        for input_level, target_level in zip(self.pyramid(input), self.pyramid(target)):
            loss = loss + weight * self.loss(input_level, target_level)
            weight *= self.loss_multiplier
        return loss


