"""
Image quality metrics for evaluating NeRF renders against ground truth.

    PSNR  (higher is better)  - pixel-wise error, 10 * log10(1 / MSE)
    SSIM  (higher is better)  - structural similarity in local patches
    LPIPS (lower is better)   - perceptual distance from a pretrained CNN

All functions take a single image pair as torch tensors of shape (H, W, 3)
with values in [0, 1].
"""
import math

import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure

from src.repo_util.LoadConfigurations import LPIPS_NET


def psnr(prediction, target):
    """
    Peak signal-to-noise ratio in dB for images in [0, 1] (max value 1).
    PSNR = 10 * log10(1 / MSE)  (equivalently 20 * log10(1 / sqrt(MSE))).
    """
    mse = F.mse_loss(prediction, target).item()
    if mse == 0:
        return math.inf
    return 10.0 * math.log10(1.0 / mse)


def ssim(prediction, target):
    """
    Structural similarity with torchmetrics' defaults (11x11 Gaussian window,
    sigma 1.5), the standard SSIM setting also used in the NeRF literature.
    """
    pred = prediction.permute(2, 0, 1)[None]
    targ = target.permute(2, 0, 1)[None].to(pred.device)
    return float(structural_similarity_index_measure(pred, targ, data_range=1.0))


class LPIPSMetric:
    """
    Wrapper around the `lpips` package. The network is loaded once (the first
    time pretrained weights are downloaded) and reused for every image.
    """

    def __init__(self, device, net=LPIPS_NET):
        import lpips  # imported here so PSNR/SSIM work without lpips installed

        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, prediction, target):
        # (H, W, 3) -> (1, 3, H, W); normalize=True maps [0, 1] to the [-1, 1] LPIPS expects.
        pred = prediction.permute(2, 0, 1)[None].to(self.device)
        targ = target.permute(2, 0, 1)[None].to(self.device)
        return float(self.model(pred, targ, normalize=True).item())