"""
Image quality metrics that compare a rendered image with its ground truth.
Every function takes one image pair as tensors of shape (H, W, 3) with values between 0 and 1.
"""
import math

import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure

from src.repo_util.LoadConfigurations import LPIPS_NET


def psnr(prediction, target):
    """Peak signal-to-noise ratio in dB, higher is better."""
    mse = F.mse_loss(prediction, target).item()
    # Identical images have no error, so PSNR is infinite
    if mse == 0:
        return math.inf
    # The maximum pixel value is 1, so PSNR = 10 * log10(1 / MSE)
    return 10.0 * math.log10(1.0 / mse)


def ssim(prediction, target):
    """Structural similarity in local patches, higher is better and 1 means identical."""
    # torchmetrics expects (batch, channels, height, width)
    pred = prediction.permute(2, 0, 1)[None]
    targ = target.permute(2, 0, 1)[None].to(pred.device)
    # Uses the torchmetrics defaults, an 11x11 Gaussian window with sigma 1.5
    return float(structural_similarity_index_measure(pred, targ, data_range=1.0))


class LPIPSMetric:
    """Perceptual distance from a pretrained network, lower is better."""

    def __init__(self, device, net=LPIPS_NET):
        import lpips  # imported here so PSNR and SSIM also work without lpips installed

        # Loaded once and reused for every image, the weights are downloaded the first time
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, prediction, target):
        # LPIPS expects (batch, channels, height, width)
        pred = prediction.permute(2, 0, 1)[None].to(self.device)
        targ = target.permute(2, 0, 1)[None].to(self.device)
        # normalize=True rescales the images from 0 to 1 into the range LPIPS expects
        return float(self.model(pred, targ, normalize=True).item())