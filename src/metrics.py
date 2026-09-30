"""
Image quality metrics for evaluating NeRF renders against ground truth.

All image functions take images of shape (H, W, 3) or (B, H, W, 3) with values in [0, 1].

- PSNR  (higher is better): pixel-wise reconstruction error in dB.
- SSIM  (higher is better): structural similarity (Wang et al., 2004).
- LPIPS (lower is better):  learned perceptual distance (Zhang et al., 2018),
  using AlexNet features as in the original NeRF evaluation protocol.
- MSE / MAE (lower is better): mean squared / absolute pixel error.
- Mask IoU (higher is better): intersection-over-union between the rendered
  opacity and the ground-truth silhouette (only for datasets with masks).
"""

import csv

import torch
from torchmetrics.functional.image import structural_similarity_index_measure


def _to_bchw(img):
    """(H, W, 3) or (B, H, W, 3) -> (B, 3, H, W)."""
    if img.dim() == 3:
        img = img[None]
    return img.permute(0, 3, 1, 2).clamp(0, 1)


def mse(pred, gt):
    return ((_to_bchw(pred) - _to_bchw(gt)) ** 2).mean().item()


def mae(pred, gt):
    return (_to_bchw(pred) - _to_bchw(gt)).abs().mean().item()


def psnr(pred, gt):
    """Peak signal-to-noise ratio in dB, averaged over the batch."""
    pred, gt = _to_bchw(pred), _to_bchw(gt)
    per_image_mse = ((pred - gt) ** 2).flatten(1).mean(dim=1)
    return (-10 * torch.log10(per_image_mse)).mean().item()


def ssim(pred, gt):
    """Structural similarity index (11x11 Gaussian window), averaged over the batch."""
    return structural_similarity_index_measure(
        _to_bchw(pred), _to_bchw(gt), data_range=1.0
    ).item()


def mask_iou(pred_alpha, gt_mask, threshold=0.5):
    """IoU between a rendered opacity map and a ground-truth mask, both (H, W)."""
    pred = pred_alpha > threshold
    gt = gt_mask > threshold
    union = (pred | gt).sum()
    if union == 0:
        return 1.0
    return ((pred & gt).sum() / union).item()


class LPIPS:
    """Wrapper around the `lpips` package that accepts images in [0, 1]."""

    def __init__(self, net="alex", device="cpu"):
        import lpips  # imported lazily, since it loads pretrained weights

        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred, gt):
        # lpips expects inputs in [-1, 1]
        pred = _to_bchw(pred).to(self.device) * 2 - 1
        gt = _to_bchw(gt).to(self.device) * 2 - 1
        return self.model(pred, gt).mean().item()


class Evaluator:
    """Computes all image metrics for a rendered view against its ground truth."""

    def __init__(self, device="cpu"):
        self.lpips = LPIPS(device=device)

    @torch.no_grad()
    def __call__(self, pred, gt, pred_alpha=None, gt_mask=None):
        scores = {
            "psnr": psnr(pred, gt),
            "ssim": ssim(pred, gt),
            "lpips": self.lpips(pred, gt),
            "mse": mse(pred, gt),
            "mae": mae(pred, gt),
        }
        if pred_alpha is not None and gt_mask is not None:
            scores["mask_iou"] = mask_iou(pred_alpha, gt_mask)
        return scores


def write_csv(path, rows):
    """Write a list of dicts (same keys) to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
