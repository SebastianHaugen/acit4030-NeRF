"""
Training script for the NeRF baseline on the lego and poster datasets.
Adapted from Chapter 6 of the course book / the PyTorch3D NeRF tutorial.

Usage (from the repo root):
    py src/train.py --dataset lego
    py src/train.py --dataset poster
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # save figures to disk without opening windows
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
)

from model import NeuralRadianceField
from load_nerf_data import load_nerf_data
from metrics import Evaluator, write_csv

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1. Dataset settings
# ---------------------------------------------------------------------------
DATASETS = {
    "lego": {
        "path": ROOT / "data" / "lego",
        "resize_to": (200, 200),   # (height, width)
        "min_depth": 2.0,          # cameras are ~4.03 units from the object
        "max_depth": 6.0,
        "use_silhouette": True,    # alpha channel available, black background
        "has_splits": True,        # transforms_train.json / transforms_test.json
    },
    "poster": {
        "path": ROOT / "data" / "poster",
        "resize_to": (320, 180),   # portrait, keeps the original 16:9 aspect ratio
        # From projecting sparse_pc.ply into all cameras: visible point depths
        # are 2.2 (1st pct), 4.2 (median), 11.0 (95th pct), 13.8 (99th pct)
        "min_depth": 1.5,
        "max_depth": 12.0,
        "use_silhouette": False,   # real photos, no masks
        "has_splits": False,       # only transforms.json
        "test_every": 8,           # every 8th image is held out for testing
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def huber(x, y, scaling=0.1):
    """Smooth L1 (Huber) loss between rendered and target values."""
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss


def sample_images_at_mc_locs(target_images, sampled_rays_xy, ndc_scale=(1.0, 1.0)):
    """
    Sample `target_images` (B, H, W, C) at the NDC locations of the
    Monte Carlo rays (B, ..., 2).

    For non-square images, PyTorch3D's NDC range is [-1, 1] on the shorter
    side and [-s, s] on the longer side. `ndc_scale` = (sx, sy) rescales the
    coordinates to [-1, 1] as expected by grid_sample. The sign is inverted
    because PyTorch3D's NDC has +X left / +Y up, while grid_sample has
    +X right / +Y down.
    """
    ba = target_images.shape[0]
    dim = target_images.shape[-1]
    spatial_size = sampled_rays_xy.shape[1:-1]
    scale = sampled_rays_xy.new_tensor(ndc_scale)
    grid = -sampled_rays_xy.view(ba, -1, 1, 2) / scale
    images_sampled = F.grid_sample(
        target_images.permute(0, 3, 1, 2), grid, align_corners=True
    )
    return images_sampled.permute(0, 2, 3, 1).view(ba, *spatial_size, dim)


def fix_fov_for_aspect(cameras, height, width):
    """
    PyTorch3D applies the FoV of FoVPerspectiveCameras to the *shorter* image
    side, but load_nerf_data computes the vertical FoV. For portrait images
    (width < height) we convert it to the horizontal FoV. Landscape and square
    images are unchanged.
    """
    if width >= height:
        return cameras
    fov_y = torch.deg2rad(cameras.fov)
    fov_short = 2 * torch.atan(torch.tan(fov_y / 2) * width / height)
    return FoVPerspectiveCameras(
        R=cameras.R,
        T=cameras.T,
        fov=torch.rad2deg(fov_short),
        znear=cameras.znear,
        zfar=cameras.zfar,
        device=cameras.device,
    )


def load_split(cfg, split, device):
    """Load the train or test split. For datasets without split files, every
    `test_every`-th image is held out as the test set."""
    if cfg["has_splits"]:
        imgs, sils, cams = load_nerf_data(
            str(cfg["path"]), split=split, resize_to=cfg["resize_to"]
        )
    else:
        imgs, sils, cams = load_nerf_data(
            str(cfg["path"]), split="train", resize_to=cfg["resize_to"]
        )
        idx = torch.arange(len(imgs))
        is_test = idx % cfg["test_every"] == 0
        keep = idx[is_test] if split == "test" else idx[~is_test]
        imgs, sils, cams = imgs[keep], sils[keep], cams[keep.tolist()]

    height, width = imgs.shape[1:3]
    cams = fix_fov_for_aspect(cams, height, width)
    # Bicubic resizing overshoots slightly outside [0, 1]
    return imgs.clamp(0, 1).to(device), sils.clamp(0, 1).to(device), cams.to(device)


@torch.no_grad()
def render_full(model, renderer_grid, camera, n_batches=32):
    """Render a full image for a single camera in chunks to save GPU memory."""
    out, _ = renderer_grid(
        cameras=camera,
        volumetric_function=model.batched_forward,
        n_batches=n_batches,
    )
    rgb, alpha = out[0].split([3, 1], dim=-1)
    return rgb.clamp(0, 1), alpha


def save_intermediate(path, rendered, target, hist_color, hist_sil, iteration):
    mse = ((rendered - target) ** 2).mean()
    psnr = -10 * torch.log10(mse)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(rendered.cpu().numpy())
    ax[0].set_title(f"Render (iter {iteration}, PSNR {psnr:.2f} dB)")
    ax[1].imshow(target.cpu().numpy())
    ax[1].set_title("Ground truth")
    ax[2].plot(hist_color, label="colour")
    if hist_sil:
        ax[2].plot(hist_sil, label="silhouette")
    ax[2].set_yscale("log")
    ax[2].set_title("Training loss")
    ax[2].legend()
    for a in ax[:2]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="lego")
    parser.add_argument("--n_iter", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=2)   # 6 in the book, reduced for a 4 GB GPU
    parser.add_argument("--n_rays", type=int, default=750)     # rays per image per iteration
    parser.add_argument("--n_pts", type=int, default=128)      # samples per ray
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vis_every", type=int, default=100)
    parser.add_argument("--n_eval_views", type=int, default=5)  # held-out views scored at every vis step
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    out_dir = ROOT / "outputs" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    print(f"Device: {device}")

    # 3. Data
    imgs, sils, cams = load_split(cfg, "train", device)
    n_views, height, width = imgs.shape[:3]
    print(f"Training on {n_views} views of size {height}x{width}")

    # A few evenly spaced held-out views, scored during training
    test_imgs, test_sils, test_cams = load_split(cfg, "test", device)
    eval_idx = torch.linspace(0, len(test_imgs) - 1, args.n_eval_views).round().long().tolist()
    # Building LPIPS initialises AlexNet with random weights before loading the
    # pretrained ones. Fork the RNG so that this does not change the NeRF
    # initialisation (the baseline is sensitive to it and can collapse to an
    # empty scene with some seeds).
    with torch.random.fork_rng(devices=[]):
        evaluator = Evaluator(device=device)

    # NDC extent: [-1, 1] on the shorter side, [-s, s] on the longer side
    sx = max(width / height, 1.0)
    sy = max(height / width, 1.0)

    # 4. Raysamplers and renderers
    raysampler_mc = MonteCarloRaysampler(
        min_x=-sx, max_x=sx,
        min_y=-sy, max_y=sy,
        n_rays_per_image=args.n_rays,
        n_pts_per_ray=args.n_pts,
        min_depth=cfg["min_depth"],
        max_depth=cfg["max_depth"],
    )
    raysampler_grid = NDCMultinomialRaysampler(
        image_height=height,
        image_width=width,
        n_pts_per_ray=args.n_pts,
        min_depth=cfg["min_depth"],
        max_depth=cfg["max_depth"],
    )
    raymarcher = EmissionAbsorptionRaymarcher()
    renderer_mc = ImplicitRenderer(raysampler=raysampler_mc, raymarcher=raymarcher).to(device)
    renderer_grid = ImplicitRenderer(raysampler=raysampler_grid, raymarcher=raymarcher).to(device)

    # 5. Model and optimizer
    model = NeuralRadianceField().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # 6. Training loop
    hist_color, hist_sil = [], []
    loss_rows, eval_rows = [], []
    vis_idx = 0  # fixed view for the intermediate renders, so they are comparable
    current_lr = args.lr
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()
    t_eval = 0.0  # time spent on evaluation, excluded from the training time

    for iteration in tqdm(range(args.n_iter)):
        if iteration == round(args.n_iter * 0.75):
            # As in the baseline: re-creating the optimizer also resets Adam's state
            tqdm.write("Decreasing LR 10-fold ...")
            current_lr = args.lr * 0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)

        optimizer.zero_grad()
        batch_idx = torch.randperm(n_views)[: args.batch_size]
        batch_cams = cams[batch_idx.tolist()]

        rendered, sampled_rays = renderer_mc(cameras=batch_cams, volumetric_function=model)
        rendered_rgb, rendered_alpha = rendered.split([3, 1], dim=-1)

        colors_at_rays = sample_images_at_mc_locs(
            imgs[batch_idx], sampled_rays.xys, (sx, sy)
        )
        color_err = huber(rendered_rgb, colors_at_rays).abs().mean()
        loss = color_err
        hist_color.append(float(color_err))

        if cfg["use_silhouette"]:
            sils_at_rays = sample_images_at_mc_locs(
                sils[batch_idx, ..., None], sampled_rays.xys, (sx, sy)
            )
            sil_err = huber(rendered_alpha, sils_at_rays).abs().mean()
            loss = loss + sil_err
            hist_sil.append(float(sil_err))

        loss.backward()
        optimizer.step()

        loss_rows.append({
            "iteration": iteration,
            "lr": current_lr,
            "loss_total": float(loss),
            "loss_color": float(color_err),
            "loss_silhouette": hist_sil[-1] if hist_sil else "",
        })

        if iteration % args.vis_every == 0 or iteration == args.n_iter - 1:
            t0 = time.perf_counter()
            rgb, _ = render_full(model, renderer_grid, cams[[vis_idx]])
            save_intermediate(
                out_dir / f"intermediate_{iteration:05d}.png",
                rgb, imgs[vis_idx], hist_color, hist_sil, iteration,
            )

            # Score the held-out views
            scores = []
            for i in eval_idx:
                rgb, alpha = render_full(model, renderer_grid, test_cams[[i]])
                gt_mask = test_sils[i] if cfg["use_silhouette"] else None
                scores.append(evaluator(rgb, test_imgs[i], alpha[..., 0], gt_mask))
            t_eval += time.perf_counter() - t0

            row = {
                "iteration": iteration,
                "train_time_s": round(time.perf_counter() - t_start - t_eval, 2),
                "lr": current_lr,
                "loss_color_avg100": sum(hist_color[-100:]) / len(hist_color[-100:]),
            }
            row.update({f"test_{k}": sum(s[k] for s in scores) / len(scores) for k in scores[0]})
            eval_rows.append(row)
            write_csv(out_dir / "training_eval.csv", eval_rows)
            tqdm.write(
                f"[{iteration:6d}] held-out PSNR {row['test_psnr']:.2f}  "
                f"SSIM {row['test_ssim']:.3f}  LPIPS {row['test_lpips']:.3f}"
            )

    train_time = time.perf_counter() - t_start - t_eval
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0

    # 7. Save
    torch.save(model.state_dict(), out_dir / "model.pth")
    config = {**vars(args), **{k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "losses.json", "w") as f:
        json.dump({"color": hist_color, "silhouette": hist_sil}, f)
    write_csv(out_dir / "losses.csv", loss_rows)
    write_csv(out_dir / "training_summary.csv", [{
        "dataset": args.dataset,
        "n_train_views": n_views,
        "image_height": height,
        "image_width": width,
        "n_iter": args.n_iter,
        "batch_size": args.batch_size,
        "n_rays": args.n_rays,
        "n_pts": args.n_pts,
        "model_parameters": n_params,
        "train_time_s": round(train_time, 1),
        "iterations_per_s": round(args.n_iter / train_time, 2),
        "peak_gpu_memory_mb": round(peak_mem_mb, 1),
        "final_loss_color_avg100": sum(hist_color[-100:]) / len(hist_color[-100:]),
    }])

    print(f"Done. Model and renders saved to {out_dir}")


if __name__ == "__main__":
    main()