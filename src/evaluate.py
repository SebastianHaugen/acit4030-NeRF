"""
Quantitative and qualitative evaluation of a trained NeRF on the held-out views.

Renders every test view, computes PSNR / SSIM / LPIPS / MSE / MAE (and mask IoU
for lego) against the ground truth, and saves:
    outputs/<dataset>/metrics_<split>.csv          per-view metrics
    outputs/<dataset>/metrics_<split>_summary.csv  mean / std / min / median / max per metric
    outputs/<dataset>/metrics_<split>.json         the same, as JSON
    outputs/results.csv                            mean metrics, one row per dataset and split
    outputs/<dataset>/renders_<split>/             rendered test views (PNG)
    report/figures/<dataset>_comparison.png        ground truth / render / error for a few views
    report/figures/<dataset>_loss.png              training loss curve

Usage (from the repo root, after training):
    python src/evaluate.py --dataset lego
    python src/evaluate.py --dataset poster
"""

import argparse
import csv
import json
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pytorch3d.renderer import (
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
)

from model import NeuralRadianceField
from metrics import Evaluator, write_csv
from train import DATASETS, ROOT, load_split, render_full


def update_results_table(path, row):
    """Insert or replace the (dataset, split) row in the combined results CSV."""
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if (r["dataset"], r["split"]) != (row["dataset"], row["split"])]
    rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], r["split"]))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def save_comparison(path, gts, renders, psnrs, n_show=4):
    """Grid with one column per view and rows: ground truth, render, abs. error."""
    idx = np.linspace(0, len(gts) - 1, n_show).round().astype(int)
    h, w = gts[0].shape[:2]
    fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 3 * 3 * h / w))
    for col, i in enumerate(idx):
        err = (renders[i] - gts[i]).abs().mean(-1)
        axes[0, col].imshow(gts[i].numpy())
        axes[0, col].set_title(f"Test view {i}")
        axes[1, col].imshow(renders[i].numpy())
        axes[1, col].set_title(f"PSNR {psnrs[i]:.2f} dB")
        axes[2, col].imshow(err.numpy(), cmap="inferno", vmin=0, vmax=0.5)
    for row, name in enumerate(["Ground truth", "NeRF render", "|Error|"]):
        axes[row, 0].set_ylabel(name, fontsize=12)
    for a in axes.flat:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_loss_curve(path, losses, dataset):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses["color"], label="colour (Huber)", alpha=0.8)
    if losses["silhouette"]:
        ax.plot(losses["silhouette"], label="silhouette (Huber)", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training loss ({dataset})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="lego")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    out_dir = ROOT / "outputs" / args.dataset
    fig_dir = ROOT / "report" / "figures"
    render_dir = out_dir / f"renders_{args.split}"
    render_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json") as f:
        train_cfg = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    imgs, sils, cams = load_split(cfg, args.split, device)
    n_views, height, width = imgs.shape[:3]
    print(f"Evaluating on {n_views} {args.split} views of size {height}x{width}")

    raysampler = NDCMultinomialRaysampler(
        image_height=height,
        image_width=width,
        n_pts_per_ray=train_cfg["n_pts"],
        min_depth=cfg["min_depth"],
        max_depth=cfg["max_depth"],
    )
    renderer = ImplicitRenderer(
        raysampler=raysampler, raymarcher=EmissionAbsorptionRaymarcher()
    ).to(device)

    model = NeuralRadianceField().to(device)
    model.load_state_dict(torch.load(out_dir / "model.pth", map_location=device))
    model.eval()

    evaluator = Evaluator(device=device)
    per_view, gts, renders = [], [], []
    for i in range(n_views):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        rgb, alpha = render_full(model, renderer, cams[[i]])
        if device.type == "cuda":
            torch.cuda.synchronize()
        render_time = time.perf_counter() - t0

        gt_mask = sils[i] if cfg["use_silhouette"] else None
        scores = evaluator(rgb, imgs[i], alpha[..., 0], gt_mask)
        scores["render_time_s"] = render_time
        per_view.append({"view": i, **scores})
        print(f"view {i:3d}: " + ", ".join(f"{k}={v:.4f}" for k, v in scores.items()))

        rgb, gt = rgb.cpu(), imgs[i].cpu()
        Image.fromarray((rgb.numpy() * 255).round().astype(np.uint8)).save(
            render_dir / f"{i:03d}.png"
        )
        renders.append(rgb)
        gts.append(gt)

    keys = [k for k in per_view[0] if k != "view"]
    summary = {}
    for k in keys:
        vals = np.array([v[k] for v in per_view])
        summary[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "median": float(np.median(vals)),
            "max": float(vals.max()),
        }
    with open(out_dir / f"metrics_{args.split}.json", "w") as f:
        json.dump({"summary": summary, "per_view": per_view}, f, indent=2)
    write_csv(out_dir / f"metrics_{args.split}.csv", per_view)
    write_csv(
        out_dir / f"metrics_{args.split}_summary.csv",
        [{"metric": k, **s} for k, s in summary.items()],
    )
    update_results_table(ROOT / "outputs" / "results.csv", {
        "dataset": args.dataset,
        "split": args.split,
        "n_views": n_views,
        **{f"{k}_mean": summary[k]["mean"] for k in keys},
        **{f"{k}_std": summary[k]["std"] for k in keys},
    })

    print(f"\n{args.dataset} ({args.split}, {n_views} views)")
    for k in keys:
        print(f"  {k:14s} {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}")

    if args.split == "test":
        save_comparison(
            fig_dir / f"{args.dataset}_comparison.png",
            gts, renders, [v["psnr"] for v in per_view],
        )
        with open(out_dir / "losses.json") as f:
            save_loss_curve(fig_dir / f"{args.dataset}_loss.png", json.load(f), args.dataset)
        print(f"Figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
