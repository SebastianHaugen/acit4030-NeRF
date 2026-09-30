"""
Train the baseline NeRF on the active dataset (lego or poster).

Adapted from baseline/train_nerf.py (course book, Chapter 6). All settings
come from config.yaml via repo_util/LoadConfigurations.py; the file is only read,
never changed. Run from the project root:

    python -m src.train                          # dataset = active_dataset in config.yaml
    NERF_DATASET=poster python -m src.train      # choose the dataset for this run
"""
import csv
import shutil

import matplotlib
matplotlib.use("Agg")  # save figures to disk without opening windows
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
)

from src.load_nerf_data import load_nerf_data
from src.model import NeuralRadianceField
from src.repo_util.LoadConfigurations import (
    CONFIG_PATH,
    ACTIVE_DATASET,
    DATA_DIR,
    OUTPUT_DIR,
    RESIZE_TO,
    CAMERA_ZNEAR,
    CAMERA_ZFAR,
    RAY_MIN_DEPTH,
    RAY_MAX_DEPTH,
    USE_SILHOUETTE_LOSS,
    TEST_EVERY,
    N_RAYS_PER_IMAGE,
    N_PTS_PER_RAY,
    N_ITERATIONS,
    BATCH_SIZE,
    LEARNING_RATE,
    LR_DECAY_AT,
    LR_DECAY_FACTOR,
    HUBER_SCALING,
    RANDOM_SEED,
    VISUALIZE_EVERY,
    CHECKPOINT_EVERY,
)

PREVIEW_VIEW = 0  # fixed training view for the previews, so they are comparable over time


# ------------------------------------------------------------
# Helpers (huber and sample_images_at_mc_locs are from the baseline)
# ------------------------------------------------------------
def huber(x, y, scaling=HUBER_SCALING):
    """
    A helper function for evaluating the smooth L1 (huber) loss
    between the rendered silhouettes and colors.
    """
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss


def sample_images_at_mc_locs(target_images, sampled_rays_xy, ndc_extent=(1.0, 1.0)):
    """
    Given a set of Monte Carlo pixel locations `sampled_rays_xy`,
    this method samples the tensor `target_images` at the
    respective 2D locations.

    Change from the baseline: `ndc_extent` rescales the NDC coordinates
    to the [-1, 1] range grid_sample expects. For square images it is
    (1, 1), which gives exactly the baseline behaviour.
    """
    ba = target_images.shape[0]
    dim = target_images.shape[-1]
    spatial_size = sampled_rays_xy.shape[1:-1]
    extent = torch.tensor(ndc_extent, device=sampled_rays_xy.device)
    # Note the sign inversion: PyTorch3D NDC (+x left, +y up) vs grid_sample.
    grid = -(sampled_rays_xy / extent)
    images_sampled = torch.nn.functional.grid_sample(
        target_images.permute(0, 3, 1, 2),
        grid.view(ba, -1, 1, 2),
        align_corners=True,
    )
    return images_sampled.permute(0, 2, 3, 1).view(ba, *spatial_size, dim)


def ndc_extent_for(height, width):
    """
    PyTorch3D NDC convention: the shorter image side spans [-1, 1],
    the longer side spans [-r, r] with r = long / short.
    Returns (x_extent, y_extent).
    """
    if width >= height:
        return (width / height, 1.0)
    return (1.0, height / width)


def fix_fov_for_aspect(cameras, height, width):
    """
    PyTorch3D applies the FoV of FoVPerspectiveCameras to the *shorter* image
    side, but load_nerf_data computes the vertical FoV. For portrait images
    (width < height, e.g. poster) we convert it to the horizontal FoV.
    Landscape and square images (lego) are unchanged.
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


def select_cameras(cameras, idx):
    """Build a camera batch from a subset of indices (same as the baseline loop)."""
    return FoVPerspectiveCameras(
        R=cameras.R[idx],
        T=cameras.T[idx],
        znear=cameras.znear[idx],
        zfar=cameras.zfar[idx],
        aspect_ratio=cameras.aspect_ratio[idx],
        fov=cameras.fov[idx],
        device=cameras.device,
    )


def load_train_split(device):
    """
    Load the training images, silhouettes and cameras for the active dataset.
    lego has its own train split. poster has one transforms.json, so we hold
    out every TEST_EVERY-th image ourselves (load_nerf_data stays untouched).
    """
    images, silhouettes, cameras = load_nerf_data(
        str(DATA_DIR),
        split="train",
        device=device,
        znear=CAMERA_ZNEAR,
        zfar=CAMERA_ZFAR,
        resize_to=RESIZE_TO,
    )
    # Bicubic resizing can overshoot slightly outside [0, 1].
    images = images.clamp(0.0, 1.0)
    silhouettes = silhouettes.clamp(0.0, 1.0)

    if TEST_EVERY:
        all_idx = torch.arange(len(images))
        train_idx = all_idx[all_idx % TEST_EVERY != 0]
        images = images[train_idx]
        silhouettes = silhouettes[train_idx]
        cameras = select_cameras(cameras, train_idx.to(device))
        print(f"Held out every {TEST_EVERY}th image: {len(train_idx)} train images left.")

    height, width = images.shape[1:3]
    cameras = fix_fov_for_aspect(cameras, height, width)
    return images, silhouettes, cameras


def save_preview(model, camera, target_image, renderer_grid, hist_color, hist_sil, iteration, path):
    """
    Render one full training view and save it next to its target and the loss
    curves. Always the same view, so previews over training are comparable.
    """
    with torch.no_grad():
        rendered, _ = renderer_grid(
            cameras=camera, volumetric_function=model.batched_forward
        )
    rendered_image = rendered[0, ..., :3].clamp(0.0, 1.0)
    target_image = target_image.clamp(0.0, 1.0)
    psnr = -10 * torch.log10(((rendered_image - target_image) ** 2).mean())

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(rendered_image.cpu().numpy())
    ax[0].set_title(f"Render (iter {iteration}, PSNR {psnr:.2f} dB)")
    ax[1].imshow(target_image.cpu().numpy())
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


def save_checkpoint(model, iteration, loss_history, path):
    torch.save(
        {
            "iteration": iteration,
            "dataset": ACTIVE_DATASET,
            "model_state_dict": model.state_dict(),
            "loss_history": loss_history,
        },
        path,
    )


def save_loss_history_csv(color, silhouette, total, path):
    """
    One row per iteration. Opens directly in Excel.
    The silhouette column is empty when the silhouette loss is not used (poster).
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "color_loss", "silhouette_loss", "total_loss"])
        for i, (c, t) in enumerate(zip(color, total)):
            s = silhouette[i] if i < len(silhouette) else ""
            writer.writerow([i, c, s, t])


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def main(
    n_iter=N_ITERATIONS,
    output_dir=OUTPUT_DIR,
    vis_every=VISUALIZE_EVERY,
    checkpoint_every=CHECKPOINT_EVERY,
):
    """
    Train on the active dataset. The arguments default to config.yaml and are
    only overridden by smoke_test.py (short run in a separate folder).
    Returns the loss histories.
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(RANDOM_SEED)

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    # Record the exact settings this run used (read-only copy, not a second config).
    config_copy = output_dir / "config_used.yaml"
    shutil.copy(CONFIG_PATH, config_copy)
    with open(config_copy, "a", encoding="utf-8") as f:
        f.write(f"\n# This run: dataset = {ACTIVE_DATASET}, n_iter = {n_iter}\n")

    # Data
    print(f"Dataset: {ACTIVE_DATASET}  ({DATA_DIR})")
    target_images, target_silhouettes, target_cameras = load_train_split(device)
    n_images, height, width, _ = target_images.shape
    x_extent, y_extent = ndc_extent_for(height, width)
    print(f"{n_images} training images at {width}x{height}, {n_iter} iterations")

    # Renderers
    raymarcher = EmissionAbsorptionRaymarcher()

    # Monte Carlo sampler: random rays for each training step.
    raysampler_mc = MonteCarloRaysampler(
        min_x=-x_extent,
        max_x=x_extent,
        min_y=-y_extent,
        max_y=y_extent,
        n_rays_per_image=N_RAYS_PER_IMAGE,
        n_pts_per_ray=N_PTS_PER_RAY,
        min_depth=RAY_MIN_DEPTH,
        max_depth=RAY_MAX_DEPTH,
    )
    renderer_mc = ImplicitRenderer(raysampler=raysampler_mc, raymarcher=raymarcher).to(device)

    # Grid sampler: one ray per pixel, for full-image previews.
    raysampler_grid = NDCMultinomialRaysampler(
        image_height=height,
        image_width=width,
        n_pts_per_ray=N_PTS_PER_RAY,
        min_depth=RAY_MIN_DEPTH,
        max_depth=RAY_MAX_DEPTH,
    )
    renderer_grid = ImplicitRenderer(raysampler=raysampler_grid, raymarcher=raymarcher).to(device)

    # Model and optimizer
    model = NeuralRadianceField().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_decay_iteration = round(n_iter * LR_DECAY_AT)

    loss_history_color, loss_history_sil, loss_history_total = [], [], []

    for iteration in tqdm(range(n_iter), desc="Training"):
        if iteration == lr_decay_iteration:
            # Same as the baseline: a fresh optimizer with a lower LR.
            tqdm.write(f"Decreasing LR by factor {LR_DECAY_FACTOR} ...")
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * LR_DECAY_FACTOR)

        optimizer.zero_grad()
        batch_idx = torch.randperm(n_images)[:BATCH_SIZE]
        batch_cameras = select_cameras(target_cameras, batch_idx)

        rendered, sampled_rays = renderer_mc(
            cameras=batch_cameras, volumetric_function=model
        )
        rendered_images, rendered_silhouettes = rendered.split([3, 1], dim=-1)

        colors_at_rays = sample_images_at_mc_locs(
            target_images[batch_idx], sampled_rays.xys, (x_extent, y_extent)
        )
        color_err = huber(rendered_images, colors_at_rays).abs().mean()
        loss = color_err

        if USE_SILHOUETTE_LOSS:
            silhouettes_at_rays = sample_images_at_mc_locs(
                target_silhouettes[batch_idx, ..., None], sampled_rays.xys, (x_extent, y_extent)
            )
            sil_err = huber(rendered_silhouettes, silhouettes_at_rays).abs().mean()
            loss = loss + sil_err
            loss_history_sil.append(sil_err.item())

        loss_history_color.append(color_err.item())
        loss_history_total.append(loss.item())

        loss.backward()
        optimizer.step()

        if iteration % vis_every == 0 or iteration == n_iter - 1:
            save_preview(
                model,
                select_cameras(target_cameras, [PREVIEW_VIEW]),
                target_images[PREVIEW_VIEW],
                renderer_grid,
                loss_history_color,
                loss_history_sil,
                iteration,
                preview_dir / f"iter_{iteration:05d}.png",
            )

        if iteration > 0 and iteration % checkpoint_every == 0:
            save_checkpoint(model, iteration, loss_history_total, output_dir / "checkpoint.pt")

    # Final checkpoint and loss history (for the report)
    save_checkpoint(model, n_iter, loss_history_total, output_dir / "checkpoint.pt")
    save_loss_history_csv(
        loss_history_color,
        loss_history_sil,
        loss_history_total,
        output_dir / "loss_history.csv",
    )
    print(f"Done. Outputs in {output_dir}")
    return {"color": loss_history_color, "silhouette": loss_history_sil, "total": loss_history_total}


if __name__ == "__main__":
    main()