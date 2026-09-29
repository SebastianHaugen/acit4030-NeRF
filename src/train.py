"""
Train the baseline NeRF on the active dataset (lego or poster).

Adapted from baseline/train_nerf.py (course book, Chapter 6). All settings
come from config.yaml via repo_util/LoadConfigurations.py. Pick the dataset
with `active_dataset` in config.yaml, then run from the project root:

    python -m src.train
"""
import json
import shutil

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

    return images, silhouettes, cameras


def save_preview(model, camera, target_image, renderer_grid, loss_history, path):
    """Render one full training view and save it next to its target + loss curve."""
    with torch.no_grad():
        rendered, _ = renderer_grid(
            cameras=camera, volumetric_function=model.batched_forward
        )
    rendered_image = rendered[0, ..., :3].clamp(0.0, 1.0).cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(loss_history, linewidth=1)
    ax[0].set_title("loss")
    ax[0].set_yscale("log")
    ax[1].imshow(rendered_image)
    ax[1].set_title("rendered image")
    ax[2].imshow(target_image.clamp(0.0, 1.0).cpu().numpy())
    ax[2].set_title("target image")
    for a in ax[1:]:
        a.axis("off")
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


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preview_dir = OUTPUT_DIR / "previews"
    preview_dir.mkdir(exist_ok=True)
    # Record the exact settings this run used (read-only copy, not a second config).
    shutil.copy(CONFIG_PATH, OUTPUT_DIR / "config_used.yaml")

    # Data
    print(f"Dataset: {ACTIVE_DATASET}  ({DATA_DIR})")
    target_images, target_silhouettes, target_cameras = load_train_split(device)
    n_images, height, width, _ = target_images.shape
    x_extent, y_extent = ndc_extent_for(height, width)
    print(f"{n_images} training images at {width}x{height}")

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
    lr_decay_iteration = round(N_ITERATIONS * LR_DECAY_AT)

    loss_history_color, loss_history_sil, loss_history_total = [], [], []

    for iteration in tqdm(range(N_ITERATIONS), desc="Training"):
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
            loss_history_sil.append(float(sil_err))

        loss_history_color.append(float(color_err))
        loss_history_total.append(float(loss))

        loss.backward()
        optimizer.step()

        if iteration % VISUALIZE_EVERY == 0:
            show_idx = torch.randperm(n_images)[:1]
            save_preview(
                model,
                select_cameras(target_cameras, show_idx),
                target_images[show_idx][0],
                renderer_grid,
                loss_history_total,
                preview_dir / f"iter_{iteration:05d}.png",
            )

        if iteration > 0 and iteration % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, iteration, loss_history_total, OUTPUT_DIR / "checkpoint.pt")

    # Final checkpoint and loss history (for the report)
    save_checkpoint(model, N_ITERATIONS, loss_history_total, OUTPUT_DIR / "checkpoint.pt")
    with open(OUTPUT_DIR / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "color": loss_history_color,
                "silhouette": loss_history_sil,
                "total": loss_history_total,
            },
            f,
        )
    print(f"Done. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()