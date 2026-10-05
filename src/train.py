"""
Trains the baseline NeRF on the active dataset, adapted from baseline/train_nerf.py (course book Chapter 6).
All settings are read from config.yaml, and the NERF_DATASET variable picks the dataset for one run.
"""
import csv
import shutil

import matplotlib
matplotlib.use("Agg")  # saves figures to disk without opening windows
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

PREVIEW_VIEW = 0  # the same training view in every preview, so they can be compared over time


# Helper functions, huber and sample_images_at_mc_locs come from the baseline
def huber(x, y, scaling=HUBER_SCALING):
    """Smooth L1 (Huber) loss between the rendered and the real colours or silhouettes."""
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss


def sample_images_at_mc_locs(target_images, sampled_rays_xy, ndc_extent=(1.0, 1.0)):
    """Reads the real pixel colours at the random ray positions used in training."""
    # ndc_extent is our addition for non-square images, square images give the baseline behaviour
    ba = target_images.shape[0]
    dim = target_images.shape[-1]
    spatial_size = sampled_rays_xy.shape[1:-1]
    extent = torch.tensor(ndc_extent, device=sampled_rays_xy.device)
    # The sign is flipped because PyTorch3D and grid_sample use opposite x and y directions
    grid = -(sampled_rays_xy / extent)
    images_sampled = torch.nn.functional.grid_sample(
        target_images.permute(0, 3, 1, 2),
        grid.view(ba, -1, 1, 2),
        align_corners=True,
    )
    return images_sampled.permute(0, 2, 3, 1).view(ba, *spatial_size, dim)


def ndc_extent_for(height, width):
    """Returns how far the NDC coordinates reach in x and y for this image size."""
    # In PyTorch3D the shorter side reaches 1 and the longer side reaches long / short
    if width >= height:
        return (width / height, 1.0)
    return (1.0, height / width)


def fix_fov_for_aspect(cameras, height, width):
    """Converts the field of view so portrait images such as poster are rendered correctly."""
    # PyTorch3D applies the field of view to the shorter side, but load_nerf_data gives the vertical one
    # Square and landscape images such as lego need no change
    if width >= height:
        return cameras
    # Converts the vertical field of view into the horizontal one
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
    """Builds a camera batch from the given indices, the same way as the baseline."""
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
    """Loads the training images, silhouettes and cameras of the active dataset."""
    images, silhouettes, cameras = load_nerf_data(
        str(DATA_DIR),
        split="train",
        device=device,
        znear=CAMERA_ZNEAR,
        zfar=CAMERA_ZFAR,
        resize_to=RESIZE_TO,
    )
    # Bicubic resizing can give values slightly below 0 or above 1
    images = images.clamp(0.0, 1.0)
    silhouettes = silhouettes.clamp(0.0, 1.0)

    # Poster has no test split, so every 8th image (TEST_EVERY) is held out here
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
    """Saves a full render of one training view next to the ground truth and the loss curves."""
    with torch.no_grad():
        rendered, _ = renderer_grid(
            cameras=camera, volumetric_function=model.batched_forward
        )
    rendered_image = rendered[0, ..., :3].clamp(0.0, 1.0)
    target_image = target_image.clamp(0.0, 1.0)
    # Quick PSNR for the preview title
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
    """Saves the model weights together with the dataset name, so evaluate.py can check them."""
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
    """Saves the losses with one row per iteration"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "color_loss", "silhouette_loss", "total_loss"])
        for i, (c, t) in enumerate(zip(color, total)):
            s = silhouette[i] if i < len(silhouette) else ""  # empty for poster, which has no silhouette loss
            writer.writerow([i, c, s, t])


# Training
def main(
    n_iter=N_ITERATIONS,
    output_dir=OUTPUT_DIR,
    vis_every=VISUALIZE_EVERY,
    checkpoint_every=CHECKPOINT_EVERY,
):
    """Trains on the active dataset and returns the loss histories."""
    # The arguments come from config.yaml, only smoke_test.py passes other values
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(RANDOM_SEED)

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    # Saves a copy of the settings this run used, the copy is never read as a config
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

    # Random rays for each training step
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

    # One ray per pixel for the full image previews
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
    lr_decay_iteration = round(n_iter * LR_DECAY_AT)  # iteration where the learning rate drops

    loss_history_color, loss_history_sil, loss_history_total = [], [], []

    for iteration in tqdm(range(n_iter), desc="Training"):
        if iteration == lr_decay_iteration:
            # Same as the baseline, a new optimizer with a lower learning rate
            tqdm.write(f"Decreasing LR by factor {LR_DECAY_FACTOR} ...")
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * LR_DECAY_FACTOR)

        optimizer.zero_grad()
        # Picks BATCH_SIZE random training images
        batch_idx = torch.randperm(n_images)[:BATCH_SIZE]
        batch_cameras = select_cameras(target_cameras, batch_idx)

        rendered, sampled_rays = renderer_mc(
            cameras=batch_cameras, volumetric_function=model
        )
        # The render holds three colour channels and one opacity channel
        rendered_images, rendered_silhouettes = rendered.split([3, 1], dim=-1)

        # Real colours at the same pixels as the rendered rays
        colors_at_rays = sample_images_at_mc_locs(
            target_images[batch_idx], sampled_rays.xys, (x_extent, y_extent)
        )
        color_err = huber(rendered_images, colors_at_rays).abs().mean()
        loss = color_err

        # Only lego has an alpha channel for the silhouette loss
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

        # Saves a preview regularly and at the last iteration
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

    # Final checkpoint and loss history for the report
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