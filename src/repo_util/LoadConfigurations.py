# Imports
from pathlib import Path
import yaml


# Configuration file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


# Retrieving configurations as constants

# Preview
PREVIEW_MESSAGE = CONFIG["preview"]["the-preview"]

# Active dataset (lego | poster)
ACTIVE_DATASET = CONFIG["active_dataset"]
_DATASET = CONFIG["datasets"][ACTIVE_DATASET]

# Model (src/model.py)
N_HARMONIC_FUNCTIONS = CONFIG["model"]["n_harmonic_functions"]
HARMONIC_OMEGA0 = CONFIG["model"]["omega0"]
N_HIDDEN_NEURONS = CONFIG["model"]["n_hidden_neurons"]

# Rendering
N_RAYS_PER_IMAGE = CONFIG["render"]["n_rays_per_image"]
N_PTS_PER_RAY = CONFIG["render"]["n_pts_per_ray"]
N_BATCHES_FULL_RENDER = CONFIG["render"]["n_batches_full_render"]

# Training
N_ITERATIONS = CONFIG["train"]["n_iter"]
BATCH_SIZE = CONFIG["train"]["batch_size"]
LEARNING_RATE = CONFIG["train"]["learning_rate"]
LR_DECAY_AT = CONFIG["train"]["lr_decay_at"]
LR_DECAY_FACTOR = CONFIG["train"]["lr_decay_factor"]
HUBER_SCALING = CONFIG["train"]["huber_scaling"]
RANDOM_SEED = CONFIG["train"]["seed"]
VISUALIZE_EVERY = CONFIG["train"]["vis_every"]
CHECKPOINT_EVERY = CONFIG["train"]["checkpoint_every"]

# Dataset-specific (values of the active dataset)
DATA_DIR = PROJECT_ROOT / _DATASET["data_dir"]
OUTPUT_DIR = PROJECT_ROOT / _DATASET["output_dir"]
RESIZE_TO = tuple(_DATASET["resize_to"]) if _DATASET["resize_to"] else None
CAMERA_ZNEAR = _DATASET["znear"]
CAMERA_ZFAR = _DATASET["zfar"]
RAY_MIN_DEPTH = _DATASET["min_depth"]
RAY_MAX_DEPTH = _DATASET["max_depth"]
USE_SILHOUETTE_LOSS = _DATASET["use_silhouette_loss"]
TEST_EVERY = _DATASET["test_every"]


if __name__ == "__main__":
    print(PREVIEW_MESSAGE)
    print(f"Active dataset: {ACTIVE_DATASET}")
    print(f"Data dir:       {DATA_DIR}")
    print(f"Output dir:     {OUTPUT_DIR}")