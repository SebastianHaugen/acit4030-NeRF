# Imports
from pathlib import Path
import yaml


# Configuration file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


# Preview configuration
PREVIEW_MESSAGE = CONFIG["preview"]["the-preview"]

















if __name__ == "__main__":
    print(PREVIEW_MESSAGE)