"""Project-wide paths and settings.  Import from notebooks: ``from config import *``."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PAIRS_PATH = PROCESSED_DIR / "pairs.parquet"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

SEED = 0

# One entry per dataset built from configs/.  A notebook sets SAMPLE = "main" (or another key)
# at the top and takes every input and output location from sample_paths(SAMPLE), so results
# from different samples never overwrite each other.
SAMPLES = {
    "main": {"config": "main_2016_2020.yaml", "processed": "processed"},                  # all quoted options
    "traded_only": {"config": "traded_only_2016_2020.yaml", "processed": "processed_traded_only"},  # volume > 0
}


def sample_paths(sample: str = "main") -> dict:
    """Input and output locations for one sample."""
    if sample not in SAMPLES:
        raise KeyError(f"unknown sample {sample!r}; choose from {list(SAMPLES)}")
    processed = ROOT / "data" / SAMPLES[sample]["processed"]
    return {
        "config": ROOT / "configs" / SAMPLES[sample]["config"],
        "processed": processed,
        "pairs": processed / "pairs.parquet",
        "predictions": processed / "predictions",     # git-ignored, one small file per model
        "tables": TABLES_DIR / sample,                 # committed
        "figures": FIGURES_DIR / sample,               # committed
    }
