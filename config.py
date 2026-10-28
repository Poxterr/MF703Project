"""Project-wide paths and settings.  Import from notebooks: ``from config import *``."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PAIRS_PATH = PROCESSED_DIR / "pairs.parquet"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

SEED = 0
