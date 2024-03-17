from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(Path(__file__).resolve().parent)

MODELS_DIR = ROOT_DIR / "../models"