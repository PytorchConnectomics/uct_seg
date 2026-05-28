"""Add repo root to sys.path so demos work without `pip install -e .`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
