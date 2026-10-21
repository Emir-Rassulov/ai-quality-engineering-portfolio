"""Makes phase3-ml and phase2-api-data importable for pytest."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase2-api-data"))
