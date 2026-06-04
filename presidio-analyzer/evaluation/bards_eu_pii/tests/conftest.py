"""Put the evaluation harness dir on sys.path so ``import metrics`` works."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
