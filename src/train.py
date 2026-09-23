from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import train_and_evaluate

if __name__ == "__main__":
    report = train_and_evaluate()
    print("Training completed")
    print(report)
