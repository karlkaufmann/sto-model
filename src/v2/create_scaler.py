#!/usr/bin/env python3
"""
Legacy helper kept for backward compatibility.

V2 training writes a real fitted StandardScaler to `scaler.pkl` directly from
`model3.py`. This script intentionally does not overwrite it.
"""

from pathlib import Path

scaler_path = Path("scaler.pkl")
if scaler_path.exists():
    print("scaler.pkl already exists. Keep the fitted scaler produced by model3.py.")
else:
    print("No scaler.pkl found. Run `./venv/bin/python model3.py` to generate artifacts.")
