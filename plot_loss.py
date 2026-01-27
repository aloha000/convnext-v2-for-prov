#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot loss curve from JSON Lines (JSONL) logs.

Usage:
  python plot_loss.py --log path/to/log.jsonl --out loss_curve.png --csv parsed_logs.csv

The log file should contain one JSON object per line, e.g.:
{"train_lr": 3.3e-06, "train_loss": 3.22, "epoch": 0}
{"train_lr": 1.08e-05, "train_loss": 3.18, "epoch": 1}
...
"""
import argparse
import json
import sys
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError as e:
                print(f"[warn] skip line {i}: {e}", file=sys.stderr)
    if not records:
        raise ValueError("No valid JSON lines found in the log.")
    return records

def main():
    parser = argparse.ArgumentParser(description="Plot training loss from JSONL logs.")
    parser.add_argument("--log", required=True, help="Path to JSONL log file.")
    parser.add_argument("--out", default="loss_curve.png", help="Output PNG path for the loss curve.")
    parser.add_argument("--csv", default="training_logs_parsed.csv", help="Output CSV path for parsed logs.")
    parser.add_argument("--loss-key", default="train_loss", help="JSON key for the loss value (default: train_loss).")
    parser.add_argument("--epoch-key", default="epoch", help="JSON key for the epoch index (default: epoch).")
    args = parser.parse_args()

    # Parse logs
    records = read_jsonl(args.log)
    df = pd.DataFrame(records)

    # Fallbacks if keys are missing
    if args.loss_key not in df.columns:
        # Try common alternatives
        for k in ["loss", "train/loss", "training_loss", "val_loss", "valid_loss"]:
            if k in df.columns:
                args.loss_key = k
                break
        else:
            raise KeyError(f"Loss key '{args.loss_key}' not found and no common alternatives present in columns: {df.columns.tolist()}")

    if args.epoch_key not in df.columns:
        # Create a 0..N-1 epoch index if absent
        df[args.epoch_key] = range(len(df))

    # Keep only necessary columns and sort by epoch
    keep_cols = [args.epoch_key, args.loss_key] + [c for c in ("train_lr", "n_parameters") if c in df.columns]
    df = df[keep_cols].sort_values(args.epoch_key).reset_index(drop=True)

    # Save CSV
    df.to_csv(args.csv, index=False)

    # Plot loss curve (single chart as required)
    plt.figure()
    plt.plot(df[args.epoch_key], df[args.loss_key], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(args.loss_key)
    plt.title("Loss vs. Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)

    print(f"[ok] Saved loss curve to: {args.out}")
    print(f"[ok] Saved parsed CSV to: {args.csv}")

if __name__ == "__main__":
    main()
