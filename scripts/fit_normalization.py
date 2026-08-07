#!/usr/bin/env python3
"""Fit variable normalization on the configured training years only."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from stormengine_dl.data import fit_era5_normalization


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(_resolve(root, args.config).read_text(encoding="utf-8"))
    data = config["data"]
    variables = list(dict.fromkeys(data["input_variables"] + data["target_variables"]))
    stats = fit_era5_normalization(
        _resolve(root, data["era5_manifest"]),
        _resolve(root, data["era5_root"]),
        data["train_years"],
        variables,
    )
    output = _resolve(root, data["normalization_stats"])
    stats.save(output)
    print(f"Normalization statistics: {output}")
    for name, stat in sorted(stats.variables.items()):
        print(f"  {name:6s} mean={stat.mean:12.6f} std={stat.std:12.6f} n={stat.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
