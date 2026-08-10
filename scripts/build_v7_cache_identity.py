#!/usr/bin/env python3
"""Bind the immutable V6 cache arrays to the exact station registry used by V7."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stormengine_dl.data import build_cache_identity  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--registry", default=ROOT / "data" / "stations_registry.csv", type=Path)
    parser.add_argument(
        "--output",
        default=ROOT / "data" / "manifests" / "v7_cache_identity_2010_2017.json",
        type=Path,
    )
    args = parser.parse_args()
    identity = build_cache_identity(args.cache, args.registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
