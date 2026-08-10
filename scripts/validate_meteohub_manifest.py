#!/usr/bin/env python3
"""Validate external MeteoHub files against their committed manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stormengine_dl.data.meteohub_manifest import validate_meteohub_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifests" / "meteohub_20260801_20260808.json",
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument(
        "--qc-only",
        action="store_true",
        help="Require only files whose manifest entry is quality-controlled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = validate_meteohub_manifest(
        args.manifest,
        args.raw_root,
        qc_only=args.qc_only,
    )
    for result in results:
        status = "OK" if result.ok else "FAIL:" + ",".join(result.problems)
        print(
            f"{status:16} {result.logical_name:32} {result.filename} "
            f"bytes={result.actual_bytes}/{result.expected_bytes} "
            f"records={result.actual_records}/{result.expected_records}"
        )
    passed = sum(result.ok for result in results)
    print(f"Validated {passed}/{len(results)} selected files")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
