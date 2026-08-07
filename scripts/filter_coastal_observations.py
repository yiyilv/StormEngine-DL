#!/usr/bin/env python3
"""Reapply the project coastal polygon to measurement-level CSV coordinates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stormengine_dl.data.coastal_filter import (
    COASTAL_FILTER_VERSION,
    DEFAULT_COASTAL_BUFFER_KM,
    is_in_adriatic_coastal_area,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lat-column", default="lat")
    parser.add_argument("--lon-column", default="lon")
    parser.add_argument("--buffer-km", type=float, default=DEFAULT_COASTAL_BUFFER_KM)
    args = parser.parse_args()

    input_path, output_path = Path(args.input), Path(args.output)
    kept: list[dict[str, str]] = []
    rejected = 0
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("measurement CSV has no header")
        for row in reader:
            try:
                accepted = is_in_adriatic_coastal_area(
                    float(row[args.lat_column]),
                    float(row[args.lon_column]),
                    buffer_km=args.buffer_km,
                )
            except (KeyError, TypeError, ValueError):
                accepted = False
            if accepted:
                kept.append(row)
            else:
                rejected += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept)
    print(f"Coastal filter: {COASTAL_FILTER_VERSION}; buffer={args.buffer_km:g} km")
    print(f"Measurements kept: {len(kept)}; rejected: {rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
