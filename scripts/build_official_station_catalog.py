#!/usr/bin/env python3
"""Build a compact official station snapshot from downloaded API responses."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from stormengine_dl.data.official_station_catalog import (
    collect_abruzzo_stations,
    collect_meteohub_stations,
    load_json,
    write_official_station_catalog,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meteohub-json", nargs="+", required=True)
    parser.add_argument("--abruzzo-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    snapshots = [(Path(path).stem, load_json(path)) for path in args.meteohub_json]
    stations = collect_meteohub_stations(snapshots)
    if args.abruzzo_json:
        stations.extend(collect_abruzzo_stations(load_json(args.abruzzo_json)))
    records = write_official_station_catalog(stations, args.output)
    counts = Counter(record.network for record in records)
    print(f"Official station coordinates: {len(records)}")
    for network, count in sorted(counts.items()):
        print(f"  {network}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
