#!/usr/bin/env python3
"""Build the versioned StormEngine station registry from audited source catalogs."""

from __future__ import annotations

import argparse
from collections import Counter

from stormengine_dl.data import build_station_registry
from stormengine_dl.data.coastal_filter import DEFAULT_COASTAL_BUFFER_KM


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpc")
    parser.add_argument("--official-catalog")
    parser.add_argument("--virtual", required=True)
    parser.add_argument("--legacy-coastal", nargs="*", default=[])
    parser.add_argument("--meteohub-json", nargs="*", default=[])
    parser.add_argument("--meteohub-network", action="append", default=[])
    parser.add_argument("--coastal-buffer-km", type=float, default=DEFAULT_COASTAL_BUFFER_KM)
    parser.add_argument("--no-coastal-filter", action="store_true")
    parser.add_argument("--output", default="data/stations_registry.csv")
    args = parser.parse_args()
    if not args.dpc and not args.official_catalog:
        parser.error("one of --official-catalog or --dpc is required")
    records = build_station_registry(
        args.dpc,
        args.virtual,
        args.output,
        legacy_coastal_paths=args.legacy_coastal,
        meteohub_json_paths=args.meteohub_json,
        meteohub_networks=args.meteohub_network,
        official_catalog_path=args.official_catalog,
        coastal_buffer_km=None if args.no_coastal_filter else args.coastal_buffer_km,
    )
    counts = Counter(record.station_type for record in records)
    print(f"Station records: {len(records)}")
    for station_type, count in sorted(counts.items()):
        print(f"  {station_type}: {count}")
    print(f"Enabled dpc_plus_sea: {sum(record.profile_dpc_plus_sea for record in records)}")
    print(f"Registry: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
