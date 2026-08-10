#!/usr/bin/env python3
"""Audit legacy or newly downloaded MeteoHub JSON Lines observations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stormengine_dl.data.official_observations import (  # noqa: E402
    audit_observations,
    audit_observations_sqlite,
    iter_meteohub_observations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", type=Path, help="MeteoHub JSON Lines exports")
    parser.add_argument("--registry", type=Path, default=ROOT / "data" / "stations_registry.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-observations", action="store_true")
    parser.add_argument(
        "--streaming-database",
        type=Path,
        help="Use this new SQLite file for memory-bounded exact deduplication.",
    )
    return parser.parse_args()


def selected_station_registry(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row["station_id"].removeprefix("LAND::"): row
            for row in csv.DictReader(handle)
            if row["station_type"] == "physical_land"
            and row["enabled"].lower() == "true"
            and row["profile_dpc_plus_sea"].lower() == "true"
        }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    registry = selected_station_registry(args.registry)
    selected = set(registry)
    if args.streaming_database is not None:
        if args.write_observations:
            raise ValueError("--write-observations is not supported with --streaming-database")
        report = audit_observations_sqlite(
            iter_meteohub_observations(args.jsonl),
            args.streaming_database,
            selected,
        )
    else:
        report = audit_observations(
            iter_meteohub_observations(args.jsonl),
            selected,
            include_observations=args.write_observations,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("variable_summary", "station_variable_summary", "station_summary", "timerange_summary"):
        write_csv(args.output_dir / f"{name}.csv", report[name])  # type: ignore[arg-type]
    observed = {row["station_id"]: row for row in report["station_summary"]}  # type: ignore[union-attr]
    coverage_rows = []
    for station_id, registered in sorted(registry.items(), key=lambda item: (item[1]["network"], item[1]["station_name"])):
        sample = observed.get(station_id)
        coverage_rows.append({
            "station_id": registered["station_id"],
            "station_name": registered["station_name"],
            "network": registered["network"],
            "latitude": registered["latitude"],
            "longitude": registered["longitude"],
            "observed_in_sample": sample is not None,
            "observed_variable_count": sample["variable_count"] if sample else 0,
            "observed_variables": sample["variables"] if sample else "",
        })
    write_csv(args.output_dir / "project_station_coverage.csv", coverage_rows)
    write_csv(
        args.output_dir / "verified_stations_candidate.csv",
        [row for row in report["station_summary"] if row["selected_for_project"]],  # type: ignore[union-attr]
    )
    summary = report["summary"]
    summary["registered_project_station_count"] = len(registry)  # type: ignore[index]
    summary["registered_observed_by_network"] = dict(sorted(Counter(  # type: ignore[index]
        row["network"] for row in coverage_rows if row["observed_in_sample"]
    ).items()))
    summary["registered_total_by_network"] = dict(sorted(Counter(  # type: ignore[index]
        row["network"] for row in coverage_rows
    ).items()))
    if args.write_observations:
        write_csv(args.output_dir / "normalized_observations.csv", report["observations"])  # type: ignore[arg-type]
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
