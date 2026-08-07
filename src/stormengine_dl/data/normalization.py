"""Train-only variable normalization for ERA5 inputs and targets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr

from .era5_dataset import ACCUM_VARIABLES, INSTANT_VARIABLES, convert_era5_units


@dataclass(frozen=True)
class VariableStat:
    mean: float
    std: float
    count: int


class NormalizationStats:
    def __init__(self, variables: dict[str, VariableStat], metadata: dict[str, object] | None = None):
        self.variables = variables
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        variables = {
            name: VariableStat(float(values["mean"]), float(values["std"]), int(values["count"]))
            for name, values in payload["variables"].items()
        }
        return cls(variables, {key: value for key, value in payload.items() if key != "variables"})

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.metadata)
        payload["variables"] = {
            name: {"mean": stat.mean, "std": stat.std, "count": stat.count}
            for name, stat in sorted(self.variables.items())
        }
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def normalize(self, variable: str, values: np.ndarray) -> np.ndarray:
        stat = self.variables[variable]
        return (values - stat.mean) / stat.std

    def denormalize(self, variable: str, values: np.ndarray) -> np.ndarray:
        stat = self.variables[variable]
        return values * stat.std + stat.mean


def fit_era5_normalization(
    manifest_path: str | Path,
    data_root: str | Path,
    years: Iterable[int],
    variables: Sequence[str],
) -> NormalizationStats:
    """Fit full-grid statistics using only valid months in the requested years."""
    selected_years = set(int(year) for year in years)
    requested = tuple(dict.fromkeys(variables))
    unknown = set(requested) - INSTANT_VARIABLES - ACCUM_VARIABLES
    if unknown:
        raise ValueError(f"unsupported ERA5 variables: {sorted(unknown)}")

    totals = {name: [0, 0.0, 0.0] for name in requested}
    months: list[str] = []
    root = Path(data_root).expanduser().resolve()
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        year = int(row["year"])
        if row["valid"].strip().lower() != "true" or year not in selected_years:
            continue
        months.append(f"{year}-{int(row['month']):02d}")
        paths = {
            "instant": root / row["instant_path"],
            "accum": root / row["accum_path"],
        }
        with xr.open_dataset(paths["instant"]) as instant, xr.open_dataset(paths["accum"]) as accum:
            for name in requested:
                source = instant if name in INSTANT_VARIABLES else accum
                values = convert_era5_units(name, np.asarray(source[name].values)).astype(
                    np.float64, copy=False
                )
                finite = values[np.isfinite(values)]
                totals[name][0] += int(finite.size)
                totals[name][1] += float(finite.sum(dtype=np.float64))
                totals[name][2] += float(np.square(finite).sum(dtype=np.float64))

    if not months:
        raise ValueError("no valid manifest months match the normalization years")
    fitted: dict[str, VariableStat] = {}
    for name, (count, total, square_total) in totals.items():
        if count == 0:
            raise ValueError(f"no finite values found for {name}")
        mean = total / count
        variance = max(square_total / count - mean * mean, 0.0)
        fitted[name] = VariableStat(mean, max(variance**0.5, 1e-8), count)
    return NormalizationStats(
        fitted,
        {
            "fit_scope": "ERA5 full grid; training years only",
            "fit_years": sorted(selected_years),
            "fit_months": months,
            "unit_conversion": "stormengine_dl.data.era5_dataset.convert_era5_units",
        },
    )
