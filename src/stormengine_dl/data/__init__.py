"""Data discovery, validation, and training datasets with lazy imports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "Era5MonthFiles",
    "CachedEra5SequenceDataset",
    "CachedReconstructionDataset",
    "Era5SequenceDataset",
    "StationRecord",
    "StaticFields",
    "NormalizationStats",
    "OperationalTensorBatch",
    "VariableStat",
    "build_manifest",
    "build_operational_tensors",
    "build_station_registry",
    "convert_era5_units",
    "fit_era5_normalization",
    "build_station_distance_field",
    "scan_month_files",
    "load_station_coordinates",
    "load_fixed_registry",
    "meteorological_wind_to_uv",
    "validate_month",
]

_EXPORTS = {
    "CachedEra5SequenceDataset": (".cached_dataset", "CachedEra5SequenceDataset"),
    "CachedReconstructionDataset": (".reconstruction_dataset", "CachedReconstructionDataset"),
    "Era5MonthFiles": (".manifest", "Era5MonthFiles"),
    "Era5SequenceDataset": (".era5_dataset", "Era5SequenceDataset"),
    "NormalizationStats": (".normalization", "NormalizationStats"),
    "OperationalTensorBatch": (".operational_adapter", "OperationalTensorBatch"),
    "StationRecord": (".station_registry", "StationRecord"),
    "StaticFields": (".static_fields", "StaticFields"),
    "VariableStat": (".normalization", "VariableStat"),
    "build_manifest": (".manifest", "build_manifest"),
    "build_operational_tensors": (".operational_adapter", "build_operational_tensors"),
    "build_station_distance_field": (".static_fields", "build_station_distance_field"),
    "build_station_registry": (".station_registry", "build_station_registry"),
    "convert_era5_units": (".era5_dataset", "convert_era5_units"),
    "fit_era5_normalization": (".normalization", "fit_era5_normalization"),
    "load_station_coordinates": (".station_registry", "load_station_coordinates"),
    "load_fixed_registry": (".operational_adapter", "load_fixed_registry"),
    "meteorological_wind_to_uv": (
        ".operational_adapter",
        "meteorological_wind_to_uv",
    ),
    "scan_month_files": (".manifest", "scan_month_files"),
    "validate_month": (".manifest", "validate_month"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
