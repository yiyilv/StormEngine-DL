"""Data discovery, validation, and training datasets."""

from .cached_dataset import CachedEra5SequenceDataset
from .era5_dataset import Era5SequenceDataset, convert_era5_units
from .manifest import Era5MonthFiles, build_manifest, scan_month_files, validate_month
from .normalization import NormalizationStats, VariableStat, fit_era5_normalization
from .static_fields import StaticFields, build_station_distance_field
from .station_registry import StationRecord, build_station_registry, load_station_coordinates

__all__ = [
    "Era5MonthFiles",
    "CachedEra5SequenceDataset",
    "Era5SequenceDataset",
    "StationRecord",
    "StaticFields",
    "NormalizationStats",
    "VariableStat",
    "build_manifest",
    "build_station_registry",
    "convert_era5_units",
    "fit_era5_normalization",
    "build_station_distance_field",
    "scan_month_files",
    "load_station_coordinates",
    "validate_month",
]
