"""Reference forecasts for evaluating sparse-to-grid forecast skill."""

from __future__ import annotations

import numpy as np
import torch


def geographic_station_coordinates(
    normalized_coordinates: np.ndarray,
    grid_latitudes: np.ndarray,
    grid_longitudes: np.ndarray,
) -> np.ndarray:
    """Convert normalized ``(latitude, longitude)`` coordinates back to degrees."""
    coordinates = np.asarray(normalized_coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("normalized_coordinates must have shape [N, 2]")
    latitudes = np.asarray(grid_latitudes, dtype=np.float64)
    longitudes = np.asarray(grid_longitudes, dtype=np.float64)
    latitude = latitudes.min() + coordinates[:, 0] * (latitudes.max() - latitudes.min())
    longitude = longitudes.min() + coordinates[:, 1] * (
        longitudes.max() - longitudes.min()
    )
    return np.stack((latitude, longitude), axis=-1)


def build_idw_neighbors(
    station_coordinates: np.ndarray,
    grid_latitudes: np.ndarray,
    grid_longitudes: np.ndarray,
    *,
    neighbors: int = 8,
    power: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute metric-aware inverse-distance neighbors for every grid cell."""
    stations = np.asarray(station_coordinates, dtype=np.float64)
    latitudes = np.asarray(grid_latitudes, dtype=np.float64)
    longitudes = np.asarray(grid_longitudes, dtype=np.float64)
    if stations.ndim != 2 or stations.shape[1] != 2 or stations.shape[0] == 0:
        raise ValueError("station_coordinates must have shape [N, 2]")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    if power <= 0:
        raise ValueError("power must be positive")

    count = min(int(neighbors), stations.shape[0])
    grid_latitude, grid_longitude = np.meshgrid(latitudes, longitudes, indexing="ij")
    mean_latitude = float(latitudes.mean())
    longitude_scale = np.cos(np.deg2rad(mean_latitude))
    grid = np.stack((grid_latitude.ravel(), grid_longitude.ravel() * longitude_scale), axis=-1)
    projected_stations = np.stack(
        (stations[:, 0], stations[:, 1] * longitude_scale), axis=-1
    )
    distances = 111.32 * np.sqrt(
        np.square(grid[:, None, :] - projected_stations[None, :, :]).sum(axis=-1)
    )
    indices = np.argpartition(distances, count - 1, axis=1)[:, :count]
    selected = np.take_along_axis(distances, indices, axis=1)
    order = np.argsort(selected, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    selected = np.take_along_axis(selected, order, axis=1)

    exact = selected <= 1e-3
    has_exact = exact.any(axis=1, keepdims=True)
    inverse = 1.0 / np.maximum(selected, 1e-3) ** float(power)
    weights = np.where(has_exact, exact.astype(np.float64), inverse)
    weights /= weights.sum(axis=1, keepdims=True)
    return torch.from_numpy(indices.astype(np.int64)), torch.from_numpy(weights.astype(np.float32))


def idw_interpolate(
    station_values: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_weights: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Interpolate ``[B, N, C]`` station values into ``[B, C, H, W]`` grids."""
    if station_values.ndim != 3:
        raise ValueError("station_values must have shape [B, N, C]")
    if neighbor_indices.shape != neighbor_weights.shape:
        raise ValueError("neighbor indices and weights must have matching shapes")
    if neighbor_indices.shape[0] != height * width:
        raise ValueError("neighbor grid size does not match height and width")
    indices = neighbor_indices.to(station_values.device)
    weights = neighbor_weights.to(device=station_values.device, dtype=station_values.dtype)
    selected = station_values[:, indices, :]
    interpolated = (selected * weights[None, :, :, None]).sum(dim=2)
    return interpolated.permute(0, 2, 1).reshape(
        station_values.shape[0], station_values.shape[2], height, width
    )


def dense_grid_persistence(current_grid: torch.Tensor, forecast_hours: int) -> torch.Tensor:
    """Repeat the last observed dense grid for every forecast lead."""
    if current_grid.ndim != 4 or forecast_hours < 1:
        raise ValueError("current_grid must be [B, C, H, W] and forecast_hours positive")
    return current_grid[:, None].expand(-1, forecast_hours, -1, -1, -1).clone()


def sparse_idw_persistence(
    current_points: torch.Tensor,
    input_variables: list[str],
    target_variables: list[str],
    neighbor_indices: torch.Tensor,
    neighbor_weights: torch.Tensor,
    height: int,
    width: int,
    forecast_hours: int,
) -> torch.Tensor:
    """IDW the last sparse observations and hold the resulting grid constant.

    Variables present in both input and target are interpolated. Hourly accumulated
    precipitation is set to zero when ``tp`` is absent from the model input, which is
    an explicit no-rain baseline rather than hidden access to a dense precipitation grid.
    """
    if current_points.shape[-1] != len(input_variables):
        raise ValueError("current_points channel count does not match input_variables")
    common = [name for name in target_variables if name in input_variables]
    common_indices = [input_variables.index(name) for name in common]
    common_grids = idw_interpolate(
        current_points[:, :, common_indices],
        neighbor_indices,
        neighbor_weights,
        height,
        width,
    )
    output = torch.zeros(
        (current_points.shape[0], len(target_variables), height, width),
        dtype=current_points.dtype,
        device=current_points.device,
    )
    for common_channel, name in enumerate(common):
        output[:, target_variables.index(name)] = common_grids[:, common_channel]
    unsupported = [name for name in target_variables if name not in common and name != "tp"]
    if unsupported:
        raise ValueError(f"no sparse persistence rule for target variables: {unsupported}")
    return dense_grid_persistence(output, forecast_hours)


def rmse_skill_scores(
    model_metrics: dict[str, object], baseline_metrics: dict[str, object]
) -> dict[str, object]:
    """Return ``1 - model_RMSE / baseline_RMSE`` with the metric hierarchy preserved."""

    def compare_regions(
        model_regions: dict[str, object], baseline_regions: dict[str, object]
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for region, model_variables_object in model_regions.items():
            model_variables = dict(model_variables_object)
            baseline_variables = dict(baseline_regions[region])
            result[region] = {}
            for variable, model_values_object in model_variables.items():
                model_rmse = float(dict(model_values_object)["rmse"])
                baseline_rmse = float(dict(baseline_variables[variable])["rmse"])
                skill = None if baseline_rmse == 0.0 else 1.0 - model_rmse / baseline_rmse
                result[region][variable] = {
                    "v6_rmse": model_rmse,
                    "baseline_rmse": baseline_rmse,
                    "skill": skill,
                }
        return result

    model_by_lead = dict(model_metrics["by_lead_hour"])
    baseline_by_lead = dict(baseline_metrics["by_lead_hour"])
    return {
        "definition": "1 - v6_rmse / baseline_rmse; positive means V6 is better",
        "aggregate": compare_regions(
            dict(model_metrics["aggregate"]), dict(baseline_metrics["aggregate"])
        ),
        "by_lead_hour": {
            lead: compare_regions(dict(regions), dict(baseline_by_lead[lead]))
            for lead, regions in model_by_lead.items()
        },
    }
