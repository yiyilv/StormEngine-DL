"""StormEngine-DL package with lazy public imports."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "StormEngineForecastModel",
    "StormEngineReconstructionModel",
    "StormEngineV7ForecastModel",
]

_EXPORTS = {
    "StormEngineForecastModel": (".models.system", "StormEngineForecastModel"),
    "StormEngineReconstructionModel": (".models.reconstruction", "StormEngineReconstructionModel"),
    "StormEngineV7ForecastModel": (".models.mask_aware", "StormEngineV7ForecastModel"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
