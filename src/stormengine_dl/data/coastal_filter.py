"""Shared Adriatic coastal geometry for station and measurement filtering."""

from __future__ import annotations

from functools import lru_cache

from shapely.affinity import scale
from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep


DEFAULT_COASTAL_BUFFER_KM = 20.0
COASTAL_FILTER_VERSION = "italian-adriatic-coastline-v1"
VIRTUAL_SUPPORT_FILTER_VERSION = "adriatic-bilateral-support-v1"

# Virtual coordinates are support nodes rather than observing stations.  The
# model target covers the full Adriatic, so the support set includes the
# eastern shore as well as points over the sea.  The southern Albanian coast,
# Greece, and the explicitly Ionian group lie beyond the Adriatic boundary at
# the Strait of Otranto and are excluded.
_ADRIATIC_VIRTUAL_GROUPS = frozenset(
    {
        "VIRTUAL_ADRIATIC_SEA",
        "VIRTUAL_SLOVENIA",
        "VIRTUAL_CROATIA",
        "VIRTUAL_MONTENEGRO",
    }
)
_ALBANIA_ADRIATIC_MIN_LAT = 40.45

# Approximate Italian Adriatic shoreline, north to south. Coordinates are
# (longitude, latitude). The line is deliberately limited to the Italian side;
# regional/DPC catalog membership is checked separately.
ITALIAN_ADRIATIC_COASTLINE = (
    (13.77, 45.65),  # Trieste
    (13.39, 45.68),  # Grado
    (13.10, 45.68),  # Lignano
    (12.88, 45.60),  # Caorle
    (12.33, 45.44),  # Venezia
    (12.30, 45.22),  # Chioggia
    (12.48, 44.95),  # Po delta
    (12.28, 44.42),  # Ravenna
    (12.40, 44.20),  # Cesenatico
    (12.57, 44.06),  # Rimini
    (12.91, 43.91),  # Pesaro
    (13.51, 43.62),  # Ancona
    (13.89, 42.95),  # San Benedetto del Tronto
    (14.22, 42.47),  # Pescara
    (14.71, 42.11),  # Vasto
    (14.99, 42.00),  # Termoli
    (15.75, 41.93),  # Gargano north coast
    (16.18, 41.88),  # Vieste
    (15.92, 41.63),  # Manfredonia
    (16.28, 41.32),  # Barletta
    (16.87, 41.13),  # Bari
    (17.94, 40.64),  # Brindisi
    (18.49, 40.14),  # Otranto
    (18.36, 39.80),  # Santa Maria di Leuca
)

# A local equirectangular projection is sufficient for constructing this
# deliberately approximate corridor. It avoids treating one longitude degree
# as the same distance as one latitude degree.
_KM_PER_DEGREE_LAT = 111.32
_KM_PER_DEGREE_LON_AT_43N = 81.43


@lru_cache(maxsize=8)
def _projected_coastline() -> LineString:
    return LineString(
        [
            (lon * _KM_PER_DEGREE_LON_AT_43N, lat * _KM_PER_DEGREE_LAT)
            for lon, lat in ITALIAN_ADRIATIC_COASTLINE
        ]
    )


@lru_cache(maxsize=8)
def adriatic_coastal_polygon(buffer_km: float = DEFAULT_COASTAL_BUFFER_KM) -> Polygon:
    """Return a Shapely polygon around the Italian Adriatic shoreline."""
    if buffer_km <= 0:
        raise ValueError("coastal buffer must be positive")
    corridor_km = _projected_coastline().buffer(
        buffer_km, cap_style="round", join_style="round"
    )
    return scale(
        corridor_km,
        xfact=1.0 / _KM_PER_DEGREE_LON_AT_43N,
        yfact=1.0 / _KM_PER_DEGREE_LAT,
        origin=(0.0, 0.0),
    )


@lru_cache(maxsize=8)
def _prepared_coastal_polygon(buffer_km: float):
    return prep(adriatic_coastal_polygon(buffer_km))


def is_in_adriatic_coastal_area(
    lat: float,
    lon: float,
    *,
    buffer_km: float = DEFAULT_COASTAL_BUFFER_KM,
) -> bool:
    """Apply the shared point-in-polygon test used at both ingestion stages."""
    return bool(_prepared_coastal_polygon(buffer_km).covers(Point(float(lon), float(lat))))


def distance_to_adriatic_coast_km(lat: float, lon: float) -> float:
    """Return approximate distance to the shared Italian Adriatic shoreline."""
    point = Point(float(lon) * _KM_PER_DEGREE_LON_AT_43N, float(lat) * _KM_PER_DEGREE_LAT)
    return float(_projected_coastline().distance(point))


def is_adriatic_virtual_support(lat: float, lon: float, source_group: str) -> bool:
    """Return whether a designed virtual node supports the Adriatic target.

    The source groups encode whether coastal points belong to the Adriatic or
    Ionian side.  Albania crosses that boundary, so its northern points are
    retained using the approximate latitude of the Strait of Otranto.
    """
    del lon  # retained in the signature because this is a coordinate rule
    group = str(source_group).strip().upper()
    if group in _ADRIATIC_VIRTUAL_GROUPS:
        return True
    return group == "VIRTUAL_ALBANIA" and float(lat) >= _ALBANIA_ADRIATIC_MIN_LAT
