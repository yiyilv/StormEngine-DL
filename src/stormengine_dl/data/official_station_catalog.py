"""Normalize official DPC/regional station metadata into a versioned snapshot."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_DOMAIN = (39.0, 46.5, 12.0, 20.0)

# MeteoHub uses a dpcn-* dataset for most regional networks. FVG and
# Emilia-Romagna publish the same civil-protection observations under their
# regional network identifiers instead.
REGIONAL_DPC_NETWORKS = frozenset(
    {
        "arpafvg",
        "agrmet",
        "boa",
        "claster",
        "locali",
        "marefe",
        "simnbo",
        "simnpr",
        "spdsra",
        "urbane",
    }
)


@dataclass(frozen=True)
class OfficialStation:
    station_id: str
    station_name: str
    lat: float
    lon: float
    network: str
    coordinate_source: str
    catalog_status: str
    observed_snapshots: int
    variables: str
    license: str
    notes: str


def is_dpc_network(network: str) -> bool:
    """Return whether a MeteoHub network belongs to the project DPC federation."""
    return network.startswith("dpcn-") or network in REGIONAL_DPC_NETWORKS


def _inside(lat: float, lon: float, domain: tuple[float, float, float, float]) -> bool:
    lat_min, lat_max, lon_min, lon_max = domain
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def _station_name(details: Iterable[Mapping[str, object]]) -> str:
    for detail in details:
        if detail.get("var") == "B01019" and detail.get("val"):
            return str(detail["val"])
    return "unnamed station"


def collect_meteohub_stations(
    snapshots: Iterable[tuple[str, Mapping[str, object]]],
    *,
    domain: tuple[float, float, float, float] = DEFAULT_DOMAIN,
) -> list[OfficialStation]:
    """Union stations present in official MeteoHub observation snapshots."""
    found: dict[tuple[str, float, float], dict[str, object]] = {}
    for snapshot_name, payload in snapshots:
        for block in payload.get("data", []):  # type: ignore[union-attr]
            stat = block.get("stat", {})
            network = str(stat.get("net", ""))
            if not is_dpc_network(network):
                continue
            lat, lon = float(stat["lat"]), float(stat["lon"])
            if not _inside(lat, lon, domain):
                continue
            key = (network, round(lat, 6), round(lon, 6))
            item = found.setdefault(
                key,
                {
                    "name": _station_name(stat.get("details", [])),
                    "snapshots": set(),
                    "variables": set(),
                },
            )
            item["snapshots"].add(snapshot_name)  # type: ignore[union-attr]
            item["variables"].update(  # type: ignore[union-attr]
                str(product["var"])
                for product in block.get("prod", [])
                if product.get("var")
            )

    stations: list[OfficialStation] = []
    for (network, lat, lon), item in sorted(found.items()):
        identity = f"meteohub|{network}|{lat:.6f}|{lon:.6f}"
        digest = hashlib.sha1(identity.encode()).hexdigest()[:12]
        stations.append(
            OfficialStation(
                station_id=f"MH::{network}::{digest}",
                station_name=str(item["name"]),
                lat=lat,
                lon=lon,
                network=network,
                coordinate_source="MeteoHub /api/observations",
                catalog_status="observed_in_snapshot",
                observed_snapshots=len(item["snapshots"]),  # type: ignore[arg-type]
                variables="|".join(sorted(item["variables"])),  # type: ignore[arg-type]
                license="CC BY 4.0 (dataset attribution applies)",
                notes=(
                    "Official DPC/regional station observed in at least one selected "
                    "MeteoHub time window; an offline station may be absent."
                ),
            )
        )
    return stations


def collect_abruzzo_stations(
    payload: Mapping[str, object],
    *,
    domain: tuple[float, float, float, float] = DEFAULT_DOMAIN,
) -> list[OfficialStation]:
    """Extract the public Polaris-linked subset from Abruzzo's official API."""
    stations: list[OfficialStation] = []
    for row in payload.get("data", []):  # type: ignore[union-attr]
        if row.get("source") != "Regione Abruzzo" or row.get("polaris_id") is None:
            continue
        lat, lon = float(row["lat"]), float(row["lon"])
        if not _inside(lat, lon, domain):
            continue
        stations.append(
            OfficialStation(
                station_id=f"ABR::POLARIS::{row['polaris_id']}",
                station_name=str(row["name"]),
                lat=lat,
                lon=lon,
                network="regione-abruzzo-polaris",
                coordinate_source="Regione Abruzzo Agroambiente station API",
                catalog_status="official_public_subset",
                observed_snapshots=int(row.get("last_week", 0) or 0),
                variables="",
                license="public regional data; verify reuse terms",
                notes=(
                    "Official Polaris-linked station. This public endpoint exposes 47 of "
                    "the 119 stations stated for the full Abruzzo telemetered network."
                ),
            )
        )
    return sorted(stations, key=lambda item: (item.lat, item.lon, item.station_id))


def write_official_station_catalog(
    stations: Iterable[OfficialStation], output_path: str | Path
) -> list[OfficialStation]:
    """Deduplicate coordinates and write a stable CSV snapshot."""
    unique: dict[tuple[float, float], OfficialStation] = {}
    for station in stations:
        unique.setdefault((round(station.lat, 6), round(station.lon, 6)), station)
    records = sorted(unique.values(), key=lambda item: (item.lat, item.lon, item.station_id))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(OfficialStation.__dataclass_fields__),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return records


def load_json(path: str | Path) -> Mapping[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
