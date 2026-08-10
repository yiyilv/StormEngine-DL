#!/usr/bin/env python3
"""Resumable downloader for V7-B ICON-2I marine support inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from stormengine_dl.data.open_meteo import coordinate_manifest_bytes, load_marine_points, sha256_bytes  # noqa: E402

ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation,pressure_msl,surface_pressure,relative_humidity_2m"


def request(points, start_date: str, end_date: str, timeout: int, retries: int):
    params = {
        "latitude": ",".join(str(p.latitude) for p in points), "longitude": ",".join(str(p.longitude) for p in points),
        "start_date": start_date, "end_date": end_date, "hourly": HOURLY,
        "models": "italia_meteo_arpae_icon_2i", "timezone": "GMT", "temperature_unit": "celsius",
        "wind_speed_unit": "ms", "precipitation_unit": "mm", "cell_selection": "sea", "elevation": ",".join("nan" for _ in points),
    }
    url = ENDPOINT + "?" + urllib.parse.urlencode(params, safe=",")
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                body = response.read(); status = response.status
            if status != 200: raise RuntimeError(f"HTTP {status}")
            return params, url, status, body
        except (urllib.error.URLError, TimeoutError, RuntimeError):
            if attempt == retries: raise
            time.sleep(2 ** attempt)


def valid_existing(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value.get("http_status") == 200 and bool(value.get("response"))
    except (OSError, ValueError): return False


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("sample", "full"), required=True)
    parser.add_argument("--registry", default="data/stations_registry.csv"); parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=120); parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--output-dir", default="data_external/open_meteo/raw/20260801_20260808/icon2i")
    args = parser.parse_args(); points = load_marine_points(ROOT / args.registry)
    start, end = ("2026-08-01", "2026-08-01") if args.mode == "sample" else ("2026-08-01", "2026-08-08")
    selected = [points[0], points[len(points)//2], points[-1]] if args.mode == "sample" else list(points)
    output = ROOT / args.output_dir / ("sample" if args.mode == "sample" else "")
    output.mkdir(parents=True, exist_ok=True)
    coordinate_bytes = coordinate_manifest_bytes(points); (output / "coordinates.csv").write_bytes(coordinate_bytes)
    chunks = [selected[i:i+args.chunk_size] for i in range(0, len(selected), args.chunk_size)]; records=[]
    for index, chunk in enumerate(chunks):
        path = output / f"chunk_{index:03d}.json"
        if valid_existing(path):
            print(f"skip validated {path.name}", flush=True); wrapper=json.loads(path.read_text(encoding="utf-8"))
        else:
            params, url, status, body = request(chunk, start, end, args.timeout, args.retries)
            response = json.loads(body); response_list = response if isinstance(response, list) else [response]
            if len(response_list) != len(chunk): raise RuntimeError(f"Expected {len(chunk)} responses, got {len(response_list)}")
            wrapper = {"request_parameters": params, "request_url": url, "download_time_utc": datetime.now(timezone.utc).isoformat(), "http_status": status, "station_ids": [p.station_id for p in chunk], "response": response}
            path.write_text(json.dumps(wrapper, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            print(f"saved {path.name}: {len(chunk)} coordinates", flush=True)
        content=path.read_bytes(); records.append({"file":path.name,"stations":len(chunk),"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest(),"station_ids":[p.station_id for p in chunk]})
    manifest={"schema_version":1,"mode":args.mode,"endpoint":ENDPOINT,"model":"italia_meteo_arpae_icon_2i","start_date":start,"end_date":end,"coordinate_count":len(selected),"registry_coordinate_count":len(points),"coordinate_manifest_sha256":sha256_bytes(coordinate_bytes),"chunks":records}
    (output / "download_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
