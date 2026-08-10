#!/usr/bin/env python3
"""Audit downloaded ICON-2I chunks and build the V7-B marine contract tensor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from stormengine_dl.data.open_meteo import haversine_km, load_download_chunks, load_marine_points  # noqa: E402


def sha256(path: Path) -> str:
    digest=hashlib.sha256();
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--raw-dir",default="data_external/open_meteo/raw/20260801_20260808/icon2i")
    parser.add_argument("--registry",default="data/stations_registry.csv"); parser.add_argument("--processed",default="data_external/open_meteo/processed/20260801_20260808/icon2i_hourly_marine.npz")
    parser.add_argument("--audit-dir",default="data/audits/open_meteo_20260801_20260808"); parser.add_argument("--manifest",default="data/manifests/open_meteo_20260801_20260808.json")
    args=parser.parse_args(); raw=ROOT/args.raw_dir; points=load_marine_points(ROOT/args.registry); batch=load_download_chunks(raw,points)
    manifest=json.loads((raw/"download_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["chunks"]:
        path=raw/item["file"]
        if sha256(path)!=item["sha256"]: raise ValueError(f"SHA-256 mismatch: {path}")
    requested=np.asarray([[p.latitude,p.longitude] for p in points]); distances=haversine_km(requested,batch.returned_coordinates)
    audit=ROOT/args.audit_dir; audit.mkdir(parents=True,exist_ok=True)
    with (audit/"coordinate_coverage.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["station_index","station_id","requested_latitude","requested_longitude","returned_latitude","returned_longitude","returned_elevation","coordinate_distance_km","distance_over_10km","elevation_over_10m"])
        for p,returned,elevation,distance in zip(points,batch.returned_coordinates,batch.returned_elevation,distances,strict=True):
            writer.writerow([p.station_index,p.station_id,p.latitude,p.longitude,float(returned[0]),float(returned[1]),float(elevation),float(distance),bool(distance>10),bool(np.isfinite(elevation) and elevation>10)])
    coverage=[]
    with (audit/"variable_coverage.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["variable","valid_cells","total_cells","coverage_fraction","stations_with_any","hours_with_any","unit"])
        units=("m/s","m/s","m/s","degC","mm")
        for channel,(name,unit) in enumerate(zip(batch.variable_names,units,strict=True)):
            mask=batch.value_mask[:,:,channel]; row={"variable":name,"valid_cells":int(mask.sum()),"total_cells":int(mask.size),"coverage_fraction":float(mask.mean()),"stations_with_any":int(mask.any(0).sum()),"hours_with_any":int(mask.any(1).sum()),"unit":unit}; coverage.append(row); writer.writerow(row.values())
    processed=ROOT/args.processed; processed.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(processed,times=batch.times,station_ids=np.asarray(batch.station_ids),variable_names=np.asarray(batch.variable_names),values=batch.values,value_mask=batch.value_mask,observation_age=batch.observation_age,coordinates=batch.coordinates,station_static=batch.station_static,source_type=np.asarray(batch.source_type),returned_coordinates=batch.returned_coordinates,returned_elevation=batch.returned_elevation)
    anomalies={"distance_over_10km":int((distances>10).sum()),"elevation_over_10m":int((np.isfinite(batch.returned_elevation)&(batch.returned_elevation>10)).sum()),"max_coordinate_distance_km":float(distances.max())}
    light={"schema_version":1,"product":"Open-Meteo Historical Forecast API","model":"italia_meteo_arpae_icon_2i","role":"model-derived marine support inputs","time_start":str(batch.times[0]),"time_end":str(batch.times[-1]),"hour_count":len(batch.times),"station_count":len(batch.station_ids),"shape":list(batch.values.shape),"variables":list(batch.variable_names),"source_type":"model_derived_open_meteo","coordinate_manifest_sha256":manifest["coordinate_manifest_sha256"],"raw_chunks":manifest["chunks"],"processed_external":{"path":args.processed,"bytes":processed.stat().st_size,"sha256":sha256(processed)},"coverage":coverage,"coordinate_audit":anomalies,"tp_semantics":"hourly precipitation at valid_time; no interpolation or forward fill","model_run_time":"unavailable; not fabricated","future_leakage_rule":"V7-B input windows may use only valid_time <= forecast origin"}
    manifest_path=ROOT/args.manifest; manifest_path.parent.mkdir(parents=True,exist_ok=True); manifest_path.write_text(json.dumps(light,indent=2)+"\n",encoding="utf-8")
    lines=["# Open-Meteo ICON-2I marine audit","",f"- Coverage: {len(batch.station_ids)}/151 registered virtual sea points",f"- Time: {batch.times[0]} through {batch.times[-1]} ({len(batch.times)} hourly valid times)",f"- Contract tensor: `{tuple(batch.values.shape)}` in `u10,v10,i10fg,t2m,tp` order",f"- Coordinate distance: max {distances.max():.3f} km; {anomalies['distance_over_10km']} points over 10 km",f"- Returned elevation over 10 m: {anomalies['elevation_over_10m']} points","", "Open-Meteo is a model-derived marine support source, not a physical observation network. TP is hourly precipitation and is never interpolated or forward-filled. Model run time is unavailable in this stitched historical product and is not fabricated. The processed tensor and raw JSON remain outside Git.","", "## Variable coverage","", "| variable | valid/total | coverage |","|---|---:|---:|"]
    lines += [f"| {row['variable']} | {row['valid_cells']}/{row['total_cells']} | {100*row['coverage_fraction']:.2f}% |" for row in coverage]
    (audit/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(light,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
