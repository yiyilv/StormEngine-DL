#!/usr/bin/env python3
"""Freeze a completed V7-A run into a small, auditable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/v7_a_2010_2017/best.pt")
    parser.add_argument("--config", default="configs/v7_a.yaml")
    parser.add_argument("--output", default="artifacts/v7_a_2010_2017/frozen_manifest.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = (root / args.checkpoint).resolve(); config = (root / args.config).resolve()
    required = [checkpoint, config]
    evaluations = sorted(checkpoint.parent.glob("evaluation_2017_*/metrics.json"))
    replay = checkpoint.parent / "dpc_replay_20260801_20260808" / "replay_summary.json"
    dpc_evaluation = checkpoint.parent / "dpc_observation_evaluation_20260801_20260808" / "metrics.json"
    required.extend(evaluations); required.extend((replay, dpc_evaluation))
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not evaluations:
        raise FileNotFoundError(f"V7-A cannot be frozen before checkpoint, 2017 evaluation, and DPC replay exist: {missing}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest = {
        "schema_version": 1,
        "model": "V7-A",
        "status": "frozen",
        "definition": "239 physical stations; 12h history; u10,v10,i10fg,t2m,tp; +1..+6h grid forecast",
        "git_commit": commit,
        "checkpoint": {"path": str(checkpoint.relative_to(root)).replace("\\", "/"), "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size},
        "config": {"path": str(config.relative_to(root)).replace("\\", "/"), "sha256": sha256(config)},
        "evaluation_files": [{"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": sha256(path)} for path in evaluations],
        "dpc_replay": {"path": str(replay.relative_to(root)).replace("\\", "/"), "sha256": sha256(replay)},
        "dpc_observation_evaluation": {"path": str(dpc_evaluation.relative_to(root)).replace("\\", "/"), "sha256": sha256(dpc_evaluation)},
        "note": "Checkpoint remains a local external artifact and should not be committed to ordinary Git.",
    }
    output = (root / args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output); return 0


if __name__ == "__main__":
    raise SystemExit(main())
