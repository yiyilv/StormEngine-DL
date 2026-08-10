import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from stormengine_dl.data.meteohub_manifest import validate_meteohub_manifest


class MeteoHubManifestTests(unittest.TestCase):
    def _fixture(self, directory: str) -> tuple[Path, Path, bytes]:
        root = Path(directory)
        raw_root = root / "external"
        raw_root.mkdir()
        content = b'{"record": 1}\n{"record": 2}\n'
        (raw_root / "qc.json").write_bytes(content)
        manifest = {
            "raw_root": "external",
            "files": [
                {
                    "logical_name": "qc",
                    "filename": "qc.json",
                    "quality_controlled_only": True,
                    "bytes": len(content),
                    "jsonl_records": 2,
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
                {
                    "logical_name": "control",
                    "filename": "control.json",
                    "quality_controlled_only": False,
                    "bytes": 1,
                    "jsonl_records": 1,
                    "sha256": "0" * 64,
                },
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, raw_root, content

    def test_qc_only_accepts_complete_qc_file_and_skips_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, raw_root, _ = self._fixture(directory)
            results = validate_meteohub_manifest(manifest, raw_root, qc_only=True)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].ok)

    def test_reports_corrupted_and_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, raw_root, _ = self._fixture(directory)
            (raw_root / "qc.json").write_bytes(b"corrupt")
            results = validate_meteohub_manifest(manifest, raw_root)
            self.assertEqual(results[0].problems, ("size", "records", "sha256"))
            self.assertEqual(results[1].problems, ("missing",))


if __name__ == "__main__":
    unittest.main()
