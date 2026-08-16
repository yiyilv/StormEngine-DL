import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from stormengine_dl import MaskAwareReconstructionModel
from stormengine_dl.models.mask_aware_reconstruction import (
    V8_RECONSTRUCTION_CONTRACT,
    restore_spatial_training_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparison = load_script("compare_v8_spatial_screens_test", "scripts/compare_v8_spatial_screens.py")
development_comparison = load_script(
    "compare_v8_spatial_development_test", "scripts/compare_v8_spatial_development.py"
)


class V8RefinementTests(unittest.TestCase):
    def make_training_objects(self):
        model = MaskAwareReconstructionModel(
            1, 1, include_age=False, point_hidden=4, latent_channels=4,
            height=3, width=3,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        contract = {"version": V8_RECONSTRUCTION_CONTRACT, "candidate": "same"}
        return model, optimizer, scheduler, scaler, contract

    def test_cross_directory_resume_preserves_selected_best(self) -> None:
        with TemporaryDirectory() as value:
            root = Path(value)
            source = root / "source"
            output = root / "continued"
            source.mkdir(); output.mkdir()
            model, optimizer, scheduler, scaler, contract = self.make_training_objects()
            history = [
                {"epoch": 1, "train_loss": 1.0, "validation_loss": 0.5},
                {"epoch": 2, "train_loss": 0.9, "validation_loss": 0.6},
            ]
            common = {
                "model_contract": contract,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_validation_loss": 0.5,
                "epochs_without_improvement": 1,
                "config": {},
            }
            torch.save({**common, "epoch": 1, "history": history[:1]}, source / "best.pt")
            torch.save({**common, "epoch": 2, "history": history}, source / "last.pt")

            restored = restore_spatial_training_checkpoint(
                source / "last.pt", output, model, optimizer, scheduler, scaler,
                contract, torch.device("cpu"),
            )
            self.assertEqual(restored[:3], (2, 0.5, 1))
            self.assertEqual(restored[3], history)
            self.assertTrue((output / "best.pt").is_file())
            copied = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(copied["epoch"], 1)

    def make_screen_summary(self, sigma: float, loss: float) -> dict:
        metric = lambda value: {"mae": value / 2, "rmse": value}
        variables = {name: metric(loss + index) for index, name in enumerate(
            ("msl", "u10", "v10", "t2m", "tp")
        )}
        return {
            "mode": "screen",
            "scientific_status": "candidate_screening_only",
            "seed": 42,
            "initial_epoch": 0,
            "target_max_epoch": 15,
            "train_batches_per_epoch": 1000,
            "validation_batches_per_epoch": 200,
            "train_years": [2010, 2011, 2012, 2013, 2014, 2015],
            "validation_years": [2016],
            "best_epoch": 10,
            "best_validation_loss": loss,
            "validation_metrics": {
                "full": variables,
                "land": variables,
                "sea": variables,
            },
            "contract": {
                "version": V8_RECONSTRUCTION_CONTRACT,
                "spatial_model": {
                    "gaussian_sigma": sigma,
                    "latent_channels": 64,
                },
            },
        }

    def test_sigma_comparison_ranks_equal_budget_candidates(self) -> None:
        with TemporaryDirectory() as value:
            root = Path(value)
            paths = []
            for sigma, loss in ((0.07, 0.4), (0.10, 0.3), (0.15, 0.35)):
                path = root / f"sigma{sigma}.json"
                path.write_text(json.dumps(self.make_screen_summary(sigma, loss)), encoding="utf-8")
                paths.append(str(path))
            result = comparison.compare(paths)
            self.assertEqual(result["screening_winner_sigma"], 0.10)
            self.assertEqual([row["gaussian_sigma"] for row in result["ranking"]], [0.10, 0.15, 0.07])

    def test_sigma_comparison_rejects_non_sigma_contract_change(self) -> None:
        with TemporaryDirectory() as value:
            root = Path(value)
            first = self.make_screen_summary(0.07, 0.4)
            second = self.make_screen_summary(0.10, 0.3)
            second["contract"]["spatial_model"]["latent_channels"] = 96
            paths = []
            for index, summary in enumerate((first, second)):
                path = root / f"candidate{index}.json"
                path.write_text(json.dumps(summary), encoding="utf-8")
                paths.append(str(path))
            with self.assertRaisesRegex(ValueError, "beyond gaussian_sigma"):
                comparison.compare(paths)

    def make_development_summary(self, sigma: float, loss: float) -> dict:
        summary = self.make_screen_summary(sigma, loss)
        summary.update({
            "mode": "develop",
            "scientific_status": "development_candidate_only",
            "train_years": [2013, 2014, 2015],
            "target_max_epoch": 25,
            "completed_epochs": 21,
            "early_stopping_patience": 10,
            "stopped_early": True,
            "epochs_without_improvement": 10,
            "train_batches_per_epoch": 1643,
            "validation_batches_per_epoch": 548,
        })
        return summary

    def test_three_year_comparison_ranks_equal_candidates(self) -> None:
        with TemporaryDirectory() as value:
            root = Path(value)
            paths = []
            for index, summary in enumerate((
                self.make_development_summary(0.10, 0.35),
                self.make_development_summary(0.15, 0.33),
            )):
                path = root / f"development{index}.json"
                path.write_text(json.dumps(summary), encoding="utf-8")
                paths.append(str(path))
            result = development_comparison.compare(paths)
            self.assertEqual(result["development_winner_sigma"], 0.15)
            self.assertEqual(result["train_years"], [2013, 2014, 2015])

    def test_three_year_comparison_rejects_different_years(self) -> None:
        with TemporaryDirectory() as value:
            root = Path(value)
            first = self.make_development_summary(0.10, 0.35)
            second = self.make_development_summary(0.15, 0.33)
            second["train_years"] = [2010, 2011, 2012]
            paths = []
            for index, summary in enumerate((first, second)):
                path = root / f"development{index}.json"
                path.write_text(json.dumps(summary), encoding="utf-8")
                paths.append(str(path))
            with self.assertRaisesRegex(ValueError, "budget differs"):
                development_comparison.compare(paths)

    def test_three_year_comparison_rejects_unconverged_candidate(self) -> None:
        with TemporaryDirectory() as value:
            root = Path(value)
            first = self.make_development_summary(0.10, 0.35)
            second = self.make_development_summary(0.15, 0.33)
            second["stopped_early"] = False
            paths = []
            for index, summary in enumerate((first, second)):
                path = root / f"development{index}.json"
                path.write_text(json.dumps(summary), encoding="utf-8")
                paths.append(str(path))
            with self.assertRaisesRegex(ValueError, "has not demonstrated validation convergence"):
                development_comparison.compare(paths)

if __name__ == "__main__":
    unittest.main()
