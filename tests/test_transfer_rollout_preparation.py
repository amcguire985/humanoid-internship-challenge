"""Colab handoff contracts tested without simulator modules."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_transfer_rollouts import load_config, claim_attempt
from record_transfer_demos import run, combine


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config/data_003_rollouts.json").read_text(encoding="utf-8"))

    def check_config(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            return load_config(path)

    def test_manifest_preserves_reverse_strategy_and_baseline_selection(self):
        config = self.check_config(self.config)
        self.assertEqual([d["episode_id"] for d in config["demos"]], ["episode_004", "episode_005", "episode_006"])
        self.assertEqual(config["demos"][1]["direction"], "target -> object")
        self.assertEqual(config["baseline_episode_ids"], ["episode_001", "episode_002"])

    def test_multiple_attempts_and_episode_collisions_are_rejected(self):
        for mutation in ("retries", "count", "duplicate", "baseline"):
            config = copy.deepcopy(self.config)
            if mutation == "retries": config["automatic_retries"] = True
            if mutation == "count": config["rollouts_per_demo"] = 2
            if mutation == "duplicate": config["demos"][1]["episode_id"] = "episode_004"
            if mutation == "baseline": config["demos"][1]["episode_id"] = "episode_001"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.check_config(config)

    def test_attempt_directory_prevents_retry_even_without_status(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempts/demo_001"
            self.assertTrue(claim_attempt(attempt))
            self.assertFalse(claim_attempt(attempt))

    def test_prepare_only_never_probes_or_imports_simulator(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            config["rollout_output"] = str(Path(directory)/"rollouts")
            config["prepared_root"] = str(Path(directory)/"prepared")
            bundle = {"demos": [{**entry, "ready_for_rollout": True} for entry in config["demos"]]}
            simulator_before = {name for name in sys.modules if name.split(".")[0] in {"libero", "robosuite", "mujoco"}}
            with patch("prepare_transfer_rollouts.prepare_bundle", return_value=(config, bundle)), patch("importlib.util.find_spec") as probe, patch("builtins.print"):
                result = run(Path("manifest.json"), prepare_only=True)
            probe.assert_not_called()
            self.assertEqual(result, bundle)
            self.assertFalse((Path(directory)/"rollouts/attempts").exists())
            self.assertEqual(simulator_before, {name for name in sys.modules if name.split(".")[0] in {"libero", "robosuite", "mujoco"}})

    def test_unrelated_robot_episode_cannot_enter_combined_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"new/episodes").mkdir(parents=True)
            with h5py.File(root/"new/episodes/episode_099.h5", "w") as file:
                file.attrs["human_demo_id"] = "unrelated_demo"
                file.attrs["episode_id"] = "episode_099"
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                combine(root/"baseline", root/"new", root/"combined")
            self.assertFalse((root/"combined").exists())

    def test_user_event_values_preserved_exactly(self):
        expected = [(5.3, 18.5, 5.9, 16.7), (24.5, 36.5, 25.4, 35.5), (41.1, 55.0, 42.5, 53.2)]
        for entry, times in zip(self.config["demos"], expected):
            annotation = json.loads((ROOT/entry["annotation_file"]).read_text(encoding="utf-8"))
            self.assertEqual(tuple(annotation[k] for k in ("grasp_time_seconds", "release_time_seconds", "transport_start_time_seconds", "transport_end_time_seconds")), times)
            self.assertIn("automatic_estimates", annotation)

    def test_prepared_input_changes_only_transport_timeout(self):
        base = json.loads((ROOT/self.config["robot_config"]).read_text(encoding="utf-8"))
        for entry in self.config["demos"]:
            prepared = json.loads((ROOT/entry["prepared_demo"]/"retargeting_input.json").read_text(encoding="utf-8"))
            robot = prepared["robot_settings"]
            self.assertEqual(robot["phase_max_steps"]["TRANSPORT"], entry["transport_step_budget"])
            original = copy.deepcopy(robot)
            original["phase_max_steps"]["TRANSPORT"] = base["phase_max_steps"]["TRANSPORT"]
            self.assertEqual(original, base)
            self.assertTrue(prepared["ready_for_rollout"])
            self.assertLess(prepared["mapping_preview"]["transport_steps"], entry["transport_step_budget"])
            self.assertLess(prepared["mapping_preview"]["retargeting"]["endpoint_correction_norm_m"], .06)


if __name__ == "__main__": unittest.main()
