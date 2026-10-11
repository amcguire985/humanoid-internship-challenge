import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from video_paths import resolve_video, output_video, video_root, REPO_ROOT


class VideoStorageTests(unittest.TestCase):
    def test_external_recording_and_legacy_metadata(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HUMANOID_VIDEO_ROOT": directory}):
            recording = Path(directory) / "data_003.MOV"
            recording.touch()
            for reference in ["data_003.MOV", "videos/data_003.MOV", "videos\\data_003.MOV", recording]:
                self.assertEqual(resolve_video(reference), recording.resolve())

    def test_overlay_is_external_and_parent_is_created(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HUMANOID_VIDEO_ROOT": directory}):
            output = output_video(REPO_ROOT / "results" / "data_003_raw")
            self.assertEqual(output, Path(directory).resolve() / "processed/data_003_raw/annotated.mp4")
            self.assertTrue(output.parent.is_dir())

    def test_repo_storage_rejected(self):
        with patch.dict(os.environ, {"HUMANOID_VIDEO_ROOT": str(REPO_ROOT / "videos")}):
            with self.assertRaises(ValueError): video_root()
        with self.assertRaises(ValueError): output_video("run", explicit=REPO_ROOT / "results/overlay.mp4")
        with self.assertRaises(ValueError): resolve_video(REPO_ROOT / "videos/data_003.MOV")

    def test_missing_configuration_and_missing_recording_are_clear(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "HUMANOID_VIDEO_ROOT"): resolve_video("data_003.MOV")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HUMANOID_VIDEO_ROOT": directory}):
            with self.assertRaises(FileNotFoundError): resolve_video("missing.MOV")

    def test_absolute_input_and_output_need_no_root(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            recording = Path(directory) / "input.MOV"
            recording.touch()
            self.assertEqual(resolve_video(recording), recording.resolve())
            self.assertEqual(output_video("run", explicit=Path(directory)/"overlay.mp4"), Path(directory).resolve()/"overlay.mp4")


if __name__ == "__main__": unittest.main()
