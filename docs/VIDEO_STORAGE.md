# External video storage

Keep original phone recordings and generated MP4 overlays in Google Drive, outside this repository. Git retains code, calibration, editable annotations, trajectory CSVs, diagnostic images, metrics and robot episode datasets.

## Windows

Move or copy the existing recordings from `videos/` into a Google Drive folder. Copy generated `results/<run>/*.mp4` into `processed/<run>/` in that folder. Verify the Drive copies before removing any local originals. This change preserves the existing local video files but removes them from Git tracking.

```powershell
$env:HUMANOID_VIDEO_ROOT = "G:\My Drive\human-demonstration-videos"
.\.venv\Scripts\python.exe scripts/track_apriltags.py --video data_003.MOV --help
```

Use your actual Drive location. Add your normal tracking/calibration options to run tracking; `--help` above only displays options. The existing AprilTag tracking and cleaning algorithms are unchanged.

## Colab

Mount Google Drive in a notebook cell, then set the same environment variable:

```python
from google.colab import drive
import os
drive.mount('/content/drive')
os.environ['HUMANOID_VIDEO_ROOT'] = '/content/drive/MyDrive/human-demonstration-videos'
```

Keep recordings directly in that folder, with their original names. Video names in commands and historical `videos/<name>` metadata resolve against this root. Absolute external input paths also work. Historical absolute metadata paths from another machine must be replaced with a recording name before overlay regeneration.

Tracking overlays default to `$HUMANOID_VIDEO_ROOT/processed/<run-directory-name>/annotated.mp4`. The smoothed overlay uses the same directory and `sg7_gap_filled_annotated.mp4`. Use `--video-output` to choose another absolute external destination, or tracking's `--no-video` to produce only trajectory data and plots. CSV/JSON/PNG outputs continue to use `--output`. Distinct run directory names avoid overwriting overlays.

All common video extensions are ignored repository-wide. Tools reject video inputs/outputs located inside the checkout. Existing source-video strings in annotation files and archived results remain provenance references; the `videos/` prefix does not require a repository folder.

Removing tracked videos changes the current Git tree, not old commits. Previously committed video blobs remain in history. Cleaning that history would require a separate coordinated history rewrite; the existing local main branch still contains the earlier oversized commits and cannot be pushed as-is. The published Colab branch avoids those oversized ancestors.
