"""External video storage shared by the existing processing tools."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def outside_repo(path):
    path = Path(path).expanduser().resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError("Videos must live outside the repository; configure HUMANOID_VIDEO_ROOT.")
    return path


def video_root():
    value = os.environ.get("HUMANOID_VIDEO_ROOT")
    if not value:
        raise ValueError("Set HUMANOID_VIDEO_ROOT to your external Google Drive video folder, or supply an absolute video path.")
    return outside_repo(value)


def resolve_video(value):
    # Retain compatibility with historical metadata containing videos/<name>.
    value = Path(str(value).replace("\\", "/")).expanduser()
    if value.is_absolute():
        path = outside_repo(value)
    else:
        parts = value.parts[1:] if value.parts and value.parts[0] == "videos" else value.parts
        path = outside_repo(video_root().joinpath(*parts))
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def output_video(run_directory, filename="annotated.mp4", explicit=None):
    path = outside_repo(explicit) if explicit else outside_repo(video_root() / "processed" / Path(run_directory).resolve().name / filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
