"""
Utility helpers for VideoSeek AI.
"""

import shutil
import subprocess
from pathlib import Path


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def cleanup_temp_files(directory: str | Path) -> None:
    """Remove all files in a temporary directory."""
    directory = Path(directory)
    if directory.exists() and directory.is_dir():
        for item in directory.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except OSError:
                pass


def youtube_embed_url(youtube_url: str, start_seconds: int = 0) -> str:
    """Convert a YouTube URL to an embeddable URL with start time."""
    video_id = None
    if "youtu.be/" in youtube_url:
        video_id = youtube_url.split("youtu.be/")[-1].split("?")[0]
    elif "watch?v=" in youtube_url:
        video_id = youtube_url.split("watch?v=")[-1].split("&")[0]
    elif "youtube.com/embed/" in youtube_url:
        video_id = youtube_url.split("embed/")[-1].split("?")[0]

    if video_id:
        return f"https://www.youtube.com/embed/{video_id}?start={start_seconds}&autoplay=1"
    return youtube_url
