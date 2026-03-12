"""
Video processing: download YouTube videos, extract frames, extract audio.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
from PIL import Image

import config


def download_youtube(url: str, output_dir: Optional[str] = None) -> str:
    """
    Download a YouTube video using yt-dlp.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the video. Defaults to TEMP_DIR.

    Returns:
        Path to the downloaded video file.
    """
    import yt_dlp

    output_dir = output_dir or str(config.TEMP_DIR)
    os.makedirs(output_dir, exist_ok=True)

    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "video")
        ext = info.get("ext", "mp4")
        return os.path.join(output_dir, f"{video_id}.{ext}")


def extract_frames(
    video_path: str,
    fps: float = None,
) -> list[tuple[float, Image.Image]]:
    """
    Extract frames from a video at a given FPS rate.

    Args:
        video_path: Path to the video file.
        fps: Frames per second to extract. Defaults to config value.

    Returns:
        List of (timestamp_seconds, PIL.Image) tuples.
    """
    fps = fps or config.FRAME_EXTRACT_FPS
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    frame_interval = max(1, int(video_fps / fps))
    frames: list[tuple[float, Image.Image]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append((timestamp, pil_image))

        frame_idx += 1

    cap.release()
    return frames


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio track from a video file to WAV format.

    Args:
        video_path: Path to the video file.
        output_path: Output WAV path. Defaults to temp file.

    Returns:
        Path to the extracted WAV file.
    """
    import subprocess

    if output_path is None:
        output_path = os.path.join(
            config.TEMP_DIR,
            Path(video_path).stem + "_audio.wav",
        )

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # WAV format
        "-ar", "16000",           # 16kHz for Whisper
        "-ac", "1",               # mono
        "-y",                     # overwrite
        output_path,
    ]

    subprocess.run(cmd, capture_output=True, check=True, timeout=300)
    return output_path
