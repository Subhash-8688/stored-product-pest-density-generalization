#!/usr/bin/env python3
"""Render an annotated object-detection video from a trained YOLO checkpoint."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poster", type=Path, help="Optional path for a representative annotated frame.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--ffmpeg", help="Optional ffmpeg executable path. Required for browser-compatible H.264 output.")
    return parser.parse_args()


def output_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Return an even-sized output resolution that preserves the aspect ratio."""
    if width <= max_width:
        return width - width % 2, height - height % 2
    scale = max_width / width
    return max_width - max_width % 2, int(height * scale) // 2 * 2


def select_h264_encoder(ffmpeg: str) -> str:
    """Choose an H.264 encoder exposed by the local ffmpeg installation."""
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    for encoder in ("libx264", "libopenh264"):
        if encoder in completed.stdout:
            return encoder
    raise RuntimeError("ffmpeg must provide libx264 or libopenh264 for browser-compatible H.264 output")


def transcode_h264(source: Path, output: Path, ffmpeg_arg: str | None) -> None:
    """Transcode an OpenCV-rendered MP4 into an H.264 file suitable for browser playback."""
    ffmpeg = ffmpeg_arg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found; install ffmpeg or pass --ffmpeg /path/to/ffmpeg")
    encoder = select_h264_encoder(ffmpeg)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if encoder == "libx264":
        command.extend(["-crf", "20", "-preset", "medium"])
    else:
        command.extend(["-b:v", "2M"])
    command.append(str(output))
    subprocess.run(command, check=True)
    source.unlink()


def main() -> None:
    """Run video inference and save an annotated MP4."""
    args = parse_args()
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    render_width, render_height = output_size(width, height, args.max_width)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.poster:
        args.poster.parent.mkdir(parents=True, exist_ok=True)
    intermediate = args.output.with_name(f"{args.output.stem}.opencv{args.output.suffix}")
    intermediate.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (render_width, render_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create video writer: {args.output}")

    model = YOLO(str(args.weights))
    poster_written = False
    processed = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != (render_width, render_height):
            frame = cv2.resize(frame, (render_width, render_height), interpolation=cv2.INTER_AREA)
        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
        annotated = result.plot(line_width=2, font_size=14)
        writer.write(annotated)
        processed += 1
        if args.poster and not poster_written and processed >= max(1, frame_count // 3):
            cv2.imwrite(str(args.poster), annotated)
            poster_written = True
        if processed % 25 == 0:
            print(f"Processed {processed}/{frame_count} frames", flush=True)

    capture.release()
    writer.release()
    if processed == 0:
        raise RuntimeError("no frames were read from the source video")
    transcode_h264(intermediate, args.output, args.ffmpeg)
    print(f"Saved {processed} browser-compatible annotated frames to {args.output}")


if __name__ == "__main__":
    main()
