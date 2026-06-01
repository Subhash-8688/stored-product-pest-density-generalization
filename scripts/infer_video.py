#!/usr/bin/env python3
"""Render an annotated object-detection video from a trained YOLO checkpoint."""

from __future__ import annotations

import argparse
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
    return parser.parse_args()


def output_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Return an even-sized output resolution that preserves the aspect ratio."""
    if width <= max_width:
        return width - width % 2, height - height % 2
    scale = max_width / width
    return max_width - max_width % 2, int(height * scale) // 2 * 2


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
    writer = cv2.VideoWriter(
        str(args.output),
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
    print(f"Saved {processed} annotated frames to {args.output}")


if __name__ == "__main__":
    main()
