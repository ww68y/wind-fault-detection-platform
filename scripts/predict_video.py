from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wind_fault.video.video_detector import detect_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLO detection on a wind blade inspection video.")
    parser.add_argument("--model", required=True, type=Path, help="Trained YOLO model path.")
    parser.add_argument("--source", required=True, type=Path, help="Video file path, such as mp4/avi/mov.")
    parser.add_argument("--conf", default=0.3, type=float, help="Confidence threshold.")
    parser.add_argument("--project", default="runs/video_detect", type=Path, help="Output root directory.")
    parser.add_argument("--name", default=None, help="Run name. Default uses video name and timestamp.")
    parser.add_argument("--device", default=None, help="Inference device, for example 0 for GPU or cpu.")
    parser.add_argument("--vid-stride", default=1, type=int, help="Process every Nth frame.")
    parser.add_argument("--max-keyframes", default=12, type=int, help="Maximum saved key frame screenshots.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = detect_video(
        model=args.model,
        source=args.source,
        output_root=args.project,
        name=args.name,
        conf=args.conf,
        device=args.device,
        vid_stride=args.vid_stride,
        max_keyframes=args.max_keyframes,
    )
    print(f"Output video: {report.get('output_video')}")
    print(f"Video report: {report['report_json']}")
    print(f"Markdown report: {report['report_markdown']}")


if __name__ == "__main__":
    main()
