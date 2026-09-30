from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Run this script with the training venv.") from exc
    return YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an isolated YOLO-P2 small-object model.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model-config", default=Path("configs/yolov8n-p2-wind.yaml"), type=Path)
    parser.add_argument("--pretrained", default=None, type=Path, help="Optional weights for partial initialization.")
    parser.add_argument("--imgsz", default=1024, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--project", default=ROOT / "runs" / "detect" / "runs" / "wind_fault" / "yolop2", type=Path)
    parser.add_argument("--name", default="yolov8n_p2_from_v04_1024")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", default=0.0005, type=float)
    parser.add_argument("--lrf", default=0.05, type=float)
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine learning-rate schedule.")
    parser.add_argument("--close-mosaic", default=15, type=int)
    parser.add_argument("--freeze", default=None, type=int, help="Freeze first N layers, for head warmup.")
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--cache", default=None, choices=["ram", "disk"], help="Optional Ultralytics cache mode.")
    parser.add_argument("--fraction", default=1.0, type=float, help="Fraction of the training split to use.")
    parser.add_argument("--amp", default="false", choices=["true", "false"], help="Enable AMP. Default false avoids offline AMP checks.")
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    YOLO = _load_yolo()
    model = YOLO(str(args.model_config))
    if args.pretrained is not None:
        model.load(str(args.pretrained))

    train_args = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "project": str(args.project),
        "name": args.name,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "cos_lr": args.cos_lr,
        "close_mosaic": args.close_mosaic,
        "freeze": args.freeze,
        "patience": args.patience,
        "fraction": args.fraction,
        "amp": args.amp == "true",
        "exist_ok": args.exist_ok,
    }
    if args.cache is not None:
        train_args["cache"] = args.cache

    model.train(**train_args)


if __name__ == "__main__":
    main()
