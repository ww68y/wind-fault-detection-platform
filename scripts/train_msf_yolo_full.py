from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_ultralytics() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    try:
        from torch.utils.data import WeightedRandomSampler
        from ultralytics import YOLO
        from ultralytics.data.build import InfiniteDataLoader, seed_worker
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import LOGGER, RANK
        from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
        from ultralytics.utils.metrics import bbox_iou
        from ultralytics.utils.tal import bbox2dist
        from ultralytics.utils.torch_utils import torch_distributed_zero_first
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed. Run this script with the training venv.") from exc
    return (
        YOLO,
        DetectionTrainer,
        DetectionModel,
        BboxLoss,
        v8DetectionLoss,
        bbox_iou,
        bbox2dist,
        InfiniteDataLoader,
        seed_worker,
        WeightedRandomSampler,
        LOGGER,
        RANK,
        torch_distributed_zero_first,
    )


(
    YOLO,
    DetectionTrainer,
    DetectionModel,
    BboxLoss,
    v8DetectionLoss,
    bbox_iou,
    bbox2dist,
    InfiniteDataLoader,
    seed_worker,
    WeightedRandomSampler,
    LOGGER,
    RANK,
    torch_distributed_zero_first,
) = _load_ultralytics()


class ShapeAwareBboxLoss(BboxLoss):
    """YOLO bbox loss with extra weight for small and slender boxes."""

    def __init__(
        self,
        reg_max: int,
        area_threshold: float,
        small_gain: float,
        slender_gain: float,
        slender_ratio: float,
        max_shape_weight: float,
    ) -> None:
        super().__init__(reg_max)
        self.area_threshold = area_threshold
        self.small_gain = small_gain
        self.slender_gain = slender_gain
        self.slender_ratio = slender_ratio
        self.max_shape_weight = max_shape_weight

    def _shape_weight(
        self,
        target_bboxes: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> torch.Tensor:
        if not (self.small_gain or self.slender_gain):
            return torch.ones((int(fg_mask.sum().item()), 1), device=target_bboxes.device)

        stride_for_box = stride.view(1, -1, 1).expand(target_bboxes.shape[0], -1, 1)[fg_mask]
        target_px = target_bboxes[fg_mask] * stride_for_box
        wh = (target_px[:, 2:4] - target_px[:, 0:2]).clamp(min=1e-6)
        area_ratio = (wh[:, 0] * wh[:, 1]) / (imgsz[0] * imgsz[1]).clamp(min=1.0)

        small_term = ((self.area_threshold - area_ratio) / max(self.area_threshold, 1e-9)).clamp(0.0, 1.0)
        aspect = torch.maximum(wh[:, 0] / wh[:, 1], wh[:, 1] / wh[:, 0])
        slender_term = ((aspect - self.slender_ratio) / max(self.slender_ratio, 1e-9)).clamp(0.0, 1.0)

        shape_weight = 1.0 + self.small_gain * small_term + self.slender_gain * slender_term
        return shape_weight.clamp(max=self.max_shape_weight).unsqueeze(-1)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        weight = weight * self._shape_weight(target_bboxes, fg_mask, imgsz, stride)

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.zeros((), device=pred_dist.device)

        return loss_iou, loss_dfl


class MSFShapeAwareDetectionLoss(v8DetectionLoss):
    def __init__(self, model: torch.nn.Module, shape_config: dict[str, float]):
        super().__init__(model)
        self.bbox_loss = ShapeAwareBboxLoss(
            self.reg_max,
            area_threshold=shape_config["area_threshold"],
            small_gain=shape_config["small_gain"],
            slender_gain=shape_config["slender_gain"],
            slender_ratio=shape_config["slender_ratio"],
            max_shape_weight=shape_config["max_shape_weight"],
        ).to(self.device)


class MSFDetectionModel(DetectionModel):
    def __init__(self, *args: Any, shape_config: dict[str, float], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.msf_shape_config = shape_config

    def init_criterion(self):
        return MSFShapeAwareDetectionLoss(self, self.msf_shape_config)


class MSFFullInnovationTrainer(DetectionTrainer):
    shape_config: dict[str, float] = {}
    sampler_config: dict[str, Any] = {}

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = MSFDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            shape_config=self.shape_config,
        )
        if weights:
            model.load(weights)
        return model

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        if mode != "train" or not self.sampler_config.get("enabled", True) or rank != -1:
            return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        weights = self._sample_weights(dataset)
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)

        batch_size = min(batch_size, len(dataset))
        nd = torch.cuda.device_count()
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        nw = min(os.cpu_count() // max(nd, 1), workers)
        LOGGER.info(
            "MSF hard-sample sampler: "
            f"n={len(weights)}, min={float(weights.min()):.3f}, "
            f"mean={float(weights.mean()):.3f}, max={float(weights.max()):.3f}"
        )
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            sampler=sampler,
            prefetch_factor=4 if nw > 0 else None,
            pin_memory=nd > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=self.args.compile and len(dataset) % batch_size != 0,
        )

    def _sample_weights(self, dataset: Any) -> torch.Tensor:
        cfg = self.sampler_config
        small_classes = set(int(item) for item in cfg["small_target_class_ids"])
        weights: list[float] = []
        for label in getattr(dataset, "labels", []):
            cls = np.asarray(label.get("cls", []), dtype=np.float32).reshape(-1)
            bboxes = np.asarray(label.get("bboxes", []), dtype=np.float32).reshape(-1, 4)
            if cls.size == 0:
                weights.append(float(cfg["negative_weight"]))
                continue

            class_ids = {int(item) for item in cls}
            weight = 1.0
            if 3 in class_ids:
                weight *= float(cfg["corrosion_weight"])
            if 0 in class_ids:
                weight *= float(cfg["crack_weight"])

            if bboxes.size:
                small = False
                for class_id, bbox in zip(cls.astype(int), bboxes):
                    if class_id not in small_classes:
                        continue
                    area = float(max(bbox[2], 0.0) * max(bbox[3], 0.0))
                    if area <= float(cfg["small_target_max_area"]):
                        small = True
                        break
                if small:
                    weight *= float(cfg["small_target_weight"])

            weights.append(min(float(cfg["max_sample_weight"]), max(0.05, weight)))

        if not weights:
            raise RuntimeError("Training dataset produced no labels for MSF sampler.")
        return torch.as_tensor(weights, dtype=torch.double)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an isolated MSF-YOLO full-innovation experiment.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model-config", default=Path("configs/yolov8n-p2-wind.yaml"), type=Path)
    parser.add_argument("--pretrained", default=None, type=Path)
    parser.add_argument("--imgsz", default=1024, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--project", default=ROOT / "runs" / "detect" / "runs" / "wind_fault" / "msf_yolo_full_innovation", type=Path)
    parser.add_argument("--name", default="msf_yolo_full_innovation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", default=0.00025, type=float)
    parser.add_argument("--lrf", default=0.05, type=float)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--close-mosaic", default=20, type=int)
    parser.add_argument("--freeze", default=None, type=int)
    parser.add_argument("--patience", default=30, type=int)
    parser.add_argument("--cache", default=None, choices=["ram", "disk"])
    parser.add_argument("--fraction", default=1.0, type=float)
    parser.add_argument("--amp", default="true", choices=["true", "false"])
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--cls-pw", default=0.35, type=float)

    parser.add_argument("--hard-reweight", default="true", choices=["true", "false"])
    parser.add_argument("--corrosion-weight", default=3.0, type=float)
    parser.add_argument("--crack-weight", default=2.0, type=float)
    parser.add_argument("--small-target-weight", default=1.5, type=float)
    parser.add_argument("--negative-weight", default=1.0, type=float)
    parser.add_argument("--max-sample-weight", default=6.0, type=float)
    parser.add_argument("--small-target-max-area", default=0.012, type=float)
    parser.add_argument("--small-target-class-ids", default=[0, 3], nargs="*", type=int)

    parser.add_argument("--shape-aware", default="true", choices=["true", "false"])
    parser.add_argument("--shape-area-threshold", default=0.012, type=float)
    parser.add_argument("--shape-small-gain", default=0.35, type=float)
    parser.add_argument("--shape-slender-gain", default=0.25, type=float)
    parser.add_argument("--shape-slender-ratio", default=3.0, type=float)
    parser.add_argument("--shape-max-weight", default=1.7, type=float)

    parser.add_argument("--multi-scale", default=0.20, type=float)
    parser.add_argument("--mosaic", default=1.0, type=float)
    parser.add_argument("--copy-paste", default=0.05, type=float)
    parser.add_argument("--scale", default=0.70, type=float)
    parser.add_argument("--fliplr", default=0.5, type=float)
    return parser


def _shape_config(args: argparse.Namespace) -> dict[str, float]:
    if args.shape_aware == "false":
        return {
            "area_threshold": args.shape_area_threshold,
            "small_gain": 0.0,
            "slender_gain": 0.0,
            "slender_ratio": args.shape_slender_ratio,
            "max_shape_weight": 1.0,
        }
    return {
        "area_threshold": args.shape_area_threshold,
        "small_gain": args.shape_small_gain,
        "slender_gain": args.shape_slender_gain,
        "slender_ratio": args.shape_slender_ratio,
        "max_shape_weight": args.shape_max_weight,
    }


def main() -> None:
    args = build_parser().parse_args()

    MSFFullInnovationTrainer.shape_config = _shape_config(args)
    MSFFullInnovationTrainer.sampler_config = {
        "enabled": args.hard_reweight == "true",
        "corrosion_weight": args.corrosion_weight,
        "crack_weight": args.crack_weight,
        "small_target_weight": args.small_target_weight,
        "negative_weight": args.negative_weight,
        "max_sample_weight": args.max_sample_weight,
        "small_target_max_area": args.small_target_max_area,
        "small_target_class_ids": args.small_target_class_ids,
    }

    model = YOLO(str(args.model_config))
    if args.pretrained is not None:
        model.load(str(args.pretrained))

    train_args: dict[str, Any] = {
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
        "cls_pw": args.cls_pw,
        "multi_scale": args.multi_scale,
        "mosaic": args.mosaic,
        "copy_paste": args.copy_paste,
        "scale": args.scale,
        "fliplr": args.fliplr,
    }
    if args.cache is not None:
        train_args["cache"] = args.cache

    config_payload = {
        "experiment": "MSF-YOLO full innovation",
        "training_integrated": [
            "YOLO-P2 small-object detection head",
            "hard-sample weighted sampler",
            "class-weighted classification loss via cls_pw",
            "Shape-Aware bbox/DFL loss weighting",
            "small-object-oriented augmentation parameters",
        ],
        "post_training_integrated": ["Blade-Slice inference/evaluation"],
        "shape_config": MSFFullInnovationTrainer.shape_config,
        "sampler_config": MSFFullInnovationTrainer.sampler_config,
        "train_args": train_args,
        "pretrained": str(args.pretrained) if args.pretrained is not None else None,
        "model_config": str(args.model_config),
    }

    model.train(trainer=MSFFullInnovationTrainer, **train_args)

    run_dir = Path(model.trainer.save_dir)
    (run_dir / "msf_full_innovation_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
