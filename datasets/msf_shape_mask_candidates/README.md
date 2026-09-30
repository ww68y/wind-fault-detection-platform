# MSF-YOLO shape mask candidate set

Purpose:

- Prepare high-value defect crops for later pixel-level morphology supervision.
- Keep original YOLO boxes as rectangular masks and optional GrabCut pseudo masks.
- Use overlays for fast manual review before converting reviewed masks to training labels.

Source:

- data_yaml: `datasets\wind_blade_defect_v05_conservative\data.yaml`
- max_per_class: `60`

Counts:

| Class | Candidates |
| --- | ---: |
| corrosion | 60 |
| crack | 60 |
| hole | 60 |
| spalling | 60 |
| total | 240 |

Review workflow:

1. Open `overlays/<class>/` and inspect each candidate.
2. Edit masks in `masks_grabcut/<class>/` or start from `masks_rect/<class>/` when GrabCut is noisy.
3. Mark `review_status` in `manifest.csv` as `accepted`, `edited`, or `rejected`.
4. Use reviewed masks later to build YOLO-seg/COCO masks or edge/centerline targets for MSF-YOLO v2.
