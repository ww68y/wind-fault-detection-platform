# 数据集目录说明

原始数据建议放在 `datasets/raw/`，统一整理后的 YOLO 数据集放在 `datasets/wind_blade_defect/`。

## 推荐结构

```text
datasets/
├── raw/
│   ├── blade_surface_dataset/
│   │   ├── images/
│   │   └── annotations/
│   └── dtu/
│       ├── images/
│       └── annotations/
└── wind_blade_defect/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
```

## VOC XML 转 YOLO

```powershell
python -m wind_fault.data.convert_voc_xml `
  --images datasets/raw/blade_surface_dataset/images `
  --annotations datasets/raw/blade_surface_dataset/annotations `
  --output datasets/wind_blade_defect `
  --classes defect
```

## LabelMe JSON 转 YOLO

当前这批数据是 LabelMe JSON 标注。建议忽略 `blade`，因为它表示叶片区域本身，不是缺陷；`ff` 目前只有 1 条，像误标，也先忽略。

```powershell
python -m wind_fault.data.convert_labelme_json `
  --images datasets/raw/blade_surface_dataset/images `
  --annotations datasets/raw/blade_surface_dataset/annotations `
  --output datasets/wind_blade_defect `
  --classes crack,hole,spalling,corrosion `
  --ignore-classes blade,ff
```

## COCO/DTU JSON 转 YOLO

DTU 仓库通常按 train/val/test 提供标注文件，建议分开转换：

```powershell
python -m wind_fault.data.convert_coco_json `
  --annotation datasets/raw/dtu/annotations/train1024-s.json `
  --images datasets/raw/dtu/images `
  --output datasets/wind_blade_defect `
  --split train `
  --classes defect
```

## 高分辨率图片切片

```powershell
python -m wind_fault.data.slice_images `
  --images datasets/wind_blade_defect/images/train `
  --labels datasets/wind_blade_defect/labels/train `
  --output datasets/wind_blade_defect_1024 `
  --split train `
  --tile-size 1024 `
  --overlap 0.2
```
