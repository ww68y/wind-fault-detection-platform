# v03 recall dataset summary

Targets:

- Recall >= 0.95
- mAP50-95 >= 0.85
- corrosion should not regress below the old model
- hard case missed samples should keep dropping
- false positives should drop after curated negative samples are added

Counts:

| Item | Value |
| --- | ---: |
| base_test_images | 543 |
| base_test_labels | 543 |
| base_train_images | 2866 |
| base_train_labels | 2866 |
| base_val_images | 571 |
| base_val_labels | 571 |
| corrosion_oversampled | 344 |
| data_yaml | C:\Users\yys\Desktop\风机\wind-fault-detection-platform\datasets\wind_blade_defect_v03_recall\data.yaml |
| low_confidence_sample_added | 0 |
| manifest | C:\Users\yys\Desktop\风机\wind-fault-detection-platform\datasets\wind_blade_defect_v03_recall\v03_recall_manifest.csv |
| manifest_rows | 1145 |
| missed_sample_added | 29 |
| missing_labels_skipped | 0 |
| negative_samples_added | 0 |
| small_target_oversampled | 772 |

Negative-sample note:

Put reviewed normal blade/background images in `hard_cases/negative_samples` or `hard_cases/false_positive_samples`, then rebuild with `--overwrite`. The script will add them with empty YOLO label files.
