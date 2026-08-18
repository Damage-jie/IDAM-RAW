# CR7-RAW Dataset Card

## Summary

CR7-RAW is a paired real-world low-light RAW dataset acquired with a Canon R7 camera using auto exposure bracketing (AEB). Each retained scene contains an underexposed Bayer RAW observation and a corresponding well-exposed ISP-processed RGB reference.

| Property | Value |
|---|---|
| Total paired scenes | 1,159 |
| Training split | 1,019 scenes |
| Validation split | 140 scenes |
| Test split | None |
| Object categories | 7 |
| Annotation tool | LabelImg |
| Annotation format | Normalized YOLO bounding boxes |

The categories are person, car, truck, electric vehicle, tricycle, garbage bin, and roadblock.

## Download

Dataset files are not stored in this GitHub repository.

> **Baidu Netdisk URL and extraction code: `BAIDU_NETDISK_LINK_TO_BE_ADDED`**

The authors should replace this placeholder only after uploading and validating the final archive.

## Acquisition, alignment, and annotation

AEB observations were captured consecutively with the Canon R7 mounted on a tripod. All pairs were manually inspected, and samples showing visible camera displacement, inconsistent framing, or object motion that caused apparent spatial misalignment were excluded. The retained samples provide scene-level and bounding-box-level correspondence; subpixel or pixel-perfect registration is not claimed.

One author independently annotated the ISP-processed RGB references with LabelImg because their higher visibility supports reliable object identification and boundary localization. Category labels and boxes were transferred to the corresponding spatially paired RAW observations.

The split is performed at the physical-scene level. RAW and RGB observations from the same scene remain in the same subset, and no physical scene appears in both training and validation.

## Archive integrity

Before publishing the Baidu Netdisk archive, verify that:

1. `images/train` and `labels/train` contain exactly 1,019 matching stems.
2. `images/val` and `labels/val` contain exactly 140 matching stems.
3. No physical scene occurs in both splits.
4. Every image has one label file and every label file has one image.
5. Each label row follows `class x_center y_center width height`, with normalized coordinates in `[0, 1]`.
6. All RAW samples decode to the expected four-channel Bayer representation.
7. Private metadata and samples without distribution permission are removed.
8. The archive includes checksums and final dataset terms.

Run the structural checker after extraction:

```bash
python scripts/check_dataset.py --root datasets/CR7_RAW --expected-train 1019 --expected-val 140
```

## Terms

The software license does not cover CR7-RAW. The downloadable archive must include the final dataset terms, attribution requirements, and permitted-use statement.

## Contact

Please use the repository issue tracker or email `2023021001000611@ecjtu.edu.cn`.
