# Models

MLVision runs third-party models that are downloaded once into a local cache
(`~/Library/Caches/mlvision/models` on macOS, `~/.cache/mlvision/models` on
Linux, or `MLVISION_MODEL_DIR`). The same files are committed under `models/files/`, so a
clone can run without any download: `mlvision scan -models models/files photo.jpg`.
Each file is pinned by SHA-256 in
`models/models.go`. Nothing in the inference path touches the network.

| Model | Role | Profile | Size | Licence | Source |
|---|---|---|---|---|---|
| `nsfw-marqo-vit-tiny-384` | whole-image NSFW classifier | both | 23 MB | Apache-2.0 | [Marqo/nsfw-image-detection-384](https://huggingface.co/Marqo/nsfw-image-detection-384) |
| `nudenet-yolov8n-320` | body-part detector | both | 12 MB | AGPL-3.0 | [notAI-tech/NudeNet](https://github.com/notAI-tech/NudeNet) |
| `face-ultraface-rfb-320` | face detector | commercial | 1 MB | MIT | [Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) |
| `age-fairface-vit-224-int8` | age (9 buckets) | both | 87 MB | Apache-2.0 | [dima806/fairface_age_image_detection](https://huggingface.co/dima806/fairface_age_image_detection) |
| `gender-fairface-vit-224-int8` | gender | both | 87 MB | Apache-2.0 | [dima806/fairface_gender_image_detection](https://huggingface.co/dima806/fairface_gender_image_detection) |
| `face-scrfd-10g` | face detector | research | 17 MB | non-commercial research | [deepinsight/insightface](https://github.com/deepinsight/insightface) `buffalo_l` |

The Marqo and FairFace checkpoints are published as PyTorch weights; the ONNX
files were produced with `tools/export_models.py`, which also checks each
export against the original model. The FairFace models are dynamically
quantised to int8 to cut their size by 4x.

## Profiles

- **commercial** (default): every model carries a permissive licence
  (Apache-2.0, MIT) or the AGPL. Fine for products, subject to the AGPL's
  terms for NudeNet.
- **research**: swaps in InsightFace's SCRFD face detector, licensed for
  non-commercial research only. Stronger in crowds and at odd angles. Age and
  gender still come from FairFace: its bucket probabilities expose uncertainty
  on small or blurry faces, where a bare regressor returns a confident wrong
  number.

## Attribution

NudeNet is © notAI.tech, AGPL-3.0. InsightFace models are © InsightFace
contributors, for non-commercial research use. Marqo and FairFace
checkpoints are Apache-2.0. UltraFace is MIT. MLVision itself is GPL-3.0.
