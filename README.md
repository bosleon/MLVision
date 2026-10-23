# MLVision

**Offline age, gender and NSFW detection for images. Pure Go. No cloud.**

MLVision is a Go library and a one-binary CLI. Give it an image and it tells you
three things: is this explicit, how old is each person, and what is their likely
gender. Everything runs on the local CPU and nothing leaves the machine.

Simplicity and elegance are the design motifs. Where other projects stack Python,
Docker, Caffe and a REST layer to answer one question, MLVision answers it with one
`go install` and one function call.

## About the author

MLVision is written and maintained by Will Bohan, who has spent the last ten years
working in machine learning and computer vision, building and shipping production
classifiers, detectors and the vision libraries that wrap them. This project distils
that experience into a tool that does one job well: best-of-breed content filtering
without the deployment weight the field has grown used to.

## Quick start

MLVision needs the ONNX Runtime shared library, the only dependency outside Go.

```bash
brew install onnxruntime           # macOS
# apt install libonnxruntime      # Debian/Ubuntu, or download a release from
#                                 # github.com/microsoft/onnxruntime and set MLVISION_ORT_LIB

go install github.com/bosleon/MLVision/cmd/mlvision@latest

mlvision models pull               # one-time download (~210 MB) into the local cache
mlvision scan photo.jpg
mlvision scan ./uploads --json
```

Sample output:

```
beach.jpg      safe 0.07  faces: F 26
group.jpg      safe 0.06  faces: M 25, M 32, F 28, M 15 minor  [face: male ~15]
example.jpg    NSFW 0.97  faces: F 31  [classifier: nsfw 0.97; detector: FEMALE_BREAST_EXPOSED 0.88]
```

## Library

```go
import mlvision "github.com/bosleon/MLVision"

a, err := mlvision.New(ctx, mlvision.Options{})
defer a.Close()

v, err := a.AnalyzeFile(ctx, "photo.jpg")
if v.NSFW || v.MinorPresent {
    reject(v.Reasons)
}
```

`Verdict` carries the classifier score, the body parts the detector found, every
face with its age, age range and gender, and a list of human-readable reasons. The
`Analyzer` is safe for concurrent use.

## How it works

```
image
  │
  ├──► NSFW classifier (ViT, whole image) ──► score
  │          │ score ≥ 0.2
  │          └──► nudity detector (YOLOv8) ──► parts, boxes, confidence
  │
  └──► face detector ──► per face: age, gender, box
                                │
                                ▼
                  Verdict { nsfw, score, reasons, parts, faces, minor_present }
```

- **Two-stage NSFW decision.** The classifier settles the easy cases. The part
  detector resolves the uncertain band and explains *why* an image was flagged,
  which is what moderators and auditors actually need. A classifier score of 0.8
  or an exposed-part detection of 0.5 makes an image NSFW; both thresholds are
  tunable.
- **Age and gender from faces.** Face detection, then per-face estimation on a
  1.5x crop. A face under 18 marks the verdict `minor_present`. Faces narrower
  than 24 px are reported box-only rather than with a confident guess; lower
  `-min-face` to estimate them anyway.
- **Two model profiles.** `commercial` (default) uses only permissively licensed
  models. `research` swaps in InsightFace's face detector, which is stronger but
  licensed for non-commercial use. See [MODELS.md](MODELS.md).
- **One runtime.** Every model is ONNX, executed through a single thin wrapper.
  No TensorFlow, no PyTorch, no Caffe, no Node.
- **Offline by construction.** Models are fetched once, verified by SHA-256 and
  cached. There is no network code in the inference path.

## Design principles

1. **Library first.** The CLI is a thin wrapper over the package.
2. **One struct in, one struct out.** `Analyze(ctx, img) (Verdict, error)`.
3. **Thresholds are data, not code.** Defaults ship in the binary; override them in
   `Options.Thresholds`.
4. **Boring dependencies.** Standard library, `x/image`, ONNX Runtime, nothing else.
5. **Deterministic and testable.** Fixed preprocessing, pinned model hashes,
   `go test` runs without network access.

## CLI

```
mlvision scan [-json] [-profile commercial|research] [-no-faces] [-minor-age 18] [-min-face 24] [-models dir] <image|dir>...
mlvision models [list|pull|verify] [-profile ...]
mlvision inspect <model.onnx>...
```

Running from a checkout needs no download: the model files are committed under
`models/files`, so pass `-models models/files` or set `MLVISION_MODEL_DIR`.

Environment: `MLVISION_MODEL_DIR` overrides the cache directory,
`MLVISION_ORT_LIB` points at the ONNX Runtime shared library.

## Layout

```
mlvision.go          public API: Options, Analyzer, Verdict
models/              model registry, pinned hashes, cache and download
internal/ort/        ONNX Runtime wrapper
internal/vision/     decode, resize, letterbox, tensor packing, NMS
internal/nsfw/       whole-image classifier
internal/nudity/     NudeNet part detector
internal/face/       face detectors and age/gender estimators, both profiles
cmd/mlvision/        CLI
tools/               one-time PyTorch to ONNX export script
```

## Roadmap

- [x] Core inference path (ONNX Runtime, image preprocessing)
- [x] NSFW whole-image classifier
- [x] Nudity region detector and reason reporting
- [x] Face detection, age and gender, commercial and research profiles
- [x] Model files committed in-repo and published as a GitHub release
- [x] CLI: `scan`, `models`, `inspect`, JSON output
- [x] Model cache with SHA-256 verification
- [ ] Video and stream support with frame sampling (ffmpeg)
- [ ] Benchmarks and a labelled evaluation set
- [ ] Optional GPU execution providers

## License

GPL-3.0. See [LICENSE](LICENSE). Model licences are listed in [MODELS.md](MODELS.md).
