# MLVision

**Offline age, gender and NSFW detection for images and video. Pure Go. No Docker. No cloud.**

MLVision is a single static binary and a library. Point it at an image, a directory or a video
stream and it tells you three things: is this explicit, how old is the person, and what is their
likely gender. Everything runs on the local CPU (GPU optional) and nothing ever leaves the machine.

Simplicity and elegance are the design motifs. Where other projects stack Python, Docker, Caffe
and a REST layer to answer one question, MLVision answers it with one `go install` and one call.

## About the author

MLVision is written and maintained by Will Bohan, who has spent the last ten years working in
machine learning and computer vision, building and shipping production classifiers, detectors
and the vision libraries that wrap them. This project distills that experience into a tool that
does one job well: best-of-breed content filtering, with none of the deployment weight that the
field has grown used to.

## Why another NSFW detector

The open source landscape is wide but fragmented. Yahoo Open NSFW, NudeNet, NSFWJS, OpenNSFW2,
NsfwSpy, the Falconsai, AdamCodd, Marqo and SigLIP2 classifiers on Hugging Face: each is a
single model with its own runtime, its own preprocessing quirks and its own definition of
"explicit". None of them do age. Almost none run without Python.

MLVision takes the best ideas from that body of work and unifies them:

- **Two-stage NSFW decision.** A fast whole-image classifier gives a confident verdict on the
  easy cases. A body-part detector resolves the hard ones and explains *why* an image was
  flagged, which is what moderators and auditors actually need.
- **Age and gender from faces.** Face detection, then per-face age regression and gender
  classification. A frame with a minor in it is handled as a first-class signal, not an
  afterthought.
- **One runtime.** Every model is shipped as ONNX and executed through a single inference path.
  No TensorFlow, no PyTorch, no Caffe, no Node.
- **Offline by construction.** Models are bundled or fetched once into a local cache. There is
  no network code in the inference path, so the privacy claim is verifiable by reading the source.

## Design

```
image / frame
     │
     ├──► NSFW classifier (whole image) ──► score + label
     │          │ (uncertain band)
     │          └──► nudity detector (regions) ──► parts, boxes, confidence
     │
     └──► face detector ──► per face: age, gender, box
                                   │
                                   ▼
                           Verdict { nsfw, reasons, faces[], minor_present }
```

Principles:

1. **Library first.** The CLI is a thin wrapper over a Go package. If you can call a function,
   you can integrate MLVision.
2. **One struct in, one struct out.** `Analyze(img) (Verdict, error)`. No callbacks, no
   configuration trees.
3. **Thresholds are data, not code.** Sensible defaults ship in the binary. Override them with a
   single small config if your use case disagrees.
4. **Boring dependencies.** Standard library wherever possible, ONNX Runtime for inference,
   nothing else that needs justifying.
5. **Deterministic and testable.** Fixed preprocessing, pinned model hashes, golden tests on a
   curated set.

## Planned usage

```bash
go install github.com/bosleon/MLVision/cmd/mlvision@latest

mlvision scan photo.jpg
mlvision scan ./uploads --json
mlvision watch rtsp://camera/stream
```

```go
import "github.com/bosleon/MLVision"

v, err := mlvision.Analyze(ctx, img)
if v.NSFW || v.MinorPresent {
    reject(v.Reasons)
}
```

## Roadmap

- [x] Repository and design
- [ ] Core inference path (ONNX Runtime, image preprocessing)
- [ ] NSFW whole-image classifier
- [ ] Nudity region detector and reason reporting
- [ ] Face detection, age and gender
- [ ] CLI: `scan`, `watch`, JSON output
- [ ] Video and stream support with frame sampling
- [ ] Model cache and integrity verification
- [ ] Benchmarks and evaluation set
- [ ] Optional GPU execution providers

## Status

Early. The README is the spec. Code follows.

## License

GPL-3.0. See [LICENSE](LICENSE).
