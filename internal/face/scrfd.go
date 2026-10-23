package face

import (
	"image"
	"image/color"
	"image/draw"

	"github.com/bosleon/MLVision/internal/ort"
	"github.com/bosleon/MLVision/internal/vision"
)

// SCRFD is the InsightFace det_10g detector (research licence).
type SCRFD struct{ m *ort.Model }

const (
	scrfdSize    = 640
	scrfdScore   = 0.5
	scrfdIoU     = 0.4
	scrfdAnchors = 2
)

var scrfdStrides = []int{8, 16, 32}

// LoadSCRFD opens the model at path.
func LoadSCRFD(path string, threads int) (*SCRFD, error) {
	m, err := ort.Load(path, threads)
	if err != nil {
		return nil, err
	}
	return &SCRFD{m: m}, nil
}

// Detect returns face boxes with score ≥ 0.5. The model is trained on faces
// well below its 640 px input, so a portrait that fills the frame is missed
// at full size; when nothing is found the image is retried at half size.
func (d *SCRFD) Detect(img image.Image) ([]vision.Box, error) {
	for _, fit := range []int{scrfdSize, scrfdSize / 2} {
		boxes, err := d.detect(img, fit)
		if err != nil || len(boxes) > 0 {
			return boxes, err
		}
	}
	return nil, nil
}

func (d *SCRFD) detect(img image.Image, fit int) ([]vision.Box, error) {
	small, scale := vision.Letterbox(img, fit, fit, color.Black)
	canvas := small
	if fit != scrfdSize {
		canvas = image.NewRGBA(image.Rect(0, 0, scrfdSize, scrfdSize))
		draw.Draw(canvas, small.Rect, small, image.Point{}, draw.Src)
	}
	data, shape := vision.CHW(canvas, vision.Norm{Mean: [3]float32{127.5, 127.5, 127.5}, Std: [3]float32{128, 128, 128}})
	out, err := d.m.Run(ort.Tensor{Shape: shape, Data: data})
	if err != nil {
		return nil, err
	}
	// Outputs: scores for each stride, then box distances for each stride
	// (keypoints follow and are unused). Anchor order is row-major over the
	// stride grid with scrfdAnchors consecutive anchors per cell.
	var boxes []vision.Box
	for si, stride := range scrfdStrides {
		scores, dists := out[si].Data, out[si+len(scrfdStrides)].Data
		cells := scrfdSize / stride
		for i, s := range scores {
			if s < scrfdScore {
				continue
			}
			cell := i / scrfdAnchors
			cx := float64((cell%cells)*stride) / scale
			cy := float64((cell/cells)*stride) / scale
			d := dists[4*i : 4*i+4]
			k := float64(stride) / scale
			r := vision.RectF(cx-float64(d[0])*k, cy-float64(d[1])*k, cx+float64(d[2])*k, cy+float64(d[3])*k)
			boxes = append(boxes, vision.Box{Rect: r.Add(img.Bounds().Min).Intersect(img.Bounds()), Score: s})
		}
	}
	return vision.NMS(boxes, scrfdIoU, false), nil
}

// Close releases the model.
func (d *SCRFD) Close() error { return d.m.Close() }
