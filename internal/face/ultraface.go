package face

import (
	"image"

	"github.com/bosleon/MLVision/internal/ort"
	"github.com/bosleon/MLVision/internal/vision"
)

// UltraFace is the 1 MB RFB-320 detector (MIT licence).
type UltraFace struct{ m *ort.Model }

const (
	ultraW, ultraH = 320, 240
	ultraScore     = 0.7
	ultraIoU       = 0.3
)

// LoadUltraFace opens the model at path.
func LoadUltraFace(path string, threads int) (*UltraFace, error) {
	m, err := ort.Load(path, threads)
	if err != nil {
		return nil, err
	}
	return &UltraFace{m: m}, nil
}

// Detect returns face boxes with score ≥ 0.7.
func (d *UltraFace) Detect(img image.Image) ([]vision.Box, error) {
	data, shape := vision.CHW(vision.Resize(img, ultraW, ultraH), vision.Norm{Mean: [3]float32{127, 127, 127}, Std: [3]float32{128, 128, 128}})
	out, err := d.m.Run(ort.Tensor{Shape: shape, Data: data})
	if err != nil {
		return nil, err
	}
	scores, rects := out[0].Data, out[1].Data // [1,N,2] and [1,N,4] normalised x1,y1,x2,y2
	b := img.Bounds()
	w, h := float64(b.Dx()), float64(b.Dy())
	var boxes []vision.Box
	for i := 0; i < len(scores)/2; i++ {
		s := scores[2*i+1]
		if s < ultraScore {
			continue
		}
		r := rects[4*i : 4*i+4]
		rect := vision.RectF(float64(r[0])*w, float64(r[1])*h, float64(r[2])*w, float64(r[3])*h).Add(b.Min)
		boxes = append(boxes, vision.Box{Rect: rect.Intersect(b), Score: s})
	}
	return vision.NMS(boxes, ultraIoU, false), nil
}

// Close releases the model.
func (d *UltraFace) Close() error { return d.m.Close() }
