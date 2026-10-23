// Package nudity localises body parts with the NudeNet YOLOv8 detector so a
// verdict can say what was seen and where.
package nudity

import (
	"image"
	"image/color"

	"github.com/bosleon/MLVision/internal/ort"
	"github.com/bosleon/MLVision/internal/vision"
)

// Labels are NudeNet's 18 classes, in output order.
var Labels = []string{
	"FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
	"FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED",
	"BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE",
	"BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
	"BUTTOCKS_COVERED",
}

// Explicit lists the classes that make an image NSFW on their own.
var Explicit = map[string]bool{
	"BUTTOCKS_EXPOSED": true, "FEMALE_BREAST_EXPOSED": true, "FEMALE_GENITALIA_EXPOSED": true,
	"ANUS_EXPOSED": true, "MALE_GENITALIA_EXPOSED": true,
}

const (
	size      = 320
	minScore  = 0.25
	nmsIoU    = 0.45
	numLabels = 18
)

// Part is one detected region.
type Part struct {
	Label string
	Score float32
	Box   image.Rectangle
}

// Detector wraps the NudeNet model.
type Detector struct{ m *ort.Model }

// Load opens the model at path.
func Load(path string, threads int) (*Detector, error) {
	m, err := ort.Load(path, threads)
	if err != nil {
		return nil, err
	}
	return &Detector{m: m}, nil
}

// Detect returns every body part found with score ≥ 0.25 after NMS.
func (d *Detector) Detect(img image.Image) ([]Part, error) {
	canvas, scale := vision.Letterbox(img, size, size, color.Black)
	data, shape := vision.CHW(canvas, vision.Unit)
	out, err := d.m.Run(ort.Tensor{Shape: shape, Data: data})
	if err != nil {
		return nil, err
	}
	// output0 is [1, 4+classes, anchors]: rows are cx, cy, w, h, then class scores.
	o := out[0]
	rows, anchors := int(o.Shape[1]), int(o.Shape[2])
	at := func(r, a int) float32 { return o.Data[r*anchors+a] }
	var boxes []vision.Box
	for a := 0; a < anchors; a++ {
		best, cls := float32(0), 0
		for c := 0; c < rows-4 && c < numLabels; c++ {
			if s := at(4+c, a); s > best {
				best, cls = s, c
			}
		}
		if best < minScore {
			continue
		}
		cx, cy, w, h := float64(at(0, a)), float64(at(1, a)), float64(at(2, a)), float64(at(3, a))
		r := vision.RectF((cx-w/2)/scale, (cy-h/2)/scale, (cx+w/2)/scale, (cy+h/2)/scale)
		boxes = append(boxes, vision.Box{Rect: r.Intersect(img.Bounds()), Score: best, Class: cls})
	}
	var parts []Part
	for _, b := range vision.NMS(boxes, nmsIoU, false) {
		parts = append(parts, Part{Label: Labels[b.Class], Score: b.Score, Box: b.Rect})
	}
	return parts, nil
}

// Close releases the model.
func (d *Detector) Close() error { return d.m.Close() }
