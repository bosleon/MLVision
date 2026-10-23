package face

import (
	"image"
	"math"

	"github.com/bosleon/MLVision/internal/ort"
	"github.com/bosleon/MLVision/internal/vision"
)

// Insight is the InsightFace genderage model (research licence): one 96×96
// crop in, [female, male, age/100] out.
type Insight struct{ m *ort.Model }

// LoadInsight opens the model at path.
func LoadInsight(path string, threads int) (*Insight, error) {
	m, err := ort.Load(path, threads)
	if err != nil {
		return nil, err
	}
	return &Insight{m: m}, nil
}

// Estimate predicts age and gender for the face in box.
func (e *Insight) Estimate(img image.Image, box image.Rectangle) (Attributes, error) {
	data, shape := vision.CHW(faceCrop(img, box, 96), vision.Norm{Std: [3]float32{1, 1, 1}})
	out, err := e.m.Run(ort.Tensor{Shape: shape, Data: data})
	if err != nil {
		return Attributes{}, err
	}
	o := out[0].Data
	g := vision.Softmax([]float32{o[0], o[1]})
	a := Attributes{Age: float32(math.Round(float64(o[2] * 100))), Gender: "female", GenderScore: g[0]}
	if g[1] > g[0] {
		a.Gender, a.GenderScore = "male", g[1]
	}
	// The regressor is typically within ±5 years.
	a.AgeLow, a.AgeHigh = max(0, int(a.Age)-5), int(a.Age)+5
	return a, nil
}

// Close releases the model.
func (e *Insight) Close() error { return e.m.Close() }
