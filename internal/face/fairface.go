package face

import (
	"image"

	"github.com/bosleon/MLVision/internal/ort"
	"github.com/bosleon/MLVision/internal/vision"
)

// FairFace pairs two Apache-licensed ViT classifiers trained on the FairFace
// dataset: nine age buckets and binary gender.
type FairFace struct {
	age, gender *ort.Model
}

var (
	fairAgeLow  = []int{0, 3, 10, 20, 30, 40, 50, 60, 70}
	fairAgeHigh = []int{2, 9, 19, 29, 39, 49, 59, 69, 100}
	fairAgeMid  = []float32{1, 6, 15, 25, 35, 45, 55, 65, 77}
)

// LoadFairFace opens the age and gender models.
func LoadFairFace(agePath, genderPath string, threads int) (*FairFace, error) {
	age, err := ort.Load(agePath, threads)
	if err != nil {
		return nil, err
	}
	gender, err := ort.Load(genderPath, threads)
	if err != nil {
		age.Close()
		return nil, err
	}
	return &FairFace{age: age, gender: gender}, nil
}

// Estimate predicts age and gender for the face in box. Age is the
// probability-weighted bucket midpoint; the range is the most likely bucket.
func (e *FairFace) Estimate(img image.Image, box image.Rectangle) (Attributes, error) {
	data, shape := vision.CHW(faceCrop(img, box, 224), vision.Half)
	in := ort.Tensor{Shape: shape, Data: data}
	ao, err := e.age.Run(in)
	if err != nil {
		return Attributes{}, err
	}
	go_, err := e.gender.Run(in)
	if err != nil {
		return Attributes{}, err
	}
	ap := vision.Softmax(ao[0].Data)
	var a Attributes
	best := 0
	for i, p := range ap {
		a.Age += p * fairAgeMid[i]
		if p > ap[best] {
			best = i
		}
	}
	a.AgeLow, a.AgeHigh = fairAgeLow[best], fairAgeHigh[best]
	gp := vision.Softmax(go_[0].Data)
	a.Gender, a.GenderScore = "female", gp[0]
	if gp[1] > gp[0] {
		a.Gender, a.GenderScore = "male", gp[1]
	}
	return a, nil
}

// Close releases both models.
func (e *FairFace) Close() error {
	err := e.age.Close()
	if err2 := e.gender.Close(); err == nil {
		err = err2
	}
	return err
}
