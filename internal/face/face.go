// Package face detects faces and estimates age and gender for each one.
// Two detectors exist, one permissively licensed and one research-only;
// attributes always come from the FairFace classifiers because their bucket
// probabilities expose uncertainty, which a bare regressor cannot.
package face

import (
	"image"

	"github.com/bosleon/MLVision/internal/vision"
)

// Detector finds faces.
type Detector interface {
	Detect(img image.Image) ([]vision.Box, error)
	Close() error
}

// Attributes are the per-face estimates.
type Attributes struct {
	Age         float32 // point estimate in years
	AgeLow      int     // range the estimate falls in
	AgeHigh     int
	Gender      string // "female" or "male"
	GenderScore float32
}

// Estimator predicts Attributes for one face box.
type Estimator interface {
	Estimate(img image.Image, box image.Rectangle) (Attributes, error)
	Close() error
}

// cropMargin is how much context to keep around a face box before
// estimating attributes; 1.5 matches the FairFace recipe.
const cropMargin = 1.5

func faceCrop(img image.Image, box image.Rectangle, size int) *image.RGBA {
	return vision.Resize(vision.Crop(img, vision.SquareAround(box, cropMargin)), size, size)
}
