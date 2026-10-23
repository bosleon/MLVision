// Package nsfw runs a whole-image NSFW classifier.
package nsfw

import (
	"image"

	"github.com/bosleon/MLVision/internal/ort"
	"github.com/bosleon/MLVision/internal/vision"
)

// Spec adapts a classifier checkpoint: input size, normalisation and which
// logit means "nsfw".
type Spec struct {
	Size  int
	Norm  vision.Norm
	Index int // index of the nsfw logit
}

// Marqo is Marqo/nsfw-image-detection-384 (ViT-tiny, labels [NSFW, SFW]).
var Marqo = Spec{Size: 384, Norm: vision.Half, Index: 0}

// AdamCodd is AdamCodd/vit-base-nsfw-detector (ViT-base, labels [sfw, nsfw]).
var AdamCodd = Spec{Size: 384, Norm: vision.Half, Index: 1}

// Classifier scores images for explicit content.
type Classifier struct {
	m    *ort.Model
	spec Spec
}

// Load opens the model at path.
func Load(path string, spec Spec, threads int) (*Classifier, error) {
	m, err := ort.Load(path, threads)
	if err != nil {
		return nil, err
	}
	return &Classifier{m: m, spec: spec}, nil
}

// Score returns the probability that img is NSFW.
func (c *Classifier) Score(img image.Image) (float32, error) {
	data, shape := vision.CHW(vision.Resize(img, c.spec.Size, c.spec.Size), c.spec.Norm)
	out, err := c.m.Run(ort.Tensor{Shape: shape, Data: data})
	if err != nil {
		return 0, err
	}
	return vision.Softmax(out[0].Data)[c.spec.Index], nil
}

// Close releases the model.
func (c *Classifier) Close() error { return c.m.Close() }
