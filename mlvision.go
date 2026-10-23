// Package mlvision is offline age, gender and NSFW detection for images.
//
// One struct in, one struct out:
//
//	a, err := mlvision.New(ctx, mlvision.Options{})
//	v, err := a.AnalyzeFile(ctx, "photo.jpg")
//	if v.NSFW || v.MinorPresent { ... }
package mlvision

import (
	"context"
	"errors"
	"fmt"
	"image"

	"github.com/bosleon/MLVision/internal/face"
	"github.com/bosleon/MLVision/internal/nsfw"
	"github.com/bosleon/MLVision/internal/nudity"
	"github.com/bosleon/MLVision/internal/vision"
	"github.com/bosleon/MLVision/models"
)

// Profile selects which models are used, by licence.
type Profile string

const (
	// Commercial uses only permissively licensed models (Apache, MIT, AGPL).
	Commercial Profile = "commercial"
	// Research adds InsightFace face and age models, which are licensed
	// for non-commercial research use only. Better age accuracy.
	Research Profile = "research"
)

// Thresholds tune the decision. Zero values take the defaults.
type Thresholds struct {
	NSFW     float32 // classifier score at or above which an image is NSFW on its own (0.8)
	Safe     float32 // classifier score below which the part detector is skipped (0.2)
	Part     float32 // minimum detector score for an exposed part to count (0.5)
	MinorAge int     // ages below this mark a face as a minor (18)
}

// Options configure an Analyzer. The zero value is a sensible default.
type Options struct {
	Profile    Profile
	ModelDir   string // model cache; defaults to the user cache directory
	Threads    int    // intra-op threads per model; 0 lets the runtime decide
	SkipFaces  bool   // skip face, age and gender estimation
	Thresholds Thresholds
}

// Box is a rectangle in image pixel coordinates.
type Box struct {
	X, Y, W, H int
}

// Part is a body-part region found by the nudity detector.
type Part struct {
	Label string  `json:"label"`
	Score float32 `json:"score"`
	Box   Box     `json:"box"`
}

// Face is one detected face with its estimated attributes.
type Face struct {
	Box         Box     `json:"box"`
	Score       float32 `json:"score"`
	Age         float32 `json:"age"`
	AgeRange    [2]int  `json:"age_range"`
	Gender      string  `json:"gender"`
	GenderScore float32 `json:"gender_score"`
	Minor       bool    `json:"minor"`
}

// Verdict is the result of analysing one image.
type Verdict struct {
	NSFW         bool     `json:"nsfw"`
	Score        float32  `json:"score"` // whole-image classifier probability
	Reasons      []string `json:"reasons,omitempty"`
	Parts        []Part   `json:"parts,omitempty"`
	Faces        []Face   `json:"faces,omitempty"`
	MinorPresent bool     `json:"minor_present"`
}

// Analyzer holds loaded models. It is safe for concurrent use.
type Analyzer struct {
	opts  Options
	cls   *nsfw.Classifier
	det   *nudity.Detector
	faces face.Detector
	attrs face.Estimator
}

func (t *Thresholds) fill() {
	if t.NSFW == 0 {
		t.NSFW = 0.8
	}
	if t.Safe == 0 {
		t.Safe = 0.2
	}
	if t.Part == 0 {
		t.Part = 0.5
	}
	if t.MinorAge == 0 {
		t.MinorAge = 18
	}
}

// New loads the models for the profile, downloading any that are missing.
func New(ctx context.Context, opts Options) (*Analyzer, error) {
	opts.Thresholds.fill()
	if opts.Profile == "" {
		opts.Profile = Commercial
	}
	dir, err := models.Dir(opts.ModelDir)
	if err != nil {
		return nil, err
	}
	get := func(m models.Model) (string, error) { return models.Ensure(ctx, dir, m) }

	a := &Analyzer{opts: opts}
	ok := false
	defer func() {
		if !ok {
			a.Close()
		}
	}()

	p, err := get(models.NSFWMarqo)
	if err != nil {
		return nil, err
	}
	if a.cls, err = nsfw.Load(p, nsfw.Marqo, opts.Threads); err != nil {
		return nil, err
	}
	if p, err = get(models.NudeNet); err != nil {
		return nil, err
	}
	if a.det, err = nudity.Load(p, opts.Threads); err != nil {
		return nil, err
	}
	if opts.SkipFaces {
		ok = true
		return a, nil
	}
	switch opts.Profile {
	case Commercial:
		if p, err = get(models.FaceUltra); err != nil {
			return nil, err
		}
		if a.faces, err = face.LoadUltraFace(p, opts.Threads); err != nil {
			return nil, err
		}
		ap, err := get(models.AgeFairFace)
		if err != nil {
			return nil, err
		}
		gp, err := get(models.GenderFairFace)
		if err != nil {
			return nil, err
		}
		if a.attrs, err = face.LoadFairFace(ap, gp, opts.Threads); err != nil {
			return nil, err
		}
	case Research:
		if p, err = get(models.FaceSCRFD); err != nil {
			return nil, err
		}
		if a.faces, err = face.LoadSCRFD(p, opts.Threads); err != nil {
			return nil, err
		}
		if p, err = get(models.GenderAgeInsight); err != nil {
			return nil, err
		}
		if a.attrs, err = face.LoadInsight(p, opts.Threads); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unknown profile %q", opts.Profile)
	}
	ok = true
	return a, nil
}

// Analyze runs every stage on img.
func (a *Analyzer) Analyze(ctx context.Context, img image.Image) (Verdict, error) {
	t := a.opts.Thresholds
	var v Verdict
	score, err := a.cls.Score(img)
	if err != nil {
		return v, err
	}
	v.Score = score
	if score >= t.NSFW {
		v.NSFW = true
		v.Reasons = append(v.Reasons, fmt.Sprintf("classifier: nsfw %.2f", score))
	}
	if err := ctx.Err(); err != nil {
		return v, err
	}
	if score >= t.Safe {
		parts, err := a.det.Detect(img)
		if err != nil {
			return v, err
		}
		for _, p := range parts {
			v.Parts = append(v.Parts, Part{Label: p.Label, Score: p.Score, Box: box(p.Box)})
			if nudity.Explicit[p.Label] && p.Score >= t.Part {
				v.NSFW = true
				v.Reasons = append(v.Reasons, fmt.Sprintf("detector: %s %.2f", p.Label, p.Score))
			}
		}
	}
	if a.faces == nil {
		return v, nil
	}
	if err := ctx.Err(); err != nil {
		return v, err
	}
	boxes, err := a.faces.Detect(img)
	if err != nil {
		return v, err
	}
	for _, b := range boxes {
		at, err := a.attrs.Estimate(img, b.Rect)
		if err != nil {
			return v, err
		}
		f := Face{Box: box(b.Rect), Score: b.Score, Age: at.Age, AgeRange: [2]int{at.AgeLow, at.AgeHigh},
			Gender: at.Gender, GenderScore: at.GenderScore, Minor: at.Age < float32(t.MinorAge)}
		if f.Minor {
			v.MinorPresent = true
			v.Reasons = append(v.Reasons, fmt.Sprintf("face: %s ~%.0f", f.Gender, f.Age))
		}
		v.Faces = append(v.Faces, f)
	}
	return v, nil
}

// AnalyzeFile decodes the image at path and analyses it.
func (a *Analyzer) AnalyzeFile(ctx context.Context, path string) (Verdict, error) {
	img, err := vision.DecodeFile(path)
	if err != nil {
		return Verdict{}, fmt.Errorf("%s: %w", path, err)
	}
	return a.Analyze(ctx, img)
}

// Close releases every loaded model.
func (a *Analyzer) Close() error {
	var errs []error
	for _, c := range []interface{ Close() error }{a.cls, a.det, a.faces, a.attrs} {
		if c != nil && !isNil(c) {
			errs = append(errs, c.Close())
		}
	}
	return errors.Join(errs...)
}

func isNil(c interface{ Close() error }) bool {
	switch x := c.(type) {
	case *nsfw.Classifier:
		return x == nil
	case *nudity.Detector:
		return x == nil
	}
	return false
}

func box(r image.Rectangle) Box { return Box{X: r.Min.X, Y: r.Min.Y, W: r.Dx(), H: r.Dy()} }
