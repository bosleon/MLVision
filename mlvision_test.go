package mlvision

import (
	"context"
	"image"
	"image/color"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/bosleon/MLVision/models"
)

// newAnalyzer skips the test when the profile's models are not cached, so
// `go test` works on a machine that has never downloaded anything.
func newAnalyzer(t *testing.T, opts Options) *Analyzer {
	t.Helper()
	dir, err := models.Dir(opts.ModelDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []models.Model{models.NSFWMarqo, models.NudeNet, models.FaceUltra, models.AgeFairFace, models.GenderFairFace} {
		if _, err := os.Stat(models.Path(dir, m)); err != nil {
			t.Skipf("model %s not cached in %s", m.Name, dir)
		}
	}
	a, err := New(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { a.Close() })
	return a
}

func noise(w, h int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	r := rand.New(rand.NewSource(1))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetRGBA(x, y, color.RGBA{uint8(r.Intn(256)), uint8(r.Intn(256)), uint8(r.Intn(256)), 255})
		}
	}
	return img
}

func TestNoiseIsSafe(t *testing.T) {
	a := newAnalyzer(t, Options{Thresholds: Thresholds{Safe: 0.001}}) // force the detector to run
	v, err := a.Analyze(context.Background(), noise(640, 480))
	if err != nil {
		t.Fatal(err)
	}
	if v.NSFW || v.MinorPresent || len(v.Faces) != 0 {
		t.Fatalf("noise should be safe and faceless: %+v", v)
	}
	for _, p := range v.Parts {
		if p.Score >= a.opts.Thresholds.Part {
			t.Fatalf("confident part on noise: %+v", p)
		}
	}
}

func TestDefaults(t *testing.T) {
	var th Thresholds
	th.fill()
	if th.NSFW != 0.8 || th.Safe != 0.2 || th.Part != 0.5 || th.MinorAge != 18 || th.MinFace != 24 {
		t.Fatalf("%+v", th)
	}
}

// TestSamples runs every image in MLVISION_TEST_IMAGES and logs the verdicts
// with the part detector forced on. It is a manual aid, not an assertion.
func TestSamples(t *testing.T) {
	dir := os.Getenv("MLVISION_TEST_IMAGES")
	if dir == "" {
		t.Skip("MLVISION_TEST_IMAGES not set")
	}
	a := newAnalyzer(t, Options{Thresholds: Thresholds{Safe: 0.001}})
	paths, _ := filepath.Glob(filepath.Join(dir, "*.jpg"))
	for _, p := range paths {
		v, err := a.AnalyzeFile(context.Background(), p)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: nsfw=%v score=%.2f parts=%d faces=%d", filepath.Base(p), v.NSFW, v.Score, len(v.Parts), len(v.Faces))
		for _, part := range v.Parts {
			t.Logf("   %-26s %.2f %+v", part.Label, part.Score, part.Box)
		}
	}
}
