package vision

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestIoU(t *testing.T) {
	a := image.Rect(0, 0, 10, 10)
	if got := IoU(a, a); got != 1 {
		t.Fatalf("identical boxes: %v", got)
	}
	if got := IoU(a, image.Rect(20, 20, 30, 30)); got != 0 {
		t.Fatalf("disjoint boxes: %v", got)
	}
	if got := IoU(a, image.Rect(5, 0, 15, 10)); math.Abs(got-1.0/3) > 1e-9 {
		t.Fatalf("half overlap: %v", got)
	}
}

func TestNMS(t *testing.T) {
	boxes := []Box{
		{Rect: image.Rect(0, 0, 10, 10), Score: 0.5, Class: 0},
		{Rect: image.Rect(1, 1, 11, 11), Score: 0.9, Class: 0},
		{Rect: image.Rect(1, 1, 11, 11), Score: 0.8, Class: 1},
		{Rect: image.Rect(50, 50, 60, 60), Score: 0.3, Class: 0},
	}
	got := NMS(append([]Box(nil), boxes...), 0.5, false)
	if len(got) != 2 || got[0].Score != 0.9 || got[1].Score != 0.3 {
		t.Fatalf("class-agnostic: %+v", got)
	}
	got = NMS(append([]Box(nil), boxes...), 0.5, true)
	if len(got) != 3 || got[1].Class != 1 {
		t.Fatalf("per-class: %+v", got)
	}
}

func TestLetterbox(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 200, 100))
	for i := range src.Pix {
		src.Pix[i] = 255
	}
	dst, scale := Letterbox(src, 100, 100, color.Black)
	if scale != 0.5 {
		t.Fatalf("scale %v", scale)
	}
	if dst.RGBAAt(10, 10) != (color.RGBA{255, 255, 255, 255}) || dst.RGBAAt(10, 80) != (color.RGBA{0, 0, 0, 255}) {
		t.Fatal("content should sit top-left over black padding")
	}
}

func TestCHW(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.SetRGBA(0, 0, color.RGBA{255, 0, 0, 255})
	img.SetRGBA(1, 0, color.RGBA{0, 0, 255, 255})
	data, shape := CHW(img, Half)
	want := []float32{1, -1, -1, -1, -1, 1} // R plane, G plane, B plane
	if len(shape) != 4 || shape[1] != 3 || shape[2] != 1 || shape[3] != 2 {
		t.Fatalf("shape %v", shape)
	}
	for i := range want {
		if data[i] != want[i] {
			t.Fatalf("data %v want %v", data, want)
		}
	}
}

func TestSoftmax(t *testing.T) {
	p := Softmax([]float32{0, 0})
	if math.Abs(float64(p[0]-0.5)) > 1e-6 {
		t.Fatalf("%v", p)
	}
	p = Softmax([]float32{1000, 0})
	if p[0] != 1 || p[1] != 0 {
		t.Fatalf("overflow guard: %v", p)
	}
}

func TestSquareAround(t *testing.T) {
	r := SquareAround(image.Rect(10, 10, 30, 50), 1.5)
	if r.Dx() != 60 || r.Dy() != 60 || r.Min.X != -10 || r.Min.Y != 0 {
		t.Fatalf("%v", r)
	}
}
