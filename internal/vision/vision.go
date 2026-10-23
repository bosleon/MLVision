// Package vision holds the small amount of image plumbing every model needs:
// decoding, resizing, cropping, tensor packing and non-maximum suppression.
package vision

import (
	"image"
	"image/color"
	"image/draw"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"
	"sort"

	_ "golang.org/x/image/bmp"
	xdraw "golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
)

// Decode reads a JPEG, PNG, GIF, WebP or BMP image.
func Decode(r io.Reader) (image.Image, error) {
	img, _, err := image.Decode(r)
	return img, err
}

// DecodeFile reads an image from disk.
func DecodeFile(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Decode(f)
}

// Resize stretches img to w×h using bilinear interpolation.
func Resize(img image.Image, w, h int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	xdraw.BiLinear.Scale(dst, dst.Bounds(), img, img.Bounds(), xdraw.Src, nil)
	return dst
}

// Letterbox scales img to fit inside w×h, anchored at the top-left and padded
// with pad. It returns the canvas and the scale applied to the source.
func Letterbox(img image.Image, w, h int, pad color.Color) (*image.RGBA, float64) {
	b := img.Bounds()
	scale := math.Min(float64(w)/float64(b.Dx()), float64(h)/float64(b.Dy()))
	nw := int(math.Round(float64(b.Dx()) * scale))
	nh := int(math.Round(float64(b.Dy()) * scale))
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(dst, dst.Bounds(), image.NewUniform(pad), image.Point{}, draw.Src)
	xdraw.BiLinear.Scale(dst, image.Rect(0, 0, nw, nh), img, b, xdraw.Src, nil)
	return dst, scale
}

// Crop returns the part of img inside r, clamped to the image bounds.
func Crop(img image.Image, r image.Rectangle) *image.RGBA {
	r = r.Intersect(img.Bounds())
	dst := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(dst, dst.Bounds(), img, r.Min, draw.Src)
	return dst
}

// SquareAround returns a square of side scale×max(w,h) centred on r.
func SquareAround(r image.Rectangle, scale float64) image.Rectangle {
	side := int(math.Round(scale * float64(max(r.Dx(), r.Dy()))))
	cx := (r.Min.X + r.Max.X) / 2
	cy := (r.Min.Y + r.Max.Y) / 2
	return image.Rect(cx-side/2, cy-side/2, cx-side/2+side, cy-side/2+side)
}

// Norm describes a per-channel affine normalisation: (x/255 - Mean) / Std
// when Scaled is true, or (x - Mean) / Std on raw 0..255 values otherwise.
type Norm struct {
	Mean, Std [3]float32
	Scaled    bool
}

// Unit maps 0..255 to 0..1.
var Unit = Norm{Std: [3]float32{1, 1, 1}, Scaled: true}

// Half maps 0..255 to -1..1, the convention of ViT and SigLIP checkpoints.
var Half = Norm{Mean: [3]float32{.5, .5, .5}, Std: [3]float32{.5, .5, .5}, Scaled: true}

// CHW packs an RGBA image into a [1,3,H,W] float32 tensor.
func CHW(img *image.RGBA, n Norm) ([]float32, []int64) {
	w, h := img.Rect.Dx(), img.Rect.Dy()
	out := make([]float32, 3*w*h)
	plane := w * h
	for y := 0; y < h; y++ {
		row := img.Pix[y*img.Stride : y*img.Stride+4*w]
		for x := 0; x < w; x++ {
			for c := 0; c < 3; c++ {
				v := float32(row[4*x+c])
				if n.Scaled {
					v /= 255
				}
				out[c*plane+y*w+x] = (v - n.Mean[c]) / n.Std[c]
			}
		}
	}
	return out, []int64{1, 3, int64(h), int64(w)}
}

// Softmax converts logits to probabilities in place and returns them.
func Softmax(x []float32) []float32 {
	m := x[0]
	for _, v := range x {
		m = max(m, v)
	}
	var sum float32
	for i, v := range x {
		x[i] = float32(math.Exp(float64(v - m)))
		sum += x[i]
	}
	for i := range x {
		x[i] /= sum
	}
	return x
}

// Box is a scored, labelled axis-aligned rectangle in pixel coordinates.
type Box struct {
	Rect  image.Rectangle
	Score float32
	Class int
}

// IoU is the intersection-over-union of two rectangles.
func IoU(a, b image.Rectangle) float64 {
	inter := a.Intersect(b)
	ia := float64(inter.Dx() * inter.Dy())
	if inter.Empty() {
		ia = 0
	}
	ua := float64(a.Dx()*a.Dy()+b.Dx()*b.Dy()) - ia
	if ua <= 0 {
		return 0
	}
	return ia / ua
}

// NMS performs greedy non-maximum suppression and returns the survivors in
// descending score order. When perClass is set, boxes of different classes
// never suppress each other.
func NMS(boxes []Box, iou float64, perClass bool) []Box {
	sort.SliceStable(boxes, func(i, j int) bool { return boxes[i].Score > boxes[j].Score })
	var keep []Box
	for _, b := range boxes {
		ok := true
		for _, k := range keep {
			if (!perClass || k.Class == b.Class) && IoU(k.Rect, b.Rect) > iou {
				ok = false
				break
			}
		}
		if ok {
			keep = append(keep, b)
		}
	}
	return keep
}

// RectF builds a rectangle from float coordinates, rounding outward.
func RectF(x0, y0, x1, y1 float64) image.Rectangle {
	return image.Rect(int(math.Floor(x0)), int(math.Floor(y0)), int(math.Ceil(x1)), int(math.Ceil(y1)))
}
