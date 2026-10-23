// Package models is the registry of model files MLVision can use and the
// local cache that holds them. Every entry is pinned by SHA-256, downloaded
// once and never touched by the inference path again.
package models

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// EnvDir overrides the cache directory.
const EnvDir = "MLVISION_MODEL_DIR"

// ReleaseURL is where pinned model files are hosted.
const ReleaseURL = "https://github.com/bosleon/MLVision/releases/download/models-v1/"

// Model is one pinned model file.
type Model struct {
	Name    string // stable identifier, also the file name without .onnx
	SHA256  string
	Size    int64
	Source  string // upstream project
	License string
}

// File is the model's file name inside the cache.
func (m Model) File() string { return m.Name + ".onnx" }

// URL is where the model is downloaded from.
func (m Model) URL() string { return ReleaseURL + m.File() }

// The registry.
var (
	NSFWMarqo = Model{Name: "nsfw-marqo-vit-tiny-384", SHA256: "792208ea335d37531427a426075976d7d0a76fdbf80d5a6b09983d4baf10acdb", Size: 22520605,
		Source: "Marqo/nsfw-image-detection-384", License: "Apache-2.0"}
	NudeNet = Model{Name: "nudenet-yolov8n-320", SHA256: "c15d8273adad2d0a92f014cc69ab2d6c311a06777a55545f2c4eb46f51911f0f", Size: 12150158,
		Source: "notAI-tech/NudeNet", License: "AGPL-3.0"}
	FaceUltra = Model{Name: "face-ultraface-rfb-320", SHA256: "34cd7e60aeff28744c657de7a3dc64e872d506741de66987f3426f2b79f88017", Size: 1270727,
		Source: "Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB", License: "MIT"}
	AgeFairFace = Model{Name: "age-fairface-vit-224-int8", SHA256: "8078fc62da3e58051b35b7167e505728cca226c4a91c57b2912d1b3d9889353e", Size: 86920807,
		Source: "dima806/fairface_age_image_detection", License: "Apache-2.0"}
	GenderFairFace = Model{Name: "gender-fairface-vit-224-int8", SHA256: "bff1bc0ddd9fd714df604af4fbd1decce7f2d072703b395acac1c536e6a5ff5a", Size: 86915402,
		Source: "dima806/fairface_gender_image_detection", License: "Apache-2.0"}
	FaceSCRFD = Model{Name: "face-scrfd-10g", SHA256: "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91", Size: 16923827,
		Source: "deepinsight/insightface buffalo_l", License: "non-commercial research only"}
)

// All lists every registered model.
var All = []Model{NSFWMarqo, NudeNet, FaceUltra, AgeFairFace, GenderFairFace, FaceSCRFD}

// Dir returns the cache directory: override, then MLVISION_MODEL_DIR, then
// the user cache directory.
func Dir(override string) (string, error) {
	if override != "" {
		return override, nil
	}
	if d := os.Getenv(EnvDir); d != "" {
		return d, nil
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "mlvision", "models"), nil
}

// Path returns where the model lives in dir without checking for it.
func Path(dir string, m Model) string { return filepath.Join(dir, m.File()) }

// Ensure returns the model's path, downloading and verifying it first if it
// is not already in dir.
func Ensure(ctx context.Context, dir string, m Model) (string, error) {
	p := Path(dir, m)
	if _, err := os.Stat(p); err == nil {
		return p, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, m.URL(), nil)
	if err != nil {
		return "", err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", m.Name, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download %s: %s", m.Name, resp.Status)
	}
	tmp, err := os.CreateTemp(dir, m.Name+".*.part")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())
	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(tmp, h), resp.Body); err != nil {
		tmp.Close()
		return "", err
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != m.SHA256 {
		return "", fmt.Errorf("%s: checksum mismatch (got %s)", m.Name, got)
	}
	return p, os.Rename(tmp.Name(), p)
}

// Verify checks that the file in dir matches the pinned hash.
func Verify(dir string, m Model) error {
	f, err := os.Open(Path(dir, m))
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != m.SHA256 {
		return errors.New(m.Name + ": checksum mismatch")
	}
	return nil
}
