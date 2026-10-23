// Package ort is a thin wrapper over ONNX Runtime: one function to locate and
// load the shared library, one type to run a model on float32 tensors.
package ort

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// EnvLibrary names the environment variable that overrides the shared
// library location.
const EnvLibrary = "MLVISION_ORT_LIB"

var (
	initOnce sync.Once
	initErr  error
)

// Init loads the ONNX Runtime shared library. An empty path triggers
// discovery via MLVISION_ORT_LIB and a short list of conventional locations.
// It is safe to call repeatedly.
func Init(path string) error {
	initOnce.Do(func() {
		if path == "" {
			path = os.Getenv(EnvLibrary)
		}
		if path == "" {
			path = discover()
		}
		if path == "" {
			initErr = errors.New("onnxruntime shared library not found; install it (brew install onnxruntime / apt install libonnxruntime) or set " + EnvLibrary)
			return
		}
		ort.SetSharedLibraryPath(path)
		if err := ort.InitializeEnvironment(); err != nil {
			initErr = fmt.Errorf("load onnxruntime from %s: %w", path, err)
		}
	})
	return initErr
}

func discover() string {
	var names []string
	switch runtime.GOOS {
	case "darwin":
		names = []string{"libonnxruntime.dylib"}
	case "windows":
		names = []string{"onnxruntime.dll"}
	default:
		names = []string{"libonnxruntime.so", "libonnxruntime.so.1"}
	}
	dirs := []string{"/opt/homebrew/lib", "/usr/local/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"}
	if exe, err := os.Executable(); err == nil {
		dirs = append([]string{filepath.Dir(exe)}, dirs...)
	}
	for _, d := range dirs {
		for _, n := range names {
			p := filepath.Join(d, n)
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	return ""
}

// IO describes one model input or output.
type IO struct {
	Name  string
	Shape []int64
	Type  string
}

// Inspect returns a model's inputs and outputs without creating a session.
func Inspect(path string) (inputs, outputs []IO, err error) {
	if err := Init(""); err != nil {
		return nil, nil, err
	}
	in, out, err := ort.GetInputOutputInfo(path)
	if err != nil {
		return nil, nil, err
	}
	conv := func(xs []ort.InputOutputInfo) []IO {
		r := make([]IO, len(xs))
		for i, x := range xs {
			r[i] = IO{Name: x.Name, Shape: []int64(x.Dimensions), Type: x.DataType.String()}
		}
		return r
	}
	return conv(in), conv(out), nil
}

// Model is a loaded ONNX model with float32 inputs and outputs. Run is safe
// for concurrent use.
type Model struct {
	sess    *ort.DynamicAdvancedSession
	Inputs  []IO
	Outputs []IO
}

// Load creates a session for the model at path using the given number of
// intra-op threads (0 lets the runtime decide).
func Load(path string, threads int) (*Model, error) {
	if err := Init(""); err != nil {
		return nil, err
	}
	in, out, err := Inspect(path)
	if err != nil {
		return nil, fmt.Errorf("inspect %s: %w", path, err)
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()
	if threads > 0 {
		if err := opts.SetIntraOpNumThreads(threads); err != nil {
			return nil, err
		}
	}
	if err := opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		return nil, err
	}
	names := func(xs []IO) []string {
		r := make([]string, len(xs))
		for i, x := range xs {
			r[i] = x.Name
		}
		return r
	}
	sess, err := ort.NewDynamicAdvancedSession(path, names(in), names(out), opts)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", filepath.Base(path), err)
	}
	return &Model{sess: sess, Inputs: in, Outputs: out}, nil
}

// Tensor is a dense float32 tensor in row-major order.
type Tensor struct {
	Shape []int64
	Data  []float32
}

// Run feeds the inputs (in model input order) and returns every output.
func (m *Model) Run(inputs ...Tensor) ([]Tensor, error) {
	if len(inputs) != len(m.Inputs) {
		return nil, fmt.Errorf("model expects %d inputs, got %d", len(m.Inputs), len(inputs))
	}
	ins := make([]ort.Value, len(inputs))
	for i, t := range inputs {
		v, err := ort.NewTensor(ort.Shape(t.Shape), t.Data)
		if err != nil {
			return nil, err
		}
		defer v.Destroy()
		ins[i] = v
	}
	outs := make([]ort.Value, len(m.Outputs))
	if err := m.sess.Run(ins, outs); err != nil {
		return nil, err
	}
	res := make([]Tensor, len(outs))
	for i, v := range outs {
		defer v.Destroy()
		t, ok := v.(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("output %s is not float32", m.Outputs[i].Name)
		}
		data := make([]float32, len(t.GetData()))
		copy(data, t.GetData())
		res[i] = Tensor{Shape: []int64(t.GetShape()), Data: data}
	}
	return res, nil
}

// Close releases the session.
func (m *Model) Close() error {
	if m.sess == nil {
		return nil
	}
	err := m.sess.Destroy()
	m.sess = nil
	return err
}
