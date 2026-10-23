// Command mlvision is a thin CLI over the mlvision library.
//
//	mlvision scan [-json] [-profile commercial|research] [-no-faces] <image|dir>...
//	mlvision models [list|pull|verify] [-profile ...]
//	mlvision inspect <model.onnx>...
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	mlvision "github.com/bosleon/MLVision"
	"github.com/bosleon/MLVision/internal/ort"
	"github.com/bosleon/MLVision/models"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	var err error
	switch os.Args[1] {
	case "scan":
		err = scan(os.Args[2:])
	case "models":
		err = modelsCmd(os.Args[2:])
	case "inspect":
		err = inspect(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "mlvision:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage:\n  mlvision scan [-json] [-profile commercial|research] [-no-faces] [-models dir] <image|dir>...\n  mlvision models [list|pull|verify] [-profile ...] [-models dir]\n  mlvision inspect <model.onnx>...")
	os.Exit(2)
}

func commonFlags(set *flag.FlagSet, opts *mlvision.Options) {
	set.Func("profile", "commercial (default) or research", func(s string) error {
		opts.Profile = mlvision.Profile(s)
		return nil
	})
	set.StringVar(&opts.ModelDir, "models", "", "model cache directory")
	set.IntVar(&opts.Threads, "threads", 0, "intra-op threads per model")
}

var imageExt = map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".gif": true, ".webp": true, ".bmp": true}

func scan(args []string) error {
	var opts mlvision.Options
	set := flag.NewFlagSet("scan", flag.ExitOnError)
	commonFlags(set, &opts)
	asJSON := set.Bool("json", false, "one JSON object per line")
	set.BoolVar(&opts.SkipFaces, "no-faces", false, "skip face, age and gender")
	set.IntVar(&opts.Thresholds.MinorAge, "minor-age", 18, "age below which a face counts as a minor")
	set.Parse(args)
	if set.NArg() == 0 {
		usage()
	}

	ctx := context.Background()
	a, err := mlvision.New(ctx, opts)
	if err != nil {
		return err
	}
	defer a.Close()

	enc := json.NewEncoder(os.Stdout)
	var failed int
	for _, root := range set.Args() {
		err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() || !imageExt[strings.ToLower(filepath.Ext(path))] {
				return nil
			}
			v, err := a.AnalyzeFile(ctx, path)
			if err != nil {
				failed++
				fmt.Fprintln(os.Stderr, "mlvision:", err)
				return nil
			}
			if *asJSON {
				return enc.Encode(struct {
					Path string `json:"path"`
					mlvision.Verdict
				}{path, v})
			}
			fmt.Println(format(path, v))
			return nil
		})
		if err != nil {
			return err
		}
	}
	if failed > 0 {
		return fmt.Errorf("%d file(s) could not be analysed", failed)
	}
	return nil
}

func format(path string, v mlvision.Verdict) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%-40s ", path)
	if v.NSFW {
		fmt.Fprintf(&b, "NSFW %.2f", v.Score)
	} else {
		fmt.Fprintf(&b, "safe %.2f", v.Score)
	}
	if len(v.Faces) > 0 {
		var list []string
		for _, f := range v.Faces {
			s := fmt.Sprintf("%s %.0f", strings.ToUpper(f.Gender[:1]), f.Age)
			if f.Minor {
				s += " minor"
			}
			list = append(list, s)
		}
		fmt.Fprintf(&b, "  faces: %s", strings.Join(list, ", "))
	}
	if len(v.Reasons) > 0 {
		fmt.Fprintf(&b, "  [%s]", strings.Join(v.Reasons, "; "))
	}
	return b.String()
}

func profileModels(p mlvision.Profile) []models.Model {
	base := []models.Model{models.NSFWMarqo, models.NudeNet}
	switch p {
	case mlvision.Research:
		return append(base, models.FaceSCRFD, models.GenderAgeInsight)
	default:
		return append(base, models.FaceUltra, models.AgeFairFace, models.GenderFairFace)
	}
}

func modelsCmd(args []string) error {
	sub := "list"
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		sub, args = args[0], args[1:]
	}
	var opts mlvision.Options
	set := flag.NewFlagSet("models", flag.ExitOnError)
	commonFlags(set, &opts)
	set.Parse(args)
	dir, err := models.Dir(opts.ModelDir)
	if err != nil {
		return err
	}
	if opts.Profile == "" {
		opts.Profile = mlvision.Commercial
	}
	switch sub {
	case "list":
		fmt.Println("cache:", dir)
		for _, m := range models.All {
			state := "missing"
			if _, err := os.Stat(models.Path(dir, m)); err == nil {
				state = "present"
			}
			fmt.Printf("  %-32s %7.1f MB  %-8s %-30s %s\n", m.Name, float64(m.Size)/1e6, state, m.License, m.Source)
		}
	case "pull":
		for _, m := range profileModels(opts.Profile) {
			fmt.Printf("  %s ... ", m.Name)
			if _, err := models.Ensure(context.Background(), dir, m); err != nil {
				fmt.Println("FAILED")
				return err
			}
			fmt.Println("ok")
		}
	case "verify":
		for _, m := range profileModels(opts.Profile) {
			fmt.Printf("  %s ... ", m.Name)
			if err := models.Verify(dir, m); err != nil {
				fmt.Println("FAILED")
				return err
			}
			fmt.Println("ok")
		}
	default:
		usage()
	}
	return nil
}

func inspect(paths []string) error {
	for _, p := range paths {
		in, out, err := ort.Inspect(p)
		if err != nil {
			return err
		}
		fmt.Println(p)
		for _, x := range in {
			fmt.Printf("  in  %-16s %-8s %v\n", x.Name, strings.TrimPrefix(x.Type, "ONNX_TENSOR_ELEMENT_DATA_TYPE_"), x.Shape)
		}
		for _, x := range out {
			fmt.Printf("  out %-16s %-8s %v\n", x.Name, strings.TrimPrefix(x.Type, "ONNX_TENSOR_ELEMENT_DATA_TYPE_"), x.Shape)
		}
	}
	return nil
}
