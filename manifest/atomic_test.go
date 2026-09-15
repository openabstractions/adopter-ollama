package manifest

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/types/model"
)

const (
	writeHelperMode = "OLLAMA_MANIFEST_WRITE_HELPER"
	writeHelperPath = "OLLAMA_MANIFEST_WRITE_PATH"
	writeHelperData = "OLLAMA_MANIFEST_WRITE_DATA"
)

// TestManifestWriteHelper runs only as the child process of the kill tests. It
// writes half the manifest, reports it, and waits to be killed before the rest.
func TestManifestWriteHelper(t *testing.T) {
	mode := os.Getenv(writeHelperMode)
	if mode == "" {
		t.Skip("child process of the manifest kill tests")
	}
	data, err := os.ReadFile(os.Getenv(writeHelperData))
	if err != nil {
		t.Fatal(err)
	}
	halfway := func(w io.Writer, data []byte) error {
		half := len(data) / 2
		if _, err := w.Write(data[:half]); err != nil {
			return err
		}
		if f, ok := w.(*os.File); ok {
			f.Sync()
		}
		fmt.Println("HALFWAY")
		os.Stdout.Sync()
		time.Sleep(2 * time.Minute)
		_, err := w.Write(data[half:])
		return err
	}
	path := os.Getenv(writeHelperPath)
	switch mode {
	case "atomic":
		stageWrite = halfway
		err = WriteFile(path, data)
	case "direct":
		// Upstream's manifest write: create the final name and encode into it.
		var f *os.File
		if f, err = os.Create(path); err == nil {
			err = halfway(f, data)
			f.Close()
		}
	default:
		t.Fatalf("unknown helper mode %q", mode)
	}
	if err != nil {
		t.Fatal(err)
	}
}

// killDuringWrite starts the helper, waits until half the new manifest is on
// disk, and kills the process.
func killDuringWrite(t *testing.T, mode, path string, next []byte) {
	t.Helper()
	dataFile := filepath.Join(t.TempDir(), "next.json")
	if err := os.WriteFile(dataFile, next, 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestManifestWriteHelper$", "-test.count=1")
	cmd.Env = append(os.Environ(), writeHelperMode+"="+mode, writeHelperPath+"="+path, writeHelperData+"="+dataFile)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	reached := make(chan bool, 1)
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			if strings.TrimSpace(scanner.Text()) == "HALFWAY" {
				reached <- true
				io.Copy(io.Discard, stdout)
				return
			}
		}
		reached <- false
	}()
	select {
	case ok := <-reached:
		if !ok {
			cmd.Wait()
			t.Fatal("helper exited before writing half the manifest")
		}
	case <-time.After(30 * time.Second):
		cmd.Process.Kill()
		t.Fatal("helper did not reach the halfway point")
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	cmd.Wait()
}

func testManifests(t *testing.T) (model.Name, string, []byte, []byte) {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	name := model.ParseName("registry.ollama.ai/library/atomic-test:latest")
	path, err := PathForName(name)
	if err != nil {
		t.Fatal(err)
	}
	encode := func(n int) []byte {
		layers := make([]Layer, n)
		for i := range layers {
			layers[i] = Layer{MediaType: "application/vnd.ollama.image.model", Digest: fmt.Sprintf("sha256:%064x", i), Size: int64(i)}
		}
		var buf bytes.Buffer
		if err := json.NewEncoder(&buf).Encode(Manifest{SchemaVersion: 2, MediaType: "application/vnd.docker.distribution.manifest.v2+json",
			Config: Layer{MediaType: "application/vnd.docker.container.image.v1+json", Digest: fmt.Sprintf("sha256:%064x", 999999), Size: 2}, Layers: layers}); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}
	previous, next := encode(2), encode(2000)
	if err := WriteFile(path, previous); err != nil {
		t.Fatal(err)
	}
	return name, path, previous, next
}

func TestManifestKilledDuringAtomicWriteKeepsThePreviousManifest(t *testing.T) {
	name, path, previous, next := testManifests(t)
	for i := 0; i < 5; i++ {
		killDuringWrite(t, "atomic", path, next)
		got, err := os.ReadFile(path)
		if err != nil || !bytes.Equal(got, previous) {
			t.Fatalf("kill %d: manifest changed (%v, %d bytes, want the previous %d bytes)", i, err, len(got), len(previous))
		}
		if _, err := ParseNamedManifest(name); err != nil {
			t.Fatalf("kill %d: manifest no longer parses: %v", i, err)
		}
		all, err := Manifests(false)
		if err != nil || len(all) != 1 {
			t.Fatalf("kill %d: enumeration %d manifests, %v", i, len(all), err)
		}
	}
	manifests, _ := Path()
	staged, _ := os.ReadDir(filepath.Join(manifests, stagingDir))
	if len(staged) != 5 {
		t.Fatalf("each killed write leaves its staged residue outside enumeration; found %d", len(staged))
	}
	if err := WriteFile(path, next); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(path); !bytes.Equal(got, next) {
		t.Fatal("a completed write did not replace the manifest")
	}
	if all, err := Manifests(false); err != nil || len(all) != 1 {
		t.Fatalf("enumeration after completed write: %d, %v", len(all), err)
	}
}

// The control: the same kill during upstream's direct write tears the manifest.
func TestManifestKilledDuringDirectWriteIsTorn(t *testing.T) {
	name, path, _, next := testManifests(t)
	killDuringWrite(t, "direct", path, next)
	got, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(got, next[:len(next)/2]) {
		t.Fatalf("direct write: %v, %d bytes", err, len(got))
	}
	if _, err := ParseNamedManifest(name); err == nil {
		t.Fatal("a torn manifest parsed")
	}
	if _, err := Manifests(false); err == nil {
		t.Fatal("enumeration accepted a torn manifest")
	}
}

func TestWriteFileRefusesPathsOutsideManifests(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	manifests, err := Path()
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{filepath.Join(t.TempDir(), "elsewhere"), manifests, filepath.Join(manifests, stagingDir, "x")} {
		if err := WriteFile(path, []byte("{}")); err == nil {
			t.Fatalf("wrote %s", path)
		}
	}
}

func TestWriteFileRemovesStaleStagedFiles(t *testing.T) {
	_, path, previous, _ := testManifests(t)
	manifests, _ := Path()
	stale := filepath.Join(manifests, stagingDir, "manifest-stale")
	fresh := filepath.Join(manifests, stagingDir, "manifest-fresh")
	for _, p := range []string{stale, fresh} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	old := time.Now().Add(-2 * staleStaging)
	if err := os.Chtimes(stale, old, old); err != nil {
		t.Fatal(err)
	}
	if err := WriteFile(path, previous); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("stale staged file kept: %v", err)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Fatalf("fresh staged file removed: %v", err)
	}
}
