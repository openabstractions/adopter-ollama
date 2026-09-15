package manifest

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"
)

// stagingDir holds manifests being written, under the manifests root. Manifests
// enumerates exactly four levels below the root, so staged files at the second
// level are never read as models, and the rename stays on one filesystem.
const stagingDir = ".staging"

// staleStaging is the age after which a staged file can only be the residue of
// a write that never finished.
const staleStaging = time.Hour

// stageWrite writes the staged bytes; tests replace it to stop a write halfway.
var stageWrite = func(w io.Writer, data []byte) error {
	_, err := w.Write(data)
	return err
}

// WriteFile replaces the manifest at path with data. A concurrent reader, or a
// restart after the process was killed at any point, finds either the previous
// manifest or the complete new one: the bytes are written and synced to a staged
// file, which is then renamed over path.
func WriteFile(path string, data []byte) error {
	manifests, err := Path()
	if err != nil {
		return err
	}
	rel, err := filepath.Rel(manifests, path)
	if err != nil || rel == "." || strings.HasPrefix(rel, "..") || strings.HasPrefix(rel, stagingDir) {
		return fmt.Errorf("manifest path %s is not under %s", path, manifests)
	}
	staging := filepath.Join(manifests, stagingDir)
	if err := os.MkdirAll(staging, 0o755); err != nil {
		return err
	}
	removeStaleStaging(staging)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	f, err := os.CreateTemp(staging, "manifest-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	err = stageWrite(f, data)
	if err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err == nil {
		err = os.Chmod(tmp, 0o644)
	}
	if err == nil {
		err = replaceFile(tmp, path)
	}
	if err != nil {
		os.Remove(tmp)
		return err
	}
	syncDir(filepath.Dir(path))
	return nil
}

// replaceFile renames over an existing file. On Windows a reader holding the
// destination open makes the rename fail with a sharing error for as long as it
// holds it, so the rename is retried briefly.
func replaceFile(from, to string) error {
	var err error
	for attempt := 0; attempt < 50; attempt++ {
		if err = os.Rename(from, to); err == nil || !transientRename(err) {
			return err
		}
		time.Sleep(10 * time.Millisecond)
	}
	return err
}

func transientRename(err error) bool {
	if runtime.GOOS != "windows" {
		return false
	}
	var errno syscall.Errno
	// ERROR_ACCESS_DENIED (5) and ERROR_SHARING_VIOLATION (32).
	return errors.As(err, &errno) && (errno == 5 || errno == 32)
}

// syncDir makes the rename durable where the platform supports syncing a
// directory. Windows cannot open a directory for sync, and NTFS journals the rename.
func syncDir(dir string) {
	if runtime.GOOS == "windows" {
		return
	}
	if d, err := os.Open(dir); err == nil {
		d.Sync()
		d.Close()
	}
}

func removeStaleStaging(staging string) {
	entries, err := os.ReadDir(staging)
	if err != nil {
		return
	}
	for _, entry := range entries {
		info, err := entry.Info()
		if err == nil && !entry.IsDir() && time.Since(info.ModTime()) > staleStaging {
			os.Remove(filepath.Join(staging, entry.Name()))
		}
	}
}
