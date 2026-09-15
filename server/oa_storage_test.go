package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	downloadserve "github.com/openabstractions/abstraction-download/go/serve"
	oaclient "github.com/openabstractions/abstraction-facade/go/client"
	oahost "github.com/openabstractions/abstraction-facade/go/runtime"
	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-identity/listen"
	storage "github.com/openabstractions/abstraction-storage/go"
)

type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// captureLog routes slog to a buffer for the test.
func captureLog(t *testing.T) *lockedBuffer {
	t.Helper()
	buf := &lockedBuffer{}
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return buf
}

// oaStorageService starts a real Go OA runtime with the job service and a
// content store holding seed, readable under policy.
func oaStorageService(t *testing.T, home string, seed map[string][]byte, policy func(context.Context, *identity.Peer, string) error) *oaFixture {
	t.Helper()
	store, err := storage.NewContentStore("oa-test-store", filepath.Join(home, "oa-store"))
	if err != nil {
		t.Fatal(err)
	}
	for digest, body := range seed {
		ref, err := store.Place(digest, int64(len(body)))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(store.Path(ref), body, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := store.Commit(ref); err != nil {
			t.Fatal(err)
		}
	}
	prefix := fmt.Sprintf("ollama-oa-st-%d-%d", os.Getpid(), time.Now().UnixNano())
	o := oahost.Options{
		Endpoint: listen.Endpoint(prefix), LogEndpoint: listen.Endpoint(prefix + "l"), ConfigEndpoint: listen.Endpoint(prefix + "c"),
		JobEndpoint: listen.Endpoint(prefix + "j"), JobRoot: filepath.Join(home, "oa-private"), JobOwner: "ollama-seam-owner",
		JobExecutor: downloadserve.HTTPExecution{OnError: func(err error) { t.Logf("OA execution: %v", err) }},
		Storage:     store, StoragePolicy: policy, StorageEndpoint: listen.Endpoint(prefix + "s"),
	}
	f := &oaFixture{options: o}
	f.start(t)
	oaMachine = func() *oaclient.Machine { return oaclient.NewUnverified(o.Endpoint) }
	return f
}

func allowContent(context.Context, *identity.Peer, string) error { return nil }

func assertNoStorageResidue(t *testing.T) {
	t.Helper()
	dir, _ := oaRequestsDir()
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".storage") {
			t.Fatalf("storage delivery residue %s", e.Name())
		}
	}
}

func TestOAPullReusesStoredBlobsWithoutSourceRequests(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaDownloadEnv, "service")
	r := newOARegistry(t)
	log := captureLog(t)
	f := oaStorageService(t, home, map[string][]byte{r.layerDigest: r.layer, r.configDig: r.config}, allowContent)

	pullOK(t, r)
	assertBlob(t, r.layerDigest, r.layer)
	assertBlob(t, r.configDig, r.config)
	if n := r.blobRequests.Load(); n != 0 {
		t.Fatalf("stored blobs were transferred again: %d blob requests", n)
	}
	if n := len(f.operations(t)); n != 0 {
		t.Fatalf("stored blobs submitted %d download operations", n)
	}
	if strings.Contains(log.String(), "OA storage did not supply") {
		t.Fatalf("reuse reported a miss:\n%s", log.String())
	}
	dir, _ := oaRequestsDir()
	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		t.Fatalf("records or residue after reuse: %v", entries)
	}
}

func TestOAPullStorageMissDownloadsWithTypedReason(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaDownloadEnv, "service")
	r := newOARegistry(t)
	log := captureLog(t)
	f := oaStorageService(t, home, map[string][]byte{r.configDig: r.config}, allowContent)

	pullOK(t, r)
	assertBlob(t, r.layerDigest, r.layer)
	assertBlob(t, r.configDig, r.config)
	if r.layerGets.Load() != 1 {
		t.Fatalf("missing layer: %d layer GETs, want 1 job transfer", r.layerGets.Load())
	}
	if n := len(f.operations(t)); n != 1 {
		t.Fatalf("operations %d, want 1 for the layer only", n)
	}
	text := log.String()
	if !strings.Contains(text, "reason="+oaReasonStorageNotFound) || !strings.Contains(text, r.layerDigest) {
		t.Fatalf("storage miss not reported with its reason:\n%s", text)
	}
	assertNoStorageResidue(t)
}

func TestOAPullStorageRefusalDownloadsWithTypedReason(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaDownloadEnv, "service")
	r := newOARegistry(t)
	log := captureLog(t)
	deny := func(context.Context, *identity.Peer, string) error { return errors.New("content denied") }
	f := oaStorageService(t, home, map[string][]byte{r.layerDigest: r.layer, r.configDig: r.config}, deny)

	pullOK(t, r)
	assertBlob(t, r.layerDigest, r.layer)
	if r.layerGets.Load() != 1 || len(f.operations(t)) != 2 {
		t.Fatalf("refused storage: layer GETs %d, operations %d", r.layerGets.Load(), len(f.operations(t)))
	}
	text := log.String()
	if strings.Count(text, "reason="+oaReasonStorageRefused) != 2 || !strings.Contains(text, "open: forbidden") || !strings.Contains(text, "level=WARN") {
		t.Fatalf("storage refusal not reported per blob:\n%s", text)
	}
	assertNoStorageResidue(t)
}

func TestOAPullStorageAbsentDownloadsWithTypedReason(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaDownloadEnv, "service")
	r := newOARegistry(t)
	log := captureLog(t)
	oaService(t, home, true) // no storage provider

	pullOK(t, r)
	assertBlob(t, r.layerDigest, r.layer)
	if r.layerGets.Load() != 1 {
		t.Fatalf("layer GETs %d", r.layerGets.Load())
	}
	if text := log.String(); !strings.Contains(text, "reason="+oaReasonStorageUnavailable) {
		t.Fatalf("absent storage not reported:\n%s", text)
	}
}

func TestOAPullStorageWrongBytesAreRejectedAndDownloaded(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaDownloadEnv, "service")
	r := newOARegistry(t)
	log := captureLog(t)
	tampered := append([]byte("tampered"), r.layer[8:]...)
	f := oaStorageService(t, home, map[string][]byte{r.layerDigest: tampered, r.configDig: r.config}, allowContent)

	pullOK(t, r)
	assertBlob(t, r.layerDigest, r.layer)
	if r.layerGets.Load() != 1 || len(f.operations(t)) != 1 {
		t.Fatalf("wrong stored bytes: layer GETs %d, operations %d", r.layerGets.Load(), len(f.operations(t)))
	}
	if text := log.String(); !strings.Contains(text, "reason="+oaReasonStorageDigestMismatch) {
		t.Fatalf("digest mismatch not reported:\n%s", text)
	}
	assertNoStorageResidue(t)
}
