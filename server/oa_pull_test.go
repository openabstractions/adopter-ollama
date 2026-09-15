package server

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
	downloadserve "github.com/openabstractions/abstraction-download/go/serve"
	oaclient "github.com/openabstractions/abstraction-facade/go/client"
	oahost "github.com/openabstractions/abstraction-facade/go/runtime"
	"github.com/openabstractions/abstraction-identity/listen"
	job "github.com/openabstractions/abstraction-job/go"
)

// oaRegistry is a local registry serving one GGUF layer and one config blob.
// Like the public registry, blob URLs redirect to a separate content host.
type oaRegistry struct {
	server      *httptest.Server
	cdn         string
	name        string
	layer       []byte
	layerDigest string
	config      []byte
	configDig   string
	layerGets   atomic.Int32 // layer byte transfers from any client
	blobHeads   atomic.Int32
	ollamaGets  atomic.Int32 // registry blob requests from Ollama's own downloader
	started     chan struct{}
	startOnce   sync.Once
	release     chan struct{} // nil serves immediately
	corrupt     atomic.Bool
	missing     atomic.Bool
}

func newOARegistry(t *testing.T) *oaRegistry {
	t.Helper()
	r := &oaRegistry{layer: make([]byte, 300*1024), config: []byte(`{"model_format":"gguf"}`), started: make(chan struct{})}
	for i := range r.layer {
		r.layer[i] = byte(i * 7)
	}
	r.layerDigest = fmt.Sprintf("sha256:%x", sha256.Sum256(r.layer))
	r.configDig = fmt.Sprintf("sha256:%x", sha256.Sum256(r.config))
	r.server = httptest.NewServer(http.HandlerFunc(r.serve))
	t.Cleanup(r.server.Close)
	u, err := url.Parse(r.server.URL)
	if err != nil {
		t.Fatal(err)
	}
	r.cdn = "http://localhost:" + u.Port()
	n := model.ParseName(u.Host + "/library/oa-seam")
	n.ProtocolScheme = "http"
	r.name = n.String()
	return r
}

func (r *oaRegistry) serve(w http.ResponseWriter, req *http.Request) {
	switch {
	case strings.Contains(req.URL.Path, "/manifests/"):
		w.Header().Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
		fmt.Fprintf(w, `{"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json",
			"config":{"mediaType":"application/vnd.docker.container.image.v1+json","digest":%q,"size":%d},
			"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":%q,"size":%d}]}`,
			r.configDig, len(r.config), r.layerDigest, len(r.layer))
	case strings.HasPrefix(req.URL.Path, "/v2/") && strings.Contains(req.URL.Path, "/blobs/"):
		// Ollama's downloader always resolves blobs here with its User-Agent,
		// before fetching the redirected content without one.
		if strings.HasPrefix(req.Header.Get("User-Agent"), "ollama/") {
			r.ollamaGets.Add(1)
		}
		digest := req.URL.Path[strings.LastIndex(req.URL.Path, "/")+1:]
		http.Redirect(w, req, r.cdn+"/cdn/"+digest, http.StatusTemporaryRedirect)
	case strings.HasPrefix(req.URL.Path, "/cdn/"):
		if req.Method == http.MethodHead {
			r.blobHeads.Add(1)
		}
		body := r.config
		if strings.HasSuffix(req.URL.Path, r.layerDigest) && r.missing.Load() {
			http.NotFound(w, req)
			return
		}
		if strings.HasSuffix(req.URL.Path, r.layerDigest) {
			body = r.layer
			if req.Method == http.MethodGet {
				r.layerGets.Add(1)
				r.startOnce.Do(func() { close(r.started) })
				if r.release != nil {
					select {
					case <-r.release:
					case <-req.Context().Done():
						return
					}
				}
			}
		}
		if r.corrupt.Load() {
			body = append([]byte("tampered"), body[8:]...)
		}
		w.Header().Set("Content-Length", fmt.Sprint(len(body)))
		if req.Method == http.MethodGet {
			w.Write(body)
		}
	default:
		http.NotFound(w, req)
	}
}

type oaFixture struct {
	options oahost.Options
	stop    func()
}

// setupOA isolates Ollama's models directory and the OA runtime's user state.
func setupOA(t *testing.T) string {
	t.Helper()
	if runtime.GOOS == "darwin" {
		t.Skip("OA Program-bound local calls remain unproven on Darwin")
	}
	home := t.TempDir()
	setTestHome(t, home)
	for _, key := range []string{"APPDATA", "XDG_CONFIG_HOME"} {
		t.Setenv(key, home)
	}
	t.Setenv("ProgramData", filepath.Join(home, "machine"))
	for _, key := range []string{"ABSTRACTION_STORE", "ABSTRACTION_NAS_STORE", "ABSTRACTION_LOG", "ABSTRACTION_LOG_SERVICE"} {
		t.Setenv(key, "")
	}
	t.Setenv("OLLAMA_MODELS", filepath.Join(home, "models"))
	envconfig.ReloadServerConfig()
	previous, interval := oaMachine, oaPollInterval
	oaPollInterval = 10 * time.Millisecond
	t.Cleanup(func() { oaMachine, oaPollInterval = previous, interval })
	return home
}

// oaService starts a real Go OA runtime with the durable job service and HTTP
// download execution. jobs=false starts the resolver without a job service.
func oaService(t *testing.T, home string, jobs bool) *oaFixture {
	t.Helper()
	prefix := fmt.Sprintf("ollama-oa-%d-%d", os.Getpid(), time.Now().UnixNano())
	o := oahost.Options{Endpoint: listen.Endpoint(prefix), LogEndpoint: listen.Endpoint(prefix + "l"), ConfigEndpoint: listen.Endpoint(prefix + "c")}
	if jobs {
		o.JobEndpoint = listen.Endpoint(prefix + "j")
		o.JobRoot = filepath.Join(home, "oa-private")
		o.JobOwner = "ollama-seam-owner"
		o.JobExecutor = downloadserve.HTTPExecution{OnError: func(err error) { t.Logf("OA execution: %v", err) }}
	}
	f := &oaFixture{options: o}
	f.start(t)
	oaMachine = func() *oaclient.Machine { return oaclient.NewUnverified(o.Endpoint) }
	return f
}

func (f *oaFixture) start(t *testing.T) {
	t.Helper()
	h, err := oahost.Listen(f.options)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- h.Serve(ctx) }()
	var once sync.Once
	f.stop = func() {
		once.Do(func() {
			cancel()
			h.Close()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Error("OA runtime cleanup timed out")
			}
		})
	}
	t.Cleanup(f.stop)
}

func (f *oaFixture) operations(t *testing.T) []*job.Record {
	t.Helper()
	// Private store inspection is fixture evidence; the seam only uses the SDK.
	store, err := job.NewFileStore(f.options.JobRoot)
	if err != nil {
		t.Fatal(err)
	}
	all, err := store.List()
	if err != nil {
		t.Fatal(err)
	}
	return all
}

func pullOA(ctx context.Context, r *oaRegistry, opts *registryOptions, fn func(api.ProgressResponse)) error {
	if opts == nil {
		opts = &registryOptions{}
	}
	if fn == nil {
		fn = func(api.ProgressResponse) {}
	}
	opts.Insecure = true
	ctx, cancel := context.WithTimeout(ctx, 40*time.Second)
	defer cancel()
	return PullModel(ctx, r.name, opts, fn)
}

// pullOK accepts success. On Windows it also accepts upstream's manifest write
// failure for a registry name with a port ("host:port" is not a valid directory
// name); blobs are committed before that step and are asserted separately.
func pullOK(t *testing.T, r *oaRegistry) {
	t.Helper()
	err := pullOA(t.Context(), r, nil, nil)
	if err == nil {
		return
	}
	if runtime.GOOS == "windows" && strings.Contains(err.Error(), filepath.Join("models", "manifests")) {
		t.Logf("upstream Windows manifest path limitation after blob commit: %v", err)
		return
	}
	t.Fatalf("pull: %v (layer GETs %d)", err, r.layerGets.Load())
}

func assertBlob(t *testing.T, digest string, want []byte) {
	t.Helper()
	fp, err := manifest.BlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(fp)
	if err != nil || string(got) != string(want) {
		t.Fatalf("blob %s: %v, %d bytes", digest, err, len(got))
	}
}

func assertNoBlob(t *testing.T, digest string) {
	t.Helper()
	fp, err := manifest.BlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(fp); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("blob %s committed: %v", digest, err)
	}
}

func oaReason(t *testing.T, err error, reason string) *OAError {
	t.Helper()
	var typed *OAError
	if !errors.As(err, &typed) || typed.Reason != reason {
		t.Fatalf("want OA %s, got %v", reason, err)
	}
	return typed
}

func TestOAPullServicePresent(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaDownloadEnv, "service")
	r := newOARegistry(t)
	f := oaService(t, home, true)

	pullOK(t, r)
	assertBlob(t, r.layerDigest, r.layer)
	assertBlob(t, r.configDig, r.config)
	if r.ollamaGets.Load() != 0 || r.blobHeads.Load() != 0 || r.layerGets.Load() != 1 {
		t.Fatalf("transfer owner: ollama GETs %d, HEADs %d, layer GETs %d", r.ollamaGets.Load(), r.blobHeads.Load(), r.layerGets.Load())
	}
	if n := len(f.operations(t)); n != 2 {
		t.Fatalf("operations: %d", n)
	}
	dir, _ := oaRequestsDir()
	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		t.Fatalf("recovery records left after commit: %v", entries)
	}
	if runtime.GOOS != "windows" {
		if _, err := manifest.ParseNamedManifest(model.ParseName(r.name)); err != nil {
			t.Fatalf("manifest: %v", err)
		}
	}
}

func TestOAPullServiceAbsentIsTyped(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaDownloadEnv, "service")
	r := newOARegistry(t)

	oaService(t, home, false) // resolver present, job service absent
	err := pullOA(t.Context(), r, nil, nil)
	oaReason(t, err, oaReasonUnavailable)
	var binding *oaclient.BindingError
	if !errors.As(err, &binding) || binding.Status != "unavailable" {
		t.Fatalf("binding refusal: %v", err)
	}

	oaMachine = func() *oaclient.Machine { return oaclient.NewUnverified(listen.Endpoint("ollama-oa-no-runtime")) }
	oaReason(t, pullOA(t.Context(), r, nil, nil), oaReasonUnavailable)

	if r.layerGets.Load() != 0 || r.ollamaGets.Load() != 0 || r.blobHeads.Load() != 0 {
		t.Fatalf("fallback transfer: layer GETs %d, ollama GETs %d, HEADs %d", r.layerGets.Load(), r.ollamaGets.Load(), r.blobHeads.Load())
	}
	assertNoBlob(t, r.layerDigest)
}

func TestOAPullDisabledKeepsUpstream(t *testing.T) {
	setupOA(t)
	t.Setenv(oaDownloadEnv, "")
	r := newOARegistry(t)
	oaMachine = func() *oaclient.Machine {
		t.Error("OA resolved while disabled")
		return oaclient.NewUnverified("")
	}
	pullOK(t, r)
	assertBlob(t, r.layerDigest, r.layer)
	if r.ollamaGets.Load() == 0 {
		t.Fatal("upstream downloader did not fetch")
	}
	if _, err := os.Stat(filepath.Join(envconfig.Models(), "oa-requests")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("OA state created while disabled: %v", err)
	}
}

func TestOAPullInvalidSwitchIsTyped(t *testing.T) {
	setupOA(t)
	t.Setenv(oaDownloadEnv, "yes")
	r := newOARegistry(t)
	oaReason(t, pullOA(t.Context(), r, nil, nil), oaReasonInvalidConfiguration)
	if r.layerGets.Load() != 0 {
		t.Fatal("blob fetched with invalid switch")
	}
}

func TestOAPullCredentialsRefused(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaDownloadEnv, "service")
	r := newOARegistry(t)
	f := oaService(t, home, true)
	oaReason(t, pullOA(t.Context(), r, &registryOptions{Token: "registry-token"}, nil), oaReasonCredentialsRequired)
	if r.layerGets.Load() != 0 || len(f.operations(t)) != 0 {
		t.Fatalf("credentialed pull reached the service: GETs %d", r.layerGets.Load())
	}
	assertNoBlob(t, r.layerDigest)
}

// A digest mismatch ends the operation with a typed cause [JOB-A8]. Nothing is
// committed, and the next pull retries as attempt 1 of the same key [JOB-A7].
func TestOAPullCorruptSourceFailsTypedThenRetriesAttempt(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaDownloadEnv, "service")
	r := newOARegistry(t)
	r.corrupt.Store(true)
	f := oaService(t, home, true)

	err := pullOA(t.Context(), r, nil, nil)
	typed := oaReason(t, err, oaReasonDigestMismatch)
	if typed.Cause != "digest_mismatch" || !errors.Is(err, errDigestMismatch) {
		t.Fatalf("corrupt source: %+v", typed)
	}
	assertNoBlob(t, r.layerDigest)
	dir, _ := oaRequestsDir()
	record, err := oaLoadRecord(oaRecordPath(dir, r.layerDigest))
	if err != nil || record == nil || !record.Terminal || record.Identity.Attempt != 0 {
		t.Fatalf("terminal record: %+v %v", record, err)
	}

	r.corrupt.Store(false)
	pullOK(t, r)
	assertBlob(t, r.layerDigest, r.layer)
	if n := len(f.operations(t)); n != 3 {
		t.Fatalf("operations: %d (want failed attempt 0, attempt 1, config)", n)
	}
}

// A blob the registry does not have ends typed as not_found, with no retry loop.
func TestOAPullMissingBlobIsTyped(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaDownloadEnv, "service")
	r := newOARegistry(t)
	r.missing.Store(true)
	oaService(t, home, true)
	typed := oaReason(t, pullOA(t.Context(), r, nil, nil), oaReasonNotFound)
	if typed.Cause != "not_found" {
		t.Fatalf("missing blob: %+v", typed)
	}
	assertNoBlob(t, r.layerDigest)
}

// Restoring a saved binding against a service with another logical owner is
// refused, so accepted work is never silently resubmitted elsewhere.
func TestOAPullRestoreRefusesDifferentOwner(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaDownloadEnv, "service")
	r := newOARegistry(t)
	r.release = make(chan struct{})
	f := oaService(t, home, true)

	ctx, cancel := context.WithCancel(t.Context())
	first := make(chan error, 1)
	go func() { first <- pullOA(ctx, r, nil, nil) }()
	select {
	case <-r.started:
	case <-time.After(15 * time.Second):
		t.Fatal("service never started the transfer")
	}
	cancel()
	<-first
	f.stop()
	close(r.release)
	f.options.JobRoot = filepath.Join(home, "other-private")
	f.options.JobOwner = "another-owner"
	f.start(t)

	oaReason(t, pullOA(t.Context(), r, nil, nil), oaReasonProviderChanged)
	assertNoBlob(t, r.layerDigest)
	if n := len(f.operations(t)); n != 0 {
		t.Fatalf("work resubmitted to another owner: %d", n)
	}
}

// An interrupted pull followed by a new pull, as after an Ollama restart,
// reconciles the accepted operation and never starts a second transfer.
func TestOAPullRestartReconcilesWithoutSecondTransfer(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaDownloadEnv, "service")
	r := newOARegistry(t)
	r.release = make(chan struct{})
	f := oaService(t, home, true)

	ctx, cancel := context.WithCancel(t.Context())
	first := make(chan error, 1)
	go func() { first <- pullOA(ctx, r, nil, nil) }()
	select {
	case <-r.started:
	case <-time.After(15 * time.Second):
		t.Fatal("service never started the transfer")
	}
	cancel()
	if err := <-first; !errors.Is(err, context.Canceled) {
		t.Fatalf("interrupted pull: %v", err)
	}
	dir, _ := oaRequestsDir()
	record, err := oaLoadRecord(oaRecordPath(dir, r.layerDigest))
	if err != nil || record == nil || record.OperationID == "" {
		t.Fatalf("recovery record: %+v %v", record, err)
	}
	close(r.release)

	pullOK(t, r)
	assertBlob(t, r.layerDigest, r.layer)
	if r.layerGets.Load() != 1 || r.ollamaGets.Load() != 0 {
		t.Fatalf("second transfer: layer GETs %d, ollama GETs %d", r.layerGets.Load(), r.ollamaGets.Load())
	}
	all := f.operations(t)
	var reconciled int
	for _, op := range all {
		if op.ID == record.OperationID {
			reconciled++
		}
	}
	if len(all) != 2 || reconciled != 1 {
		t.Fatalf("operations: %d total, reconciled %d", len(all), reconciled)
	}
}

// The OA runtime restarting under an interrupted pull keeps one operation.
// The service resumes its own attempt; Ollama never submits a second request.
func TestOAPullServiceRestartKeepsOneOperation(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaDownloadEnv, "service")
	r := newOARegistry(t)
	r.release = make(chan struct{})
	f := oaService(t, home, true)

	ctx, cancel := context.WithCancel(t.Context())
	first := make(chan error, 1)
	go func() { first <- pullOA(ctx, r, nil, nil) }()
	select {
	case <-r.started:
	case <-time.After(15 * time.Second):
		t.Fatal("service never started the transfer")
	}
	cancel()
	<-first
	f.stop()
	close(r.release)
	f.start(t)

	dir, _ := oaRequestsDir()
	record, err := oaLoadRecord(oaRecordPath(dir, r.layerDigest))
	if err != nil || record == nil || record.OperationID == "" {
		t.Fatalf("recovery record: %+v %v", record, err)
	}
	pullOK(t, r)
	assertBlob(t, r.layerDigest, r.layer)
	all := f.operations(t)
	var reconciled int
	for _, op := range all {
		if op.ID == record.OperationID {
			reconciled++
		}
	}
	if len(all) != 2 || reconciled != 1 || r.ollamaGets.Load() != 0 {
		t.Fatalf("operations after runtime restart: %d total, reconciled %d, ollama GETs %d", len(all), reconciled, r.ollamaGets.Load())
	}
	t.Logf("service layer GETs across its own restart: %d", r.layerGets.Load())
}
