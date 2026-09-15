package server

// OpenAbstractions job/download seam. This fork patch routes GGUF blob pulls
// through the resolved OA durable job service when OLLAMA_OA_DOWNLOAD=service.
// It is kept in this file so the upstream diff stays at two call sites in
// images.go. Unset keeps upstream behavior. Any other value is an error.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	request "github.com/openabstractions/abstraction-download/go/abstraction/download/request"
	oaclient "github.com/openabstractions/abstraction-facade/go/client"
	acceptance "github.com/openabstractions/abstraction-job/go/abstraction/job/acceptance"
)

const oaDownloadEnv = "OLLAMA_OA_DOWNLOAD"

// OAError is the typed, visible failure of the OA download seam. Reason is one
// of the oaReason* words; Detail carries the service's own reason text.
type OAError struct {
	Reason string
	Digest string
	Detail string
	Err    error
}

const (
	oaReasonInvalidConfiguration = "invalid_configuration"
	oaReasonUnavailable          = "unavailable"
	oaReasonUnknown              = "unknown"
	oaReasonRefused              = "refused"
	oaReasonProviderChanged      = "provider_changed"
	oaReasonCredentialsRequired  = "credentials_required"
	oaReasonUnsupported          = "unsupported"
	oaReasonFailed               = "failed"
	oaReasonResultUnavailable    = "result_unavailable"
	oaReasonDigestMismatch       = "digest_mismatch"
)

func (e *OAError) Error() string {
	var b strings.Builder
	b.WriteString("openabstractions download (" + oaDownloadEnv + "=service)")
	if e.Digest != "" {
		b.WriteString(" " + e.Digest)
	}
	b.WriteString(": " + e.Reason)
	if e.Detail != "" {
		b.WriteString(": " + e.Detail)
	}
	if e.Err != nil {
		b.WriteString(": " + e.Err.Error())
	}
	return b.String()
}

func (e *OAError) Unwrap() error { return e.Err }

// oaMachine selects the installed runtime with verified trust. Tests replace it
// with an isolated fixture binding.
var oaMachine = oaclient.Discover

var oaPollInterval = 250 * time.Millisecond

func oaDownloadEnabled() (bool, error) {
	switch v := os.Getenv(oaDownloadEnv); v {
	case "":
		return false, nil
	case "service":
		return true, nil
	default:
		return false, &OAError{Reason: oaReasonInvalidConfiguration, Detail: fmt.Sprintf("%s=%q; supported values are unset and \"service\"", oaDownloadEnv, v)}
	}
}

// oaCheckTransfer refuses tensor-layer pulls when adoption is enabled, because
// they use Ollama's own transfer client, which this seam does not cover.
func oaCheckTransfer() error {
	enabled, err := oaDownloadEnabled()
	if err != nil || !enabled {
		return err
	}
	return &OAError{Reason: oaReasonUnsupported, Detail: "tensor-layer models use Ollama's transfer client; the OA seam covers GGUF blob layers only"}
}

// pullBlob is the images.go call site. Disabled adoption calls upstream downloadBlob.
func pullBlob(ctx context.Context, size int64, opts downloadOpts) (bool, error) {
	enabled, err := oaDownloadEnabled()
	if err != nil {
		return false, err
	}
	if !enabled {
		return downloadBlob(ctx, opts)
	}
	return oaDownloadBlob(ctx, size, opts)
}

// oaRecord is the caller recovery record, persisted before Submit.
type oaRecord struct {
	Format       string   `json:"format"`
	Digest       string   `json:"digest"`
	Attempt      int      `json:"attempt"`
	Endpoint     string   `json:"endpoint"`
	LogicalOwner string   `json:"logical_owner"`
	Key          string   `json:"key"`
	HistoryEpoch string   `json:"history_epoch"`
	Kind         string   `json:"kind"`
	Request      []byte   `json:"request"`
	Guarantees   []string `json:"required_guarantees"`
	OperationID  string   `json:"operation_id,omitempty"`
}

const oaRecordFormat = "ollama-oa-download/1"

func (r *oaRecord) identity() acceptance.RequestIdentity {
	return acceptance.RequestIdentity{Key: r.Key, HistoryEpoch: r.HistoryEpoch}
}

// oaRequestsDir sits beside blobs/ so PruneLayers never removes records.
func oaRequestsDir() (string, error) {
	dir := filepath.Join(envconfig.Models(), "oa-requests")
	return dir, os.MkdirAll(dir, 0o755)
}

func oaRecordPath(dir, digest string) string {
	return filepath.Join(dir, strings.ReplaceAll(digest, ":", "-")+".json")
}

func oaLoadRecord(path string) (*oaRecord, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	var r oaRecord
	// A record without an endpoint only names the next attempt after terminal work.
	if err := json.Unmarshal(data, &r); err != nil || r.Format != oaRecordFormat || r.Attempt < 0 || r.Endpoint != "" && (r.Key == "" || r.HistoryEpoch == "") {
		return nil, fmt.Errorf("unreadable OA request record %s: %v", path, err)
	}
	return &r, nil
}

// oaNextAttempt retires a terminal identity so the next pull submits a new one.
func oaNextAttempt(path string, r *oaRecord) error {
	return oaSaveRecord(path, &oaRecord{Format: oaRecordFormat, Digest: r.Digest, Attempt: r.Attempt + 1})
}

func oaSaveRecord(path string, r *oaRecord) error {
	data, err := json.Marshal(r)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func oaKey(digest string, attempt int) string {
	key := "ollama-blob-" + strings.ReplaceAll(digest, ":", "-")
	if attempt > 0 {
		key += fmt.Sprintf("-attempt-%d", attempt)
	}
	return key
}

func oaDownloadBlob(ctx context.Context, size int64, opts downloadOpts) (bool, error) {
	digest := opts.digest
	fail := func(reason, detail string, err error) (bool, error) {
		return false, &OAError{Reason: reason, Digest: digest, Detail: detail, Err: err}
	}
	if digest == "" {
		return false, fmt.Errorf("%s: %s", opts.n.DisplayNamespaceModel(), "digest is empty")
	}
	fp, err := manifest.BlobsPath(digest)
	if err != nil {
		return false, err
	}
	if fi, err := os.Stat(fp); err == nil {
		opts.fn(api.ProgressResponse{Status: fmt.Sprintf("pulling %s", digest[7:19]), Digest: digest, Total: fi.Size(), Completed: fi.Size()})
		return true, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return false, err
	}
	// Registry credentials cannot travel in the portable download request.
	if r := opts.regOpts; r != nil && (r.Token != "" || r.Username != "" || r.Password != "") {
		return fail(oaReasonCredentialsRequired, "this registry pull needs credentials and abstraction.download requests are anonymous", nil)
	}

	source := opts.n.BaseURL().JoinPath("v2", opts.n.DisplayNamespaceModel(), "blobs", digest)
	if opts.regOpts != nil && opts.regOpts.Insecure {
		source.Scheme = "http" // as makeRequest does for upstream requests
	}
	payload := request.Encode(&request.Request{
		Artifact: request.Artifact{Digest: digest, Size: size},
		Sources:  []request.Source{{Scheme: source.Scheme, Locator: source.String()}},
	})

	dir, err := oaRequestsDir()
	if err != nil {
		return false, err
	}
	recordPath := oaRecordPath(dir, digest)
	record, err := oaLoadRecord(recordPath)
	if err != nil {
		return false, err
	}

	jobs, err := oaMachine().ResolveJobOperations(ctx, oaclient.Requirements{Guarantees: acceptance.AdmissionGuarantees, Scope: "local"})
	if err != nil {
		return fail(oaReasonUnavailable, "job service did not resolve", err)
	}

	if record == nil || record.Endpoint == "" {
		attempt := 0
		if record != nil {
			attempt = record.Attempt
		}
		window, err := jobs.GetHistoryWindow(ctx)
		if err != nil {
			return fail(oaReasonUnavailable, "history window", err)
		}
		record = &oaRecord{
			Format: oaRecordFormat, Digest: digest, Attempt: attempt,
			Endpoint: jobs.Endpoint(), LogicalOwner: window.LogicalOwner,
			Key: oaKey(digest, attempt), HistoryEpoch: window.HistoryEpoch,
			Kind: "download", Request: payload, Guarantees: acceptance.AdmissionGuarantees,
		}
		if err := oaSaveRecord(recordPath, record); err != nil {
			return false, err
		}
	} else if record.Endpoint != jobs.Endpoint() {
		return fail(oaReasonProviderChanged, fmt.Sprintf("request was accepted by %s and the resolver selected %s", record.Endpoint, jobs.Endpoint()), nil)
	}

	id := record.identity()
	accepted, err := oaAdmit(ctx, jobs, record)
	if err != nil {
		var typed *OAError
		if errors.As(err, &typed) {
			typed.Digest = digest
		}
		return false, err
	}
	if accepted.Receipt.LogicalOwner != record.LogicalOwner {
		return fail(oaReasonProviderChanged, "logical owner changed", nil)
	}
	if record.OperationID != accepted.Receipt.OperationId {
		record.OperationID = accepted.Receipt.OperationId
		if err := oaSaveRecord(recordPath, record); err != nil {
			return false, err
		}
	}

	status := fmt.Sprintf("pulling %s", digest[7:19])
	var lastFailure string
	for {
		observed, err := jobs.ObserveWork(ctx, id)
		if err != nil {
			if ctx.Err() != nil {
				// A caller disconnect stops waiting only; accepted work continues.
				return false, ctx.Err()
			}
			return fail(oaReasonUnknown, "observe work", err)
		}
		if observed.Outcome != "observed" {
			return fail(oaReasonUnknown, "observe work outcome "+observed.Outcome, nil)
		}
		s := observed.Snapshot
		total := s.Progress.Total
		if total == 0 {
			total = size
		}
		current := status
		if s.Failure != nil && s.State != "failed" {
			// Nonterminal failure: the service retries. Show it; stop only on permanent.
			if s.Failure.Classification == "permanent" {
				return fail(oaReasonFailed, "permanent: "+s.Failure.Message, nil)
			}
			current = fmt.Sprintf("%s (OA service retrying: %s)", status, s.Failure.Message)
			if s.Failure.Message != lastFailure {
				slog.Warn("OA download retrying", "digest", digest, "state", s.State, "classification", s.Failure.Classification, "message", s.Failure.Message)
				lastFailure = s.Failure.Message
			}
		}
		opts.fn(api.ProgressResponse{Status: current, Digest: digest, Total: total, Completed: s.Progress.Done})
		if s.State == "complete" {
			break
		}
		if s.State == "failed" || s.State == "cancelled" {
			detail := s.State
			if s.Failure != nil {
				detail += " (" + s.Failure.Classification + "): " + s.Failure.Message
			}
			// The next pull submits a new attempt; this identity is terminal.
			if err := oaNextAttempt(recordPath, record); err != nil {
				return false, err
			}
			return fail(oaReasonFailed, detail, nil)
		}
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case <-time.After(oaPollInterval):
		}
	}

	tmpPath := strings.TrimSuffix(recordPath, ".json") + ".delivery"
	written, sum, err := oaCopy(ctx, jobs, id, tmpPath)
	if err != nil {
		os.Remove(tmpPath)
		if ctx.Err() != nil {
			return false, ctx.Err()
		}
		var service *acceptance.ServiceError
		if errors.As(err, &service) && service.Code != "invalid_result" {
			if saveErr := oaNextAttempt(recordPath, record); saveErr != nil {
				return false, saveErr
			}
			return fail(oaReasonResultUnavailable, "", err)
		}
		return false, err
	}
	if got := "sha256:" + sum; got != digest || (size > 0 && written != size) {
		os.Remove(tmpPath)
		return fail(oaReasonDigestMismatch, fmt.Sprintf("delivered %s, %d bytes", got, written), errDigestMismatch)
	}
	if err := os.Rename(tmpPath, fp); err != nil {
		os.Remove(tmpPath)
		return false, err
	}
	if err := os.Remove(recordPath); err != nil {
		slog.Warn("couldn't remove OA request record", "path", recordPath, "error", err)
	}
	opts.fn(api.ProgressResponse{Status: status, Digest: digest, Total: written, Completed: written})
	// Bytes were verified before commit, so PullModel need not hash them again.
	return true, nil
}

// oaAdmit reconciles a persisted identity or submits it. It never cancels work.
func oaAdmit(ctx context.Context, jobs *oaclient.JobsClient, record *oaRecord) (acceptance.AcceptanceResult, error) {
	id := record.identity()
	refused := func(reason string, result acceptance.AcceptanceResult) error {
		return &OAError{Reason: reason, Detail: result.Outcome + ": " + result.Reason}
	}
	if record.OperationID != "" {
		result, err := jobs.Reconcile(ctx, id)
		if err != nil {
			return result, &OAError{Reason: oaReasonUnknown, Detail: "reconcile", Err: err}
		}
		switch result.Outcome {
		case "accepted":
			return result, nil
		case "definitely_not_accepted":
			// Fall through to submission of the identical persisted request.
		default:
			return result, refused(oaReasonRefused, result)
		}
	}
	submission := acceptance.Submission{Identity: id, Kind: record.Kind, Spec: record.Request, RequiredGuarantees: record.Guarantees}
	result, err := jobs.Submit(ctx, submission)
	if err != nil {
		// Acceptance is unresolved; the retained record reconciles next time.
		return result, &OAError{Reason: oaReasonUnknown, Detail: "submit", Err: err}
	}
	switch result.Outcome {
	case "accepted":
		return result, nil
	case "key_conflict":
		// Blobs are shared across model names, so the same digest may have been
		// requested from another locator. The digest is verified on delivery.
		reconciled, err := jobs.Reconcile(ctx, id)
		if err != nil {
			return reconciled, &OAError{Reason: oaReasonUnknown, Detail: "reconcile after key conflict", Err: err}
		}
		if reconciled.Outcome == "accepted" {
			return reconciled, nil
		}
		return reconciled, refused(oaReasonRefused, reconciled)
	default:
		return result, refused(oaReasonRefused, result)
	}
}

type oaHashingFile struct {
	f *os.File
	h hash.Hash
}

func (w oaHashingFile) Write(p []byte) (int, error) {
	n, err := w.f.Write(p)
	w.h.Write(p[:n])
	return n, err
}

func oaCopy(ctx context.Context, jobs *oaclient.JobsClient, id acceptance.RequestIdentity, path string) (int64, string, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return 0, "", err
	}
	w := oaHashingFile{f: f, h: sha256.New()}
	written, err := jobs.CopyResult(ctx, id, w)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return written, hex.EncodeToString(w.h.Sum(nil)), err
}

var _ io.Writer = oaHashingFile{}
