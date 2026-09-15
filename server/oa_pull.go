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
	"github.com/openabstractions/abstraction-identity/listen"
	acceptance "github.com/openabstractions/abstraction-job/go/abstraction/job/acceptance"
)

const oaDownloadEnv = "OLLAMA_OA_DOWNLOAD"

// OAError is the typed, visible failure of the OA download seam. Reason is one
// of the oaReason* words, Cause the service's typed failure cause when it
// reported one, and Detail the service's own reason text.
type OAError struct {
	Reason string
	Cause  string
	Digest string
	Detail string
	Err    error
}

const (
	oaReasonInvalidConfiguration = "invalid_configuration"
	oaReasonUnavailable          = "unavailable"
	oaReasonUntrusted            = "untrusted"
	oaReasonUnknown              = "unknown"
	oaReasonRefused              = "refused"
	oaReasonProviderChanged      = "provider_changed"
	oaReasonCredentialsRequired  = "credentials_required"
	oaReasonNotFound             = "not_found"
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
	if e.Cause != "" && e.Cause != e.Reason {
		b.WriteString(" (" + e.Cause + ")")
	}
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

// oaRecord is the caller recovery record, persisted before Submit. Binding is
// the resolved selection RestoreJobs authenticates again after a restart.
// Terminal marks an identity whose operation failed, was cancelled or was
// sealed; the next pull submits the next attempt of the same key [JOB-A7].
type oaRecord struct {
	Format      string                     `json:"format"`
	Digest      string                     `json:"digest"`
	Binding     oaclient.JobsBinding       `json:"binding"`
	Identity    acceptance.RequestIdentity `json:"identity"`
	Kind        string                     `json:"kind"`
	Request     []byte                     `json:"request"`
	OperationID string                     `json:"operation_id,omitempty"`
	Terminal    bool                       `json:"terminal,omitempty"`
}

const oaRecordFormat = "ollama-oa-download/2"

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
	if err := json.Unmarshal(data, &r); err != nil || r.Format != oaRecordFormat || r.Identity.Key == "" || r.Identity.HistoryEpoch == "" || r.Identity.Attempt < 0 || r.Binding.Endpoint == "" || r.Binding.LogicalOwner == "" {
		return nil, fmt.Errorf("unreadable OA request record %s: %v", path, err)
	}
	return &r, nil
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

func oaKey(digest string) string {
	return "ollama-blob-" + strings.ReplaceAll(digest, ":", "-")
}

// oaBindError types a resolution or restoration failure.
func oaBindError(detail string, err error) error {
	var service *acceptance.ServiceError
	switch {
	case errors.Is(err, listen.ErrServerUntrusted):
		return &OAError{Reason: oaReasonUntrusted, Detail: detail, Err: err}
	case errors.As(err, &service) && service.Code == "invalid_acceptance":
		return &OAError{Reason: oaReasonProviderChanged, Detail: detail, Err: err}
	default:
		return &OAError{Reason: oaReasonUnavailable, Detail: detail, Err: err}
	}
}

// oaFailure types terminal work by the service's cause [JOB-A8].
func oaFailure(s *acceptance.OperationSnapshot) *OAError {
	e := &OAError{Reason: oaReasonFailed, Detail: s.State}
	if s.Failure == nil {
		return e
	}
	e.Cause = s.Failure.Cause
	e.Detail = s.State + " (" + s.Failure.Classification + "): " + s.Failure.Message
	switch s.Failure.Cause {
	case acceptance.FailureCauseUnauthorized:
		e.Reason = oaReasonCredentialsRequired
	case acceptance.FailureCauseNotFound:
		e.Reason = oaReasonNotFound
	case acceptance.FailureCauseDigestMismatch:
		e.Reason, e.Err = oaReasonDigestMismatch, errDigestMismatch
	}
	return e
}

func oaDownloadBlob(ctx context.Context, size int64, opts downloadOpts) (bool, error) {
	digest := opts.digest
	fail := func(err error) (bool, error) {
		var typed *OAError
		if errors.As(err, &typed) && typed.Digest == "" {
			typed.Digest = digest
		}
		return false, err
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
		return fail(&OAError{Reason: oaReasonCredentialsRequired, Detail: "this registry pull needs credentials and abstraction.download requests are anonymous"})
	}

	dir, err := oaRequestsDir()
	if err != nil {
		return false, err
	}
	recordPath := oaRecordPath(dir, digest)
	record, err := oaLoadRecord(recordPath)
	if err != nil {
		return false, err
	}

	var jobs *oaclient.JobsClient
	if record == nil {
		source := opts.n.BaseURL().JoinPath("v2", opts.n.DisplayNamespaceModel(), "blobs", digest)
		if opts.regOpts != nil && opts.regOpts.Insecure {
			source.Scheme = "http" // as makeRequest does for upstream requests
		}
		jobs, err = oaMachine().ResolveJobOperations(ctx, oaclient.Requirements{Guarantees: acceptance.AdmissionGuarantees, Scope: "local"})
		if err != nil {
			return fail(oaBindError("job service did not resolve", err))
		}
		window, err := jobs.GetHistoryWindow(ctx)
		if err != nil {
			return fail(oaBindError("history window", err))
		}
		record = &oaRecord{
			Format: oaRecordFormat, Digest: digest, Binding: jobs.Binding(),
			Identity: acceptance.RequestIdentity{Key: oaKey(digest), HistoryEpoch: window.HistoryEpoch},
			Kind:     "download",
			Request: request.Encode(&request.Request{
				Artifact: request.Artifact{Digest: digest, Size: size},
				Sources:  []request.Source{{Scheme: source.Scheme, Locator: source.String()}},
			}),
		}
	} else {
		// A restart restores the saved selection and authenticates it again.
		jobs, err = oaMachine().RestoreJobs(ctx, record.Binding)
		if err != nil {
			return fail(oaBindError("restore saved job binding", err))
		}
		if record.Terminal {
			record.Identity.Attempt++
			record.OperationID, record.Terminal = "", false
		}
	}
	if err := oaSaveRecord(recordPath, record); err != nil {
		return false, err
	}

	id := record.Identity
	accepted, err := oaAdmit(ctx, jobs, record)
	if err != nil {
		var typed *OAError
		if errors.As(err, &typed) && typed.Reason == oaReasonRefused && strings.HasPrefix(typed.Detail, "definitely_not_accepted") {
			// A sealed attempt is terminal; the next pull presents the next attempt.
			record.Terminal = true
			if saveErr := oaSaveRecord(recordPath, record); saveErr != nil {
				return false, saveErr
			}
		}
		return fail(err)
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
			return fail(&OAError{Reason: oaReasonUnknown, Detail: "observe work", Err: err})
		}
		if observed.Outcome == "unavailable" {
			// A policy decision outage reads nothing; the record is kept [JOB-A9].
			return fail(&OAError{Reason: oaReasonUnavailable, Detail: "observe work: policy decision unavailable"})
		}
		if observed.Outcome != "observed" {
			return fail(&OAError{Reason: oaReasonUnknown, Detail: "observe work outcome " + observed.Outcome})
		}
		s := observed.Snapshot
		total := s.Progress.Total
		if total == 0 {
			total = size
		}
		current := status
		if s.Failure != nil && s.Failure.Classification == "retryable" {
			// The service tries this operation again; show why it is waiting.
			reason := s.Failure.Cause
			if reason == "" {
				reason = s.Failure.Message
			}
			current = fmt.Sprintf("%s (OA service retrying: %s)", status, reason)
			if reason != lastFailure {
				slog.Warn("OA download retrying", "digest", digest, "state", s.State, "cause", s.Failure.Cause, "message", s.Failure.Message)
				lastFailure = reason
			}
		}
		opts.fn(api.ProgressResponse{Status: current, Digest: digest, Total: total, Completed: s.Progress.Done})
		if s.State == "complete" {
			break
		}
		if s.State == "failed" || s.State == "cancelled" {
			record.Terminal = true
			if err := oaSaveRecord(recordPath, record); err != nil {
				return false, err
			}
			return fail(oaFailure(s))
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
			typed := &OAError{Reason: oaReasonResultUnavailable, Err: err}
			// A recorded loss ends the operation as failed/result_lost, and the
			// next pull presents the next attempt of the key [JOB-A10, JOB-A7].
			if again, oerr := jobs.ObserveWork(ctx, id); oerr == nil && again.Outcome == "observed" &&
				again.Snapshot.State == "failed" && again.Snapshot.Failure != nil {
				typed.Cause = again.Snapshot.Failure.Cause
				record.Terminal = true
				if saveErr := oaSaveRecord(recordPath, record); saveErr != nil {
					return false, saveErr
				}
			}
			return fail(typed)
		}
		return false, err
	}
	if got := "sha256:" + sum; got != digest || (size > 0 && written != size) {
		os.Remove(tmpPath)
		return fail(&OAError{Reason: oaReasonDigestMismatch, Detail: fmt.Sprintf("delivered %s, %d bytes", got, written), Err: errDigestMismatch})
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
	id := record.Identity
	refused := func(result acceptance.AcceptanceResult) error {
		reason := oaReasonRefused
		if result.Outcome == "unavailable" {
			// No effect and no seal: the same identity is presented next time [JOB-A9].
			reason = oaReasonUnavailable
		}
		return &OAError{Reason: reason, Detail: result.Outcome + ": " + result.Reason}
	}
	if record.OperationID != "" {
		result, err := jobs.Reconcile(ctx, id)
		if err != nil {
			return result, &OAError{Reason: oaReasonUnknown, Detail: "reconcile", Err: err}
		}
		if result.Outcome == "accepted" {
			return result, nil
		}
		return result, refused(result)
	}
	submission := acceptance.Submission{Identity: id, Kind: record.Kind, Spec: record.Request, RequiredGuarantees: record.Binding.RequiredGuarantees}
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
		return reconciled, refused(reconciled)
	default:
		return result, refused(result)
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
