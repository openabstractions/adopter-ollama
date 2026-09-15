package server

// OpenAbstractions storage reuse. Before an OA-routed blob pull starts a job,
// the seam asks the resolved content storage reader for the digest. Bytes it
// supplies are delivered only after an exact size and sha256 check. Every other
// answer is a typed reason that is logged before the job download runs.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	oaclient "github.com/openabstractions/abstraction-facade/go/client"
)

// OA storage reasons.
const (
	oaReasonStorageUnavailable    = "storage_unavailable"
	oaReasonStorageNotFound       = "storage_not_found"
	oaReasonStorageRefused        = "storage_refused"
	oaReasonStorageReadFailed     = "storage_read_failed"
	oaReasonStorageDigestMismatch = "storage_digest_mismatch"
)

// oaStorageChunk is the read size, the content reader's maximum.
const oaStorageChunk int64 = 64 * 1024

// oaReuseFromStorage delivers the blob from OA storage and returns true, or
// returns a typed *OAError naming why storage did not supply it. It never leaves
// a partial blob at fp.
func oaReuseFromStorage(ctx context.Context, digest string, size int64, fp string, fn func(api.ProgressResponse)) (bool, error) {
	typed := func(reason, detail string, err error) (bool, error) {
		return false, &OAError{Reason: reason, Digest: digest, Detail: detail, Err: err}
	}
	store, err := oaMachine().ResolveStorage(ctx, oaclient.Requirements{Scope: "local"})
	if err != nil {
		return typed(oaReasonStorageUnavailable, "abstraction.storage/content-reader@1 did not resolve", err)
	}
	opened, err := store.Open(ctx, digest)
	if err != nil {
		return typed(oaReasonStorageUnavailable, "open", err)
	}
	switch opened.Outcome {
	case "opened":
	case "not_found":
		return typed(oaReasonStorageNotFound, "open: not_found", nil)
	default:
		return typed(oaReasonStorageRefused, "open: "+opened.Outcome, nil)
	}
	resource := *opened.Resource
	defer func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		store.Close(closeCtx, resource)
		cancel()
	}()
	if size > 0 && resource.Size != size {
		return typed(oaReasonStorageDigestMismatch, fmt.Sprintf("storage has %d bytes, the manifest says %d", resource.Size, size), nil)
	}

	dir, err := oaRequestsDir()
	if err != nil {
		return false, err
	}
	tmp := filepath.Join(dir, strings.ReplaceAll(digest, ":", "-")+".storage")
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return false, err
	}
	committed := false
	defer func() {
		if !committed {
			f.Close()
			os.Remove(tmp)
		}
	}()
	status := fmt.Sprintf("pulling %s", digest[7:19])
	h := sha256.New()
	var offset int64
	for {
		read, err := store.Read(ctx, resource, offset, oaStorageChunk)
		if err != nil {
			if ctx.Err() != nil {
				return false, ctx.Err()
			}
			return typed(oaReasonStorageReadFailed, fmt.Sprintf("read at %d", offset), err)
		}
		if read.Outcome != "data" || read.Chunk == nil || read.Chunk.Offset != offset || (len(read.Chunk.Data) == 0 && !read.Chunk.Eof) {
			return typed(oaReasonStorageReadFailed, fmt.Sprintf("read at %d: %s", offset, read.Outcome), nil)
		}
		if _, err := f.Write(read.Chunk.Data); err != nil {
			return false, err
		}
		h.Write(read.Chunk.Data)
		offset += int64(len(read.Chunk.Data))
		fn(api.ProgressResponse{Status: status, Digest: digest, Total: resource.Size, Completed: offset})
		if read.Chunk.Eof {
			break
		}
	}
	if err := f.Sync(); err != nil {
		return false, err
	}
	if err := f.Close(); err != nil {
		return false, err
	}
	if got := "sha256:" + hex.EncodeToString(h.Sum(nil)); got != digest || offset != resource.Size {
		os.Remove(tmp)
		committed = true
		return typed(oaReasonStorageDigestMismatch, fmt.Sprintf("storage delivered %s, %d bytes", got, offset), errDigestMismatch)
	}
	if err := os.Rename(tmp, fp); err != nil {
		os.Remove(tmp)
		committed = true
		return false, err
	}
	committed = true
	return true, nil
}

// oaReportStorage logs why storage did not supply a blob before the job
// download runs. not_found is the ordinary miss; every other reason is a warning.
func oaReportStorage(digest string, err error) {
	var typed *OAError
	reason := "unknown"
	if errors.As(err, &typed) {
		reason = typed.Reason
	}
	level := slog.LevelWarn
	if reason == oaReasonStorageNotFound {
		level = slog.LevelInfo
	}
	slog.Log(context.Background(), level, "OA storage did not supply the blob; downloading through the OA job service",
		"digest", digest, "reason", reason, "error", err.Error())
}
