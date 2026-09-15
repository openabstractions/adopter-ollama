package server

// OpenAbstractions logging seam. OLLAMA_OA_LOGGING=service sends every slog
// record Ollama writes to the resolved OA logging service as well as to the
// usual stderr handler. Unset keeps upstream logging exactly. Any other value
// is an error. The seam lives in this file; routes.go calls oaLogHandler once
// when Serve installs its logger.

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"time"

	oaclient "github.com/openabstractions/abstraction-facade/go/client"
	oalog "github.com/openabstractions/abstraction-logging/go"
)

const oaLogEnv = "OLLAMA_OA_LOGGING"

// OA logging reasons.
const (
	oaLogReasonInvalidConfiguration = "invalid_configuration"
	oaLogReasonUnavailable          = "unavailable"
	oaLogReasonWriteFailed          = "write_failed"
)

// oaLogWriteTimeout bounds one record's delivery, so an unresponsive service
// costs a log call at most this long.
var oaLogWriteTimeout = 2 * time.Second

// OALogError is the typed, visible failure of the OA logging seam.
type OALogError struct {
	Reason string
	Detail string
	Err    error
}

func (e *OALogError) Error() string {
	msg := "openabstractions logging (" + oaLogEnv + "=service): " + e.Reason
	if e.Detail != "" {
		msg += ": " + e.Detail
	}
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	return msg
}

func (e *OALogError) Unwrap() error { return e.Err }

func oaLoggingEnabled() (bool, error) {
	switch v := os.Getenv(oaLogEnv); v {
	case "":
		return false, nil
	case "service":
		return true, nil
	default:
		return false, &OALogError{Reason: oaLogReasonInvalidConfiguration, Detail: fmt.Sprintf("%s=%q; supported values are unset and \"service\"", oaLogEnv, v)}
	}
}

// oaLogHandler returns the handler Serve installs. Disabled: local, untouched.
// Enabled and resolved: local plus the OA service. Enabled and not resolved:
// local plus a typed *OALogError the caller reports.
func oaLogHandler(ctx context.Context, local slog.Handler, level slog.Level) (slog.Handler, error) {
	enabled, err := oaLoggingEnabled()
	if err != nil || !enabled {
		return local, err
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	sink, err := oaMachine().ResolveLog(ctx, oaclient.Requirements{Scope: "local"})
	if err != nil {
		return local, &OALogError{Reason: oaLogReasonUnavailable, Detail: "abstraction.logging/sink@1 did not resolve", Err: err}
	}
	state := &oaLogState{warn: local}
	// oalog.Level equals slog.Level, so the configured level converts by cast.
	remote := oalog.NewClientHandler(sink, &oalog.Options{Program: "ollama", Level: oalog.Level(level)})
	return &oaFanout{local: local, remote: remote, state: state}, nil
}

// oaLogState counts deliveries and reports each transition into and out of
// failure on the local handler only, so a report cannot recurse into the service.
type oaLogState struct {
	warn    slog.Handler
	mu      sync.Mutex
	failing bool
	written int64
	failed  int64
}

func (s *oaLogState) delivered(ctx context.Context) {
	s.mu.Lock()
	s.written++
	recovered := s.failing
	s.failing = false
	lost := s.failed
	s.mu.Unlock()
	if recovered {
		s.report(ctx, slog.LevelInfo, "OpenAbstractions logging delivers again", slog.Int64("lost", lost))
	}
}

func (s *oaLogState) failure(ctx context.Context, err error) {
	s.mu.Lock()
	s.failed++
	first := !s.failing
	s.failing = true
	lost := s.failed
	s.mu.Unlock()
	if first {
		typed := &OALogError{Reason: oaLogReasonWriteFailed, Err: err}
		s.report(ctx, slog.LevelWarn, "OpenAbstractions logging write failed; records reach stderr only until it recovers",
			slog.String("reason", typed.Reason), slog.Int64("lost", lost), slog.String("error", typed.Error()))
	}
}

func (s *oaLogState) counts() (written, failed int64, failing bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.written, s.failed, s.failing
}

func (s *oaLogState) report(ctx context.Context, level slog.Level, msg string, attrs ...slog.Attr) {
	if !s.warn.Enabled(ctx, level) {
		return
	}
	r := slog.NewRecord(time.Now(), level, msg, 0)
	r.AddAttrs(attrs...)
	_ = s.warn.Handle(ctx, r)
}

// oaFanout writes each record to the local handler and to the OA service.
// The local handler's result is the one returned: stderr logging behaves as
// upstream whatever the service does.
type oaFanout struct {
	local  slog.Handler
	remote slog.Handler
	state  *oaLogState
}

func (h *oaFanout) Enabled(ctx context.Context, level slog.Level) bool {
	return h.local.Enabled(ctx, level) || h.remote.Enabled(ctx, level)
}

func (h *oaFanout) Handle(ctx context.Context, r slog.Record) error {
	var err error
	if h.local.Enabled(ctx, r.Level) {
		err = h.local.Handle(ctx, r.Clone())
	}
	if h.remote.Enabled(ctx, r.Level) {
		remoteCtx, cancel := context.WithTimeout(ctx, oaLogWriteTimeout)
		remoteErr := h.remote.Handle(remoteCtx, r)
		cancel()
		if remoteErr != nil {
			h.state.failure(ctx, remoteErr)
		} else {
			h.state.delivered(ctx)
		}
	}
	return err
}

func (h *oaFanout) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &oaFanout{local: h.local.WithAttrs(attrs), remote: h.remote.WithAttrs(attrs), state: h.state}
}

func (h *oaFanout) WithGroup(name string) slog.Handler {
	return &oaFanout{local: h.local.WithGroup(name), remote: h.remote.WithGroup(name), state: h.state}
}

// oaServeLogger installs Ollama's logger with the OA seam applied. An invalid
// switch is returned and stops Serve; an unavailable service is reported on
// stderr and Ollama continues with its own logging.
func oaServeLogger(local *slog.Logger, level slog.Level) error {
	slog.SetDefault(local)
	handler, err := oaLogHandler(context.Background(), local.Handler(), level)
	var typed *OALogError
	if errors.As(err, &typed) && typed.Reason == oaLogReasonInvalidConfiguration {
		return err
	}
	if err != nil {
		slog.Warn("OpenAbstractions logging not adopted; Ollama logs to stderr only", "reason", typed.Reason, "error", err.Error())
		return nil
	}
	if handler != local.Handler() {
		slog.SetDefault(slog.New(handler))
		slog.Info("OpenAbstractions logging adopted", "env", oaLogEnv)
	}
	return nil
}
