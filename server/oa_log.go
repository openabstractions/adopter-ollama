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
	oaLogReasonQueueFull            = "queue_full"
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
	report := &oaLogReporter{local: local}
	queue := oalog.NewAsyncSink(oaTimeoutSink{inner: oaLogInner(sink)}, oalog.AsyncOptions{
		Capacity:   oaLogQueueCapacity,
		OnOverflow: report.overflow,
		OnFailure:  report.failure,
		OnRecovery: report.recovery,
	})
	// oalog.Level equals slog.Level, so the configured level converts by cast.
	remote := oalog.NewHandler(queue, &oalog.Options{Program: "ollama", Level: oalog.Level(level)})
	return &oaFanout{local: local, remote: remote, queue: queue, report: report}, nil
}

// oaLogQueueCapacity bounds the records waiting for the service. Ollama's startup
// burst is tens of records and a debug model load logs a few hundred GGUF keys,
// so 1024 absorbs both while the service is slow. At a few hundred bytes a
// record the queue holds well under a megabyte.
var oaLogQueueCapacity = 1024

// oaLogInner builds the delivering sink; tests replace it to stall delivery.
var oaLogInner = oalog.NewClientSink

// oaTimeoutSink bounds each delivery by oaLogWriteTimeout. The asynchronous
// sink delivers without a deadline of its own, and an unresponsive service
// would otherwise hold its one worker until Close.
type oaTimeoutSink struct{ inner oalog.Sink }

func (s oaTimeoutSink) Write(r oalog.Record) error { return s.WriteContext(context.Background(), r) }

func (s oaTimeoutSink) WriteContext(ctx context.Context, r oalog.Record) error {
	ctx, cancel := context.WithTimeout(ctx, oaLogWriteTimeout)
	defer cancel()
	if w, ok := s.inner.(interface {
		WriteContext(context.Context, oalog.Record) error
	}); ok {
		return w.WriteContext(ctx, r)
	}
	return s.inner.Write(r)
}

// oaLogReporter reports each transition of the asynchronous sink on the local
// handler only, so a report cannot recurse into the service. Failure and
// recovery reports come from the delivery goroutine after the log call that
// queued the record has returned.
type oaLogReporter struct {
	local slog.Handler
}

func (s *oaLogReporter) failure(err error, c oalog.AsyncCounts) {
	typed := &OALogError{Reason: oaLogReasonWriteFailed, Err: err}
	s.report(slog.LevelWarn, "OpenAbstractions logging write failed; records reach stderr only until it recovers",
		slog.String("reason", typed.Reason), slog.Uint64("failed", c.Failed), slog.Int("queued", c.Queued), slog.String("error", typed.Error()))
}

func (s *oaLogReporter) recovery(c oalog.AsyncCounts) {
	s.report(slog.LevelInfo, "OpenAbstractions logging delivers again", slog.Uint64("lost", c.Failed+c.Dropped))
}

func (s *oaLogReporter) overflow(c oalog.AsyncCounts) {
	s.report(slog.LevelWarn, "OpenAbstractions logging queue full; records reach stderr only until it drains",
		slog.String("reason", oaLogReasonQueueFull), slog.Uint64("dropped", c.Dropped), slog.Int("capacity", oaLogQueueCapacity))
}

func (s *oaLogReporter) report(level slog.Level, msg string, attrs ...slog.Attr) {
	ctx := context.Background()
	if !s.local.Enabled(ctx, level) {
		return
	}
	r := slog.NewRecord(time.Now(), level, msg, 0)
	r.AddAttrs(attrs...)
	_ = s.local.Handle(ctx, r)
}

// oaFanout writes each record to the local handler and queues it for the OA
// service. The local handler's result is the one returned: stderr logging
// behaves as upstream whatever the service does.
type oaFanout struct {
	local  slog.Handler
	remote slog.Handler
	queue  *oalog.AsyncSink
	report *oaLogReporter
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
		// Queueing never blocks. A full queue is reported by OnOverflow and a
		// closed one by close, so the returned error adds nothing here.
		_ = h.remote.Handle(ctx, r)
	}
	return err
}

func (h *oaFanout) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &oaFanout{local: h.local.WithAttrs(attrs), remote: h.remote.WithAttrs(attrs), queue: h.queue, report: h.report}
}

func (h *oaFanout) WithGroup(name string) slog.Handler {
	return &oaFanout{local: h.local.WithGroup(name), remote: h.remote.WithGroup(name), queue: h.queue, report: h.report}
}

// close drains the queue for up to the deadline and reports records that did
// not reach the service.
func (h *oaFanout) close(ctx context.Context) {
	err := h.queue.Close(ctx)
	c := h.queue.Counts()
	if err != nil || c.Failed+c.Dropped+c.Abandoned > 0 {
		attrs := []slog.Attr{slog.Uint64("written", c.Written), slog.Uint64("failed", c.Failed), slog.Uint64("dropped", c.Dropped), slog.Uint64("abandoned", c.Abandoned)}
		if err != nil {
			attrs = append(attrs, slog.String("error", err.Error()))
		}
		h.report.report(slog.LevelWarn, "OpenAbstractions logging closed with undelivered records", attrs...)
	}
}

// oaLogShutdown closes the installed seam; Serve defers it.
var oaLogShutdown = func() {}

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
	if fanout, ok := handler.(*oaFanout); ok {
		oaLogShutdown = func() {
			ctx, cancel := context.WithTimeout(context.Background(), oaLogWriteTimeout)
			defer cancel()
			slog.SetDefault(local)
			fanout.close(ctx)
		}
		slog.SetDefault(slog.New(handler))
		slog.Info("OpenAbstractions logging adopted", "env", oaLogEnv)
	}
	return nil
}
