package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/logutil"
	oaclient "github.com/openabstractions/abstraction-facade/go/client"
	oahost "github.com/openabstractions/abstraction-facade/go/runtime"
	"github.com/openabstractions/abstraction-identity/listen"
	oalog "github.com/openabstractions/abstraction-logging/go"
	oalogwire "github.com/openabstractions/abstraction-logging/go/abstraction/logging"
	oalogclient "github.com/openabstractions/abstraction-logging/go/client"
)

// oaLogService starts a real Go OA runtime whose logging provider keeps
// readable history, and points the seam's machine at it.
func oaLogService(t *testing.T, home string) *oaFixture {
	t.Helper()
	sink, err := oalog.OpenFileSink(filepath.Join(home, "oa-logs", "runtime.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { sink.Close() })
	prefix := fmt.Sprintf("ollama-oa-log-%d-%d", os.Getpid(), time.Now().UnixNano())
	o := oahost.Options{Endpoint: listen.Endpoint(prefix), LogEndpoint: listen.Endpoint(prefix + "l"), ConfigEndpoint: listen.Endpoint(prefix + "c"), Sink: sink}
	f := &oaFixture{options: o}
	f.start(t)
	oaMachine = func() *oaclient.Machine { return oaclient.NewUnverified(o.Endpoint) }
	return f
}

func oaLogReason(t *testing.T, err error, reason string) *OALogError {
	t.Helper()
	var typed *OALogError
	if !errors.As(err, &typed) || typed.Reason != reason {
		t.Fatalf("want OA logging reason %q, got %v", reason, err)
	}
	return typed
}

// oaLogHistory reads every retained record through the resolved history reader.
func oaLogHistory(t *testing.T) []oalogwire.Record {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	reader, err := oaMachine().ResolveLogReader(ctx, oaclient.Requirements{Scope: "local"})
	if err != nil {
		t.Fatal(err)
	}
	var records []oalogwire.Record
	cursor := ""
	for {
		page, err := reader.ReadContext(ctx, cursor, 256, 65536)
		if err != nil {
			t.Fatal(err)
		}
		if page.Outcome != "page" {
			t.Fatalf("history outcome %s", page.Outcome)
		}
		records = append(records, page.Records...)
		cursor = page.Next
		if page.AtEnd {
			return records
		}
	}
}

func oaLogFind(records []oalogwire.Record, msg string) (oalogwire.Record, bool) {
	for _, r := range records {
		if r.Msg == msg {
			return r, true
		}
	}
	return oalogwire.Record{}, false
}

func TestOALoggingDisabledKeepsUpstream(t *testing.T) {
	setupOA(t)
	t.Setenv(oaLogEnv, "")
	oaMachine = func() *oaclient.Machine {
		t.Error("OA resolved while logging is disabled")
		return oaclient.NewUnverified("")
	}
	var buf bytes.Buffer
	local := logutil.NewLogger(&buf, slog.LevelInfo)
	handler, err := oaLogHandler(t.Context(), local.Handler(), slog.LevelInfo)
	if err != nil {
		t.Fatal(err)
	}
	if handler != local.Handler() {
		t.Fatalf("disabled seam replaced the upstream handler: %T", handler)
	}
}

func TestOALoggingInvalidSwitchIsTyped(t *testing.T) {
	setupOA(t)
	t.Setenv(oaLogEnv, "stderr")
	oaMachine = func() *oaclient.Machine {
		t.Error("OA resolved with an invalid switch")
		return oaclient.NewUnverified("")
	}
	local := logutil.NewLogger(&bytes.Buffer{}, slog.LevelInfo)
	handler, err := oaLogHandler(t.Context(), local.Handler(), slog.LevelInfo)
	oaLogReason(t, err, oaLogReasonInvalidConfiguration)
	if handler != local.Handler() {
		t.Fatal("invalid switch replaced the upstream handler")
	}
	if err := oaServeLogger(local, slog.LevelInfo); err == nil {
		t.Fatal("Serve's logger accepted an invalid switch")
	}
}

func TestOALoggingAbsentServiceIsTypedAndVisible(t *testing.T) {
	setupOA(t)
	t.Setenv(oaLogEnv, "service")
	oaMachine = func() *oaclient.Machine { return oaclient.NewUnverified(listen.Endpoint("ollama-oa-log-no-runtime")) }
	var buf bytes.Buffer
	local := logutil.NewLogger(&buf, slog.LevelInfo)
	handler, err := oaLogHandler(t.Context(), local.Handler(), slog.LevelInfo)
	oaLogReason(t, err, oaLogReasonUnavailable)
	if handler != local.Handler() {
		t.Fatal("unresolved service replaced the upstream handler")
	}

	previous := slog.Default()
	t.Cleanup(func() { slog.SetDefault(previous) })
	if err := oaServeLogger(local, slog.LevelInfo); err != nil {
		t.Fatalf("an absent service must not stop Serve: %v", err)
	}
	slog.Info("after absent service")
	text := buf.String()
	for _, want := range []string{"OpenAbstractions logging not adopted", "reason=unavailable", "after absent service"} {
		if !strings.Contains(text, want) {
			t.Fatalf("stderr lacks %q:\n%s", want, text)
		}
	}
}

func TestOALoggingDeliversThroughServiceHistory(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaLogEnv, "service")
	oaLogService(t, home)
	var buf bytes.Buffer
	local := logutil.NewLogger(&buf, slog.LevelDebug)
	handler, err := oaLogHandler(t.Context(), local.Handler(), slog.LevelDebug)
	if err != nil {
		t.Fatal(err)
	}
	logger := slog.New(handler).With("component", "test")
	logger.Info("oa seam info", "model", "llama3", "size", 42)
	logger.WithGroup("gpu").Warn("oa seam warn", "vram", "8GiB")
	logger.Debug("oa seam debug")
	logger.Log(t.Context(), logutil.LevelTrace, "oa seam trace below the handler level")

	for _, want := range []string{"oa seam info", "oa seam warn", "oa seam debug", "source="} {
		if !strings.Contains(buf.String(), want) {
			t.Fatalf("upstream stderr handler lost %q:\n%s", want, buf.String())
		}
	}
	// Records reach the service from the delivery goroutine after the log calls.
	oaLogSettled(t, handler, func(c oalog.AsyncCounts) bool { return c.Accepted == 3 })
	records := oaLogHistory(t)
	info, ok := oaLogFind(records, "oa seam info")
	if !ok {
		t.Fatalf("info record not in history: %d records", len(records))
	}
	if info.Level != 0 || info.Attrs["model"] != "llama3" || info.Attrs["size"] != "42" || info.Attrs["component"] != "test" {
		t.Fatalf("info record %+v", info)
	}
	warn, ok := oaLogFind(records, "oa seam warn")
	if !ok || warn.Level != 4 || warn.Attrs["gpu.vram"] != "8GiB" {
		t.Fatalf("warn record %+v (found %v)", warn, ok)
	}
	if debug, ok := oaLogFind(records, "oa seam debug"); !ok || debug.Level != -4 {
		t.Fatalf("debug record %+v (found %v)", debug, ok)
	}
	if _, ok := oaLogFind(records, "oa seam trace below the handler level"); ok {
		t.Fatal("a record below the configured level reached the service")
	}
	if len(info.Identity) != 2 {
		t.Fatalf("identity chain %+v", info.Identity)
	}
	writer, stamp := info.Identity[0], info.Identity[1]
	if writer.By != "self" || writer.Verified || writer.Hop != 0 || writer.Program != "ollama" {
		t.Fatalf("writer claim %+v", writer)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if stamp.By != "identity/"+runtime.GOOS || !stamp.Verified || stamp.Hop != 1 || stamp.Program != "" ||
		!strings.EqualFold(filepath.Clean(stamp.Exe), filepath.Clean(executable)) {
		t.Fatalf("service stamp %+v, want verified identity/%s naming %s", stamp, runtime.GOOS, executable)
	}
	if c := handler.(*oaFanout).queue.Counts(); c.Written != 3 || c.Failed != 0 || c.Dropped != 0 || c.Failing {
		t.Fatalf("counts %+v", c)
	}
}

// oaLogSettled waits until every accepted record has been delivered or failed.
func oaLogSettled(t *testing.T, handler slog.Handler, want func(oalog.AsyncCounts) bool) oalog.AsyncCounts {
	t.Helper()
	queue := handler.(*oaFanout).queue
	deadline := time.Now().Add(15 * time.Second)
	for {
		c := queue.Counts()
		if c.Queued == 0 && c.Written+c.Failed+c.Abandoned == c.Accepted && want(c) {
			return c
		}
		if time.Now().After(deadline) {
			t.Fatalf("asynchronous sink did not settle: %+v", c)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// oaLogWaitFor waits for text to appear on the local handler's output. The
// asynchronous sink reports from its delivery goroutine after the log call.
func oaLogWaitFor(t *testing.T, buf *lockedBuffer, text string) {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for !strings.Contains(buf.String(), text) {
		if time.Now().After(deadline) {
			t.Fatalf("stderr never showed %q:\n%s", text, buf.String())
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestOALoggingWriteFailureIsReportedOnce(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaLogEnv, "service")
	fixture := oaLogService(t, home)
	buf := &lockedBuffer{}
	local := logutil.NewLogger(buf, slog.LevelInfo)
	handler, err := oaLogHandler(t.Context(), local.Handler(), slog.LevelInfo)
	if err != nil {
		t.Fatal(err)
	}
	oaLogWriteTimeout = 500 * time.Millisecond
	t.Cleanup(func() { oaLogWriteTimeout = 2 * time.Second })
	logger := slog.New(handler)
	logger.Info("before the service stops")
	oaLogSettled(t, handler, func(c oalog.AsyncCounts) bool { return c.Written == 1 })
	fixture.stop()
	logger.Info("first after stop")
	logger.Info("second after stop")
	c := oaLogSettled(t, handler, func(c oalog.AsyncCounts) bool { return c.Failed == 2 })
	oaLogWaitFor(t, buf, "OpenAbstractions logging write failed")
	text := buf.String()
	if strings.Count(text, "OpenAbstractions logging write failed") != 1 || !strings.Contains(text, "reason=write_failed") {
		t.Fatalf("write failure must be reported once with its reason:\n%s", text)
	}
	for _, want := range []string{"before the service stops", "first after stop", "second after stop"} {
		if !strings.Contains(text, want) {
			t.Fatalf("stderr lost %q while the service was down:\n%s", want, text)
		}
	}
	if c.Written != 1 || c.Failed != 2 || !c.Failing {
		t.Fatalf("counts %+v", c)
	}
	handler.(*oaFanout).close(t.Context())
	oaLogWaitFor(t, buf, "OpenAbstractions logging closed with undelivered records")
}

// stalledSink holds every delivery until release closes or its deadline passes.
type stalledSink struct {
	release chan struct{}
	started chan struct{}
	once    sync.Once
}

func (s *stalledSink) Write(r oalog.Record) error { return s.WriteContext(context.Background(), r) }

func (s *stalledSink) WriteContext(ctx context.Context, _ oalog.Record) error {
	s.once.Do(func() { close(s.started) })
	select {
	case <-s.release:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func stallOALogging(t *testing.T, capacity int, timeout time.Duration) *stalledSink {
	t.Helper()
	stalled := &stalledSink{release: make(chan struct{}), started: make(chan struct{})}
	previousInner, previousCapacity, previousTimeout := oaLogInner, oaLogQueueCapacity, oaLogWriteTimeout
	oaLogInner = func(*oalogclient.Client) oalog.Sink { return stalled }
	oaLogQueueCapacity, oaLogWriteTimeout = capacity, timeout
	t.Cleanup(func() {
		oaLogInner, oaLogQueueCapacity, oaLogWriteTimeout = previousInner, previousCapacity, previousTimeout
	})
	return stalled
}

func TestOALoggingStalledServiceOverflowIsReportedAndLogCallsStayFast(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaLogEnv, "service")
	oaLogService(t, home)
	stalled := stallOALogging(t, 4, time.Minute)
	buf := &lockedBuffer{}
	local := logutil.NewLogger(buf, slog.LevelInfo)
	handler, err := oaLogHandler(t.Context(), local.Handler(), slog.LevelInfo)
	if err != nil {
		t.Fatal(err)
	}
	logger := slog.New(handler)
	logger.Info("held in flight")
	<-stalled.started
	began := time.Now()
	for i := range 20 {
		logger.Info("burst", "i", i)
	}
	if elapsed := time.Since(began); elapsed > time.Second {
		t.Fatalf("20 log calls against a stalled service took %v", elapsed)
	}
	c := handler.(*oaFanout).queue.Counts()
	if c.Accepted != 5 || c.Dropped != 16 || !c.Overflowing {
		t.Fatalf("counts %+v, want 1 in flight, 4 queued, 16 dropped", c)
	}
	text := buf.String()
	if strings.Count(text, "OpenAbstractions logging queue full") != 1 || !strings.Contains(text, "reason=queue_full") || strings.Count(text, "msg=burst") != 20 {
		t.Fatalf("overflow must be reported once and every record must reach stderr:\n%s", text)
	}
	close(stalled.release)
	c = oaLogSettled(t, handler, func(c oalog.AsyncCounts) bool { return c.Written == 5 })
	handler.(*oaFanout).close(t.Context())
	if !strings.Contains(buf.String(), "dropped=16") {
		t.Fatalf("close did not report the dropped records:\n%s", buf.String())
	}
}

func TestOALoggingUnresponsiveServiceTimesOutAndIsReported(t *testing.T) {
	home := setupOA(t)
	t.Setenv(oaLogEnv, "service")
	oaLogService(t, home)
	stallOALogging(t, 16, 100*time.Millisecond)
	buf := &lockedBuffer{}
	local := logutil.NewLogger(buf, slog.LevelInfo)
	handler, err := oaLogHandler(t.Context(), local.Handler(), slog.LevelInfo)
	if err != nil {
		t.Fatal(err)
	}
	logger := slog.New(handler)
	logger.Info("never answered 1")
	logger.Info("never answered 2")
	c := oaLogSettled(t, handler, func(c oalog.AsyncCounts) bool { return c.Failed == 2 })
	oaLogWaitFor(t, buf, "reason=write_failed")
	if strings.Count(buf.String(), "OpenAbstractions logging write failed") != 1 || !strings.Contains(buf.String(), "context deadline exceeded") || c.Written != 0 {
		t.Fatalf("unresponsive service %+v:\n%s", c, buf.String())
	}
}
