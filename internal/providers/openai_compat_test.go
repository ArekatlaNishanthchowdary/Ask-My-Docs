package providers

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// The backoff cap is a budget for the whole call, not for one sleep. Checked
// per-sleep, four retries just under the cap could legally sit for four times
// it — the shape of the 162s p95 this was written to stop.
func TestPostBacksOffWithinItsTotalBudget(t *testing.T) {
	var attempts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		w.Header().Set("retry-after", "0.4") // +500ms slop = 900ms per sleep
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"slow down"}}`))
	}))
	defer srv.Close()

	o := &OpenAICompat{BaseURL: srv.URL, MaxBackoff: 2 * time.Second, HTTP: srv.Client()}
	start := time.Now()
	_, status, err := o.post(context.Background(), []byte(`{}`))
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("post returned an error rather than the 429: %v", err)
	}
	if status != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 429 so the caller can report the provider's own message", status)
	}
	// Two 900ms sleeps fit in the 2s budget; a third would not, so the call
	// gives up after three attempts rather than sleeping out all five.
	if attempts != 3 {
		t.Errorf("attempts = %d, want 3 (two sleeps inside a 2s budget)", attempts)
	}
	if elapsed > 2*time.Second {
		t.Errorf("post took %s, over its own %s cap", elapsed.Round(time.Millisecond), 2*time.Second)
	}
}

// A dropped connection is transient in exactly the way a 502 is. Before this,
// post returned on the first one — so a single reset failed an answer, and in
// eval silently shrank the measurement instead of failing the run.
func TestPostRetriesDroppedConnections(t *testing.T) {
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&attempts, 1) <= 2 {
			// Hijack and close without a response: the client sees EOF, which
			// is what NVIDIA NIM actually does when it drops a request.
			conn, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Errorf("hijack: %v", err)
				return
			}
			conn.Close()
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{}"}}]}`))
	}))
	defer srv.Close()

	o := &OpenAICompat{BaseURL: srv.URL, MaxBackoff: 30 * time.Second, HTTP: srv.Client()}
	raw, status, err := o.post(context.Background(), []byte(`{}`))
	if err != nil {
		t.Fatalf("post gave up on a retryable transport error: %v", err)
	}
	if status != http.StatusOK {
		t.Errorf("status = %d, want 200 once the connection held", status)
	}
	if got := atomic.LoadInt32(&attempts); got != 3 {
		t.Errorf("attempts = %d, want 3 (two drops then success)", got)
	}
	if !strings.Contains(string(raw), "choices") {
		t.Errorf("body = %q, want the successful response", truncate(string(raw), 80))
	}
}

// A cancelled context is not transient — retrying it spends the remaining
// budget failing the same way, and turns a 1s cancel into a 15s one.
func TestPostDoesNotRetryADeadContext(t *testing.T) {
	var attempts int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&attempts, 1)
		conn, _, _ := w.(http.Hijacker).Hijack()
		conn.Close()
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	o := &OpenAICompat{BaseURL: srv.URL, MaxBackoff: 30 * time.Second, HTTP: srv.Client()}
	if _, _, err := o.post(ctx, []byte(`{}`)); err == nil {
		t.Fatal("post succeeded on a cancelled context")
	}
	if got := atomic.LoadInt32(&attempts); got > 1 {
		t.Errorf("attempts = %d, want at most 1 — a dead context is not retryable", got)
	}
}
