package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// A cross-encoder has a fixed maximum sequence length, and TEI rejects the
// whole batch when one pair exceeds it rather than scoring what it can:
//
//	413: `inputs` must have less than 512 tokens. Given: 513
//
// Rerank's error fails the entire query — it does not degrade to fusion order —
// so on a 512-token model that is a hard outage for any chunk near
// MAX_CHUNK_CHARS. It went unseen because bge-reranker-v2-m3 accepts 8192, and
// that is what development ran; CI on a 512-token model failed 5 of 19 items
// the first time it ever executed.
//
// One JSON field prevents it, which is exactly why it needs a test: nothing
// else in the codebase would notice it going missing until a different model
// was configured.
func TestRerankRequestAsksForTruncation(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"index":0,"score":0.5}]`))
	}))
	defer srv.Close()

	r := NewTEIReranker(srv.URL, 32)
	if _, err := r.Rerank(context.Background(), "q", []string{"a long passage"}, 1); err != nil {
		t.Fatalf("Rerank: %v", err)
	}

	if got["truncate"] != true {
		t.Errorf("request must set truncate=true so an over-long passage is scored "+
			"on its first tokens instead of failing the whole query; got %#v", got["truncate"])
	}
	// raw_scores must stay off: MIN_RERANK_SCORE is expressed on the 0..1
	// sigmoid scale, and raw logits would silently change what the gate means.
	if got["raw_scores"] != false {
		t.Errorf("raw_scores must be false; got %#v", got["raw_scores"])
	}
}
