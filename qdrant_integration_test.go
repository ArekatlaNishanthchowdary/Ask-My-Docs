package main

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"
)

// Exercises the Qdrant wire format — collection schema, sparse vectors, point
// ids, server-side RRF fusion, ACL filtering — against a real server, using
// synthetic vectors so no embedding API key is needed.
//
//	docker compose up -d
//	QDRANT_TEST_URL=http://localhost:6333 go test -run Integration -v
//
// Skipped when QDRANT_TEST_URL is unset, so `go test ./...` stays offline.
func testQdrant(t *testing.T) (*Qdrant, string) {
	t.Helper()
	url := os.Getenv("QDRANT_TEST_URL")
	if url == "" {
		t.Skip("QDRANT_TEST_URL not set; skipping Qdrant integration test")
	}
	q := NewQdrant(url, os.Getenv("QDRANT_API_KEY"))
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := q.Health(ctx); err != nil {
		t.Fatalf("qdrant unreachable at %s: %v", url, err)
	}
	name := fmt.Sprintf("test_%d", time.Now().UnixNano())
	t.Cleanup(func() {
		c, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		_ = q.do(c, "DELETE", "/collections/"+name, nil, nil)
	})
	return q, name
}

const testDim = 8

// vec returns a deterministic unit-ish vector seeded by n, so "nearest
// neighbour" is predictable without a real embedding model.
func vec(seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	v := make([]float32, testDim)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

func TestIntegrationQdrantRoundTrip(t *testing.T) {
	q, coll := testQdrant(t)
	ctx := context.Background()

	if err := q.EnsureCollection(ctx, coll, testDim, 1); err != nil {
		t.Fatalf("EnsureCollection: %v", err)
	}
	// Must be idempotent — serve/ingest both call it on every start.
	if err := q.EnsureCollection(ctx, coll, testDim, 1); err != nil {
		t.Fatalf("EnsureCollection is not idempotent: %v", err)
	}

	target := vec(1)
	pts := []Point{
		{
			ID:     pointID("doc.md#0"),
			Vector: map[string]any{"dense": target, "sparse": SparseEncode("refunds are issued within 10 business days")},
			Payload: map[string]any{
				"doc_id": "doc.md", "chunk_id": "doc.md#0", "text": "refund text",
				"context": "situating context", "section": "Refunds", "acl": []string{"finance"},
			},
		},
		{
			ID:     pointID("doc.md#1"),
			Vector: map[string]any{"dense": vec(2), "sparse": SparseEncode("invoices are sent monthly ERR-4290")},
			Payload: map[string]any{
				"doc_id": "doc.md", "chunk_id": "doc.md#1", "text": "invoice text",
				"context": "", "section": "Invoicing", "acl": []string{"hr"},
			},
		},
	}
	if err := q.Upsert(ctx, coll, pts); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	// Re-upserting the same ids must overwrite, not duplicate — this is what
	// makes a full re-index safe to re-run.
	if err := q.Upsert(ctx, coll, pts); err != nil {
		t.Fatalf("re-Upsert: %v", err)
	}

	t.Run("hybrid RRF fusion returns both legs", func(t *testing.T) {
		hits, err := q.HybridSearch(ctx, coll, target, SparseEncode("refunds business days"), 10, nil)
		if err != nil {
			t.Fatalf("HybridSearch: %v", err)
		}
		if len(hits) != 2 {
			t.Fatalf("got %d hits, want 2 (re-upsert may have duplicated points)", len(hits))
		}
		if got := str(hits[0].Payload["chunk_id"]); got != "doc.md#0" {
			t.Errorf("top hit = %q, want doc.md#0", got)
		}
		if str(hits[0].Payload["text"]) == "" {
			t.Error("payload did not come back; with_payload is not taking effect")
		}
	})

	t.Run("sparse leg alone finds an exact identifier", func(t *testing.T) {
		// Dense vector points at chunk 0, but the identifier is only in chunk 1.
		// If the sparse leg is wired up, chunk 1 still surfaces.
		hits, err := q.HybridSearch(ctx, coll, target, SparseEncode("ERR-4290"), 10, nil)
		if err != nil {
			t.Fatalf("HybridSearch: %v", err)
		}
		var found bool
		for _, h := range hits {
			if str(h.Payload["chunk_id"]) == "doc.md#1" {
				found = true
			}
		}
		if !found {
			t.Error("exact identifier match did not surface via the sparse leg")
		}
	})

	t.Run("ACL filter excludes chunks the caller cannot see", func(t *testing.T) {
		hits, err := q.HybridSearch(ctx, coll, target, SparseEncode("refunds invoices"), 10,
			aclFilter([]string{"hr"}))
		if err != nil {
			t.Fatalf("HybridSearch: %v", err)
		}
		if len(hits) != 1 {
			t.Fatalf("got %d hits, want 1 — ACL filter is not being applied", len(hits))
		}
		if got := str(hits[0].Payload["chunk_id"]); got != "doc.md#1" {
			t.Errorf("ACL filter returned %q, want the hr-tagged chunk doc.md#1", got)
		}
	})

	t.Run("dense search backs the semantic cache", func(t *testing.T) {
		hits, err := q.DenseSearch(ctx, coll, target, 1)
		if err != nil {
			t.Fatalf("DenseSearch: %v", err)
		}
		if len(hits) != 1 || str(hits[0].Payload["chunk_id"]) != "doc.md#0" {
			t.Fatalf("dense nearest neighbour did not return the seeded vector: %+v", hits)
		}
		// The cache gate compares this score against CACHE_THRESHOLD, so an
		// exact vector match has to score ~1.0 or the cache never hits.
		if hits[0].Score < 0.99 {
			t.Errorf("exact vector match scored %v, want ~1.0", hits[0].Score)
		}
	})
}
