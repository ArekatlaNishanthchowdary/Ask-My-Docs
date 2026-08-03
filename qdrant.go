package main

import (
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// ponytail: Qdrant REST over stdlib net/http instead of the gRPC client — six
// endpoints, no codegen, no dependency. Switch to the official client if the
// per-request JSON marshalling ever shows up in a profile.
type Qdrant struct {
	URL  string
	Key  string
	HTTP *http.Client
}

func NewQdrant(url, key string) *Qdrant {
	return &Qdrant{URL: strings.TrimRight(url, "/"), Key: key,
		HTTP: &http.Client{Timeout: 30 * time.Second}}
}

func (q *Qdrant) do(ctx context.Context, method, path string, body, out any) error {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rdr = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, q.URL+path, rdr)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if q.Key != "" {
		req.Header.Set("api-key", q.Key)
	}
	resp, err := q.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		return fmt.Errorf("qdrant %s %s: %s: %s", method, path, resp.Status, truncate(string(raw), 400))
	}
	if out != nil {
		return json.Unmarshal(raw, out)
	}
	return nil
}

// SparseVec is Qdrant's sparse representation: term-hash -> term frequency.
// IDF is applied server-side via the collection's "idf" modifier, so the
// client never needs corpus statistics.
type SparseVec struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
}

// EnsureCollection creates the collection if absent. Existing collections are
// left untouched — re-indexing after a schema change is a deliberate act.
func (q *Qdrant) EnsureCollection(ctx context.Context, name string, dim int, shards int) error {
	var probe map[string]any
	if err := q.do(ctx, http.MethodGet, "/collections/"+name, nil, &probe); err == nil {
		return nil
	}
	body := map[string]any{
		"shard_number":       shards,
		"replication_factor": 1,
		"vectors": map[string]any{
			"dense": map[string]any{
				"size":     dim,
				"distance": "Cosine",
				// Scalar quantization keeps 1M+ vectors resident in RAM at ~1/4 the size.
				"quantization_config": map[string]any{
					"scalar": map[string]any{"type": "int8", "quantile": 0.99, "always_ram": true},
				},
			},
		},
		"sparse_vectors": map[string]any{
			"sparse": map[string]any{"modifier": "idf"},
		},
	}
	if err := q.do(ctx, http.MethodPut, "/collections/"+name, body, nil); err != nil {
		return err
	}
	for _, f := range []string{"doc_id", "acl", "version"} {
		idx := map[string]any{"field_name": f, "field_schema": "keyword"}
		if err := q.do(ctx, http.MethodPut, "/collections/"+name+"/index?wait=true", idx, nil); err != nil {
			return err
		}
	}
	return nil
}

type Point struct {
	ID      string         `json:"id"`
	Vector  map[string]any `json:"vector"`
	Payload map[string]any `json:"payload"`
}

func (q *Qdrant) Upsert(ctx context.Context, name string, pts []Point) error {
	if len(pts) == 0 {
		return nil
	}
	return q.do(ctx, http.MethodPut, "/collections/"+name+"/points?wait=true",
		map[string]any{"points": pts}, nil)
}

type Hit struct {
	ID      string         `json:"id"`
	Score   float32        `json:"score"`
	Payload map[string]any `json:"payload"`
}

type queryResp struct {
	Result struct {
		Points []Hit `json:"points"`
	} `json:"result"`
}

// HybridSearch fires dense ANN and sparse lexical retrieval as Qdrant prefetches
// and fuses them with Reciprocal Rank Fusion server-side — one network round
// trip instead of two concurrent ones plus client-side fusion.
func (q *Qdrant) HybridSearch(ctx context.Context, name string, dense []float32, sparse SparseVec, limit int, filter map[string]any) ([]Hit, error) {
	pre := []map[string]any{
		{"query": dense, "using": "dense", "limit": limit},
		{"query": sparse, "using": "sparse", "limit": limit},
	}
	if filter != nil {
		for _, p := range pre {
			p["filter"] = filter
		}
	}
	body := map[string]any{
		"prefetch":     pre,
		"query":        map[string]any{"fusion": "rrf"},
		"limit":        limit,
		"with_payload": true,
	}
	var out queryResp
	if err := q.do(ctx, http.MethodPost, "/collections/"+name+"/points/query", body, &out); err != nil {
		return nil, err
	}
	return out.Result.Points, nil
}

// DenseSearch is the semantic cache lookup: nearest neighbour on the query
// embedding alone, no fusion, no rerank.
func (q *Qdrant) DenseSearch(ctx context.Context, name string, dense []float32, limit int) ([]Hit, error) {
	body := map[string]any{"query": dense, "using": "dense", "limit": limit, "with_payload": true}
	var out queryResp
	if err := q.do(ctx, http.MethodPost, "/collections/"+name+"/points/query", body, &out); err != nil {
		return nil, err
	}
	return out.Result.Points, nil
}

func (q *Qdrant) Health(ctx context.Context) error {
	return q.do(ctx, http.MethodGet, "/collections", nil, nil)
}

// DropCollection removes a collection. Absent is success — the caller wants it
// gone, and it is.
func (q *Qdrant) DropCollection(ctx context.Context, name string) error {
	err := q.do(ctx, http.MethodDelete, "/collections/"+name, nil, nil)
	if err != nil && strings.Contains(err.Error(), "404") {
		return nil
	}
	return err
}

func (q *Qdrant) Count(ctx context.Context, name string) (int, error) {
	return q.CountFiltered(ctx, name, nil)
}

// CountFiltered counts points matching a filter; a nil filter counts everything.
func (q *Qdrant) CountFiltered(ctx context.Context, name string, filter map[string]any) (int, error) {
	body := map[string]any{"exact": true}
	if filter != nil {
		body["filter"] = filter
	}
	var out struct {
		Result struct {
			Count int `json:"count"`
		} `json:"result"`
	}
	err := q.do(ctx, http.MethodPost, "/collections/"+name+"/points/count", body, &out)
	return out.Result.Count, err
}

// DocIDFilter matches every chunk belonging to one document.
func DocIDFilter(docID string) map[string]any {
	return map[string]any{
		"must": []map[string]any{
			{"key": "doc_id", "match": map[string]any{"value": docID}},
		},
	}
}

// DocVersionFilter matches one document's chunks only at one version, which is
// how ingest asks "is this file already indexed as it is on disk right now?"
// without fetching a payload.
func DocVersionFilter(docID, version string) map[string]any {
	return map[string]any{
		"must": []map[string]any{
			{"key": "doc_id", "match": map[string]any{"value": docID}},
			{"key": "version", "match": map[string]any{"value": version}},
		},
	}
}

// pointID maps a stable string key onto the UUID shape Qdrant requires, so
// re-ingesting a document overwrites its chunks instead of duplicating them.
func pointID(key string) string {
	s := sha1.Sum([]byte(key))
	h := fmt.Sprintf("%x", s[:16])
	return fmt.Sprintf("%s-%s-%s-%s-%s", h[0:8], h[8:12], h[12:16], h[16:20], h[20:32])
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
