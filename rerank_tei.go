package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"time"

	"golang.org/x/sync/errgroup"
)

// TEIReranker is a real cross-encoder behind HuggingFace Text Embeddings
// Inference, which exposes a native /rerank endpoint.
//
// This exists because an LLM scoring passages is the wrong tool twice over: it
// is ~30x over the latency budget, and it miscalibrates — on the eval set it
// scored a rank-1 correct chunk 0.00, which tripped the confidence gate and
// refused an answerable question. A cross-encoder is a model trained for
// exactly this comparison, so it is both faster and better.
//
// It implements only Rerank; embeddings stay with whichever provider is
// configured. Compose it over any Embedder with rerankedEmbedder.
type TEIReranker struct {
	URL   string
	Batch int
	HTTP  *http.Client
}

func NewTEIReranker(url string, batch int) *TEIReranker {
	if batch <= 0 {
		batch = 32
	}
	return &TEIReranker{URL: trimSlash(url), Batch: batch,
		HTTP: &http.Client{Timeout: 2 * time.Minute}}
}

// Rerank scores every candidate, splitting the request to respect the server's
// client-batch limit (TEI defaults to 32, while CANDIDATE_K is typically 50).
//
// Splitting is exact rather than an approximation: a cross-encoder scores each
// (query, document) pair independently, so a document's score does not depend
// on which other documents shared its request. Batching therefore changes
// throughput, never the ranking.
func (t *TEIReranker) Rerank(ctx context.Context, query string, docs []string, topK int) ([]Scored, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	scores := make([]Scored, len(docs))
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(4)

	for start := 0; start < len(docs); start += t.Batch {
		start := start
		end := min(start+t.Batch, len(docs))
		g.Go(func() error {
			part, err := t.rerankBatch(gctx, query, docs[start:end])
			if err != nil {
				return err
			}
			for _, s := range part {
				// Indexes come back relative to the batch; shift them into the
				// caller's numbering.
				scores[start+s.Index] = Scored{Index: start + s.Index, Score: s.Score}
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}

	sort.SliceStable(scores, func(i, j int) bool { return scores[i].Score > scores[j].Score })
	if topK > 0 && len(scores) > topK {
		scores = scores[:topK]
	}
	return scores, nil
}

func (t *TEIReranker) rerankBatch(ctx context.Context, query string, docs []string) ([]Scored, error) {
	body, err := json.Marshal(map[string]any{
		"query":      query,
		"texts":      docs,
		"raw_scores": false, // sigmoid-normalised to 0..1, so MIN_RERANK_SCORE is comparable
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.URL+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := t.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("tei rerank: %w (is the reranker up at %s? `docker compose up -d reranker`)", err, t.URL)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("tei rerank: %s: %s", resp.Status, truncate(string(raw), 300))
	}

	var out []struct {
		Index int     `json:"index"`
		Score float32 `json:"score"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("tei rerank: %w: %s", err, truncate(string(raw), 200))
	}

	res := make([]Scored, 0, len(out))
	for _, s := range out {
		if s.Index < 0 || s.Index >= len(docs) {
			return nil, fmt.Errorf("tei rerank: index %d out of range", s.Index)
		}
		res = append(res, Scored{Index: s.Index, Score: s.Score})
	}
	return res, nil
}

func (t *TEIReranker) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, t.URL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := t.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("tei health: %s", resp.Status)
	}
	return nil
}

// rerankedEmbedder takes embeddings from one backend and reranking from
// another. Reranking is the stage most worth specialising, and it is the only
// one whose implementation is genuinely independent of the rest.
type rerankedEmbedder struct {
	Embedder
	reranker *TEIReranker
}

func (r rerankedEmbedder) Rerank(ctx context.Context, query string, docs []string, topK int) ([]Scored, error) {
	return r.reranker.Rerank(ctx, query, docs, topK)
}

func trimSlash(s string) string {
	for len(s) > 0 && s[len(s)-1] == '/' {
		s = s[:len(s)-1]
	}
	return s
}
