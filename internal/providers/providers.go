package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"time"

	"askmydocs/internal/rag"
)

// Voyage supplies both the dense embeddings and the cross-encoder reranker.
// ponytail: hosted rerank keeps the critical path dependency-free, but it costs
// ~600ms p95 against the plan's 150ms budget. When that budget starts binding,
// swap RerankFn for a self-hosted BGE-v2-m3 / Jina-v3 gRPC endpoint — the
// interface is one function, nothing else in the query path changes.
type Voyage struct {
	Key         string
	EmbedModel  string
	RerankModel string
	HTTP        *http.Client
}

func NewVoyage(key, embedModel, rerankModel string) *Voyage {
	return &Voyage{Key: key, EmbedModel: embedModel, RerankModel: rerankModel,
		HTTP: &http.Client{Timeout: 60 * time.Second}}
}

func (v *Voyage) post(ctx context.Context, path string, body, out any) error {
	b, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.voyageai.com/v1"+path, bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+v.Key)
	resp, err := v.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		return fmt.Errorf("voyage %s: %s: %s", path, resp.Status, truncate(string(raw), 400))
	}
	return json.Unmarshal(raw, out)
}

type embedResp struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// Embed returns one dense vector per input, in input order. inputType is
// "document" at ingest and "query" at search — asymmetric encoding is what
// makes the retrieval model work, so don't collapse the two.
func (v *Voyage) Embed(ctx context.Context, texts []string, inputType string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	var out embedResp
	err := v.post(ctx, "/embeddings", map[string]any{
		"input": texts, "model": v.EmbedModel, "input_type": inputType,
	}, &out)
	if err != nil {
		return nil, err
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("voyage: asked for %d embeddings, got %d", len(texts), len(out.Data))
	}
	vecs := make([][]float32, len(texts))
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= len(vecs) {
			return nil, fmt.Errorf("voyage: embedding index %d out of range", d.Index)
		}
		vecs[d.Index] = d.Embedding
	}
	return vecs, nil
}

type rerankResp struct {
	Data []struct {
		Index          int     `json:"index"`
		RelevanceScore float32 `json:"relevance_score"`
	} `json:"data"`
}

// Rerank scores documents against the query and returns them best-first.
func (v *Voyage) Rerank(ctx context.Context, query string, docs []string, topK int) ([]rag.Scored, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	var out rerankResp
	err := v.post(ctx, "/rerank", map[string]any{
		"query": query, "documents": docs, "model": v.RerankModel, "top_k": topK,
	}, &out)
	if err != nil {
		return nil, err
	}
	res := make([]rag.Scored, 0, len(out.Data))
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= len(docs) {
			return nil, fmt.Errorf("voyage: rerank index %d out of range", d.Index)
		}
		res = append(res, rag.Scored{Index: d.Index, Score: d.RelevanceScore})
	}
	sort.SliceStable(res, func(i, j int) bool { return res[i].Score > res[j].Score })
	return res, nil
}

// truncate shortens a string for error messages. Duplicated in rag and qdrant
// too — three lines each, not worth a shared package.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
