package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// CompatEmbedder drives any OpenAI-compatible /v1/embeddings endpoint. NVIDIA
// NIM is the reason it exists, but the wire format is the same one Voyage,
// OpenAI and vLLM speak, so a new vendor is a config row rather than a type.
//
// It implements Embedder only. Reranking stays wherever RERANKER_URL points:
// none of these endpoints serves a cross-encoder.
type CompatEmbedder struct {
	BaseURL string
	Key     string
	Model   string
	HTTP    *http.Client
}

func NewCompatEmbedder(baseURL, key, model string) *CompatEmbedder {
	return &CompatEmbedder{
		BaseURL: baseURL, Key: key, Model: model,
		HTTP: &http.Client{Timeout: 2 * time.Minute},
	}
}

// Embed returns one dense vector per input, in input order.
//
// The inputType mapping is not cosmetic. Encoding the same sentence as a query
// and as a passage on nemotron-3-embed-1b gives vectors 0.76 cosine apart, so
// sending "document" (this codebase's word) straight through — or dropping the
// field — silently degrades every retrieval instead of failing loudly.
func (e *CompatEmbedder) Embed(ctx context.Context, texts []string, inputType string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	kind := "passage"
	if inputType == "query" {
		kind = "query"
	}
	body, err := json.Marshal(map[string]any{
		"model": e.Model, "input": texts, "input_type": kind,
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.BaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.Key)

	resp, err := e.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("embeddings %s: %s: %s", e.Model, resp.Status, truncate(string(raw), 400))
	}

	var out embedResp
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("embeddings %s: %w: %s", e.Model, err, truncate(string(raw), 200))
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("embeddings %s: asked for %d, got %d", e.Model, len(texts), len(out.Data))
	}
	// Index-keyed rather than positional: the response is documented as ordered,
	// but a silently transposed corpus is not a failure you would ever notice.
	vecs := make([][]float32, len(texts))
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= len(vecs) {
			return nil, fmt.Errorf("embeddings %s: index %d out of range", e.Model, d.Index)
		}
		vecs[d.Index] = d.Embedding
	}
	return vecs, nil
}

// Rerank is not served by the chat/embeddings endpoints this type talks to.
// Returning an error rather than the input order keeps the failure loud: a
// silent identity rerank looks like a working pipeline that quietly lost its
// most accurate stage. NewApp requires RERANKER_URL for these providers, so
// rerankedEmbedder overrides this in practice and it stays unreachable.
func (e *CompatEmbedder) Rerank(context.Context, string, []string, int) ([]Scored, error) {
	return nil, fmt.Errorf("%s serves no reranker; set RERANKER_URL", e.BaseURL)
}

// EmbedDim probes the model rather than trusting a constant: creating the
// Qdrant collection at the wrong width is a silent corpus-wide failure.
func (e *CompatEmbedder) EmbedDim(ctx context.Context) (int, error) {
	v, err := e.Embed(ctx, []string{"dimension probe"}, "query")
	if err != nil {
		return 0, err
	}
	return len(v[0]), nil
}
