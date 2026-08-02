package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"net/http"
	"sort"
	"strings"
	"time"
	"unicode"
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
func (v *Voyage) Rerank(ctx context.Context, query string, docs []string, topK int) ([]Scored, error) {
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
	res := make([]Scored, 0, len(out.Data))
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= len(docs) {
			return nil, fmt.Errorf("voyage: rerank index %d out of range", d.Index)
		}
		res = append(res, Scored{Index: d.Index, Score: d.RelevanceScore})
	}
	sort.SliceStable(res, func(i, j int) bool { return res[i].Score > res[j].Score })
	return res, nil
}

type Scored struct {
	Index int
	Score float32
}

// Tokenize is the lexical half of hybrid retrieval: lowercase, split on
// non-alphanumerics, drop single characters. Deliberately dumb — no stemming,
// no stopword list — because exact-match precision on identifiers, error codes
// and part numbers is the whole point of the sparse leg.
func Tokenize(s string) []string {
	fields := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	out := fields[:0]
	for _, f := range fields {
		if len(f) > 1 {
			out = append(out, f)
		}
	}
	return out
}

// SparseEncode builds a term-frequency sparse vector. Qdrant's "idf" modifier
// supplies the inverse-document-frequency weighting server-side, which is why
// this can be a pure function of a single string.
func SparseEncode(s string) SparseVec {
	tf := map[uint32]float32{}
	for _, tok := range Tokenize(s) {
		h := fnv.New32a()
		h.Write([]byte(tok))
		tf[h.Sum32()]++
	}
	v := SparseVec{Indices: make([]uint32, 0, len(tf)), Values: make([]float32, 0, len(tf))}
	for idx := range tf {
		v.Indices = append(v.Indices, idx)
	}
	sort.Slice(v.Indices, func(i, j int) bool { return v.Indices[i] < v.Indices[j] })
	for _, idx := range v.Indices {
		v.Values = append(v.Values, tf[idx])
	}
	return v
}

func cosine(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(float64(dot) / (math.Sqrt(float64(na)) * math.Sqrt(float64(nb))))
}
