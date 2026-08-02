package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"
)

// Ollama runs the whole pipeline on local models. It satisfies both Embedder
// and LLM, so a single backend covers embedding, reranking, and generation.
//
// Two things differ from the hosted providers and shape the code below:
//   - Ollama has no rerank endpoint (/api/rerank is 404), so the cross-encoder
//     stage is LLM-scored instead. See Rerank.
//   - A 7B model will not reliably handle 40 chunks or 50 documents in one
//     prompt, so batches are small and issued concurrently.
type Ollama struct {
	URL         string
	EmbedModel  string
	ChatModel   string
	NumCtx      int
	DocPrefix   string // bge-m3 needs none; nomic-embed-text wants "search_document: "
	QueryPrefix string // ...and "search_query: "
	RerankBatch int
	CtxBatch    int
	Concurrency int
	Seed        int
	HTTP        *http.Client
}

func NewOllama(cfg Config) *Ollama {
	return &Ollama{
		URL:         strings.TrimRight(cfg.OllamaURL, "/"),
		EmbedModel:  cfg.OllamaEmbedModel,
		ChatModel:   cfg.OllamaChatModel,
		NumCtx:      cfg.OllamaNumCtx,
		DocPrefix:   cfg.OllamaDocPrefix,
		QueryPrefix: cfg.OllamaQueryPrefix,
		RerankBatch: cfg.OllamaRerankBatch,
		CtxBatch:    cfg.OllamaContextBatch,
		Concurrency: cfg.OllamaConcurrency,
		Seed:        cfg.OllamaSeed,
		// Local generation is slow; a short timeout here just turns a working
		// pipeline into a flaky one.
		HTTP: &http.Client{Timeout: 10 * time.Minute},
	}
}

func (o *Ollama) post(ctx context.Context, path string, body, out any) error {
	b, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.URL+path, bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("ollama %s: %w (is `ollama serve` running at %s?)", path, err, o.URL)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		hint := ""
		if resp.StatusCode == 501 {
			hint = fmt.Sprintf(" — %q is not an embedding model; pull one (e.g. `ollama pull bge-m3`)", o.EmbedModel)
		}
		if resp.StatusCode == 404 {
			hint = " — model not pulled? run `ollama pull <model>`"
		}
		return fmt.Errorf("ollama %s: %s: %s%s", path, resp.Status, truncate(string(raw), 300), hint)
	}
	return json.Unmarshal(raw, out)
}

// --- Embedder -------------------------------------------------------------

func (o *Ollama) Embed(ctx context.Context, texts []string, inputType string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	// Asymmetric models encode queries and passages differently. bge-m3 needs
	// no prefix, so both default to empty; set them when swapping in a model
	// that does (nomic-embed-text, E5).
	prefix := o.DocPrefix
	if inputType == "query" {
		prefix = o.QueryPrefix
	}
	in := make([]string, len(texts))
	for i, t := range texts {
		in[i] = prefix + t
	}

	var out struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := o.post(ctx, "/api/embed", map[string]any{
		"model": o.EmbedModel, "input": in, "truncate": true,
	}, &out); err != nil {
		return nil, err
	}
	if len(out.Embeddings) != len(texts) {
		return nil, fmt.Errorf("ollama embed: asked for %d vectors, got %d", len(texts), len(out.Embeddings))
	}
	return out.Embeddings, nil
}

const rerankSystem = `You score how well each numbered passage answers a specific question.

Score each passage from 0.0 to 1.0:
  1.0  the passage directly and completely answers the question
  0.7  the passage contains part of the answer
  0.3  the passage is on the same topic but does not answer the question
  0.0  the passage is unrelated

Judge only whether the passage answers THIS question. A well-written passage
about something else scores 0.0.

Return one entry per passage, each carrying the passage's own index. Every
index you were given must appear exactly once, including the ones you score 0.`

// Rerank is the cross-encoder stage, done with an LLM because Ollama exposes no
// rerank endpoint. Passages are scored in small concurrent batches: a 7B model
// loses calibration when asked to rank 50 items at once, and one call per
// passage would serialise into tens of seconds.
//
// ponytail: this is the weakest link in the local pipeline and the one to
// replace first — a real cross-encoder (BGE-v2-m3 behind ONNX Runtime) is both
// faster and better. The swap point is this method and nothing else.
func (o *Ollama) Rerank(ctx context.Context, query string, docs []string, topK int) ([]Scored, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	scores := make([]float32, len(docs))
	scored := make([]bool, len(docs))
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(o.Concurrency)

	for start := 0; start < len(docs); start += o.RerankBatch {
		start := start
		end := min(start+o.RerankBatch, len(docs))
		g.Go(func() error {
			// Two attempts: a 7B model intermittently drops or duplicates an
			// entry, and a re-roll is cheaper than degrading the whole ranking.
			for attempt := 0; attempt < 2; attempt++ {
				got, err := o.scoreBatch(gctx, query, docs[start:end])
				if err != nil {
					return err
				}
				for i, s := range got {
					if s.ok {
						scores[start+i], scored[start+i] = s.score, true
					}
				}
				if allTrue(scored[start:end]) {
					return nil
				}
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}

	// Anything the model never scored keeps a neutral score rather than zero.
	// Zeroing an unscored passage would silently delete a possibly-relevant
	// chunk on a model hiccup; neutral leaves it ordered by its RRF rank.
	missing := 0
	for i := range scores {
		if !scored[i] {
			scores[i], missing = 0.5, missing+1
		}
	}
	if missing > 0 {
		fmt.Fprintf(os.Stderr, "warn: rerank: %d/%d passages unscored by %s, left at neutral\n",
			missing, len(docs), o.ChatModel)
	}

	out := make([]Scored, len(docs))
	for i, s := range scores {
		out[i] = Scored{Index: i, Score: s}
	}
	// Stable sort keeps the fused RRF order as the tiebreak, which matters a
	// lot here because a 7B scorer emits plateaus of identical scores.
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if topK > 0 && len(out) > topK {
		out = out[:topK]
	}
	return out, nil
}

type batchScore struct {
	score float32
	ok    bool
}

// scoreBatch scores one batch, returning results positionally. Scores come back
// index-keyed rather than as a bare array: a small model that drops or reorders
// an entry then produces a detectable gap instead of silently shifting every
// subsequent passage's score onto the wrong document.
func (o *Ollama) scoreBatch(ctx context.Context, query string, docs []string) ([]batchScore, error) {
	var sb strings.Builder
	fmt.Fprintf(&sb, "Question: %s\n\n", query)
	for i, d := range docs {
		// Bound the prompt: the lead of a passage carries the signal, and a 7B
		// context window fills fast.
		fmt.Fprintf(&sb, "<passage index=\"%d\">\n%s\n</passage>\n\n", i, truncate(d, 1200))
	}
	fmt.Fprintf(&sb, "Score all %d passages, indexes 0 to %d.", len(docs), len(docs)-1)

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"scores": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"index": map[string]any{"type": "integer"},
						"score": map[string]any{"type": "number"},
					},
					"required": []string{"index", "score"},
				},
			},
		},
		"required": []string{"scores"},
	}
	var res struct {
		Scores []struct {
			Index int     `json:"index"`
			Score float32 `json:"score"`
		} `json:"scores"`
	}
	if err := o.chat(ctx, rerankSystem, sb.String(), schema, &res); err != nil {
		return nil, err
	}

	out := make([]batchScore, len(docs))
	for _, s := range res.Scores {
		if s.Index < 0 || s.Index >= len(docs) || out[s.Index].ok {
			continue // out of range, or a duplicate index — ignore, don't guess
		}
		// Models occasionally answer on a 0-10 or 0-100 scale despite the rubric.
		score := s.Score
		if score > 1 {
			score = score / 10
			if score > 1 {
				score = score / 10
			}
		}
		if score < 0 {
			score = 0
		}
		out[s.Index] = batchScore{score: min(score, 1), ok: true}
	}
	return out, nil
}

func allTrue(bs []bool) bool {
	for _, b := range bs {
		if !b {
			return false
		}
	}
	return true
}

// --- LLM ------------------------------------------------------------------

// chat issues one structured-output request. Ollama constrains generation to
// the supplied JSON schema, which is what makes the citation contract
// enforceable on a small local model.
func (o *Ollama) chat(ctx context.Context, system, user string, schema map[string]any, out any) error {
	body := map[string]any{
		"model":  o.ChatModel,
		"stream": false,
		"format": schema,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"options": map[string]any{
			"temperature": 0, // extraction and scoring, not prose
			"num_ctx":     o.NumCtx,
			// A fixed seed makes runs reproducible. Without it the CI gate
			// compares a fresh sample against the baseline every time and
			// flaps on sampling noise rather than on real regressions.
			"seed": o.Seed,
		},
	}
	var res struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := o.post(ctx, "/api/chat", body, &res); err != nil {
		return err
	}
	if strings.TrimSpace(res.Message.Content) == "" {
		return fmt.Errorf("ollama chat: empty response from %s", o.ChatModel)
	}
	if err := json.Unmarshal([]byte(res.Message.Content), out); err != nil {
		return fmt.Errorf("ollama chat: response did not match schema: %w: %s",
			err, truncate(res.Message.Content, 200))
	}
	return nil
}

func (o *Ollama) Contextualize(ctx context.Context, doc string, chunks []Chunk) ([]string, error) {
	out := make([]string, len(chunks))
	// The document is repeated in every batch's prompt. Ollama reuses the KV
	// cache for a shared prefix, so keeping it first is close to free after the
	// first call; truncation bounds the worst case on long documents.
	docView := truncate(doc, o.NumCtx*3)

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(o.Concurrency)
	for start := 0; start < len(chunks); start += o.CtxBatch {
		start := start
		end := min(start+o.CtxBatch, len(chunks))
		g.Go(func() error {
			batch := chunks[start:end]
			var sb strings.Builder
			for i, c := range batch {
				fmt.Fprintf(&sb, "<chunk index=\"%d\" section=\"%s\">\n%s\n</chunk>\n\n", i, c.Section, c.Text)
			}
			fmt.Fprintf(&sb, "Return exactly %d contexts, in order.", len(batch))

			schema := map[string]any{
				"type": "object",
				"properties": map[string]any{
					"contexts": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
				},
				"required": []string{"contexts"},
			}
			var res struct {
				Contexts []string `json:"contexts"`
			}
			err := o.chat(gctx, fmt.Sprintf(contextSystem, docView),
				fmt.Sprintf(contextUser, len(batch), len(batch), sb.String()), schema, &res)
			if err != nil {
				return err
			}
			if len(res.Contexts) != len(batch) {
				return fmt.Errorf("ollama contextualize: expected %d contexts, got %d", len(batch), len(res.Contexts))
			}
			copy(out[start:end], res.Contexts)
			return nil
		})
	}
	return out, g.Wait()
}

func (o *Ollama) Answer(ctx context.Context, question string, sources []Source) (Answer, error) {
	var sb strings.Builder
	sb.WriteString("Sources:\n\n")
	for _, s := range sources {
		fmt.Fprintf(&sb, "<source id=\"%s\" doc=\"%s\" section=\"%s\">\n%s\n</source>\n\n",
			s.ChunkID, s.DocID, s.Section, s.Text)
	}
	fmt.Fprintf(&sb, "Question: %s\n\nCite sources by their exact id, e.g. %q.",
		question, sources[0].ChunkID)

	var a Answer
	err := o.chat(ctx, answerSystem, sb.String(), answerSchema, &a)
	return a, err
}

// Verify checks one claim/evidence pair per call rather than batching them.
// There are only ever a handful of pairs, so batching saves little, and a
// single boolean is the one shape a 7B model gets right every time — which
// matters more here than anywhere else, because a dropped verdict in a batched
// array would either refuse a good answer or wave through an unverified one.
func (o *Ollama) Verify(ctx context.Context, pairs []VerifyPair) ([]bool, error) {
	if len(pairs) == 0 {
		return nil, nil
	}
	out := make([]bool, len(pairs))
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(o.Concurrency)

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"entailed": map[string]any{"type": "boolean"},
			"reason":   map[string]any{"type": "string"},
		},
		"required": []string{"entailed", "reason"},
	}
	for i, p := range pairs {
		i, p := i, p
		g.Go(func() error {
			user := fmt.Sprintf("<claim>%s</claim>\n\n<evidence>%s</evidence>\n\nDoes the evidence entail the claim?",
				p.Claim, truncate(p.Evidence, 8000))
			var res struct {
				Entailed bool `json:"entailed"`
			}
			if err := o.chat(gctx, verifySystem, user, schema, &res); err != nil {
				return err
			}
			out[i] = res.Entailed
			return nil
		})
	}
	return out, g.Wait()
}

func (o *Ollama) Judge(ctx context.Context, question, reference, candidate string) (float64, error) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"score":     map[string]any{"type": "number"},
			"reasoning": map[string]any{"type": "string"},
		},
		"required": []string{"score", "reasoning"},
	}
	var res struct {
		Score float64 `json:"score"`
	}
	user := fmt.Sprintf("Question: %s\n\nReference answer: %s\n\nCandidate answer: %s",
		question, reference, candidate)
	if err := o.chat(ctx, judgeSystem, user, schema, &res); err != nil {
		return 0, err
	}
	return res.Score, nil
}

// EmbedDim probes the embedding model for its dimensionality. Creating the
// Qdrant collection with the wrong size is a silent, corpus-wide failure that
// only shows up as bad retrieval, so it is worth one round trip at startup.
func (o *Ollama) EmbedDim(ctx context.Context) (int, error) {
	v, err := o.Embed(ctx, []string{"dimension probe"}, "query")
	if err != nil {
		return 0, err
	}
	return len(v[0]), nil
}
