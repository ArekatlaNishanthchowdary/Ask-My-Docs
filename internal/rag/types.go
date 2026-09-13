package rag

import "context"

// AnthropicModel is the Claude model every stage of the cloud backend uses.
// It lives here, not in the provider client, because the CLI/HTTP wiring
// (which constructs providers) and this package (which reports what is
// running) both need to name it without importing each other.
const AnthropicModel = "claude-opus-5"

// Claim is one sentence of the answer plus the sources that entail it.
type Claim struct {
	Text      string   `json:"text"`
	Citations []string `json:"citations"`
}

// Answer carries no "sufficient" flag by design — an empty Claims list is the
// only way to decline, and the pipeline treats it as such.
type Answer struct {
	Claims []Claim `json:"claims"`
}

// VerifyPair is one claim and the evidence cited for it, checked for entailment.
type VerifyPair struct {
	Claim    string
	Evidence string
}

// Embedder is the retrieval half of a backend: dense vectors plus reranking.
type Embedder interface {
	Embed(ctx context.Context, texts []string, inputType string) ([][]float32, error)
	Rerank(ctx context.Context, query string, docs []string, topK int) ([]Scored, error)
}

// dimProber is implemented by embedders that can report their vector width.
type dimProber interface {
	EmbedDim(ctx context.Context) (int, error)
}

// unwrapper is implemented by an Embedder that wraps another one (the
// cross-encoder reranker override), so EnsureCollections can see through it
// to probe the real embedding model's dimension.
type unwrapper interface {
	Unwrap() Embedder
}

// LLM is the generative half: everything that needs a language model.
type LLM interface {
	Contextualize(ctx context.Context, doc string, chunks []Chunk) ([]string, error)
	Answer(ctx context.Context, question string, sources []Source) (Answer, error)
	Verify(ctx context.Context, pairs []VerifyPair) ([]bool, error)
	Judge(ctx context.Context, question, reference, candidate string) (float64, error)
}

// StageName records which backend and model a stage is currently using, so the
// UI can show it and a switch can be reported back accurately.
type StageName struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
}
