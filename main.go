package main

import (
	"bufio"
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Config struct {
	Provider        string // "ollama" (local, default) or "cloud" (Voyage + Claude)
	QdrantURL       string
	QdrantKey       string
	Collection      string
	CacheCollection string
	Shards          int
	EmbedModel      string
	RerankModel     string
	EmbedDim        int
	EmbedDimSet     bool // true when EMBED_DIM was set explicitly
	EmbedBatch      int
	RerankerURL     string
	RerankerBatch   int

	// LLMProvider overrides just the generative stage. Empty follows Provider.
	LLMProvider           string
	JudgeProvider         string
	VerifyProvider        string
	VerifyModel           string
	ContextProvider       string
	ContextModel          string
	OpenAIBaseURL         string
	OpenAIKey             string
	OpenAIModel           string
	OpenAIJudgeModel      string
	OpenAIMaxTokens       int
	OpenAIContextBatch    int
	OpenAIConcurrency     int
	OpenAIMaxBackoffSec   int
	OpenAIReasoningEffort string
	OpenAISystemPrefix    string
	OpenAIEmbedModel      string
	NvidiaBaseURL         string
	NvidiaKey             string
	NvidiaModel           string
	NvidiaSystemPrefix    string
	NvidiaEmbedModel      string

	OllamaURL          string
	OllamaEmbedModel   string
	OllamaChatModel    string
	OllamaNumCtx       int
	OllamaDocPrefix    string
	OllamaQueryPrefix  string
	OllamaRerankBatch  int
	OllamaContextBatch int
	OllamaConcurrency  int
	OllamaSeed         int

	MaxChunkChars     int
	ChunkOverlap      int
	IngestConcurrency int
	ContextDocChars   int

	// Ablate switches individual pipeline stages off so `eval` can measure what
	// each one is worth instead of asserting it. See ablatable.
	Ablate map[string]bool

	CorpusDir         string
	MaxUploadMB       int
	RequestTimeoutSec int
	AuthTokensFile    string
	AllowAnonymous    bool
	CandidateK        int     // fused candidates handed to the reranker
	TopK              int     // chunks handed to the model
	MinRerankScore    float32 // confidence gate
	RerankMargin      float32 // gate = max(MinRerankScore, RerankMargin x median candidate)
	CacheThreshold    float32 // semantic-cache cosine similarity

	Addr string
}

func LoadConfig() Config {
	return Config{
		Provider:        env("PROVIDER", "ollama"),
		QdrantURL:       env("QDRANT_URL", "http://localhost:6333"),
		QdrantKey:       os.Getenv("QDRANT_API_KEY"),
		Collection:      env("QDRANT_COLLECTION", "docs"),
		CacheCollection: env("QDRANT_CACHE_COLLECTION", "query_cache"),
		Shards:          envInt("QDRANT_SHARDS", 4),
		EmbedModel:      env("VOYAGE_EMBED_MODEL", "voyage-3.5"),
		RerankModel:     env("VOYAGE_RERANK_MODEL", "rerank-2.5"),
		EmbedDim:        envInt("EMBED_DIM", 1024),
		EmbedDimSet:     os.Getenv("EMBED_DIM") != "",
		EmbedBatch:      envInt("EMBED_BATCH", 64),
		RerankerURL:     os.Getenv("RERANKER_URL"),
		RerankerBatch:   envInt("RERANKER_BATCH", 32),

		LLMProvider:    os.Getenv("LLM_PROVIDER"),
		JudgeProvider:  os.Getenv("JUDGE_PROVIDER"),
		VerifyProvider: os.Getenv("VERIFY_PROVIDER"),
		// Naming the verifier's model separately is what lets it differ from the
		// generator on the *same* endpoint — otherwise both stages collapse onto
		// that provider's single configured model, and the second opinion the
		// verifier exists to give comes from the model that wrote the claim.
		VerifyModel: os.Getenv("VERIFY_MODEL"),
		// Contextualization runs once per batch of chunks at ingest and never
		// at query time, so it is the one stage where a small fast model is
		// the obvious choice regardless of what generates answers.
		ContextProvider: os.Getenv("CONTEXT_PROVIDER"),
		ContextModel:    os.Getenv("CONTEXT_MODEL"),
		// Defaults to Groq; point OPENAI_BASE_URL at OpenRouter, Together,
		// Fireworks, vLLM or LM Studio to use those instead.
		OpenAIBaseURL:         env("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
		OpenAIKey:             os.Getenv("OPENAI_API_KEY"),
		OpenAIModel:           os.Getenv("OPENAI_MODEL"),
		OpenAIJudgeModel:      os.Getenv("OPENAI_JUDGE_MODEL"),
		OpenAIMaxTokens:       envInt("OPENAI_MAX_TOKENS", 4096),
		OpenAIContextBatch:    envInt("OPENAI_CONTEXT_BATCH", 20),
		OpenAIConcurrency:     envInt("OPENAI_CONCURRENCY", 4),
		OpenAIMaxBackoffSec:   envInt("OPENAI_MAX_BACKOFF_SEC", 45),
		OpenAIReasoningEffort: os.Getenv("OPENAI_REASONING_EFFORT"),
		OpenAISystemPrefix:    os.Getenv("OPENAI_SYSTEM_PREFIX"),
		OpenAIEmbedModel:      os.Getenv("OPENAI_EMBED_MODEL"),
		NvidiaBaseURL:         env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
		NvidiaKey:             os.Getenv("NVIDIA_API_KEY"),
		NvidiaModel:           os.Getenv("NVIDIA_MODEL"),
		// Nemotron defaults to a thinking mode that exhausts max_tokens before
		// emitting JSON. Set NVIDIA_SYSTEM_PREFIX="detailed thinking on" to
		// restore it for a model where the reasoning is worth the tokens.
		NvidiaSystemPrefix: env("NVIDIA_SYSTEM_PREFIX", "detailed thinking off"),
		NvidiaEmbedModel:   env("NVIDIA_EMBED_MODEL", "nvidia/nemotron-3-embed-1b"),

		OllamaURL: env("OLLAMA_URL", "http://localhost:11434"),
		// No model defaults: running against a model you did not choose is worse
		// than a startup error that names the variable to set.
		OllamaEmbedModel: os.Getenv("OLLAMA_EMBED_MODEL"),
		OllamaChatModel:  os.Getenv("OLLAMA_CHAT_MODEL"),
		OllamaNumCtx:     envInt("OLLAMA_NUM_CTX", 8192),
		// bge-m3 is symmetric and needs no prefixes. Models that are asymmetric
		// (nomic-embed-text, E5) retrieve badly without them — set both.
		OllamaDocPrefix:    os.Getenv("OLLAMA_DOC_PREFIX"),
		OllamaQueryPrefix:  os.Getenv("OLLAMA_QUERY_PREFIX"),
		OllamaRerankBatch:  envInt("OLLAMA_RERANK_BATCH", 10),
		OllamaContextBatch: envInt("OLLAMA_CONTEXT_BATCH", 8),
		OllamaConcurrency:  envInt("OLLAMA_CONCURRENCY", 4),
		OllamaSeed:         envInt("OLLAMA_SEED", 42),

		MaxChunkChars:     envInt("MAX_CHUNK_CHARS", 1600),
		ChunkOverlap:      envInt("CHUNK_OVERLAP", 200),
		IngestConcurrency: envInt("INGEST_CONCURRENCY", 8),
		// Ceiling on how much of a document contextualization is allowed to
		// resend per batch. See docDigest: this is the knob that decides whether
		// a long document costs tokens proportional to its length or to its
		// length squared. 0 disables the bound and restores the old behaviour.
		ContextDocChars: envInt("CONTEXT_DOC_CHARS", 6000),

		CorpusDir:         env("CORPUS_DIR", "corpus"),
		MaxUploadMB:       envInt("MAX_UPLOAD_MB", 32),
		RequestTimeoutSec: envInt("REQUEST_TIMEOUT_SEC", 240),
		AuthTokensFile:    os.Getenv("AUTH_TOKENS_FILE"),
		AllowAnonymous:    os.Getenv("ALLOW_ANONYMOUS") != "",
		CandidateK:        envInt("CANDIDATE_K", 50),
		TopK:              envInt("TOP_K", 10),
		// The gate threshold is meaningless across rerankers: an LLM scorer emits
		// 0/0.3/0.7/1.0 buckets, a cross-encoder emits sigmoid scores where
		// irrelevant chunks sit near 0.001. Carrying 0.30 onto a cross-encoder
		// refuses every query. Default per backend, and run `calibrate` to
		// derive the real value for your own corpus.
		MinRerankScore: float32(envFloat("MIN_RERANK_SCORE", defaultGate(os.Getenv("RERANKER_URL")))),
		RerankMargin:   float32(envFloat("RERANK_MARGIN", 3.0)),
		CacheThreshold: float32(envFloat("CACHE_THRESHOLD", 0.97)),

		Addr: env("ADDR", ":8080"),
	}
}

//go:embed ui.html
var uiHTML []byte

// defaultGate picks a starting confidence threshold for whichever reranker is
// in play. Both values are starting points, not tuned constants — `calibrate`
// derives the right one from the golden set.
func defaultGate(rerankerURL string) float64 {
	if rerankerURL != "" {
		return 0.002 // cross-encoder floor; RERANK_MARGIN does the real work
	}
	return 0.30 // LLM scorer: coarse 0/0.3/0.7/1.0 buckets
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

// LLM is the generative half: everything that needs a language model.
type LLM interface {
	Contextualize(ctx context.Context, doc string, chunks []Chunk) ([]string, error)
	Answer(ctx context.Context, question string, sources []Source) (Answer, error)
	Verify(ctx context.Context, pairs []VerifyPair) ([]bool, error)
	Judge(ctx context.Context, question, reference, candidate string) (float64, error)
}

type App struct {
	Cfg      Config
	Qdrant   *Qdrant
	Embedder Embedder
	LLM      LLM
	// Judger scores answers during eval. It defaults to LLM, which means the
	// generator grades itself — convenient, but not a measurement. Set
	// JUDGE_PROVIDER to make answer_correctness independent of the model
	// being measured.
	Judger LLM

	// Verifier runs the entailment check; defaults to LLM.
	Verifier LLM

	// Contextualizer writes the situating preamble at ingest; defaults to LLM.
	// Not runtime-swappable, unlike the three above: it only runs at ingest,
	// and its output is already baked into the vectors of everything indexed.
	Contextualizer LLM

	// Tokens is nil when the server runs unauthenticated, which cmdServe only
	// permits with ALLOW_ANONYMOUS set explicitly.
	Tokens TokenStore

	// ingesting serialises POST /ingest against itself. Held for the whole
	// walk, which can be minutes.
	ingesting sync.Mutex

	// embedDim is the probed vector width, kept so collections can be recreated
	// (e.g. clearing the cache) without probing the model again.
	embedDim int

	// mu guards the generative backends, which can be swapped at runtime while
	// requests are in flight. Embeddings are deliberately NOT swappable: the
	// index holds vectors from one specific model, and changing it would not
	// error — it would silently retrieve nonsense.
	mu          sync.RWMutex
	llmName     stageName
	verifyName  stageName
	judgeName   stageName
	contextName stageName
}

// stageName records which backend and model a stage is currently using, so the
// UI can show it and a switch can be reported back accurately.
type stageName struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

func (a *App) llm() LLM {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.LLM
}

func (a *App) verifier() LLM {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.Verifier
}

func (a *App) judger() LLM {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.Judger
}

// compatEndpoint is one OpenAI-compatible service. Groq, NVIDIA NIM,
// OpenRouter, Together, vLLM and LM Studio all speak the same wire format, so
// they differ only by base URL, key and model — no per-vendor code.
type compatEndpoint struct {
	Name    string
	BaseURL string
	Key     string
	Model   string
	// SysPrefix is prepended to every system prompt sent to this endpoint,
	// for models that take behaviour directives in the prompt instead of a
	// request field. See OpenAICompat.request.
	SysPrefix string
	// EmbedModel is separate from Model: the chat model and the embedding
	// model on the same endpoint are different products.
	EmbedModel string
}

// compatEndpoints lists the hosted backends this build knows about. Adding
// another vendor is a row here, not a new implementation.
func (c Config) compatEndpoints() []compatEndpoint {
	return []compatEndpoint{
		{"openai", c.OpenAIBaseURL, c.OpenAIKey, c.OpenAIModel, c.OpenAISystemPrefix, c.OpenAIEmbedModel},
		{"nvidia", c.NvidiaBaseURL, c.NvidiaKey, c.NvidiaModel, c.NvidiaSystemPrefix, c.NvidiaEmbedModel},
	}
}

func (c Config) compatEndpoint(name string) (compatEndpoint, bool) {
	for _, e := range c.compatEndpoints() {
		if e.Name == name {
			return e, true
		}
	}
	return compatEndpoint{}, false
}

// buildLLM constructs one generative backend. Model may be empty to keep the
// provider's configured default.
func buildLLM(cfg Config, provider, modelID string) (LLM, stageName, error) {
	if e, ok := cfg.compatEndpoint(provider); ok {
		if modelID != "" {
			e.Model = modelID
		}
		if e.Key == "" {
			return nil, stageName{}, fmt.Errorf("%s needs an API key", provider)
		}
		if e.Model == "" {
			return nil, stageName{}, fmt.Errorf("%s needs a model", provider)
		}
		// Reuse the one OpenAI-compatible client, pointed at this endpoint.
		cfg.OpenAIBaseURL, cfg.OpenAIKey, cfg.OpenAIModel = e.BaseURL, e.Key, e.Model
		cfg.OpenAISystemPrefix = e.SysPrefix
		return NewOpenAICompat(cfg), stageName{provider, e.Model}, nil
	}
	switch provider {
	case "ollama":
		if modelID != "" {
			cfg.OllamaChatModel = modelID
		}
		if cfg.OllamaChatModel == "" {
			return nil, stageName{}, fmt.Errorf("ollama needs a chat model (OLLAMA_CHAT_MODEL)")
		}
		return NewOllama(cfg), stageName{provider, cfg.OllamaChatModel}, nil
	case "cloud":
		if os.Getenv("ANTHROPIC_API_KEY") == "" {
			return nil, stageName{}, fmt.Errorf("cloud needs ANTHROPIC_API_KEY")
		}
		return NewClaude(), stageName{provider, string(model)}, nil
	}
	return nil, stageName{}, fmt.Errorf("unknown provider %q (want ollama, openai or cloud)", provider)
}

// NewApp wires the three stages independently: embeddings, reranking, and
// generation each come from whichever backend is configured for them. They are
// separate knobs because they have genuinely different constraints — embedding
// is high-volume and cheap to keep local, reranking is latency-critical, and
// generation is the quality-critical one worth spending a large model on.
func NewApp(cfg Config) (*App, error) {
	a := &App{Cfg: cfg, Qdrant: NewQdrant(cfg.QdrantURL, cfg.QdrantKey)}
	// Which model runs each stage is the operator's choice, so nothing is
	// assumed. Fail at startup naming the variable, rather than indexing a
	// whole corpus with a model nobody picked.
	needsOllamaChat := cfg.LLMProvider == "ollama" || (cfg.LLMProvider == "" && cfg.Provider == "ollama")
	if cfg.Provider == "ollama" && cfg.OllamaEmbedModel == "" {
		return nil, fmt.Errorf("OLLAMA_EMBED_MODEL is not set — choose an embedding model " +
			"(`ollama list` to see what you have, `ollama pull bge-m3` for a good default)")
	}
	if needsOllamaChat && cfg.OllamaChatModel == "" {
		return nil, fmt.Errorf("OLLAMA_CHAT_MODEL is not set — choose a chat model " +
			"(`ollama list`), or set LLM_PROVIDER to use a hosted one instead")
	}

	if e, hosted := cfg.compatEndpoint(cfg.Provider); hosted {
		// A hosted endpoint serves both stages: embeddings from EmbedModel,
		// generation from the same client every other stage uses.
		if e.Key == "" {
			return nil, fmt.Errorf("PROVIDER=%s requires %s_API_KEY", e.Name, strings.ToUpper(e.Name))
		}
		if e.EmbedModel == "" {
			return nil, fmt.Errorf("PROVIDER=%s requires %s_EMBED_MODEL", e.Name, strings.ToUpper(e.Name))
		}
		// These endpoints serve no cross-encoder, and reranking is the stage
		// that carries retrieval quality here. Fail at startup rather than on
		// the first query.
		if cfg.RerankerURL == "" {
			return nil, fmt.Errorf("PROVIDER=%s serves no reranker; set RERANKER_URL (see docker-compose.yml)", e.Name)
		}
		llm, _, err := buildLLM(cfg, cfg.Provider, "")
		if err != nil {
			return nil, err
		}
		a.Embedder, a.LLM = NewCompatEmbedder(e.BaseURL, e.Key, e.EmbedModel), llm
	} else {
		switch cfg.Provider {
		case "ollama":
			o := NewOllama(cfg)
			a.Embedder, a.LLM = o, o
		case "cloud":
			key := os.Getenv("VOYAGE_API_KEY")
			if key == "" {
				return nil, fmt.Errorf("PROVIDER=cloud requires VOYAGE_API_KEY")
			}
			if os.Getenv("ANTHROPIC_API_KEY") == "" {
				return nil, fmt.Errorf("PROVIDER=cloud requires ANTHROPIC_API_KEY")
			}
			a.Embedder = NewVoyage(key, cfg.EmbedModel, cfg.RerankModel)
			a.LLM = NewClaude()
		default:
			return nil, fmt.Errorf("unknown PROVIDER %q (want ollama, nvidia, openai or cloud)", cfg.Provider)
		}
	}

	// A dedicated cross-encoder overrides whichever reranker the provider ships
	// with. Opt-in by setting RERANKER_URL, so the pipeline still runs with one
	// less moving part when it is not there.
	if cfg.RerankerURL != "" {
		a.Embedder = rerankedEmbedder{Embedder: a.Embedder, reranker: NewTEIReranker(cfg.RerankerURL, cfg.RerankerBatch)}
	}

	// Generation can come from somewhere else entirely — Groq, OpenRouter, or
	// any other OpenAI-compatible endpoint. Retrieval is unaffected: those
	// services have no embeddings or rerank API to offer.
	if cfg.LLMProvider != "" {
		llm, _, err := buildLLM(cfg, cfg.LLMProvider, "")
		if err != nil {
			return nil, fmt.Errorf("LLM_PROVIDER: %w", err)
		}
		a.LLM = llm
	}

	// Verification is separable from generation, and usually wants a different
	// home. It is binary classification over long evidence: cheap for a small
	// local model, but the largest token consumer of any stage — which on a
	// metered endpoint makes it the first thing to hit a rate limit, and no
	// amount of concurrency helps against a tokens-per-minute cap. Pointing it
	// at local hardware removes that pressure and uses a GPU that is otherwise
	// idle between queries.
	a.Verifier = a.LLM
	if cfg.VerifyProvider != "" || cfg.VerifyModel != "" {
		v, _, err := buildLLM(cfg, firstNonEmpty(cfg.VerifyProvider, cfg.LLMProvider), cfg.VerifyModel)
		if err != nil {
			return nil, fmt.Errorf("VERIFY_PROVIDER/VERIFY_MODEL: %w", err)
		}
		a.Verifier = v
	}

	// Contextualization is separable for the opposite reason to the verifier:
	// not because it needs to be independent, but because it is by far the
	// highest-volume LLM stage and by far the simplest task. One query runs the
	// generator once; indexing a 300-page document runs this ~100 times, and
	// what it asks for each time is a single sentence saying where a chunk sits
	// in its document. Tying that to the generator's model forces a choice
	// between a good generator and an ingest that finishes — measured on a
	// 780-chunk document, contextualization was over 95% of total ingest time
	// whichever backend ran it.
	a.Contextualizer = a.LLM
	if cfg.ContextProvider == "none" {
		// Ingest without any model call. The chunks still get a context line —
		// fallbackContext derives one from the filename and heading path — so
		// this is "no LLM", not "no context". Measured on a 302-page document:
		// 26s here against 14m14s for the same document through a local 7B.
		a.Contextualizer = nil
	} else if cfg.ContextProvider != "" || cfg.ContextModel != "" {
		c, _, err := buildLLM(cfg, firstNonEmpty(cfg.ContextProvider, cfg.LLMProvider), cfg.ContextModel)
		if err != nil {
			return nil, fmt.Errorf("CONTEXT_PROVIDER/CONTEXT_MODEL: %w", err)
		}
		a.Contextualizer = c
	}

	// The judge is independently selectable so answer_correctness can be
	// measured by a model that had no hand in producing the answer. Comparing
	// two generators is only meaningful when both are graded by the same
	// third-party judge.
	a.Judger = a.LLM
	if cfg.JudgeProvider != "" {
		// OPENAI_JUDGE_MODEL names the judge's model on a hosted endpoint;
		// empty keeps that endpoint's configured default.
		j, _, err := buildLLM(cfg, cfg.JudgeProvider, cfg.OpenAIJudgeModel)
		if err != nil {
			return nil, fmt.Errorf("JUDGE_PROVIDER: %w", err)
		}
		a.Judger = j
	}

	// Record what each stage started on, so /providers reports the truth rather
	// than re-deriving it from config that a runtime switch may have outgrown.
	a.llmName = describeStage(cfg, cfg.LLMProvider)
	a.verifyName = describeStage(cfg, firstNonEmpty(cfg.VerifyProvider, cfg.LLMProvider))
	if cfg.VerifyModel != "" {
		a.verifyName.Model = cfg.VerifyModel
	}
	a.contextName = describeStage(cfg, firstNonEmpty(cfg.ContextProvider, cfg.LLMProvider))
	if cfg.ContextModel != "" {
		a.contextName.Model = cfg.ContextModel
	}
	a.judgeName = describeStage(cfg, firstNonEmpty(cfg.JudgeProvider, cfg.LLMProvider))
	if cfg.JudgeProvider == "openai" && cfg.OpenAIJudgeModel != "" {
		a.judgeName.Model = cfg.OpenAIJudgeModel
	}
	return a, nil
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// describeStage resolves which backend and model a stage runs on, following
// the same provider-override precedence NewApp uses.
func describeStage(cfg Config, provider string) stageName {
	if provider == "" {
		provider = cfg.Provider
	}
	if e, ok := cfg.compatEndpoint(provider); ok {
		return stageName{provider, e.Model}
	}
	switch provider {
	case "cloud":
		return stageName{provider, string(model)}
	}
	return stageName{"ollama", cfg.OllamaChatModel}
}

func (a *App) EnsureCollections(ctx context.Context) error {
	dim := a.Cfg.EmbedDim
	// Ask the model how wide its vectors are rather than trusting a config
	// value. A mismatch here is not an error at write time — it is a corpus
	// that silently retrieves badly.
	//
	// Unwrap first: a reranker wrapper would otherwise hide the prober and
	// silently fall back to the configured guess.
	emb := a.Embedder
	if w, ok := emb.(rerankedEmbedder); ok {
		emb = w.Embedder
	}
	if p, ok := emb.(dimProber); ok {
		probed, err := p.EmbedDim(ctx)
		if err != nil {
			return fmt.Errorf("probing the embedding model for its dimension: %w", err)
		}
		if a.Cfg.EmbedDimSet && probed != dim {
			return fmt.Errorf("EMBED_DIM=%d but the embedding model produces %d-dimensional vectors", dim, probed)
		}
		dim = probed
	}
	a.embedDim = dim
	if err := a.Qdrant.EnsureCollection(ctx, a.Cfg.Collection, dim, a.Cfg.Shards); err != nil {
		return err
	}
	return a.Qdrant.EnsureCollection(ctx, a.Cfg.CacheCollection, dim, 1)
}

// loadDotEnv reads KEY=value lines from .env into the environment. Real
// environment variables always win, so `FOO=bar ./ask-my-docs …` still
// overrides the file — the usual precedence, and what makes one-off
// experiments possible without editing config.
func loadDotEnv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return // no .env is a perfectly normal setup
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		k = strings.TrimSpace(k)
		v = strings.Trim(strings.TrimSpace(v), `"'`)
		if _, set := os.LookupEnv(k); !set && k != "" {
			_ = os.Setenv(k, v)
		}
	}
}

// version is stamped by the release workflow with -X main.version=<tag>. A
// plain `go build` leaves it "dev" — which is itself the useful answer when
// someone reports a bug against a binary nobody can identify.
var version = "dev"

func main() {
	log.SetFlags(0)
	if len(os.Args) < 2 {
		usage()
	}
	loadDotEnv(env("ENV_FILE", ".env"))
	cfg := LoadConfig()
	ablate, err := parseAblate(os.Getenv("ABLATE"))
	if err != nil {
		log.Fatalf("error: %v", err)
	}
	cfg.Ablate = ablate
	if len(ablate) > 0 {
		// Loud, because every downstream number is now measuring a crippled
		// pipeline. An ablation run that gets mistaken for a normal one is how a
		// benchmark table ends up quietly wrong.
		fmt.Fprintf(os.Stderr, "ABLATION ACTIVE — stages disabled: %s\n", os.Getenv("ABLATE"))
	}
	ctx := context.Background()

	switch os.Args[1] {
	case "version", "-v", "--version":
		fmt.Println(version)
	case "serve":
		run(ctx, cfg, cmdServe)
	case "ingest":
		run(ctx, cfg, cmdIngest)
	case "query":
		run(ctx, cfg, cmdQuery)
	case "eval":
		run(ctx, cfg, cmdEval)
	case "ablate":
		run(ctx, cfg, cmdAblate)
	case "calibrate":
		run(ctx, cfg, cmdCalibrate)
	case "token":
		// Offline: minting a credential needs no services, and requiring them
		// would mean standing up the whole stack to add a user.
		if err := cmdToken(os.Args[2:]); err != nil {
			log.Fatalf("error: %v", err)
		}
	case "detect":
		// Offline and service-free by design: it runs before `docker compose
		// up`, which is the whole point.
		if err := cmdDetect(os.Args[2:]); err != nil {
			log.Fatalf("error: %v", err)
		}
	case "chunks":
		// Offline: no API keys, no services. Used to tune chunk boundaries and
		// to read off the chunk ids that the golden eval set has to reference.
		if err := cmdChunks(cfg, os.Args[2:]); err != nil {
			log.Fatalf("error: %v", err)
		}
	default:
		usage()
	}
}

func cmdCalibrate(ctx context.Context, a *App, args []string) error {
	fs := flag.NewFlagSet("calibrate", flag.ExitOnError)
	golden := fs.String("golden", "eval/golden.jsonl", "path to the golden eval set")
	_ = fs.Parse(args)
	items, err := LoadGolden(*golden)
	if err != nil {
		return err
	}
	if len(items) == 0 {
		return fmt.Errorf("%s contains no eval items — calibration needs both answerable "+
			"items and at least one with \"relevant_chunk_ids\": [] to bound the gate", *golden)
	}
	return a.Calibrate(ctx, items)
}

func cmdChunks(cfg Config, args []string) error {
	fs := flag.NewFlagSet("chunks", flag.ExitOnError)
	dir := fs.String("dir", "", "directory to chunk (required)")
	text := fs.Bool("text", false, "print each chunk's body, not just its id")
	_ = fs.Parse(args)
	if *dir == "" {
		fs.Usage()
		return fmt.Errorf("-dir is required")
	}
	return filepath.WalkDir(*dir, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		if !isDoc(p) {
			return nil
		}
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(*dir, p)
		docID := filepath.ToSlash(rel)
		// Same extraction the indexer uses, so this previews what will actually
		// be indexed rather than a different reading of the same file.
		body, err := LoadDocumentText(docID, raw)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", docID, err)
			return nil
		}
		for _, c := range ChunkDoc(body, cfg.MaxChunkChars, cfg.ChunkOverlap) {
			fmt.Printf("%s#%d\t%d chars\t%s\n", docID, c.Ordinal, len(c.Text), c.Section)
			if *text {
				fmt.Printf("%s\n---\n", c.Text)
			}
		}
		return nil
	})
}

func usage() {
	fmt.Fprint(os.Stderr, `ask-my-docs — hybrid-retrieval RAG with citation enforcement

  serve    Run the query API
  ingest   Index a directory of documents
  query    Ask one question from the command line
  eval     Run the golden eval set and gate on regressions
  ablate   Run the golden set once per pipeline stage and print what each is worth
  calibrate Derive MIN_RERANK_SCORE from the golden set
  token    Mint a bearer token and its AUTH_TOKENS_FILE entry (offline, no keys)
  chunks   Print chunk boundaries and ids for a directory (offline, no keys)
  detect   Pick the reranker image for this machine's GPU and write it to .env
  version  Print the build version

Run "<command> -h" for flags.
`)
	os.Exit(2)
}

func run(ctx context.Context, cfg Config, fn func(context.Context, *App, []string) error) {
	app, err := NewApp(cfg)
	if err != nil {
		log.Fatalf("error: %v", err)
	}
	if err := fn(ctx, app, os.Args[2:]); err != nil {
		log.Fatalf("error: %v", err)
	}
}

// --- serve ----------------------------------------------------------------

func cmdServe(ctx context.Context, a *App, args []string) error {
	if err := a.EnsureCollections(ctx); err != nil {
		return err
	}
	switch {
	case a.Cfg.AuthTokensFile != "":
		tokens, err := LoadTokens(a.Cfg.AuthTokensFile)
		if err != nil {
			return err
		}
		a.Tokens = tokens
		log.Printf("auth: %d principal(s) from %s", len(tokens), a.Cfg.AuthTokensFile)
	case a.Cfg.AllowAnonymous:
		// Loud, because the thing being given away is every document in the
		// index plus the ability to add more.
		log.Printf("WARNING: ALLOW_ANONYMOUS is set — %s serves every indexed document, "+
			"and accepts uploads and model switches, from anyone who can reach it. "+
			"Set AUTH_TOKENS_FILE before this is reachable by anything you do not trust.", a.Cfg.Addr)
	default:
		// Refusing to start is the only version of this that stays true. A
		// warning gets scrolled past; a default-open server does not announce
		// itself again after the day it was set up.
		return fmt.Errorf("refusing to serve without authentication: set AUTH_TOKENS_FILE " +
			"(mint an entry with `ask-my-docs token <name>`), or set ALLOW_ANONYMOUS=1 to " +
			"accept that every caller may read every indexed document, upload more, and " +
			"switch models")
	}
	mux := http.NewServeMux()

	mux.HandleFunc("POST /query", a.guard(false, func(w http.ResponseWriter, r *http.Request, p Principal) {
		var req QueryRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
			httpErr(w, http.StatusBadRequest, "invalid JSON body")
			return
		}
		if strings.TrimSpace(req.Question) == "" {
			httpErr(w, http.StatusBadRequest, "question is required")
			return
		}
		// The whole point of the auth boundary: whatever acl the caller put in
		// the body is discarded and replaced with what their token grants.
		// Merging the two would restore exactly the hole this closes.
		req.ACL = p.Tags()
		ctx, cancel := context.WithTimeout(r.Context(), time.Duration(a.Cfg.RequestTimeoutSec)*time.Second)
		defer cancel()
		resp, err := a.Query(ctx, req)
		if err != nil {
			// Return the actual cause, not a generic message. This binds to
			// localhost and every failure here is operational — a batch limit,
			// a model that is not pulled, a service that is down. Hiding that
			// behind "query failed" sends the operator to a log file to learn
			// something the UI already knew.
			log.Printf("query failed: %v", err)
			httpErr(w, http.StatusBadGateway, err.Error())
			return
		}
		writeJSON(w, http.StatusOK, resp)
	}))

	// GET /providers — what each generative stage is using, and what it could
	// use. Model lists come from the providers themselves rather than a
	// hardcoded table, so they stay true as models are pulled or retired.
	mux.HandleFunc("GET /providers", a.guard(false, func(w http.ResponseWriter, r *http.Request, _ Principal) {
		a.mu.RLock()
		cur := map[string]stageName{"llm": a.llmName, "verify": a.verifyName, "judge": a.judgeName}
		a.mu.RUnlock()
		writeJSON(w, http.StatusOK, map[string]any{
			"current":   cur,
			"available": a.availableBackends(r.Context()),
			// Surfaced so the UI can explain the omission rather than leaving
			// someone hunting for an embeddings dropdown that will never exist.
			"fixed": map[string]string{
				"embeddings": fmt.Sprintf("%s (%d-dim) — changing it requires re-indexing every document",
					a.Cfg.OllamaEmbedModel, a.embedDim),
			},
		})
	}))

	// POST /providers — switch a stage. Validated before it is applied, so a
	// bad choice returns an error instead of breaking every later query.
	// Admin-only: this changes which model answers for every caller, not just
	// the one asking.
	mux.HandleFunc("POST /providers", a.guard(true, func(w http.ResponseWriter, r *http.Request, p Principal) {
		var req struct {
			Stage    string `json:"stage"`
			Provider string `json:"provider"`
			Model    string `json:"model"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&req); err != nil {
			httpErr(w, http.StatusBadRequest, "invalid JSON body")
			return
		}
		next, name, err := buildLLM(a.Cfg, req.Provider, req.Model)
		if err != nil {
			httpErr(w, http.StatusBadRequest, err.Error())
			return
		}
		if err := a.checkModelExists(r.Context(), name); err != nil {
			httpErr(w, http.StatusBadRequest, err.Error())
			return
		}

		a.mu.Lock()
		switch req.Stage {
		case "llm":
			a.LLM, a.llmName = next, name
		case "verify":
			a.Verifier, a.verifyName = next, name
		case "judge":
			a.Judger, a.judgeName = next, name
		default:
			a.mu.Unlock()
			httpErr(w, http.StatusBadRequest, "stage must be llm, verify or judge")
			return
		}
		a.mu.Unlock()

		log.Printf("%s switched %s to %s/%s", p.Name, req.Stage, name.Provider, name.Model)
		writeJSON(w, http.StatusOK, map[string]any{"stage": req.Stage, "using": name})
	}))

	// GET /documents — what is currently in the corpus directory, restricted to
	// what the caller may actually read. A filename is itself information —
	// "you cannot open this, but it is called 2026_redundancies.docx" is the
	// same leak in a smaller package — so a document with no readable chunks is
	// omitted entirely rather than listed with a zero.
	mux.HandleFunc("GET /documents", a.guard(false, func(w http.ResponseWriter, r *http.Request, p Principal) {
		all, err := listCorpus(a.Cfg.CorpusDir)
		if err != nil {
			httpErr(w, http.StatusInternalServerError, "cannot read corpus directory")
			return
		}
		tags := p.Tags()
		docs := all[:0:0]
		for _, d := range all {
			// Report indexed chunks per document, not just what is on disk. A
			// file present with zero chunks means it was saved but never
			// indexed — a state worth being able to see rather than infer.
			n, err := a.Qdrant.CountFiltered(r.Context(), a.Cfg.Collection, DocIDFilter(d.Name, tags))
			if err != nil {
				// Unknown is not the same as permitted. For an unrestricted
				// caller the count is cosmetic and the file is listed anyway;
				// for a restricted one it is the access check, so drop it.
				if len(tags) > 0 {
					continue
				}
			}
			if len(tags) > 0 && n == 0 {
				continue
			}
			d.Chunks = n
			docs = append(docs, d)
		}
		writeJSON(w, http.StatusOK, map[string]any{"documents": docs})
	}))

	// POST /documents — upload files into the corpus directory and index them.
	//
	// ponytail: indexes synchronously and answers when done. Contextualization
	// is several seconds per document on a local model, so this is a slow
	// request by design rather than a job queue plus polling. If you routinely
	// upload dozens at once, that is the point to add one.
	//
	// Admin-only: an upload writes to the corpus directory and puts text into
	// the index that every later answer may be built from. The acl form value
	// decides who can then retrieve it, which makes this the endpoint that
	// hands out access rather than one that consumes it.
	mux.HandleFunc("POST /documents", a.guard(true, func(w http.ResponseWriter, r *http.Request, p Principal) {
		maxBytes := int64(a.Cfg.MaxUploadMB) << 20
		r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
		if err := r.ParseMultipartForm(16 << 20); err != nil {
			httpErr(w, http.StatusRequestEntityTooLarge,
				fmt.Sprintf("upload too large or malformed (limit %dMB)", a.Cfg.MaxUploadMB))
			return
		}
		defer func() { _ = r.MultipartForm.RemoveAll() }()

		files := r.MultipartForm.File["files"]
		if len(files) == 0 {
			httpErr(w, http.StatusBadRequest, "no files in the request")
			return
		}
		if err := os.MkdirAll(a.Cfg.CorpusDir, 0o755); err != nil {
			httpErr(w, http.StatusInternalServerError, "cannot create corpus directory")
			return
		}
		acl := splitCSV(r.FormValue("acl"))

		ctx, cancel := context.WithTimeout(r.Context(), 30*time.Minute)
		defer cancel()

		type result struct {
			Name   string `json:"name"`
			Chunks int    `json:"chunks,omitempty"`
			Error  string `json:"error,omitempty"`
		}
		results := make([]result, 0, len(files))
		indexed := 0

		for _, fh := range files {
			res := result{Name: fh.Filename}
			dest, err := SafeCorpusPath(a.Cfg.CorpusDir, fh.Filename)
			if err != nil {
				res.Error = err.Error()
				results = append(results, res)
				continue
			}
			res.Name = filepath.Base(dest)

			body, err := readUpload(fh, maxBytes)
			if err != nil {
				res.Error = err.Error()
				results = append(results, res)
				continue
			}
			// Extract before writing: a .docx that turns out to be corrupt
			// should be rejected outright, not left in the corpus directory to
			// fail again on the next `ingest -dir`.
			text, err := LoadDocumentText(res.Name, body)
			if err != nil {
				res.Error = err.Error()
				results = append(results, res)
				continue
			}
			if err := os.WriteFile(dest, body, 0o644); err != nil {
				res.Error = "could not save file"
				log.Printf("upload %s: %v", res.Name, err)
				results = append(results, res)
				continue
			}
			n, err := a.IngestDoc(ctx, res.Name, fmt.Sprintf("%d", time.Now().Unix()), acl, text)
			if err != nil {
				// The file is on disk; only indexing failed. Say so, because
				// `ingest -dir` will pick it up on a retry.
				res.Error = "saved, but indexing failed: " + err.Error()
				results = append(results, res)
				continue
			}
			res.Chunks = n
			indexed += n
			results = append(results, res)
		}

		if indexed > 0 {
			// New documents can change the correct answer to a question already
			// in the cache, and a cache hit skips retrieval entirely.
			if err := a.ClearCache(ctx); err != nil {
				log.Printf("clearing semantic cache after ingest: %v", err)
			}
		}
		total, _ := a.Qdrant.Count(ctx, a.Cfg.Collection)
		log.Printf("%s uploaded %d file(s), %d chunks indexed, acl=%v", p.Name, len(files), indexed, acl)
		writeJSON(w, http.StatusOK, map[string]any{
			"results": results, "chunks_added": indexed, "chunks_total": total,
		})
	}))

	// POST /ingest — index whatever is already in the corpus directory.
	//
	// Uploading and indexing are the same request in POST /documents, which
	// covers files that arrive through the UI and nothing else. Files that got
	// there any other way — copied in, restored from a backup, written by a
	// sync job — were CLI-only until now, and GET /documents would list them at
	// zero chunks with no way to act on it.
	//
	// Admin-only for the same reason uploading is: it changes what every other
	// caller can retrieve.
	mux.HandleFunc("POST /ingest", a.guard(true, func(w http.ResponseWriter, r *http.Request, p Principal) {
		// Unchanged documents are skipped by version, so the common case costs
		// a count query per document and no tokens. -force is deliberately not
		// exposed here: it re-contextualizes the whole corpus, which is the
		// most expensive thing this system can be asked to do, and a button
		// that does it by accident is worse than a flag you have to type.
		// One at a time. Two concurrent walks of the same directory would
		// contextualize and embed every document twice, which is the most
		// expensive way this system can waste a token — and a button invites
		// exactly that, because a slow one gets clicked again.
		if !a.ingesting.TryLock() {
			httpErr(w, http.StatusConflict, "an ingest is already running")
			return
		}
		defer a.ingesting.Unlock()

		acl := splitCSV(r.URL.Query().Get("acl"))
		// Deliberately NOT r.Context(). Indexing a corpus can run for minutes,
		// and tying it to the request means closing the tab aborts it partway
		// — measured here, a browser giving up cancelled the embed mid-document
		// and the work was thrown away. The response is still sent to whoever
		// is listening; the difference is that the indexing finishes either way.
		ctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), 30*time.Minute)
		defer cancel()

		indexed, skipped, chunks, err := a.IngestDir(ctx, a.Cfg.CorpusDir, acl, false)
		if err != nil {
			log.Printf("%s ingest failed: %v", p.Name, err)
			httpErr(w, http.StatusBadGateway, err.Error())
			return
		}
		if chunks > 0 {
			// Same reason as upload: a new document can change the right answer
			// to a question already in the cache, and a hit skips retrieval.
			if err := a.ClearCache(ctx); err != nil {
				log.Printf("clearing semantic cache after ingest: %v", err)
			}
		}
		total, _ := a.Qdrant.Count(ctx, a.Cfg.Collection)
		log.Printf("%s re-indexed %s: %d document(s), %d unchanged, %d chunks", p.Name, a.Cfg.CorpusDir, indexed, skipped, chunks)
		writeJSON(w, http.StatusOK, map[string]any{
			"indexed": indexed, "skipped": skipped,
			"chunks_added": chunks, "chunks_total": total,
		})
	}))

	// The UI is a single embedded file — no build step, no npm, and it ships
	// inside the same binary as the API it talks to.
	mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(uiHTML)
	})

	// healthz doubles as "what is actually running", which matters now that
	// embeddings, reranking and generation can each come from a different
	// backend and the gate threshold is corpus-specific.
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		if err := a.Qdrant.Health(r.Context()); err != nil {
			httpErr(w, http.StatusServiceUnavailable, "qdrant unreachable")
			return
		}
		chunks, _ := a.Qdrant.Count(r.Context(), a.Cfg.Collection)
		reranker := "llm"
		if a.Cfg.RerankerURL != "" {
			reranker = "cross-encoder"
		}
		// Report the live backend, not the configured one — they diverge the
		// moment anything is switched at runtime, and a header that lies about
		// which model answered is worse than no header.
		a.mu.RLock()
		llmName := a.llmName
		a.mu.RUnlock()
		writeJSON(w, http.StatusOK, map[string]any{
			"status":   "ok",
			"provider": a.Cfg.Provider,
			"llm":      llmName.Model,
			"reranker": reranker,
			"gate":     a.Cfg.MinRerankScore,
			"chunks":   chunks,
		})
	})

	srv := &http.Server{
		Addr:              a.Cfg.Addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Printf("listening on %s (collection=%s)", a.Cfg.Addr, a.Cfg.Collection)
	return srv.ListenAndServe()
}

type backendInfo struct {
	Ready  bool     `json:"ready"`
	Reason string   `json:"reason,omitempty"`
	Models []string `json:"models,omitempty"`
}

// availableBackends asks each provider what it can actually run. Unreachable or
// unconfigured providers are reported with the reason rather than hidden, so a
// missing option is explained instead of just absent.
func (a *App) availableBackends(ctx context.Context) map[string]backendInfo {
	out := map[string]backendInfo{}

	models, err := listOllamaModels(ctx, a.Cfg.OllamaURL)
	switch {
	case err != nil:
		out["ollama"] = backendInfo{Reason: "cannot reach " + a.Cfg.OllamaURL}
	default:
		// The embedding model cannot generate, so offering it as a chat choice
		// would only produce a 501 later.
		chat := models[:0:0]
		for _, m := range models {
			if !strings.EqualFold(m, a.Cfg.OllamaEmbedModel) &&
				!strings.HasPrefix(m, a.Cfg.OllamaEmbedModel+":") {
				chat = append(chat, m)
			}
		}
		out["ollama"] = backendInfo{Ready: len(chat) > 0, Models: chat}
	}

	for _, e := range a.Cfg.compatEndpoints() {
		switch {
		case e.Key == "":
			out[e.Name] = backendInfo{Reason: strings.ToUpper(e.Name) + "_API_KEY not set"}
		default:
			m, err := listOpenAIModels(ctx, e.BaseURL, e.Key)
			if err != nil {
				out[e.Name] = backendInfo{Reason: err.Error()}
				continue
			}
			out[e.Name] = backendInfo{Ready: true, Models: m}
		}
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		out["cloud"] = backendInfo{Reason: "ANTHROPIC_API_KEY not set"}
	} else {
		out["cloud"] = backendInfo{Ready: true, Models: []string{string(model)}}
	}
	return out
}

// checkModelExists rejects a switch to a model the provider does not have.
// It is a listing lookup rather than a test generation on purpose: a probe call
// would burn tokens from the very budget people are usually switching to
// escape.
func (a *App) checkModelExists(ctx context.Context, s stageName) error {
	avail := a.availableBackends(ctx)
	info, ok := avail[s.Provider]
	if !ok || !info.Ready {
		reason := "unavailable"
		if ok && info.Reason != "" {
			reason = info.Reason
		}
		return fmt.Errorf("%s is not usable: %s", s.Provider, reason)
	}
	for _, m := range info.Models {
		if m == s.Model {
			return nil
		}
	}
	return fmt.Errorf("%s has no model %q", s.Provider, s.Model)
}

func listOllamaModels(ctx context.Context, base string) ([]string, error) {
	var out struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := getJSON(ctx, strings.TrimRight(base, "/")+"/api/tags", "", &out); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(out.Models))
	for _, m := range out.Models {
		names = append(names, m.Name)
	}
	sort.Strings(names)
	return names, nil
}

func listOpenAIModels(ctx context.Context, base, key string) ([]string, error) {
	var out struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := getJSON(ctx, strings.TrimRight(base, "/")+"/models", key, &out); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(out.Data))
	for _, m := range out.Data {
		// Speech and safety models cannot serve chat completions, so offering
		// them would only produce a failure later.
		if isNonChatModel(m.ID) {
			continue
		}
		names = append(names, m.ID)
	}
	sort.Strings(names)
	return names, nil
}

// isNonChatModel filters out speech, embedding and moderation models that
// share the same listing endpoint as chat models.
func isNonChatModel(id string) bool {
	l := strings.ToLower(id)
	for _, bad := range []string{"whisper", "tts", "guard", "orpheus", "embed", "rerank", "moderation"} {
		if strings.Contains(l, bad) {
			return true
		}
	}
	return false
}

func getJSON(ctx context.Context, url, bearer string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	resp, err := (&http.Client{Timeout: 15 * time.Second}).Do(req)
	if err != nil {
		return fmt.Errorf("cannot reach %s", url)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("%s returned %s", url, resp.Status)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

type corpusDoc struct {
	Name     string `json:"name"`
	Bytes    int64  `json:"bytes"`
	Modified string `json:"modified"`
	Chunks   int    `json:"chunks"`
}

func listCorpus(dir string) ([]corpusDoc, error) {
	out := []corpusDoc{}
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return out, nil // not yet created is not an error
		}
		return nil, err
	}
	for _, e := range entries {
		if e.IsDir() || !isDoc(e.Name()) {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		out = append(out, corpusDoc{
			Name:     e.Name(),
			Bytes:    info.Size(),
			Modified: info.ModTime().UTC().Format(time.RFC3339),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

// readUpload reads one uploaded file, refusing anything over the limit rather
// than truncating it — a silently half-indexed document is worse than a
// rejected one. Office files are binary, so validation of the *content* is
// left to LoadDocumentText; only size and emptiness are checked here.
func readUpload(fh *multipart.FileHeader, limit int64) ([]byte, error) {
	f, err := fh.Open()
	if err != nil {
		return nil, fmt.Errorf("could not read upload")
	}
	defer f.Close()
	body, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, fmt.Errorf("could not read upload")
	}
	if int64(len(body)) > limit {
		return nil, fmt.Errorf("file exceeds the size limit")
	}
	if len(bytes.TrimSpace(body)) == 0 {
		return nil, fmt.Errorf("file is empty")
	}
	return body, nil
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func httpErr(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

// --- ingest ---------------------------------------------------------------

func cmdIngest(ctx context.Context, a *App, args []string) error {
	fs := flag.NewFlagSet("ingest", flag.ExitOnError)
	dir := fs.String("dir", a.Cfg.CorpusDir, "directory of .md/.txt documents to index")
	acl := fs.String("acl", "", "comma-separated ACL tags applied to every chunk")
	// Unchanged files are skipped by default because re-indexing them is pure
	// cost for an identical result. Anything that changes how a document should
	// be indexed rather than the document itself — embedding model, chunk
	// bounds, contextualization prompt — is invisible to that check and needs
	// this flag.
	force := fs.Bool("force", false, "re-index every document, including ones unchanged since the last run")
	_ = fs.Parse(args)
	if err := a.EnsureCollections(ctx); err != nil {
		return err
	}
	start := time.Now()
	docs, skipped, chunks, err := a.IngestDir(ctx, *dir, splitCSV(*acl), *force)
	if err != nil {
		return err
	}
	if chunks > 0 {
		// Cached answers predate these documents and would be served without
		// ever consulting them.
		if err := a.ClearCache(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "warn: could not clear the semantic cache: %v\n", err)
		}
	}
	fmt.Printf("indexed %d documents, %d chunks in %s", docs, chunks, time.Since(start).Round(time.Millisecond))
	if skipped > 0 {
		fmt.Printf(" (%d unchanged, skipped — -force to re-index)", skipped)
	}
	fmt.Println()
	return nil
}

// --- query ----------------------------------------------------------------

func cmdQuery(ctx context.Context, a *App, args []string) error {
	fs := flag.NewFlagSet("query", flag.ExitOnError)
	acl := fs.String("acl", "", "comma-separated ACL tags for this caller")
	noCache := fs.Bool("no-cache", false, "bypass the semantic cache")
	jsonOut := fs.Bool("json", false, "print the full response as JSON")
	_ = fs.Parse(args)
	q := strings.Join(fs.Args(), " ")
	if strings.TrimSpace(q) == "" {
		return fmt.Errorf("usage: ask-my-docs query [flags] <question>")
	}
	resp, err := a.Query(ctx, QueryRequest{Question: q, ACL: splitCSV(*acl), NoCache: *noCache})
	if err != nil {
		return err
	}
	if *jsonOut {
		b, _ := json.MarshalIndent(resp, "", "  ")
		fmt.Println(string(b))
		return nil
	}
	fmt.Println(resp.Answer)
	if len(resp.Sources) > 0 {
		fmt.Println("\nSources:")
		for _, s := range resp.Sources {
			fmt.Printf("  %-30s %.3f  %s\n", s.ChunkID, s.Score, s.Section)
		}
	}
	for _, warn := range resp.Warnings {
		fmt.Printf("warning: %s\n", warn)
	}
	fmt.Printf("\ncache_hit=%v  retrieve=%dms rerank=%dms generate=%dms verify=%dms total=%dms\n",
		resp.CacheHit, resp.Timings.Retrieve, resp.Timings.Rerank,
		resp.Timings.Generate, resp.Timings.Verify, resp.Timings.Total)
	return nil
}

// --- eval -----------------------------------------------------------------

func cmdEval(ctx context.Context, a *App, args []string) error {
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	golden := fs.String("golden", "eval/golden.jsonl", "path to the golden eval set")
	baseline := fs.String("baseline", "eval/baseline.json", "known-good metrics to compare against")
	out := fs.String("out", "", "write this run's metrics to a file (build artifact)")
	tolerance := fs.Float64("tolerance", 0.02, "allowed quality regression (absolute)")
	latencyTol := fs.Float64("latency-tolerance", 0.50, "allowed latency regression (relative); loose because wall-clock is noisy")
	judge := fs.Bool("judge", true, "score answers against gold answers with an LLM judge")
	verbose := fs.Bool("verbose", false, "print the per-item breakdown, worst first")
	update := fs.Bool("update-baseline", false, "overwrite the baseline with this run instead of gating")
	_ = fs.Parse(args)

	items, err := LoadGolden(*golden)
	if err != nil {
		return err
	}
	if len(items) == 0 {
		return fmt.Errorf("%s contains no eval items — add some before this can measure "+
			"anything (the file documents the format; `chunks -dir corpus` prints the "+
			"chunk ids to reference)", *golden)
	}
	if err := a.CheckGoldenIDs(ctx, items); err != nil {
		return err
	}
	fmt.Printf("running %d eval items...\n", len(items))

	m, results, err := a.RunEval(ctx, items, *judge)
	if err != nil {
		return err
	}
	pretty, _ := json.MarshalIndent(m, "", "  ")
	fmt.Println(string(pretty))
	if *verbose {
		ReportItems(results)
	}
	// Recall@k saturates when the corpus is smaller than k: "the top 10 contains
	// the answer" is then arithmetic, not quality. Say so rather than letting a
	// 1.00 be read as a passing grade.
	if n, err := a.Qdrant.Count(ctx, a.Cfg.Collection); err == nil && n <= a.Cfg.TopK*2 {
		fmt.Printf("\nwarning: corpus is %d chunks against TOP_K=%d — recall_at_10 is\n"+
			"saturated and meaningless at this size. Trust ndcg/mrr, and re-baseline\n"+
			"on a corpus of at least %d chunks before believing recall.\n",
			n, a.Cfg.TopK, a.Cfg.TopK*10)
	}

	if *out != "" {
		if err := os.WriteFile(*out, pretty, 0o644); err != nil {
			return err
		}
	}
	if *update {
		fmt.Printf("writing baseline to %s\n", *baseline)
		return os.WriteFile(*baseline, pretty, 0o644)
	}

	raw, err := os.ReadFile(*baseline)
	if err != nil {
		fmt.Printf("\nno baseline at %s — skipping gate. Create one with -update-baseline.\n", *baseline)
		return nil
	}
	var base Metrics
	if err := json.Unmarshal(raw, &base); err != nil {
		return fmt.Errorf("%s: %w", *baseline, err)
	}
	fails := CompareBaseline(base, m, *tolerance, *latencyTol)
	if m.Failures > 0 {
		fails = append(fails, fmt.Sprintf("%d eval items errored", m.Failures))
	}
	if len(fails) > 0 {
		fmt.Println("\nFAIL — gated metrics regressed:")
		for _, f := range fails {
			fmt.Printf("  - %s\n", f)
		}
		os.Exit(1)
	}
	fmt.Println("\nPASS — no gated metric regressed beyond tolerance.")
	return nil
}

// --- env helpers ----------------------------------------------------------

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return v
	}
	return def
}

func envFloat(k string, def float64) float64 {
	if v, err := strconv.ParseFloat(os.Getenv(k), 64); err == nil {
		return v
	}
	return def
}

// ablatable lists every stage ABLATE can switch off, in pipeline order.
//
// A closed set, because the whole point of an ablation is that the config
// matches the claim. ABLATE=rerankr measuring nothing and reporting a number
// anyway is worse than a crash: it produces a benchmark row that looks real.
var ablatable = []string{"sparse", "rerank", "gate", "llmcontext", "context", "citations", "verify"}

// parseAblate reads the ABLATE set and rejects anything it does not recognise.
func parseAblate(raw string) (map[string]bool, error) {
	out := map[string]bool{}
	for _, s := range splitCSV(raw) {
		s = strings.ToLower(s)
		if !slices.Contains(ablatable, s) {
			return nil, fmt.Errorf("ABLATE: unknown stage %q (known: %s)", s, strings.Join(ablatable, ", "))
		}
		out[s] = true
	}
	// "context" means no context at all, which necessarily removes the LLM one
	// too. Implying it here keeps the ladder's rungs from having to know that.
	if out["context"] {
		out["llmcontext"] = true
	}
	return out, nil
}

func splitCSV(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := parts[:0]
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
