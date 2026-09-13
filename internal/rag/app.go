package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"askmydocs/internal/auth"
	"askmydocs/internal/qdrant"
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

// defaultGate picks a starting confidence threshold for whichever reranker is
// in play. Both values are starting points, not tuned constants — `calibrate`
// derives the right one from the golden set.
func defaultGate(rerankerURL string) float64 {
	if rerankerURL != "" {
		return 0.002 // cross-encoder floor; RERANK_MARGIN does the real work
	}
	return 0.30 // LLM scorer: coarse 0/0.3/0.7/1.0 buckets
}

// CompatEndpoint is one OpenAI-compatible service. Groq, NVIDIA NIM,
// OpenRouter, Together, vLLM and LM Studio all speak the same wire format, so
// they differ only by base URL, key and model — no per-vendor code.
type CompatEndpoint struct {
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

// CompatEndpoints lists the hosted backends this build knows about. Adding
// another vendor is a row here, not a new implementation.
func (c Config) CompatEndpoints() []CompatEndpoint {
	return []CompatEndpoint{
		{"openai", c.OpenAIBaseURL, c.OpenAIKey, c.OpenAIModel, c.OpenAISystemPrefix, c.OpenAIEmbedModel},
		{"nvidia", c.NvidiaBaseURL, c.NvidiaKey, c.NvidiaModel, c.NvidiaSystemPrefix, c.NvidiaEmbedModel},
	}
}

func (c Config) CompatEndpoint(name string) (CompatEndpoint, bool) {
	for _, e := range c.CompatEndpoints() {
		if e.Name == name {
			return e, true
		}
	}
	return CompatEndpoint{}, false
}

type App struct {
	Cfg      Config
	Qdrant   *qdrant.Qdrant
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
	Tokens auth.TokenStore

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
	LLMName     StageName
	VerifyName  StageName
	JudgeName   StageName
	ContextName StageName
}

// NewApp constructs the shell of an App: config and the Qdrant client. The
// caller (the CLI/HTTP composition root) wires Embedder/LLM/Judger/Verifier/
// Contextualizer and the initial *Name fields, because building those means
// constructing concrete provider clients, and this package deliberately does
// not know those types exist — only the Embedder/LLM interfaces they satisfy.
func NewApp(cfg Config) *App {
	return &App{Cfg: cfg, Qdrant: qdrant.NewQdrant(cfg.QdrantURL, cfg.QdrantKey)}
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

// SwitchProvider validates and applies a runtime provider switch for one
// generative stage. next/name come from the caller's own provider construction
// (this package does not build concrete providers), already validated against
// the provider's own model listing.
func (a *App) SwitchProvider(ctx context.Context, stage string, next LLM, name StageName) error {
	if err := a.checkModelExists(ctx, name); err != nil {
		return err
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	switch stage {
	case "llm":
		a.LLM, a.LLMName = next, name
	case "verify":
		a.Verifier, a.VerifyName = next, name
	case "judge":
		a.Judger, a.JudgeName = next, name
	default:
		return fmt.Errorf("stage must be llm, verify or judge")
	}
	return nil
}

// CurrentStages reports which backend and model each runtime-swappable stage
// is using right now, not at startup — they diverge the moment anything is
// switched.
func (a *App) CurrentStages() map[string]StageName {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return map[string]StageName{"llm": a.LLMName, "verify": a.VerifyName, "judge": a.JudgeName}
}

// EmbedDim returns the probed embedding vector width, valid after
// EnsureCollections has run.
func (a *App) EmbedDim() int { return a.embedDim }

func (a *App) EnsureCollections(ctx context.Context) error {
	dim := a.Cfg.EmbedDim
	// Ask the model how wide its vectors are rather than trusting a config
	// value. A mismatch here is not an error at write time — it is a corpus
	// that silently retrieves badly.
	//
	// Unwrap first: a reranker wrapper would otherwise hide the prober and
	// silently fall back to the configured guess.
	emb := a.Embedder
	if w, ok := emb.(unwrapper); ok {
		emb = w.Unwrap()
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

// --- HTTP auth boundary -----------------------------------------------------

// Guard authenticates a request and hands the handler the principal it is
// acting for. Handlers never see the raw token.
func (a *App) Guard(requireAdmin bool, h func(http.ResponseWriter, *http.Request, auth.Principal)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if a.Tokens == nil {
			h(w, r, auth.Anonymous)
			return
		}
		p, ok := a.Tokens.Lookup(auth.BearerToken(r))
		if !ok {
			// The challenge header is what makes a browser or an HTTP client
			// retry with credentials instead of just reporting a failure.
			w.Header().Set("WWW-Authenticate", `Bearer realm="ask-my-docs"`)
			HTTPErr(w, http.StatusUnauthorized, "missing or invalid bearer token")
			return
		}
		if requireAdmin && !p.Admin {
			HTTPErr(w, http.StatusForbidden, "this endpoint requires an admin token")
			return
		}
		h(w, r, p)
	}
}

func WriteJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func HTTPErr(w http.ResponseWriter, code int, msg string) {
	WriteJSON(w, code, map[string]string{"error": msg})
}

// --- Runtime backend discovery ----------------------------------------------

type BackendInfo struct {
	Ready  bool     `json:"ready"`
	Reason string   `json:"reason,omitempty"`
	Models []string `json:"models,omitempty"`
}

// AvailableBackends asks each provider what it can actually run. Unreachable or
// unconfigured providers are reported with the reason rather than hidden, so a
// missing option is explained instead of just absent.
func (a *App) AvailableBackends(ctx context.Context) map[string]BackendInfo {
	out := map[string]BackendInfo{}

	models, err := listOllamaModels(ctx, a.Cfg.OllamaURL)
	switch {
	case err != nil:
		out["ollama"] = BackendInfo{Reason: "cannot reach " + a.Cfg.OllamaURL}
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
		out["ollama"] = BackendInfo{Ready: len(chat) > 0, Models: chat}
	}

	for _, e := range a.Cfg.CompatEndpoints() {
		switch {
		case e.Key == "":
			out[e.Name] = BackendInfo{Reason: strings.ToUpper(e.Name) + "_API_KEY not set"}
		default:
			m, err := listOpenAIModels(ctx, e.BaseURL, e.Key)
			if err != nil {
				out[e.Name] = BackendInfo{Reason: err.Error()}
				continue
			}
			out[e.Name] = BackendInfo{Ready: true, Models: m}
		}
	}

	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		out["cloud"] = BackendInfo{Reason: "ANTHROPIC_API_KEY not set"}
	} else {
		out["cloud"] = BackendInfo{Ready: true, Models: []string{AnthropicModel}}
	}
	return out
}

// checkModelExists rejects a switch to a model the provider does not have.
// It is a listing lookup rather than a test generation on purpose: a probe call
// would burn tokens from the very budget people are usually switching to
// escape.
func (a *App) checkModelExists(ctx context.Context, s StageName) error {
	avail := a.AvailableBackends(ctx)
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

// --- env helpers -------------------------------------------------------------

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
