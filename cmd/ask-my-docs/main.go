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
	"sort"
	"strings"
	"time"

	"askmydocs/internal/auth"
	"askmydocs/internal/extract"
	"askmydocs/internal/providers"
	"askmydocs/internal/qdrant"
	"askmydocs/internal/rag"
)

//go:embed ui.html
var uiHTML []byte

// --- provider construction ---------------------------------------------------
//
// This is the composition root: the only place that imports both rag (the
// interfaces and the App that consumes them) and providers (the concrete
// clients that satisfy those interfaces). rag deliberately does not know
// these types exist.

type compatEndpoint = rag.CompatEndpoint

// buildLLM constructs one generative backend. Model may be empty to keep the
// provider's configured default.
func buildLLM(cfg rag.Config, provider, modelID string) (rag.LLM, rag.StageName, error) {
	if e, ok := cfg.CompatEndpoint(provider); ok {
		if modelID != "" {
			e.Model = modelID
		}
		if e.Key == "" {
			return nil, rag.StageName{}, fmt.Errorf("%s needs an API key", provider)
		}
		if e.Model == "" {
			return nil, rag.StageName{}, fmt.Errorf("%s needs a model", provider)
		}
		// Reuse the one OpenAI-compatible client, pointed at this endpoint.
		cfg.OpenAIBaseURL, cfg.OpenAIKey, cfg.OpenAIModel = e.BaseURL, e.Key, e.Model
		cfg.OpenAISystemPrefix = e.SysPrefix
		return providers.NewOpenAICompat(cfg), rag.StageName{Provider: provider, Model: e.Model}, nil
	}
	switch provider {
	case "ollama":
		if modelID != "" {
			cfg.OllamaChatModel = modelID
		}
		if cfg.OllamaChatModel == "" {
			return nil, rag.StageName{}, fmt.Errorf("ollama needs a chat model (OLLAMA_CHAT_MODEL)")
		}
		return providers.NewOllama(cfg), rag.StageName{Provider: provider, Model: cfg.OllamaChatModel}, nil
	case "cloud":
		if os.Getenv("ANTHROPIC_API_KEY") == "" {
			return nil, rag.StageName{}, fmt.Errorf("cloud needs ANTHROPIC_API_KEY")
		}
		return providers.NewClaude(), rag.StageName{Provider: provider, Model: rag.AnthropicModel}, nil
	}
	return nil, rag.StageName{}, fmt.Errorf("unknown provider %q (want ollama, openai or cloud)", provider)
}

// describeStage resolves which backend and model a stage runs on, following
// the same provider-override precedence buildApp uses.
func describeStage(cfg rag.Config, provider string) rag.StageName {
	if provider == "" {
		provider = cfg.Provider
	}
	if e, ok := cfg.CompatEndpoint(provider); ok {
		return rag.StageName{Provider: provider, Model: e.Model}
	}
	switch provider {
	case "cloud":
		return rag.StageName{Provider: provider, Model: rag.AnthropicModel}
	}
	return rag.StageName{Provider: "ollama", Model: cfg.OllamaChatModel}
}

// buildApp wires the three stages independently: embeddings, reranking, and
// generation each come from whichever backend is configured for them. They are
// separate knobs because they have genuinely different constraints — embedding
// is high-volume and cheap to keep local, reranking is latency-critical, and
// generation is the quality-critical one worth spending a large model on.
func buildApp(cfg rag.Config) (*rag.App, error) {
	a := rag.NewApp(cfg)
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

	if e, hosted := cfg.CompatEndpoint(cfg.Provider); hosted {
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
		a.Embedder, a.LLM = providers.NewCompatEmbedder(e.BaseURL, e.Key, e.EmbedModel), llm
	} else {
		switch cfg.Provider {
		case "ollama":
			o := providers.NewOllama(cfg)
			a.Embedder, a.LLM = o, o
		case "cloud":
			key := os.Getenv("VOYAGE_API_KEY")
			if key == "" {
				return nil, fmt.Errorf("PROVIDER=cloud requires VOYAGE_API_KEY")
			}
			if os.Getenv("ANTHROPIC_API_KEY") == "" {
				return nil, fmt.Errorf("PROVIDER=cloud requires ANTHROPIC_API_KEY")
			}
			a.Embedder = providers.NewVoyage(key, cfg.EmbedModel, cfg.RerankModel)
			a.LLM = providers.NewClaude()
		default:
			return nil, fmt.Errorf("unknown PROVIDER %q (want ollama, nvidia, openai or cloud)", cfg.Provider)
		}
	}

	// A dedicated cross-encoder overrides whichever reranker the provider ships
	// with. Opt-in by setting RERANKER_URL, so the pipeline still runs with one
	// less moving part when it is not there.
	if cfg.RerankerURL != "" {
		a.Embedder = providers.NewRerankedEmbedder(a.Embedder, providers.NewTEIReranker(cfg.RerankerURL, cfg.RerankerBatch))
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
	a.LLMName = describeStage(cfg, cfg.LLMProvider)
	a.VerifyName = describeStage(cfg, firstNonEmpty(cfg.VerifyProvider, cfg.LLMProvider))
	if cfg.VerifyModel != "" {
		a.VerifyName.Model = cfg.VerifyModel
	}
	a.ContextName = describeStage(cfg, firstNonEmpty(cfg.ContextProvider, cfg.LLMProvider))
	if cfg.ContextModel != "" {
		a.ContextName.Model = cfg.ContextModel
	}
	a.JudgeName = describeStage(cfg, firstNonEmpty(cfg.JudgeProvider, cfg.LLMProvider))
	if cfg.JudgeProvider == "openai" && cfg.OpenAIJudgeModel != "" {
		a.JudgeName.Model = cfg.OpenAIJudgeModel
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
		v = strings.Trim(strings.TrimSpace(dropInlineComment(v)), `"'`)
		if _, set := os.LookupEnv(k); !set && k != "" {
			_ = os.Setenv(k, v)
		}
	}
}

// dropInlineComment removes a trailing `# ...` from a .env value.
//
// Every commented example in .env.example carries its options inline —
// `OPENAI_REASONING_EFFORT=low   # reasoning models only` — so uncommenting one
// used to set the value to "low   # reasoning models only", which the provider
// rejects with a 400 naming a value that looks correct in the file. The failure
// points at the setting rather than at the parser, which is what made it cost
// an afternoon.
//
// Only whitespace-then-# starts a comment, so a value that legitimately
// contains one (`pass#word`) survives. A quoted value is taken verbatim: that
// is the escape hatch for a value that really does end in " #".
func dropInlineComment(v string) string {
	if v = strings.TrimSpace(v); strings.HasPrefix(v, `"`) || strings.HasPrefix(v, `'`) {
		if end := strings.IndexByte(v[1:], v[0]); end >= 0 {
			return v[:end+2]
		}
		return v
	}
	for i := 1; i < len(v); i++ {
		if v[i] == '#' && (v[i-1] == ' ' || v[i-1] == '\t') {
			return strings.TrimRight(v[:i], " \t")
		}
	}
	return v
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
	cfg := rag.LoadConfig()
	ablate, err := rag.ParseAblate(os.Getenv("ABLATE"))
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
		run(ctx, cfg, rag.CmdAblate)
	case "calibrate":
		run(ctx, cfg, cmdCalibrate)
	case "token":
		// Offline: minting a credential needs no services, and requiring them
		// would mean standing up the whole stack to add a user.
		if err := auth.CmdToken(os.Args[2:]); err != nil {
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

func cmdCalibrate(ctx context.Context, a *rag.App, args []string) error {
	fs := flag.NewFlagSet("calibrate", flag.ExitOnError)
	golden := fs.String("golden", "eval/golden.jsonl", "path to the golden eval set")
	_ = fs.Parse(args)
	items, err := rag.LoadGolden(*golden)
	if err != nil {
		return err
	}
	if len(items) == 0 {
		return fmt.Errorf("%s contains no eval items — calibration needs both answerable "+
			"items and at least one with \"relevant_chunk_ids\": [] to bound the gate", *golden)
	}
	return a.Calibrate(ctx, items)
}

func cmdChunks(cfg rag.Config, args []string) error {
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
		if !rag.IsDoc(p) {
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
		body, err := extract.LoadDocumentText(docID, raw)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", docID, err)
			return nil
		}
		for _, c := range rag.ChunkDoc(body, cfg.MaxChunkChars, cfg.ChunkOverlap) {
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

func run(ctx context.Context, cfg rag.Config, fn func(context.Context, *rag.App, []string) error) {
	app, err := buildApp(cfg)
	if err != nil {
		log.Fatalf("error: %v", err)
	}
	if err := fn(ctx, app, os.Args[2:]); err != nil {
		log.Fatalf("error: %v", err)
	}
}

// --- serve ----------------------------------------------------------------

func cmdServe(ctx context.Context, a *rag.App, args []string) error {
	if err := a.EnsureCollections(ctx); err != nil {
		return err
	}
	switch {
	case a.Cfg.AuthTokensFile != "":
		tokens, err := auth.LoadTokens(a.Cfg.AuthTokensFile)
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

	mux.HandleFunc("POST /query", a.Guard(false, func(w http.ResponseWriter, r *http.Request, p auth.Principal) {
		var req rag.QueryRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
			rag.HTTPErr(w, http.StatusBadRequest, "invalid JSON body")
			return
		}
		if strings.TrimSpace(req.Question) == "" {
			rag.HTTPErr(w, http.StatusBadRequest, "question is required")
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
			rag.HTTPErr(w, http.StatusBadGateway, err.Error())
			return
		}
		rag.WriteJSON(w, http.StatusOK, resp)
	}))

	// GET /providers — what each generative stage is using, and what it could
	// use. Model lists come from the providers themselves rather than a
	// hardcoded table, so they stay true as models are pulled or retired.
	mux.HandleFunc("GET /providers", a.Guard(false, func(w http.ResponseWriter, r *http.Request, _ auth.Principal) {
		rag.WriteJSON(w, http.StatusOK, map[string]any{
			"current":   a.CurrentStages(),
			"available": a.AvailableBackends(r.Context()),
			// Surfaced so the UI can explain the omission rather than leaving
			// someone hunting for an embeddings dropdown that will never exist.
			"fixed": map[string]string{
				"embeddings": fmt.Sprintf("%s (%d-dim) — changing it requires re-indexing every document",
					a.Cfg.OllamaEmbedModel, a.EmbedDim()),
			},
		})
	}))

	// POST /providers — switch a stage. Validated before it is applied, so a
	// bad choice returns an error instead of breaking every later query.
	// Admin-only: this changes which model answers for every caller, not just
	// the one asking.
	mux.HandleFunc("POST /providers", a.Guard(true, func(w http.ResponseWriter, r *http.Request, p auth.Principal) {
		var req struct {
			Stage    string `json:"stage"`
			Provider string `json:"provider"`
			Model    string `json:"model"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&req); err != nil {
			rag.HTTPErr(w, http.StatusBadRequest, "invalid JSON body")
			return
		}
		next, name, err := buildLLM(a.Cfg, req.Provider, req.Model)
		if err != nil {
			rag.HTTPErr(w, http.StatusBadRequest, err.Error())
			return
		}
		if err := a.SwitchProvider(r.Context(), req.Stage, next, name); err != nil {
			rag.HTTPErr(w, http.StatusBadRequest, err.Error())
			return
		}

		log.Printf("%s switched %s to %s/%s", p.Name, req.Stage, name.Provider, name.Model)
		rag.WriteJSON(w, http.StatusOK, map[string]any{"stage": req.Stage, "using": name})
	}))

	// GET /documents — what is currently in the corpus directory, restricted to
	// what the caller may actually read. A filename is itself information —
	// "you cannot open this, but it is called 2026_redundancies.docx" is the
	// same leak in a smaller package — so a document with no readable chunks is
	// omitted entirely rather than listed with a zero.
	mux.HandleFunc("GET /documents", a.Guard(false, func(w http.ResponseWriter, r *http.Request, p auth.Principal) {
		all, err := listCorpus(a.Cfg.CorpusDir)
		if err != nil {
			rag.HTTPErr(w, http.StatusInternalServerError, "cannot read corpus directory")
			return
		}
		tags := p.Tags()
		docs := all[:0:0]
		for _, d := range all {
			// Report indexed chunks per document, not just what is on disk. A
			// file present with zero chunks means it was saved but never
			// indexed — a state worth being able to see rather than infer.
			n, err := a.Qdrant.CountFiltered(r.Context(), a.Cfg.Collection, qdrant.DocIDFilter(d.Name, tags))
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
		rag.WriteJSON(w, http.StatusOK, map[string]any{"documents": docs})
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
	mux.HandleFunc("POST /documents", a.Guard(true, func(w http.ResponseWriter, r *http.Request, p auth.Principal) {
		maxBytes := int64(a.Cfg.MaxUploadMB) << 20
		r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
		if err := r.ParseMultipartForm(16 << 20); err != nil {
			rag.HTTPErr(w, http.StatusRequestEntityTooLarge,
				fmt.Sprintf("upload too large or malformed (limit %dMB)", a.Cfg.MaxUploadMB))
			return
		}
		defer func() { _ = r.MultipartForm.RemoveAll() }()

		files := r.MultipartForm.File["files"]
		if len(files) == 0 {
			rag.HTTPErr(w, http.StatusBadRequest, "no files in the request")
			return
		}
		if err := os.MkdirAll(a.Cfg.CorpusDir, 0o755); err != nil {
			rag.HTTPErr(w, http.StatusInternalServerError, "cannot create corpus directory")
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
			dest, err := rag.SafeCorpusPath(a.Cfg.CorpusDir, fh.Filename)
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
			text, err := extract.LoadDocumentText(res.Name, body)
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
		rag.WriteJSON(w, http.StatusOK, map[string]any{
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
	mux.HandleFunc("POST /ingest", a.Guard(true, func(w http.ResponseWriter, r *http.Request, p auth.Principal) {
		acl := splitCSV(r.URL.Query().Get("acl"))
		// Deliberately NOT r.Context(). Indexing a corpus can run for minutes,
		// and tying it to the request means closing the tab aborts it partway
		// — measured here, a browser giving up cancelled the embed mid-document
		// and the work was thrown away. The response is still sent to whoever
		// is listening; the difference is that the indexing finishes either way.
		ctx, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), 30*time.Minute)
		defer cancel()

		// Unchanged documents are skipped by version, so the common case costs
		// a count query per document and no tokens. -force is deliberately not
		// exposed here: it re-contextualizes the whole corpus, which is the
		// most expensive thing this system can be asked to do, and a button
		// that does it by accident is worse than a flag you have to type.
		// One at a time. Two concurrent walks of the same directory would
		// contextualize and embed every document twice, which is the most
		// expensive way this system can waste a token — and a button invites
		// exactly that, because a slow one gets clicked again.
		indexed, skipped, chunks, err := a.IngestDir(ctx, a.Cfg.CorpusDir, acl, false)
		if err != nil {
			if err == rag.ErrIngestBusy {
				rag.HTTPErr(w, http.StatusConflict, "an ingest is already running")
				return
			}
			log.Printf("%s ingest failed: %v", p.Name, err)
			rag.HTTPErr(w, http.StatusBadGateway, err.Error())
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
		rag.WriteJSON(w, http.StatusOK, map[string]any{
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
			rag.HTTPErr(w, http.StatusServiceUnavailable, "qdrant unreachable")
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
		llmName := a.CurrentStages()["llm"]
		rag.WriteJSON(w, http.StatusOK, map[string]any{
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
		if e.IsDir() || !rag.IsDoc(e.Name()) {
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

// --- ingest ---------------------------------------------------------------

func cmdIngest(ctx context.Context, a *rag.App, args []string) error {
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

func cmdQuery(ctx context.Context, a *rag.App, args []string) error {
	fs := flag.NewFlagSet("query", flag.ExitOnError)
	acl := fs.String("acl", "", "comma-separated ACL tags for this caller")
	noCache := fs.Bool("no-cache", false, "bypass the semantic cache")
	jsonOut := fs.Bool("json", false, "print the full response as JSON")
	_ = fs.Parse(args)
	q := strings.Join(fs.Args(), " ")
	if strings.TrimSpace(q) == "" {
		return fmt.Errorf("usage: ask-my-docs query [flags] <question>")
	}
	resp, err := a.Query(ctx, rag.QueryRequest{Question: q, ACL: splitCSV(*acl), NoCache: *noCache})
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

func cmdEval(ctx context.Context, a *rag.App, args []string) error {
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

	items, err := rag.LoadGolden(*golden)
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
		rag.ReportItems(results)
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
	var base rag.Metrics
	if err := json.Unmarshal(raw, &base); err != nil {
		return fmt.Errorf("%s: %w", *baseline, err)
	}
	fails := rag.CompareBaseline(base, m, *tolerance, *latencyTol)
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
