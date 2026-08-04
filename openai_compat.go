package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
)

// debugLLM reports whether to log each upstream call.
//
// A function rather than a package var: package-level initialisation runs
// before main, and .env is loaded inside main, so a var here would see only
// real environment variables — DEBUG_LLM=1 written into .env, exactly as
// .env.example documents, would do nothing.
func debugLLM() bool { return os.Getenv("DEBUG_LLM") != "" }

// OpenAICompat drives any OpenAI-compatible chat-completions endpoint: Groq,
// OpenRouter, Together, Fireworks, or a self-hosted vLLM / LM Studio. One
// implementation covers all of them because the wire format is identical —
// only the base URL and model id differ.
//
// It implements LLM only. Neither Groq nor OpenRouter serves embeddings or
// reranking, so retrieval stays wherever PROVIDER and RERANKER_URL point it.
// That split is the useful one anyway: embeddings and reranking are the
// high-volume latency-critical stages worth keeping local, while generation is
// the low-volume quality-critical stage worth spending a 70B on.
type OpenAICompat struct {
	BaseURL         string
	APIKey          string
	Model           string
	JudgeModel      string
	MaxTokens       int
	Seed            int
	CtxBatch        int
	Concurrency     int
	ReasoningEffort string
	SystemPrefix    string
	MaxBackoff      time.Duration
	HTTP            *http.Client

	// Not every endpoint supports strict json_schema. The first 400 that
	// blames response_format flips this, and every later call uses the
	// json_object fallback instead of re-learning the same lesson.
	mu            sync.RWMutex
	useJSONObject bool
}

func NewOpenAICompat(cfg Config) *OpenAICompat {
	judge := cfg.OpenAIJudgeModel
	if judge == "" {
		judge = cfg.OpenAIModel
	}
	return &OpenAICompat{
		BaseURL:         strings.TrimRight(cfg.OpenAIBaseURL, "/"),
		APIKey:          cfg.OpenAIKey,
		Model:           cfg.OpenAIModel,
		JudgeModel:      judge,
		MaxTokens:       cfg.OpenAIMaxTokens,
		Seed:            cfg.OllamaSeed,
		CtxBatch:        cfg.OpenAIContextBatch,
		Concurrency:     cfg.OpenAIConcurrency,
		ReasoningEffort: cfg.OpenAIReasoningEffort,
		SystemPrefix:    cfg.OpenAISystemPrefix,
		MaxBackoff:      time.Duration(cfg.OpenAIMaxBackoffSec) * time.Second,
		HTTP:            &http.Client{Timeout: 5 * time.Minute},
	}
}

type chatChoice struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
	FinishReason string `json:"finish_reason"`
}

// chat issues one structured-output request against the chat-completions API.
func (o *OpenAICompat) chat(ctx context.Context, model, system, user string, schema map[string]any, out any) error {
	o.mu.RLock()
	fallback := o.useJSONObject
	o.mu.RUnlock()

	body, err := o.request(model, system, user, schema, fallback)
	if err != nil {
		return err
	}
	if debugLLM() {
		log.Printf("[llm] REQ %s", truncate(string(body), 1200))
	}
	started := time.Now()
	raw, status, err := o.post(ctx, body)
	if debugLLM() {
		log.Printf("[llm] model=%s fallback=%v status=%d in %s", model, fallback, status, time.Since(started).Round(time.Millisecond))
	}
	if err != nil {
		return err
	}
	// Endpoints that reject strict schemas say so with a 400. Downgrade once,
	// remember it, and retry rather than failing the whole run.
	if status == http.StatusBadRequest && !fallback && mentionsSchema(string(raw)) {
		o.mu.Lock()
		o.useJSONObject = true
		o.mu.Unlock()
		if body, err = o.request(model, system, user, schema, true); err != nil {
			return err
		}
		if raw, status, err = o.post(ctx, body); err != nil {
			return err
		}
	}
	if status == http.StatusTooManyRequests {
		// Lead with the provider's own words: it names which limit was hit and
		// how long it lasts, which is the only thing that tells you whether to
		// wait a minute or switch model.
		return fmt.Errorf("rate limited by %s: %s", o.BaseURL, providerMessage(raw))
	}
	if status >= 300 {
		return fmt.Errorf("%s: %d: %s", o.BaseURL, status, providerMessage(raw))
	}

	var res struct {
		Choices []chatChoice `json:"choices"`
		Error   *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		return fmt.Errorf("decoding response: %w: %s", err, truncate(string(raw), 200))
	}
	if res.Error != nil {
		return fmt.Errorf("%s: %s", o.BaseURL, res.Error.Message)
	}
	if len(res.Choices) == 0 {
		return fmt.Errorf("%s: no choices returned", o.BaseURL)
	}
	if res.Choices[0].FinishReason == "length" {
		return fmt.Errorf("response truncated at max_tokens=%d; raise OPENAI_MAX_TOKENS", o.MaxTokens)
	}
	content := strings.TrimSpace(res.Choices[0].Message.Content)
	// A response that parses into the right shape but the wrong contents (an
	// empty array where verdicts were expected) is invisible from the error.
	if debugLLM() {
		log.Printf("DEBUG_LLM %s -> %s", model, truncate(content, 400))
	}
	if content == "" {
		return fmt.Errorf("%s: empty response from %s", o.BaseURL, model)
	}
	if err := json.Unmarshal([]byte(stripFence(content)), out); err != nil {
		return fmt.Errorf("response did not match schema: %w: %s", err, truncate(content, 200))
	}
	return nil
}

func (o *OpenAICompat) request(model, system, user string, schema map[string]any, fallback bool) ([]byte, error) {
	// Some models read reasoning directives out of the system prompt rather
	// than a request field. NVIDIA's Nemotron family is the reason this exists:
	// left in its default thinking mode it spent 8192 output tokens reasoning
	// and never reached the JSON (197s, truncated). With "detailed thinking
	// off" the same call answered in 1.8s and 172 tokens.
	if o.SystemPrefix != "" {
		system = o.SystemPrefix + "\n" + system
	}
	req := map[string]any{
		"model":       model,
		"temperature": 0,
		"seed":        o.Seed,
		"max_tokens":  o.MaxTokens,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
	}
	// Reasoning models spend most of their latency thinking before answering.
	// Measured on gpt-oss-120b for a single entailment check: high effort took
	// 2.48s and 1009 completion tokens, low took 0.49s and 43 — for a question
	// whose answer is one boolean. Only sent when set, since not every endpoint
	// accepts the field.
	if o.ReasoningEffort != "" {
		req["reasoning_effort"] = o.ReasoningEffort
	}
	if fallback {
		// json_object mode constrains syntax but not shape, so the schema has to
		// go in the prompt. Several providers also require the literal word
		// "JSON" to appear in the messages before they accept this mode.
		pretty, _ := json.Marshal(schema)
		req["response_format"] = map[string]any{"type": "json_object"}
		req["messages"] = []map[string]string{
			{"role": "system", "content": system + "\n\nReply with JSON matching this schema:\n" + string(pretty)},
			{"role": "user", "content": user},
		}
	} else {
		req["response_format"] = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name": "response", "strict": true, "schema": schema,
			},
		}
	}
	return json.Marshal(req)
}

// post sends one request, retrying on rate limits and transient server errors.
//
// Rate limits are the normal case rather than the exception here: a free Groq
// tier allows 12k tokens/minute, and this pipeline is token-heavy (whole
// documents during contextualization, ten sources per answer). Providers say
// exactly how long to wait in `retry-after`, so honour it instead of guessing
// a backoff or failing an ingest that would have succeeded seconds later.
func (o *OpenAICompat) post(ctx context.Context, body []byte) ([]byte, int, error) {
	const maxAttempts = 5
	var raw []byte
	var status int
	var slept time.Duration // total time spent backing off, across attempts
	var transport error     // last transport failure, if the attempts ran out on one

	for attempt := 0; attempt < maxAttempts; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.BaseURL+"/chat/completions", bytes.NewReader(body))
		if err != nil {
			return nil, 0, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+o.APIKey)

		var retryAfter string
		started := time.Now()
		resp, err := o.HTTP.Do(req)
		switch {
		case err != nil:
			// A connection reset or an unexpected EOF is transient in exactly
			// the way a 502 is, and the backoff below already exists for that.
			// Returning here instead meant one dropped connection failed a
			// whole answer — and in eval, silently dropped that item from the
			// metrics rather than failing the run, so the measurement quietly
			// got smaller. Measured against NVIDIA NIM at roughly one item in
			// twenty.
			//
			// A dead context is the exception: it is not transient, and
			// retrying it spends the remaining budget failing the same way.
			if ctx.Err() != nil {
				return nil, 0, fmt.Errorf("%s: %w", o.BaseURL, err)
			}
			transport = err
			if debugLLM() {
				log.Printf("[llm] attempt %d transport error after %s: %v",
					attempt, time.Since(started).Round(time.Millisecond), err)
			}
		default:
			transport = nil
			raw, _ = io.ReadAll(resp.Body)
			status = resp.StatusCode
			retryAfter = resp.Header.Get("retry-after")
			resp.Body.Close()
			if debugLLM() {
				log.Printf("[llm] attempt %d: status=%d reqBytes=%d respBytes=%d in %s",
					attempt, status, len(body), len(raw), time.Since(started).Round(time.Millisecond))
			}
			if status != http.StatusTooManyRequests && status < 500 {
				return raw, status, nil
			}
		}

		if attempt == maxAttempts-1 {
			break
		}
		wait := parseRetryAfter(retryAfter)
		if wait <= 0 {
			wait = time.Duration(1<<attempt) * time.Second // 1s, 2s, 4s, 8s
		}
		wait += 500 * time.Millisecond // clear the window rather than land on its edge

		// A per-minute limit is worth waiting out; a per-day one is not. When the
		// provider asks for longer than any request should live, stop and return
		// the 429 so the caller reports "daily token limit reached" instead of
		// sleeping for an hour and surfacing a meaningless deadline error.
		//
		// The cap is a budget for the whole call, not for one sleep. Checked
		// per-sleep it bounded nothing useful: four retries just under a 45s cap
		// legally sat for ~180s, which is how a p50 of 21s grew a p95 of 162s.
		if slept+wait > o.MaxBackoff {
			if debugLLM() {
				log.Printf("[llm] backoff budget spent (%s + %s > %s cap); failing fast", slept, wait, o.MaxBackoff)
			}
			if transport != nil {
				// There is no response to hand back on this path, and a nil
				// error beside a zero status would read as a successful call.
				return nil, 0, fmt.Errorf("%s: %w", o.BaseURL, transport)
			}
			return raw, status, nil
		}
		slept += wait
		if debugLLM() {
			log.Printf("[llm] backing off %s (retry-after=%q)", wait, retryAfter)
		}
		select {
		case <-ctx.Done():
			return nil, 0, ctx.Err()
		case <-time.After(wait):
		}
	}
	if transport != nil {
		return nil, 0, fmt.Errorf("%s: %w (after %d attempts)", o.BaseURL, transport, maxAttempts)
	}
	return raw, status, nil
}

func parseRetryAfter(v string) time.Duration {
	if v == "" {
		return 0
	}
	if secs, err := strconv.ParseFloat(strings.TrimSpace(v), 64); err == nil && secs >= 0 {
		return time.Duration(secs * float64(time.Second))
	}
	return 0
}

func mentionsSchema(body string) bool {
	b := strings.ToLower(body)
	return strings.Contains(b, "response_format") || strings.Contains(b, "json_schema") ||
		strings.Contains(b, "schema")
}

// stripFence unwraps ```json fences, which json_object-mode providers sometimes
// emit despite being asked for bare JSON.
func stripFence(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") {
		return s
	}
	if i := strings.Index(s, "\n"); i >= 0 {
		s = s[i+1:]
	}
	return strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(s), "```"))
}

// --- LLM ------------------------------------------------------------------

func (o *OpenAICompat) Contextualize(ctx context.Context, doc string, chunks []Chunk) ([]string, error) {
	out := make([]string, len(chunks))
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
			schema := map[string]any{
				"type": "object",
				"properties": map[string]any{
					// Pinned length for the same reason as the verify schema: an
					// unconstrained array lets the model return [] and still be valid.
					"contexts": map[string]any{
						"type":     "array",
						"items":    map[string]any{"type": "string"},
						"minItems": len(batch),
						"maxItems": len(batch),
					},
				},
				"required":             []string{"contexts"},
				"additionalProperties": false,
			}
			var res struct {
				Contexts []string `json:"contexts"`
			}
			err := o.chat(gctx, o.Model, fmt.Sprintf(contextSystem, doc),
				fmt.Sprintf(contextUser, len(batch), len(batch), sb.String()), schema, &res)
			if err != nil {
				return err
			}
			if len(res.Contexts) != len(batch) {
				return fmt.Errorf("contextualize: expected %d contexts, got %d", len(batch), len(res.Contexts))
			}
			copy(out[start:end], res.Contexts)
			return nil
		})
	}
	return out, g.Wait()
}

func (o *OpenAICompat) Answer(ctx context.Context, question string, sources []Source) (Answer, error) {
	var sb strings.Builder
	sb.WriteString("Sources:\n\n")
	for _, s := range sources {
		fmt.Fprintf(&sb, "<source id=\"%s\" doc=\"%s\" section=\"%s\">\n%s\n</source>\n\n",
			s.ChunkID, s.DocID, s.Section, s.Text)
	}
	fmt.Fprintf(&sb, "Question: %s\n\nCite sources by their exact id, e.g. %q.",
		question, sources[0].ChunkID)

	var a Answer
	err := o.chat(ctx, o.Model, answerSystem, sb.String(), answerSchema, &a)
	return a, err
}

// Verify checks every claim in one request.
//
// This is a deliberate latency-for-tokens trade, not the cheapest wall clock.
// One call per claim runs the checks concurrently and finishes in about the
// time of its slowest check, but it retransmits the evidence once per claim.
// Batched, the model reasons through the claims in sequence, so the stage costs
// roughly the sum — which is why verify dominates query latency and is the
// first place to look when the tail grows. Tokens per minute, not requests, is
// what a metered endpoint limits, and that is the constraint this project is
// built under.
func (o *OpenAICompat) Verify(ctx context.Context, pairs []VerifyPair) ([]bool, error) {
	if len(pairs) == 0 {
		return nil, nil
	}
	// Claims in one answer overwhelmingly cite the same handful of chunks, so
	// sending each claim with its own copy of the evidence retransmits the same
	// text once per claim. On a six-claim answer over four chunks that is ~6x
	// the tokens for identical content — and tokens per minute, not requests,
	// is what a metered endpoint actually limits. Send each distinct piece of
	// evidence once and have the claims refer to it.
	seen := map[string]int{}
	var evidence []string
	refs := make([]int, len(pairs))
	for i, p := range pairs {
		ev := truncate(p.Evidence, 8000)
		idx, ok := seen[ev]
		if !ok {
			idx = len(evidence)
			seen[ev] = idx
			evidence = append(evidence, ev)
		}
		refs[i] = idx
	}

	var sb strings.Builder
	for i, ev := range evidence {
		fmt.Fprintf(&sb, "<evidence id=\"E%d\">\n%s\n</evidence>\n\n", i, ev)
	}
	for i, p := range pairs {
		fmt.Fprintf(&sb, "<item index=\"%d\" evidence=\"E%d\">%s</item>\n", i, refs[i], p.Claim)
	}
	fmt.Fprintf(&sb, "\nReturn exactly %d verdicts, in index order.", len(pairs))

	// minItems/maxItems are load-bearing, not decoration. Without them an empty
	// array satisfies the schema, and Nemotron reliably returned {"entailed":[]}
	// — a valid response that skips the work. Pinning the length to the number
	// of claims makes the shortcut unrepresentable.
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"entailed": map[string]any{
				"type":     "array",
				"items":    map[string]any{"type": "boolean"},
				"minItems": len(pairs),
				"maxItems": len(pairs),
			},
		},
		"required":             []string{"entailed"},
		"additionalProperties": false,
	}
	var res struct {
		Entailed []bool `json:"entailed"`
	}
	if err := o.chat(ctx, o.Model, verifySystem, sb.String(), schema, &res); err != nil {
		return nil, err
	}
	if len(res.Entailed) != len(pairs) {
		return nil, fmt.Errorf("verify: expected %d verdicts, got %d", len(pairs), len(res.Entailed))
	}
	return res.Entailed, nil
}

// Judge runs on JudgeModel, which defaults to Model. Point OPENAI_JUDGE_MODEL
// at a *different* model than the generator — a model grading its own output
// is not an independent measurement, and answer_correctness is the one metric
// that silently rewards that.
func (o *OpenAICompat) Judge(ctx context.Context, question, reference, candidate string) (float64, error) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"score":     map[string]any{"type": "number"},
			"reasoning": map[string]any{"type": "string"},
		},
		"required":             []string{"score", "reasoning"},
		"additionalProperties": false,
	}
	var res struct {
		Score float64 `json:"score"`
	}
	user := fmt.Sprintf("Question: %s\n\nReference answer: %s\n\nCandidate answer: %s",
		question, reference, candidate)
	if err := o.chat(ctx, o.JudgeModel, judgeSystem, user, schema, &res); err != nil {
		return 0, err
	}
	return res.Score, nil
}

// providerMessage pulls the human-readable message out of an OpenAI-style
// error envelope, falling back to the raw body.
func providerMessage(raw []byte) string {
	var e struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &e) == nil && e.Error.Message != "" {
		return e.Error.Message
	}
	return truncate(string(raw), 300)
}
