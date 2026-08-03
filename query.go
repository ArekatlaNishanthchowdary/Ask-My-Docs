package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

type Source struct {
	ChunkID string  `json:"chunk_id"`
	DocID   string  `json:"doc_id"`
	Section string  `json:"section"`
	Text    string  `json:"text"`
	Score   float32 `json:"score"`
}

type QueryRequest struct {
	Question string   `json:"question"`
	ACL      []string `json:"acl,omitempty"` // caller's permission tags; empty means unrestricted
	NoCache  bool     `json:"no_cache,omitempty"`
}

type QueryResponse struct {
	Answer     string   `json:"answer"`
	Claims     []Claim  `json:"claims"`
	Sources    []Source `json:"sources"`
	Sufficient bool     `json:"sufficient"`
	CacheHit   bool     `json:"cache_hit"`
	Timings    Timings  `json:"timings_ms"`
	Warnings   []string `json:"warnings,omitempty"`
}

type Timings struct {
	Embed    int64 `json:"embed"`
	Retrieve int64 `json:"retrieve"`
	Rerank   int64 `json:"rerank"`
	Generate int64 `json:"generate"`
	Verify   int64 `json:"verify"`
	Total    int64 `json:"total"`
}

// Query runs the full pipeline: semantic cache -> embed -> hybrid retrieval with
// server-side RRF -> cross-encoder rerank -> confidence gate -> cited generation
// -> citation validation -> entailment verification.
func (a *App) Query(ctx context.Context, req QueryRequest) (QueryResponse, error) {
	start := time.Now()
	var resp QueryResponse

	t := time.Now()
	qvecs, err := a.Embedder.Embed(ctx, []string{req.Question}, "query")
	if err != nil {
		return resp, fmt.Errorf("embed query: %w", err)
	}
	qvec := qvecs[0]
	resp.Timings.Embed = ms(t)

	// Semantic cache. ponytail: stored as a second Qdrant collection rather than
	// standing up Redis — the vector store we already run is the only thing that
	// can answer "is this near-duplicate of a previous query" anyway.
	if !req.NoCache && len(req.ACL) == 0 {
		if cached, ok := a.cacheGet(ctx, qvec); ok {
			cached.CacheHit = true
			cached.Timings = resp.Timings
			cached.Timings.Total = ms(start)
			return cached, nil
		}
	}

	stage, err := a.RetrieveAndRank(ctx, req.Question, qvec, req.ACL)
	if err != nil {
		return resp, err
	}
	resp.Timings.Retrieve = stage.RetrieveMS
	resp.Timings.Rerank = stage.RerankMS

	if len(stage.Candidates) == 0 {
		resp.Answer = insufficientAnswer
		resp.Timings.Total = ms(start)
		return resp, nil
	}
	top := stage.Ranked
	resp.Sources = top

	// Confidence gate: if the best chunk after reranking is weak, the corpus
	// almost certainly does not contain the answer. Refusing here is cheaper
	// and safer than letting the model improvise from marginal context.
	gate := a.gateThreshold(top)
	if len(top) == 0 || top[0].Score < gate {
		resp.Answer = insufficientAnswer
		resp.Timings.Total = ms(start)
		return resp, nil
	}

	// Apply the same threshold to every source, not just the best one. TopK is
	// a ceiling, not a quota: padding the prompt out to 10 chunks when only one
	// clears the bar buries the answer in noise, and a small model responds to
	// that by returning nothing at all. Measured on the eval set, this converts
	// several silent refusals into correct cited answers.
	kept := top[:0:0]
	for _, s := range top {
		if s.Score >= gate {
			kept = append(kept, s)
		}
	}
	// TopK is applied here rather than at rerank time, so the gate could see
	// the whole distribution first.
	if len(kept) > a.Cfg.TopK {
		kept = kept[:a.Cfg.TopK]
	}
	top = kept
	resp.Sources = top

	t = time.Now()
	ans, err := a.llm().Answer(ctx, req.Question, top)
	if err != nil {
		return resp, fmt.Errorf("generate: %w", err)
	}
	// The confidence gate above already established that the corpus covers this
	// question, so an empty answer contradicts a decision we have already made.
	// On a small local model that is usually a sampling hiccup rather than a
	// real refusal, and one re-roll recovers most of them. Genuine
	// out-of-corpus questions never reach here — the gate stops them.
	if len(ans.Claims) == 0 {
		if retry, rerr := a.llm().Answer(ctx, req.Question, top); rerr == nil && len(retry.Claims) > 0 {
			ans = retry
			resp.Warnings = append(resp.Warnings, "empty first answer; recovered on retry")
		}
	}
	resp.Timings.Generate = ms(t)

	if len(ans.Claims) == 0 {
		// Two empty answers after the gate passed. On a small model this is
		// usually a hiccup; on a capable one it is the model judging that these
		// sources do not answer the question — a second opinion on the gate.
		// Either way, record what it was given so the two are distinguishable.
		resp.Warnings = append(resp.Warnings,
			fmt.Sprintf("generator produced no claims from %d source(s), best %q at %.4f",
				len(top), top[0].ChunkID, top[0].Score))
		resp.Answer = insufficientAnswer
		resp.Timings.Total = ms(start)
		return resp, nil
	}

	// Hard constraint: a citation to a chunk that was not retrieved is dropped,
	// see resolveCitation for how near-miss transcriptions are handled.
	// not trusted. A claim left with no valid citation is an ungrounded claim
	// and the whole answer is refused rather than shipped.
	valid := map[string]Source{}
	loose := map[string][]string{}
	for _, s := range top {
		valid[s.ChunkID] = s
		k := loosenChunkID(s.ChunkID)
		loose[k] = append(loose[k], s.ChunkID)
	}
	for i := range ans.Claims {
		kept := ans.Claims[i].Citations[:0]
		for _, cid := range ans.Claims[i].Citations {
			if resolved, ok := resolveCitation(cid, valid, loose); ok {
				kept = append(kept, resolved)
			} else {
				resp.Warnings = append(resp.Warnings,
					fmt.Sprintf("dropped citation to unretrieved chunk %q", cid))
			}
		}
		ans.Claims[i].Citations = kept
		if len(kept) == 0 {
			resp.Answer = insufficientAnswer
			resp.Warnings = append(resp.Warnings, "claim had no valid citation")
			resp.Timings.Total = ms(start)
			return resp, nil
		}
	}

	t = time.Now()
	entailed, err := a.verifyClaims(ctx, ans.Claims, valid)
	if err != nil {
		return resp, fmt.Errorf("verify: %w", err)
	}
	resp.Timings.Verify = ms(t)

	// An unentailed claim means the model cited something that does not actually
	// support it — exactly the hallucination this stage exists to catch — so it
	// is dropped and never reaches the user. Dropping it rather than refusing
	// the whole answer is the same policy the citation check above already
	// applies, and it matters because verification is per-claim: a thorough
	// answer runs the verifier eight or nine times, so an all-or-nothing rule
	// refuses long answers more often than short ones even when every claim is
	// sound. If nothing survives, there is no answer left and we refuse.
	verified, dropped := keepEntailed(ans.Claims, entailed)
	for _, i := range dropped {
		resp.Warnings = append(resp.Warnings,
			fmt.Sprintf("dropped claim %d, not entailed by its citations", i))
	}
	if len(verified) == 0 {
		resp.Answer = insufficientAnswer
		resp.Timings.Total = ms(start)
		return resp, nil
	}
	ans.Claims = verified

	resp.Sufficient = true
	resp.Claims = ans.Claims
	resp.Answer = renderAnswer(ans.Claims)
	resp.Timings.Total = ms(start)

	if !req.NoCache && len(req.ACL) == 0 {
		a.cachePut(ctx, qvec, req.Question, resp)
	}
	return resp, nil
}

// keepEntailed splits claims by the verifier's verdicts, returning the ones to
// emit and the indexes of the ones dropped. Order is preserved, since the claims
// are the answer's prose and read in sequence.
//
// A verdict list shorter than the claim list would silently emit unverified
// claims, so anything unaccounted for is treated as not entailed. verifyClaims
// already rejects a length mismatch; this is the belt to that braces.
func keepEntailed(claims []Claim, entailed []bool) ([]Claim, []int) {
	kept := claims[:0:0]
	var dropped []int
	for i, c := range claims {
		if i < len(entailed) && entailed[i] {
			kept = append(kept, c)
			continue
		}
		dropped = append(dropped, i)
	}
	return kept, dropped
}

// loosenChunkID collapses everything models get wrong when retyping an id —
// case, and runs of spaces/underscores/hyphens/dots — while keeping the "#N"
// chunk index significant, since two chunks of one document are different
// evidence and must never merge.
func loosenChunkID(id string) string {
	var b strings.Builder
	sep := false
	for _, r := range strings.ToLower(id) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '#':
			if sep && b.Len() > 0 {
				b.WriteByte('_')
			}
			sep = false
			b.WriteRune(r)
		default:
			sep = true
		}
	}
	return b.String()
}

// resolveCitation maps a model-written citation onto a retrieved chunk id.
//
// Models retype ids rather than copy them, and space-vs-underscore is the slip
// they actually make: with both "Arekatla Nishanth Chowdary_Doc.docx" and
// "Arekatla_Nishanth_Chowdary_Resume_Viasat.docx" retrieved, the first was
// cited using the second's underscores and a correct, grounded answer was
// refused over punctuation.
//
// This loosens transcription, not the guard. Candidates come only from chunks
// that were actually retrieved, so an invented id still has nothing to match.
// An id that loosens onto more than one retrieved chunk is dropped rather than
// guessed — picking one would attach a claim to evidence nobody chose.
func resolveCitation(cid string, valid map[string]Source, loose map[string][]string) (string, bool) {
	if _, ok := valid[cid]; ok {
		return cid, true
	}
	if ids := loose[loosenChunkID(cid)]; len(ids) == 1 {
		return ids[0], true
	}
	return "", false
}

// verifyClaims checks that each claim is supported by the evidence it cites.
//
// The evidence is the union of a claim's cited chunks, not each chunk in turn.
// Checking citations individually looks stricter but is actually wrong: a model
// writing "the lab has an oscilloscope and a soldering iron" from two chunks
// cites both, and neither chunk alone entails the sentence — so a per-citation
// check refuses a perfectly grounded answer. What the citation list asserts is
// "these sources, together, support this", and that is what gets verified.
//
// It also costs one call per answer rather than one per citation, which matters
// on a rate-limited endpoint.
func (a *App) verifyClaims(ctx context.Context, claims []Claim, valid map[string]Source) ([]bool, error) {
	pairs := make([]VerifyPair, 0, len(claims))
	for _, c := range claims {
		var ev strings.Builder
		for _, cid := range c.Citations {
			src, ok := valid[cid]
			if !ok {
				continue
			}
			if ev.Len() > 0 {
				ev.WriteString("\n\n")
			}
			fmt.Fprintf(&ev, "[%s] %s", cid, src.Text)
		}
		pairs = append(pairs, VerifyPair{Claim: c.Text, Evidence: ev.String()})
	}
	verdicts, err := a.verifier().Verify(ctx, pairs)
	if err != nil {
		return nil, err
	}
	if len(verdicts) != len(claims) {
		return nil, fmt.Errorf("expected %d verdicts, got %d", len(claims), len(verdicts))
	}
	return verdicts, nil
}

// gateThreshold decides how good a chunk has to be to count as relevant.
//
// A fixed sigmoid threshold does not survive contact with a new corpus: what a
// cross-encoder calls 0.99 on prose that directly answers a question, it calls
// 0.008 on a comma-separated parts list that answers it just as well. Both
// times the *ranking* was correct and only the absolute number moved, so
// gating on the number alone refuses correct answers on any corpus it was not
// tuned for — which is every corpus until someone writes golden items.
//
// So gate on the margin instead: the median candidate is, by construction,
// noise (most of the 50 fused candidates are irrelevant to any one question),
// and a genuinely relevant chunk sits well above it. An out-of-corpus question
// produces no such separation — everything clusters at the noise floor.
// MIN_RERANK_SCORE remains an absolute floor beneath which nothing passes.
func (a *App) gateThreshold(candidates []Source) float32 {
	floor := a.Cfg.MinRerankScore
	if a.Cfg.RerankMargin <= 0 || len(candidates) < 4 {
		return floor // relative gating disabled, or too few samples to be meaningful
	}
	scores := make([]float32, len(candidates))
	for i, c := range candidates {
		scores[i] = c.Score
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i] < scores[j] })
	median := scores[len(scores)/2]

	if rel := median * a.Cfg.RerankMargin; rel > floor {
		return rel
	}
	return floor
}

// Stage holds both the fused candidate set and the reranked top-k. The eval
// harness needs both to report the reranker's nDCG lift over retrieval alone.
type Stage struct {
	Candidates []Source
	Ranked     []Source
	RetrieveMS int64
	RerankMS   int64
}

// RetrieveAndRank is steps 2-4 of the pipeline, factored out so the eval
// harness measures exactly the code the server runs.
func (a *App) RetrieveAndRank(ctx context.Context, question string, qvec []float32, acl []string) (Stage, error) {
	var st Stage

	t := time.Now()
	hits, err := a.Qdrant.HybridSearch(ctx, a.Cfg.Collection, qvec,
		SparseEncode(question), a.Cfg.CandidateK, aclFilter(acl))
	if err != nil {
		return st, fmt.Errorf("retrieve: %w", err)
	}
	st.RetrieveMS = ms(t)
	if len(hits) == 0 {
		return st, nil
	}

	docs := make([]string, len(hits))
	st.Candidates = make([]Source, len(hits))
	for i, h := range hits {
		st.Candidates[i] = Source{
			ChunkID: str(h.Payload["chunk_id"]),
			DocID:   str(h.Payload["doc_id"]),
			Section: str(h.Payload["section"]),
			Text:    str(h.Payload["text"]),
			Score:   h.Score,
		}
		// The reranker sees the situating context too — the same text the
		// embedding was built from, so both stages judge the same object.
		docs[i] = strings.TrimSpace(str(h.Payload["context"]) + "\n\n" + st.Candidates[i].Text)
	}

	t = time.Now()
	// Rerank everything, not just TopK. The full score distribution is what the
	// confidence gate measures its margin against, and truncating here would
	// leave only the top of the distribution — biasing the noise floor upward.
	// TopK is applied after gating.
	ranked, err := a.Embedder.Rerank(ctx, question, docs, 0)
	if err != nil {
		return st, fmt.Errorf("rerank: %w", err)
	}
	st.RerankMS = ms(t)

	st.Ranked = make([]Source, 0, len(ranked))
	for _, r := range ranked {
		s := st.Candidates[r.Index]
		s.Score = r.Score
		st.Ranked = append(st.Ranked, s)
	}
	return st, nil
}

const insufficientAnswer = "I don't have enough information in the indexed documents to answer that."

func renderAnswer(claims []Claim) string {
	var sb strings.Builder
	for i, c := range claims {
		if i > 0 {
			sb.WriteString(" ")
		}
		sb.WriteString(c.Text)
		sb.WriteString(" [")
		sb.WriteString(strings.Join(c.Citations, ", "))
		sb.WriteString("]")
	}
	return sb.String()
}

// aclFilter restricts retrieval to chunks the caller is allowed to see.
// An empty tag list means no restriction — callers that need enforcement must
// pass their tags; there is no implicit "allow all" for a non-empty list.
func aclFilter(acl []string) map[string]any {
	if len(acl) == 0 {
		return nil
	}
	return map[string]any{
		"must": []map[string]any{
			{"key": "acl", "match": map[string]any{"any": acl}},
		},
	}
}

// --- Semantic cache -------------------------------------------------------

func (a *App) cacheGet(ctx context.Context, qvec []float32) (QueryResponse, bool) {
	hits, err := a.Qdrant.DenseSearch(ctx, a.Cfg.CacheCollection, qvec, 1)
	if err != nil || len(hits) == 0 || hits[0].Score < a.Cfg.CacheThreshold {
		return QueryResponse{}, false
	}
	var out QueryResponse
	if err := json.Unmarshal([]byte(str(hits[0].Payload["response"])), &out); err != nil {
		return QueryResponse{}, false
	}
	return out, true
}

func (a *App) cachePut(ctx context.Context, qvec []float32, question string, resp QueryResponse) {
	blob, err := json.Marshal(resp)
	if err != nil {
		return
	}
	// Cache writes are best-effort: a failure here costs latency on the next
	// identical query, nothing more.
	_ = a.Qdrant.Upsert(ctx, a.Cfg.CacheCollection, []Point{{
		ID:     pointID("cache:" + question),
		Vector: map[string]any{"dense": qvec},
		Payload: map[string]any{
			"question": question,
			"response": string(blob),
			"cached":   time.Now().UTC().Format(time.RFC3339),
		},
	}})
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func ms(t time.Time) int64 { return time.Since(t).Milliseconds() }
