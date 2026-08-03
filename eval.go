package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"
)

// GoldenItem is one graded example. Tagging each item by the stage it exercises
// is what lets a regression be attributed to retrieval vs reranking vs
// generation instead of just "the number went down".
type GoldenItem struct {
	ID       string   `json:"id"`
	Query    string   `json:"query"`
	Relevant []string `json:"relevant_chunk_ids"`
	Gold     string   `json:"gold_answer,omitempty"`
	Stage    string   `json:"stage,omitempty"` // retrieval | rerank | generation
	ACL      []string `json:"acl,omitempty"`
}

type Metrics struct {
	N                 int     `json:"n"`
	RecallAt10        float64 `json:"recall_at_10"`
	NDCGAt10          float64 `json:"ndcg_at_10"`
	MRR               float64 `json:"mrr"`
	RetrievalNDCGAt10 float64 `json:"retrieval_ndcg_at_10"` // pre-rerank baseline
	RerankLift        float64 `json:"rerank_lift"`
	CitationPrecision float64 `json:"citation_precision"`
	CitationRecall    float64 `json:"citation_recall"`
	AnswerCorrectness float64 `json:"answer_correctness"`
	LatencyP50        int64   `json:"latency_p50_ms"`
	LatencyP95        int64   `json:"latency_p95_ms"`
	LatencyP99        int64   `json:"latency_p99_ms"`
	RetrieveP95       int64   `json:"retrieve_p95_ms"`
	RerankP95         int64   `json:"rerank_p95_ms"`
	Failures          int     `json:"failures"`
}

// gated lists the metrics CI enforces. Direction is "higher is better" for
// quality and "lower is better" for latency.
var gatedQuality = []string{"recall_at_10", "ndcg_at_10", "mrr", "citation_precision", "answer_correctness"}
var gatedLatency = []string{"retrieve_p95_ms", "rerank_p95_ms"}

func LoadGolden(path string) ([]GoldenItem, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var items []GoldenItem
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		var it GoldenItem
		if err := json.Unmarshal([]byte(line), &it); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		items = append(items, it)
	}
	return items, sc.Err()
}

// ItemResult is the per-item detail behind the aggregates. Averages say a
// number moved; this says which query moved it and at which stage.
type ItemResult struct {
	ID        string
	Stage     string
	FirstRank int     // 1-based rank of the first relevant chunk; 0 = not retrieved
	TopScore  float32 // reranker score of the top chunk
	Refused   bool
	Cited     []string
	Judge     float64
	Judged    bool
	TotalMS   int64
	Note      string
}

// RunEval executes every golden item through the real pipeline and aggregates.
func (a *App) RunEval(ctx context.Context, items []GoldenItem, judge bool) (Metrics, []ItemResult, error) {
	var m Metrics
	var recalls, ndcgs, rrs, baseNDCG, citP, citR, correct []float64
	var totals, retrieves, reranks []int64
	results := make([]ItemResult, 0, len(items))

	for _, it := range items {
		res := ItemResult{ID: it.ID, Stage: it.Stage}
		qvecs, err := a.Embedder.Embed(ctx, []string{it.Query}, "query")
		if err != nil {
			return m, results, err
		}
		start := time.Now()
		stage, err := a.RetrieveAndRank(ctx, it.Query, qvecs[0], it.ACL)
		if err != nil {
			fmt.Fprintf(os.Stderr, "eval %s: %v\n", it.ID, err)
			m.Failures++
			res.Note = "retrieve/rerank error"
			results = append(results, res)
			continue
		}
		retrieves = append(retrieves, stage.RetrieveMS)
		reranks = append(reranks, stage.RerankMS)
		if len(stage.Ranked) > 0 {
			res.TopScore = stage.Ranked[0].Score
		}

		rel := map[string]bool{}
		for _, r := range it.Relevant {
			rel[r] = true
		}
		for i, id := range ids(stage.Ranked) {
			if rel[id] {
				res.FirstRank = i + 1
				break
			}
		}
		// Negative-rejection items have no relevant chunks by construction.
		// Scoring them as recall 0 would punish the system for behaving
		// correctly, so they contribute to citation and answer metrics only.
		if len(rel) > 0 {
			rankedIDs := ids(stage.Ranked)
			recalls = append(recalls, recallAt(rankedIDs, rel, 10))
			ndcgs = append(ndcgs, ndcgAt(rankedIDs, rel, 10))
			rrs = append(rrs, reciprocalRank(rankedIDs, rel))
			baseNDCG = append(baseNDCG, ndcgAt(ids(stage.Candidates), rel, 10))
		}

		// Generation-stage items additionally exercise citation and answer quality.
		if it.Stage == "generation" || it.Gold != "" {
			resp, err := a.Query(ctx, QueryRequest{Question: it.Query, ACL: it.ACL, NoCache: true})
			if err != nil {
				fmt.Fprintf(os.Stderr, "eval %s: %v\n", it.ID, err)
				m.Failures++
				res.Note = "query error"
				results = append(results, res)
				continue
			}
			totals = append(totals, resp.Timings.Total)
			res.TotalMS = resp.Timings.Total
			res.Refused = !resp.Sufficient
			if len(resp.Warnings) > 0 {
				res.Note = resp.Warnings[0]
			}

			cited := map[string]bool{}
			for _, c := range resp.Claims {
				for _, id := range c.Citations {
					cited[id] = true
				}
			}
			for id := range cited {
				res.Cited = append(res.Cited, id)
			}
			sort.Strings(res.Cited)
			hitCount := 0
			for id := range cited {
				if rel[id] {
					hitCount++
				}
			}
			if len(cited) > 0 {
				citP = append(citP, float64(hitCount)/float64(len(cited)))
			}
			if len(rel) > 0 {
				citR = append(citR, float64(hitCount)/float64(len(rel)))
			}
			if judge && it.Gold != "" {
				score, err := a.judger().Judge(ctx, it.Query, it.Gold, resp.Answer)
				if err != nil {
					fmt.Fprintf(os.Stderr, "eval %s: judge: %v\n", it.ID, err)
				} else {
					correct = append(correct, score)
					res.Judge, res.Judged = score, true
				}
			}
		} else {
			totals = append(totals, ms(start))
			res.TotalMS = ms(start)
		}
		results = append(results, res)
	}

	m.N = len(items)
	m.RecallAt10 = mean(recalls)
	m.NDCGAt10 = mean(ndcgs)
	m.MRR = mean(rrs)
	m.RetrievalNDCGAt10 = mean(baseNDCG)
	m.RerankLift = m.NDCGAt10 - m.RetrievalNDCGAt10
	m.CitationPrecision = mean(citP)
	m.CitationRecall = mean(citR)
	m.AnswerCorrectness = mean(correct)
	m.LatencyP50 = pct(totals, 0.50)
	m.LatencyP95 = pct(totals, 0.95)
	m.LatencyP99 = pct(totals, 0.99)
	m.RetrieveP95 = pct(retrieves, 0.95)
	m.RerankP95 = pct(reranks, 0.95)
	return m, results, nil
}

// ReportItems prints the per-item breakdown, worst first. Aggregates tell you a
// number moved; this tells you which query moved it and at which stage, which
// is the difference between "correctness dropped" and "the reranker buried
// chunk 4 on the two paraphrase queries".
func ReportItems(results []ItemResult) {
	sorted := append([]ItemResult(nil), results...)
	sort.SliceStable(sorted, func(i, j int) bool { return itemScore(sorted[i]) < itemScore(sorted[j]) })

	fmt.Printf("\n%-24s %-11s %5s %6s %-8s %-22s %s\n",
		"ITEM", "STAGE", "RANK", "TOP", "JUDGE", "CITED", "NOTE")
	for _, r := range sorted {
		rank := "-"
		if r.FirstRank > 0 {
			rank = fmt.Sprintf("%d", r.FirstRank)
		}
		judge := "-"
		if r.Judged {
			judge = fmt.Sprintf("%.2f", r.Judge)
		}
		note := r.Note
		if r.Refused && note == "" {
			note = "refused"
		}
		fmt.Printf("%-24s %-11s %5s %6.2f %-8s %-22s %s\n",
			r.ID, r.Stage, rank, r.TopScore, judge,
			truncate(strings.Join(r.Cited, ","), 22), note)
	}
}

// itemScore ranks items by how badly they did, so the worst sort to the top.
func itemScore(r ItemResult) float64 {
	s := 0.0
	if r.Judged {
		s += r.Judge
	} else {
		s += 0.5 // unjudged items are neither evidence of success nor failure
	}
	if r.FirstRank == 1 {
		s += 1
	} else if r.FirstRank > 1 {
		s += 0.5
	}
	if r.Refused {
		s -= 1
	}
	return s
}

// Calibrate derives the confidence-gate threshold from the golden set instead
// of guessing it.
//
// The gate compares the top reranked score against MIN_RERANK_SCORE, but that
// number is meaningless across rerankers: an LLM scorer emits 0/0.3/0.7/1.0
// buckets while a cross-encoder emits sigmoid scores that can sit around 0.01.
// A threshold carried over from one to the other either refuses everything or
// gates nothing. So measure: run every golden query, collect the top score for
// answerable items and for out-of-corpus items, and report a value that
// separates them.
func (a *App) Calibrate(ctx context.Context, items []GoldenItem) error {
	var answerable, unanswerable []float32

	for _, it := range items {
		qvecs, err := a.Embedder.Embed(ctx, []string{it.Query}, "query")
		if err != nil {
			return err
		}
		stage, err := a.RetrieveAndRank(ctx, it.Query, qvecs[0], it.ACL)
		if err != nil {
			return err
		}
		if len(stage.Ranked) == 0 {
			continue
		}
		if len(it.Relevant) == 0 {
			unanswerable = append(unanswerable, stage.Ranked[0].Score)
			continue
		}
		// Score the first *relevant* chunk, not the top-ranked one. The gate has
		// to admit the chunk that actually answers the question; if the reranker
		// put something else first, that item's top score says nothing about
		// where the threshold belongs — and using it silently suggests a
		// threshold that excludes the chunk we needed.
		rel := map[string]bool{}
		for _, r := range it.Relevant {
			rel[r] = true
		}
		found := false
		for _, s := range stage.Ranked {
			if rel[s.ChunkID] {
				answerable = append(answerable, s.Score)
				found = true
				break
			}
		}
		if !found {
			fmt.Printf("note: %-24s no relevant chunk retrieved at all — excluded from calibration\n", it.ID)
		}
	}
	if len(answerable) == 0 {
		return fmt.Errorf("calibrate: no answerable items in the golden set")
	}

	sort.Slice(answerable, func(i, j int) bool { return answerable[i] < answerable[j] })
	lowestGood := answerable[0]
	fmt.Printf("answerable items   : n=%d  min=%.4f  median=%.4f  max=%.4f\n",
		len(answerable), lowestGood, answerable[len(answerable)/2], answerable[len(answerable)-1])

	var highestBad float32
	if len(unanswerable) > 0 {
		for _, s := range unanswerable {
			highestBad = max(highestBad, s)
		}
		fmt.Printf("out-of-corpus items: n=%d  max=%.4f\n", len(unanswerable), highestBad)
	} else {
		fmt.Println("out-of-corpus items: none — add negative examples to bound the gate from below")
	}

	if highestBad >= lowestGood {
		fmt.Printf("\nNo threshold separates them (worst answerable %.4f <= best out-of-corpus %.4f).\n"+
			"The gate cannot do this job on its own; rely on citation validation and\n"+
			"entailment, and improve retrieval before tightening the gate.\n", lowestGood, highestBad)
		return nil
	}
	// Geometric midpoint: these scores span orders of magnitude, so the
	// arithmetic mean would sit almost on top of the larger value.
	suggested := float32(math.Sqrt(float64(lowestGood) * float64(max(highestBad, 1e-6))))
	fmt.Printf("\nMIN_RERANK_SCORE=%.4f\n", suggested)
	fmt.Printf("(separates the two cleanly; re-run after any reranker or corpus change)\n")
	return nil
}

// --- Metric primitives ----------------------------------------------------

func ids(ss []Source) []string {
	out := make([]string, len(ss))
	for i, s := range ss {
		out[i] = s.ChunkID
	}
	return out
}

func recallAt(ranked []string, rel map[string]bool, k int) float64 {
	if len(rel) == 0 {
		return 0
	}
	found := 0
	for i, id := range ranked {
		if i >= k {
			break
		}
		if rel[id] {
			found++
		}
	}
	return float64(found) / float64(len(rel))
}

// ndcgAt uses binary relevance: gain 1 for a relevant chunk, 0 otherwise.
func ndcgAt(ranked []string, rel map[string]bool, k int) float64 {
	var dcg float64
	for i, id := range ranked {
		if i >= k {
			break
		}
		if rel[id] {
			dcg += 1 / math.Log2(float64(i+2))
		}
	}
	var idcg float64
	for i := 0; i < min(k, len(rel)); i++ {
		idcg += 1 / math.Log2(float64(i+2))
	}
	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

func reciprocalRank(ranked []string, rel map[string]bool) float64 {
	for i, id := range ranked {
		if rel[id] {
			return 1 / float64(i+1)
		}
	}
	return 0
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

// pct returns the p-th percentile using nearest-rank, which is correct on the
// small sample sizes an eval set has (interpolation invents values here).
func pct(xs []int64, p float64) int64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]int64(nil), xs...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	idx := int(math.Ceil(p*float64(len(s)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(s) {
		idx = len(s) - 1
	}
	return s[idx]
}

// --- CI gating ------------------------------------------------------------

// CompareBaseline fails the build when a gated metric regresses past tolerance.
// Quality metrics may not drop by more than qualityTol (absolute); latency
// percentiles may not rise by more than latencyTol (relative).
//
// The two tolerances are separate on purpose. Wall-clock timings on shared or
// contended hardware — a laptop GPU, a noisy CI runner — vary by tens of
// percent between identical runs, so gating them as tightly as quality just
// produces a red build that says nothing. Keep quality tight and latency loose
// enough to catch an order-of-magnitude change rather than thermal noise.
func CompareBaseline(baseline, current Metrics, qualityTol, latencyTol float64) []string {
	var fails []string
	b, c := toMap(baseline), toMap(current)
	for _, k := range gatedQuality {
		if b[k] == 0 {
			continue // no baseline recorded for this metric yet
		}
		if drop := b[k] - c[k]; drop > qualityTol {
			fails = append(fails, fmt.Sprintf("%s regressed %.4f -> %.4f (drop %.4f > tolerance %.4f)",
				k, b[k], c[k], drop, qualityTol))
		}
	}
	for _, k := range gatedLatency {
		if b[k] == 0 {
			continue
		}
		if rise := (c[k] - b[k]) / b[k]; rise > latencyTol {
			fails = append(fails, fmt.Sprintf("%s regressed %.0fms -> %.0fms (+%.1f%% > tolerance %.1f%%)",
				k, b[k], c[k], rise*100, latencyTol*100))
		}
	}
	return fails
}

func toMap(m Metrics) map[string]float64 {
	b, _ := json.Marshal(m)
	var raw map[string]float64
	_ = json.Unmarshal(b, &raw)
	return raw
}

// CheckGoldenIDs fails fast when a golden item references a chunk that is not
// in the index.
//
// A mistyped chunk id does not error anywhere — it simply never matches, so the
// item scores 0 recall for the rest of the file's life and every number built
// on it is quietly wrong. Chunk ids also change whenever chunking changes, so
// this is not a one-off typo check: it is what catches a golden set that has
// silently gone stale against its own corpus.
func (a *App) CheckGoldenIDs(ctx context.Context, items []GoldenItem) error {
	seen := map[string]bool{}
	var missing []string
	for _, it := range items {
		for _, id := range it.Relevant {
			if seen[id] {
				continue
			}
			seen[id] = true
			n, err := a.Qdrant.CountFiltered(ctx, a.Cfg.Collection,
				map[string]any{"must": []map[string]any{
					{"key": "chunk_id", "match": map[string]any{"value": id}},
				}})
			if err != nil {
				return fmt.Errorf("checking golden chunk ids: %w", err)
			}
			if n == 0 {
				missing = append(missing, fmt.Sprintf("%s (item %s)", id, it.ID))
			}
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf("%d golden chunk id(s) are not in collection %q:\n  %s\n"+
			"Re-run `ask-my-docs chunks -dir corpus` and update the golden set — "+
			"these items would score 0 recall forever without ever erroring",
			len(missing), a.Cfg.Collection, strings.Join(missing, "\n  "))
	}
	return nil
}
