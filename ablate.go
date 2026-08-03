package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
)

// The ablation ladder answers "is this better than traditional RAG?" with a
// measurement instead of a feature list.
//
// Rung 0 is the textbook baseline every RAG tutorial ships: embed, take the top
// k by cosine, put them in the prompt, generate. Each rung after it switches ONE
// stage back on, so the difference between two adjacent rows is that stage's
// contribution on this corpus — not an argument for it, a number.
//
// Contextualization is deliberately absent. It is baked into the vectors at
// ingest time, so its rungs cannot be a flag on the query; they need their own
// index. reportContextRungs prints how.
var ablationLadder = []struct {
	Name   string
	Ablate []string
}{
	{"dense top-k (baseline)", []string{"sparse", "rerank", "gate", "citations", "verify"}},
	{"+ hybrid retrieval", []string{"rerank", "gate", "citations", "verify"}},
	{"+ cross-encoder rerank", []string{"gate", "citations", "verify"}},
	{"+ confidence gate", []string{"citations", "verify"}},
	{"+ citation validation", []string{"verify"}},
	{"+ entailment verify (full)", nil},
}

type rungResult struct {
	Name    string   `json:"name"`
	Ablated []string `json:"ablated"`
	Metrics Metrics  `json:"metrics"`
}

func cmdAblate(ctx context.Context, a *App, args []string) error {
	fs := flag.NewFlagSet("ablate", flag.ExitOnError)
	golden := fs.String("golden", "eval/golden.jsonl", "path to the golden eval set")
	out := fs.String("out", "eval/ablation.json", "write every rung's metrics here")
	judge := fs.Bool("judge", true, "score answers with the LLM judge; the expensive half of the run")
	from := fs.Int("from", 0, "first rung to run, for resuming an interrupted ladder")
	_ = fs.Parse(args)

	if len(a.Cfg.Ablate) > 0 {
		// The ladder sets ABLATE itself for every rung. Honouring an inherited
		// one too would silently shift every row.
		return fmt.Errorf("ABLATE is set in the environment (%v); unset it — ablate drives the ladder itself", a.Cfg.Ablate)
	}
	items, err := LoadGolden(*golden)
	if err != nil {
		return err
	}
	if len(items) == 0 {
		return fmt.Errorf("%s contains no eval items", *golden)
	}
	if err := a.CheckGoldenIDs(ctx, items); err != nil {
		return err
	}

	rungs := ablationLadder[min(*from, len(ablationLadder)):]
	// Printed before anything is spent, because the generator and the judge are
	// metered and this runs the whole golden set once per rung.
	fmt.Printf("%d rungs x %d items = %d pipeline runs", len(rungs), len(items), len(rungs)*len(items))
	if *judge {
		fmt.Printf(", plus up to %d judge calls", len(rungs)*len(items))
	}
	fmt.Printf("\nnoise floor: one item flipping moves a mean by %.3f — treat anything smaller as zero\n\n",
		1.0/float64(len(items)))

	results := make([]rungResult, 0, len(rungs))
	for _, rung := range rungs {
		fmt.Printf("--- %s ---\n", rung.Name)
		a.Cfg.Ablate = map[string]bool{}
		for _, s := range rung.Ablate {
			a.Cfg.Ablate[s] = true
		}
		m, _, err := a.RunEval(ctx, items, *judge)
		if err != nil {
			return fmt.Errorf("%s: %w", rung.Name, err)
		}
		results = append(results, rungResult{Name: rung.Name, Ablated: rung.Ablate, Metrics: m})
	}
	a.Cfg.Ablate = nil

	printLadder(results, len(items))
	reportContextRungs(a.Cfg.Collection)

	blob, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(*out, blob, 0o644); err != nil {
		return err
	}
	fmt.Printf("\nwrote %s\n", *out)
	return nil
}

// printLadder renders the table the claim gets made from. Deltas are against
// the rung below, so each row reads as "what this stage bought".
func printLadder(rs []rungResult, n int) {
	noise := 1.0 / float64(n)
	fmt.Printf("\n%-28s %7s %7s %7s %7s %7s %9s\n",
		"stage", "ndcg10", "mrr", "cit-P", "cit-R", "correct", "p95_ms")
	fmt.Println(strings.Repeat("-", 82))
	for i, r := range rs {
		m := r.Metrics
		fmt.Printf("%-28s %7.3f %7.3f %7.3f %7.3f %7.3f %9d\n",
			r.Name, m.NDCGAt10, m.MRR, m.CitationPrecision, m.CitationRecall,
			m.AnswerCorrectness, m.LatencyP95)
		if i == 0 {
			continue
		}
		// A delta inside the noise floor is not a result, and printing it as one
		// is how a benchmark table starts lying. Say "~0" and mean it.
		d := m.AnswerCorrectness - rs[i-1].Metrics.AnswerCorrectness
		dn := m.NDCGAt10 - rs[i-1].Metrics.NDCGAt10
		fmt.Printf("%-28s %+7s %7s %7s %7s %+7s\n", "  vs previous",
			fmtDelta(dn, noise), "", "", "", fmtDelta(d, noise))
	}
	fmt.Printf("\nOne item is worth %.3f. Differences below that are noise, not findings.\n", noise)
}

func fmtDelta(d, noise float64) string {
	if math.Abs(d) < noise {
		return "~0"
	}
	return fmt.Sprintf("%+.3f", d)
}

// reportContextRungs prints the part of the ladder that cannot be a flag.
//
// Contextualization changes the vectors, so measuring it means building a
// separate index per rung. That costs an ingest each, which is the one part of
// this pipeline that spends real tokens — so it is instructions, not something
// the command runs on its own.
func reportContextRungs(collection string) {
	fmt.Printf(`
Contextualization is an index-time stage and is not in the table above: its
rungs need their own collections. Each of these is one full ingest, so run them
deliberately.

  ABLATE=context     QDRANT_COLLECTION=%[1]s_raw  ask-my-docs ingest -force
  ABLATE=llmcontext  QDRANT_COLLECTION=%[1]s_free ask-my-docs ingest -force
                     QDRANT_COLLECTION=%[1]s      ask-my-docs ingest -force

Then run eval against each, pointing the golden set at the same chunk ids:

  QDRANT_COLLECTION=%[1]s_raw ask-my-docs eval -baseline ""

Only the middle one is free of API cost; the third is what is already indexed.
`, collection)
}
