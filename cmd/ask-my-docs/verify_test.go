package main

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"askmydocs/internal/rag"
)

// Entailment verification is the last guard before an answer reaches a user,
// and on the hosted default it was also 73% of query latency (18.3s of 25.0s).
// Moving it to a small local model is only worth doing if the verdicts survive,
// so this measures both at once: accuracy on labelled pairs, and wall clock.
//
//	ollama serve
//	VERIFY_TEST=ollama:qwen2.5:1.5b go test -run Verifier -v
//	VERIFY_TEST=nvidia:minimaxai/minimax-m3 go test -run Verifier -v
//
// Skipped when VERIFY_TEST is unset, so `go test ./...` stays offline.

// verifyCase is a claim/evidence pair with a known verdict.
//
// The pairs are synthetic and generic on purpose — a verifier tuned against
// this project's own corpus would measure memorisation, not entailment.
type verifyCase struct {
	name     string
	claim    string
	evidence string
	want     bool
}

// Balanced 6/6, because the failure mode worth catching is a model that has
// learned to say yes. An always-true verifier scores exactly 0.50 here, which
// is the number to compare against — not zero.
var verifyCases = []verifyCase{
	// --- should verify ---
	{
		name:     "restated",
		claim:    "The maintenance window runs for four hours.",
		evidence: "[a#1] Scheduled maintenance begins at 01:00 and ends at 05:00 UTC.",
		want:     true,
	},
	{
		name:  "union of two sources",
		claim: "The kit ships with a power supply and a carrying case.",
		evidence: "[a#1] Each kit includes a 12V power supply.\n\n" +
			"[a#2] A padded carrying case is provided with every kit.",
		want: true,
	},
	{
		name:     "paraphrased, low lexical overlap",
		claim:    "Applicants must be at least eighteen.",
		evidence: "[a#1] Nobody under the age of 18 is eligible to apply.",
		want:     true,
	},
	{
		name:     "arithmetic within the evidence",
		claim:    "The two components together weigh 7 kg.",
		evidence: "[a#1] The base unit weighs 5 kg.\n\n[a#2] The mounting bracket weighs 2 kg.",
		want:     true,
	},
	{
		name:     "hedged claim, definite evidence",
		claim:    "The service may return a cached response.",
		evidence: "[a#1] Responses are served from cache when the entry is under 60 seconds old.",
		want:     true,
	},
	{
		name:     "negation preserved",
		claim:    "Refunds are not offered after 30 days.",
		evidence: "[a#1] The refund window closes 30 days after purchase; no refunds are issued past that point.",
		want:     true,
	},

	// --- should refuse ---
	{
		name:     "plausible but absent number",
		claim:    "The maintenance window runs for six hours.",
		evidence: "[a#1] Scheduled maintenance begins at 01:00 and ends at 05:00 UTC.",
		want:     false,
	},
	{
		name:     "some generalised to all",
		claim:    "Every region supports same-day delivery.",
		evidence: "[a#1] Same-day delivery is available in selected metropolitan regions.",
		want:     false,
	},
	{
		name:     "contradicted",
		claim:    "The API returns results in XML.",
		evidence: "[a#1] All API responses are encoded as JSON.",
		want:     false,
	},
	{
		name:     "adjacent subject",
		claim:    "The premium tier includes phone support.",
		evidence: "[a#1] The premium tier includes priority email support and a dedicated account manager.",
		want:     false,
	},
	{
		name:     "unstated causation",
		claim:    "The outage was caused by the database upgrade.",
		evidence: "[a#1] A database upgrade completed at 02:00. The service was unavailable from 02:15 to 03:40.",
		want:     false,
	},
	{
		name:     "plausible world knowledge, not in evidence",
		claim:    "The device is waterproof.",
		evidence: "[a#1] The device has an aluminium housing and operates between -10C and 45C.",
		want:     false,
	},
}

// TestVerifierAccuracy scores one verifier on the labelled pairs.
//
// It asserts only that the verifier beats always-true. Which model to run is a
// judgement call about the accuracy/latency trade this prints — the test's job
// is to make that trade visible and to fail if a verifier has stopped verifying.
func TestVerifierAccuracy(t *testing.T) {
	spec := os.Getenv("VERIFY_TEST")
	if spec == "" {
		t.Skip("VERIFY_TEST not set (want provider:model); skipping live verifier test")
	}
	provider, model, _ := strings.Cut(spec, ":")

	loadDotEnv(env("ENV_FILE", ".env"))
	cfg := rag.LoadConfig()
	v, name, err := buildLLM(cfg, provider, model)
	if err != nil {
		t.Fatalf("buildLLM(%q, %q): %v", provider, model, err)
	}

	// One call per pair, matching how verifyClaims batches in production: the
	// evidence is the union of a claim's citations, one pair per claim.
	pairs := make([]rag.VerifyPair, len(verifyCases))
	for i, c := range verifyCases {
		pairs[i] = rag.VerifyPair{Claim: c.claim, Evidence: c.evidence}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	start := time.Now()
	got, err := v.Verify(ctx, pairs)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if len(got) != len(pairs) {
		t.Fatalf("expected %d verdicts, got %d", len(pairs), len(got))
	}

	var correct, falseNeg, falsePos int
	for i, c := range verifyCases {
		if got[i] == c.want {
			correct++
			continue
		}
		// A false positive waves through an unsupported claim, which is the
		// failure this stage exists to prevent. A false negative only refuses
		// a good answer. Both are reported; they are not equally bad.
		if c.want {
			falseNeg++
		} else {
			falsePos++
		}
		t.Logf("  MISS  %-38s want=%v got=%v", c.name, c.want, got[i])
	}

	acc := float64(correct) / float64(len(verifyCases))
	t.Logf("%s/%s: accuracy %.2f (%d/%d)  false-accept %d  false-refuse %d  total %s  per-claim %s",
		name.Provider, name.Model, acc, correct, len(verifyCases),
		falsePos, falseNeg, elapsed.Round(time.Millisecond),
		(elapsed / time.Duration(len(pairs))).Round(time.Millisecond))

	// ponytail: floor, not a target. Always-true scores 0.50 on a balanced set,
	// so this only catches a verifier that has degenerated into a rubber stamp.
	// Tighten it once a model is actually chosen.
	if acc <= 0.5 {
		t.Errorf("accuracy %.2f does not beat always-true (0.50) — this verifier is not verifying", acc)
	}
}
