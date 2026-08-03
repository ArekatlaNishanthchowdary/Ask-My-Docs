package main

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// The non-trivial logic in this system that does not need a network: chunking,
// sparse encoding, the ranking metrics, and the CI gate. Everything else is
// I/O against Qdrant, Voyage, or Claude and is covered by `eval`.

func TestChunkDocRespectsHeadings(t *testing.T) {
	doc := `# Billing

Intro paragraph.

## Refunds

Refunds are issued within 30 days.

## Invoices

Invoices are sent monthly.`

	chunks := ChunkDoc(doc, 1600, 200)
	if len(chunks) != 3 {
		t.Fatalf("expected 3 section chunks, got %d: %+v", len(chunks), chunks)
	}
	want := []string{"Billing", "Billing > Refunds", "Billing > Invoices"}
	for i, w := range want {
		if chunks[i].Section != w {
			t.Errorf("chunk %d section = %q, want %q", i, chunks[i].Section, w)
		}
		if chunks[i].Ordinal != i {
			t.Errorf("chunk %d ordinal = %d, want %d", i, chunks[i].Ordinal, i)
		}
	}
	// A chunk must not carry text from a sibling section.
	if strings.Contains(chunks[1].Text, "Invoices are sent") {
		t.Error("Refunds chunk leaked text from the Invoices section")
	}
}

func TestChunkDocSplitsOversizedSectionsWithOverlap(t *testing.T) {
	para := strings.Repeat("word ", 60) // ~300 chars
	doc := "# Big\n\n" + strings.Repeat(para+"\n\n", 10)

	chunks := ChunkDoc(doc, 600, 100)
	if len(chunks) < 2 {
		t.Fatalf("expected the oversized section to split, got %d chunk(s)", len(chunks))
	}
	for i, c := range chunks {
		if len(c.Text) > 600+100 {
			t.Errorf("chunk %d is %d chars, over the budget", i, len(c.Text))
		}
		if c.Section != "Big" {
			t.Errorf("chunk %d lost its section path: %q", i, c.Section)
		}
	}
}

func TestChunkDocHardSplitDoesNotCutMidWord(t *testing.T) {
	// A CSV or other table rendered to markdown is one giant blob of
	// "\n"-joined rows with no blank line anywhere, so packText's paragraph
	// splitter never fires and the whole thing falls into the hard-split
	// path. Each row here stands in for a table row.
	var rows []string
	for i := range 40 {
		rows = append(rows, "| this is table row number "+strings.Repeat("x", i%7)+" with some content |")
	}
	doc := "# Data\n\n" + strings.Join(rows, "\n")

	chunks := ChunkDoc(doc, 300, 50)
	if len(chunks) < 2 {
		t.Fatalf("expected the blob to hard-split, got %d chunk(s)", len(chunks))
	}
	// Every row ends in "|"; a chunk that got cut mid-row ends somewhere else.
	for i, c := range chunks {
		end := strings.TrimRight(c.Text, "\n")
		if !strings.HasSuffix(end, "|") {
			t.Errorf("chunk %d cut mid-row instead of on a row boundary: %q", i, end[max(0, len(end)-30):])
		}
	}
}

func TestSparseEncodeIsDeterministicAndCountsTerms(t *testing.T) {
	a := SparseEncode("Error CODE-500 error")
	b := SparseEncode("error code-500 ERROR")
	if len(a.Indices) != len(b.Indices) {
		t.Fatalf("case change altered the sparse vector: %d vs %d terms", len(a.Indices), len(b.Indices))
	}
	for i := range a.Indices {
		if a.Indices[i] != b.Indices[i] || a.Values[i] != b.Values[i] {
			t.Fatalf("sparse encoding is not case-insensitive at position %d", i)
		}
	}
	// "error" appears twice, "code" and "500" once each.
	total := float32(0)
	for _, v := range a.Values {
		total += v
	}
	if len(a.Indices) != 3 || total != 4 {
		t.Errorf("got %d distinct terms totalling %v, want 3 terms totalling 4", len(a.Indices), total)
	}
	// Indices must be ascending — Qdrant requires sorted sparse vectors.
	for i := 1; i < len(a.Indices); i++ {
		if a.Indices[i] <= a.Indices[i-1] {
			t.Fatal("sparse indices are not strictly ascending")
		}
	}
}

func TestRankingMetrics(t *testing.T) {
	rel := map[string]bool{"a": true, "b": true}

	if got := recallAt([]string{"a", "x", "b"}, rel, 10); got != 1.0 {
		t.Errorf("recall@10 = %v, want 1.0", got)
	}
	if got := recallAt([]string{"a", "x", "y"}, rel, 10); got != 0.5 {
		t.Errorf("recall@10 = %v, want 0.5", got)
	}
	if got := recallAt([]string{"x", "a"}, rel, 1); got != 0 {
		t.Errorf("recall@1 = %v, want 0 (cutoff must be honoured)", got)
	}

	if got := reciprocalRank([]string{"x", "a"}, rel); got != 0.5 {
		t.Errorf("MRR = %v, want 0.5", got)
	}
	if got := reciprocalRank([]string{"x", "y"}, rel); got != 0 {
		t.Errorf("MRR with no hit = %v, want 0", got)
	}

	// A perfect ranking scores 1.0; demoting a relevant item must score lower.
	perfect := ndcgAt([]string{"a", "b", "x"}, rel, 10)
	if math.Abs(perfect-1.0) > 1e-9 {
		t.Errorf("nDCG of a perfect ranking = %v, want 1.0", perfect)
	}
	worse := ndcgAt([]string{"a", "x", "b"}, rel, 10)
	if worse >= perfect {
		t.Errorf("nDCG did not penalise the worse ranking: %v >= %v", worse, perfect)
	}
}

func TestPercentileNearestRank(t *testing.T) {
	xs := []int64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	if got := pct(xs, 0.95); got != 100 {
		t.Errorf("p95 = %d, want 100", got)
	}
	if got := pct(xs, 0.50); got != 50 {
		t.Errorf("p50 = %d, want 50", got)
	}
	if got := pct(nil, 0.95); got != 0 {
		t.Errorf("p95 of empty = %d, want 0", got)
	}
}

func TestCompareBaselineGatesRegressions(t *testing.T) {
	base := Metrics{RecallAt10: 0.90, NDCGAt10: 0.85, MRR: 0.80,
		CitationPrecision: 0.95, AnswerCorrectness: 0.90, RetrieveP95: 50, RerankP95: 150}

	// Within tolerance: a 1% drop at a 2% tolerance must pass.
	ok := base
	ok.RecallAt10 = 0.89
	if fails := CompareBaseline(base, ok, 0.02, 0.50); len(fails) != 0 {
		t.Errorf("expected no failures within tolerance, got %v", fails)
	}

	// Past tolerance: a 5% quality drop must fail the build.
	bad := base
	bad.RecallAt10 = 0.85
	if fails := CompareBaseline(base, bad, 0.02, 0.50); len(fails) != 1 {
		t.Errorf("expected 1 quality failure, got %v", fails)
	}

	// Latency is gated relatively, in the opposite direction, and against its
	// own looser tolerance — wall-clock noise on shared hardware must not turn
	// the build red.
	slow := base
	slow.RetrieveP95 = 60 // +20%
	if fails := CompareBaseline(base, slow, 0.02, 0.50); len(fails) != 0 {
		t.Errorf("a 20%% latency rise must pass a 50%% tolerance, got %v", fails)
	}
	if fails := CompareBaseline(base, slow, 0.02, 0.10); len(fails) != 1 {
		t.Errorf("a 20%% latency rise must fail a 10%% tolerance, got %v", fails)
	}
	// An order-of-magnitude regression must fail even the loose tolerance.
	crawl := base
	crawl.RerankP95 = 1500 // 10x
	if fails := CompareBaseline(base, crawl, 0.02, 0.50); len(fails) != 1 {
		t.Errorf("expected the 10x rerank regression to fail, got %v", fails)
	}
	// Quality and latency tolerances must not be conflated: a quality drop
	// inside the loose latency tolerance still has to fail.
	qDrop := base
	qDrop.MRR = 0.70 // -0.10, well inside 0.50 but outside 0.02
	if fails := CompareBaseline(base, qDrop, 0.02, 0.50); len(fails) != 1 {
		t.Errorf("expected the quality drop to fail on the quality tolerance, got %v", fails)
	}
	fast := base
	fast.RetrieveP95 = 10
	if fails := CompareBaseline(base, fast, 0.02, 0.50); len(fails) != 0 {
		t.Errorf("an improvement must never fail the build, got %v", fails)
	}

	// An absent baseline metric must not fail the build.
	empty := Metrics{}
	if fails := CompareBaseline(empty, Metrics{}, 0.02, 0.50); len(fails) != 0 {
		t.Errorf("empty baseline should gate nothing, got %v", fails)
	}
}

func TestRenderAnswerEmitsCitations(t *testing.T) {
	got := renderAnswer([]Claim{
		{Text: "Refunds take 30 days.", Citations: []string{"policy.md#2"}},
		{Text: "Invoices are monthly.", Citations: []string{"policy.md#3", "policy.md#4"}},
	})
	want := "Refunds take 30 days. [policy.md#2] Invoices are monthly. [policy.md#3, policy.md#4]"
	if got != want {
		t.Errorf("renderAnswer =\n  %q\nwant\n  %q", got, want)
	}
}

func TestACLFilterShape(t *testing.T) {
	if aclFilter(nil) != nil {
		t.Error("an empty ACL list must produce no filter")
	}
	b, _ := json.Marshal(aclFilter([]string{"finance", "hr"}))
	want := `{"must":[{"key":"acl","match":{"any":["finance","hr"]}}]}`
	if string(b) != want {
		t.Errorf("acl filter = %s, want %s", b, want)
	}
}

func TestPointIDIsStableAndUUIDShaped(t *testing.T) {
	a, b := pointID("policy.md#3"), pointID("policy.md#3")
	if a != b {
		t.Fatal("point IDs must be deterministic or re-ingest duplicates chunks")
	}
	if a == pointID("policy.md#4") {
		t.Fatal("distinct chunks collided")
	}
	parts := strings.Split(a, "-")
	if len(parts) != 5 || len(parts[0]) != 8 || len(parts[4]) != 12 {
		t.Errorf("point ID %q is not UUID-shaped; Qdrant will reject it", a)
	}
}

// The bug this guards: a corpus holding both "Arekatla Nishanth Chowdary_Doc.docx"
// and "Arekatla_Nishanth_Chowdary_Resume_Viasat.docx" had the first cited with
// the second's underscores, and a grounded answer was refused over punctuation.
func TestResolveCitation(t *testing.T) {
	top := []string{
		"Arekatla Nishanth Chowdary_Doc.docx#1",
		"Arekatla_Nishanth_Chowdary_Resume_Viasat.docx#0",
		"notes.txt#0",
		"notes.txt#1",
	}
	valid := map[string]Source{}
	loose := map[string][]string{}
	for _, id := range top {
		valid[id] = Source{ChunkID: id}
		k := loosenChunkID(id)
		loose[k] = append(loose[k], id)
	}

	for _, tc := range []struct {
		name, cite, want string
		ok               bool
	}{
		{"exact", "notes.txt#0", "notes.txt#0", true},
		{"underscored spaces", "Arekatla_Nishanth_Chowdary_Doc.docx#1", "Arekatla Nishanth Chowdary_Doc.docx#1", true},
		{"case and spacing", "arekatla nishanth chowdary doc.docx#1", "Arekatla Nishanth Chowdary_Doc.docx#1", true},
		{"invented id still rejected", "made_up.docx#0", "", false},
		{"wrong chunk index is a different chunk", "notes.txt#7", "", false},
		{"chunk index never merges", "notes.txt#0", "notes.txt#0", true},
	} {
		got, ok := resolveCitation(tc.cite, valid, loose)
		if ok != tc.ok || got != tc.want {
			t.Errorf("%s: resolveCitation(%q) = (%q, %v), want (%q, %v)",
				tc.name, tc.cite, got, ok, tc.want, tc.ok)
		}
	}

	// Ambiguity must drop, not guess: two retrieved chunks that differ only by
	// the characters we collapse cannot be told apart, so neither is chosen.
	amb := map[string]Source{"a b.txt#0": {}, "a_b.txt#0": {}}
	ambLoose := map[string][]string{loosenChunkID("a b.txt#0"): {"a b.txt#0", "a_b.txt#0"}}
	if got, ok := resolveCitation("a-b.txt#0", amb, ambLoose); ok {
		t.Errorf("ambiguous citation resolved to %q, want dropped", got)
	}
}

func TestKeepEntailedDropsOnlyUnsupportedClaims(t *testing.T) {
	claims := []Claim{{Text: "a"}, {Text: "b"}, {Text: "c"}}

	kept, dropped := keepEntailed(claims, []bool{true, false, true})
	if len(kept) != 2 || kept[0].Text != "a" || kept[1].Text != "c" {
		t.Errorf("kept = %v, want the entailed claims in order", kept)
	}
	if len(dropped) != 1 || dropped[0] != 1 {
		t.Errorf("dropped = %v, want [1]", dropped)
	}

	// Nothing entailed leaves no answer, which the caller turns into a refusal.
	if kept, _ := keepEntailed(claims, []bool{false, false, false}); len(kept) != 0 {
		t.Errorf("kept %d claims with no verdicts true, want 0", len(kept))
	}

	// A short verdict list must not let unverified claims through.
	kept, dropped = keepEntailed(claims, []bool{true})
	if len(kept) != 1 || kept[0].Text != "a" {
		t.Errorf("kept = %v, want only the claim an actual verdict covered", kept)
	}
	if len(dropped) != 2 {
		t.Errorf("dropped = %v, want the two uncovered claims", dropped)
	}

	// The input must not be aliased: kept shares no array with claims.
	if kept, _ := keepEntailed(claims, []bool{false, true, true}); len(kept) > 0 && claims[0].Text != "a" {
		t.Errorf("keepEntailed overwrote its input: claims[0] = %q", claims[0].Text)
	}
}

func TestFallbackContextNamesTheDocument(t *testing.T) {
	// The point of the fallback is that a filename usually carries the entity a
	// chunk refers to only by pronoun, so the separators have to become words.
	got := fallbackContext("Arekatla_Nishanth_Chowdary_Resume-Viasat.docx", "Education")
	want := `From the document "Arekatla Nishanth Chowdary Resume Viasat", section "Education".`
	if got != want {
		t.Errorf("fallbackContext = %q, want %q", got, want)
	}

	// Chunks before the first heading have no section; the line must still be
	// a well-formed sentence rather than a dangling "section".
	if got := fallbackContext("notes/q3 plan.md", ""); got != `From the document "q3 plan".` {
		t.Errorf("sectionless fallbackContext = %q", got)
	}
}
