package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
)

const model = anthropic.Model("claude-opus-5")

type Claude struct{ c anthropic.Client }

func NewClaude() *Claude {
	c := anthropic.NewClient()
	return &Claude{c: c}
}

// call issues a structured-output request and unmarshals the single text block
// into out. Structured outputs (rather than tool-calling) are what make the
// citation contract enforceable: the model cannot emit a shape we can't parse.
func (cl *Claude) call(ctx context.Context, system string, sysCache bool, user string, schema map[string]any, effort anthropic.OutputConfigEffort, maxTokens int64, out any) error {
	sysBlock := anthropic.TextBlockParam{Text: system}
	if sysCache {
		// The document body lives in the system prompt and is reused across every
		// chunk of that document, so cache it rather than re-billing per call.
		sysBlock.CacheControl = anthropic.NewCacheControlEphemeralParam()
	}
	resp, err := cl.c.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     model,
		MaxTokens: maxTokens,
		System:    []anthropic.TextBlockParam{sysBlock},
		OutputConfig: anthropic.OutputConfigParam{
			Effort: effort,
			Format: anthropic.JSONOutputFormatParam{Schema: schema},
		},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(user)),
		},
	})
	if err != nil {
		return err
	}
	if resp.StopReason == anthropic.StopReasonRefusal {
		return fmt.Errorf("claude refused: %s", resp.StopDetails.Explanation)
	}
	for _, block := range resp.Content {
		if t, ok := block.AsAny().(anthropic.TextBlock); ok {
			return json.Unmarshal([]byte(t.Text), out)
		}
	}
	return fmt.Errorf("claude: no text block in response (stop_reason=%s)", resp.StopReason)
}

// --- Contextual retrieval -------------------------------------------------

const contextSystem = `You situate excerpts within the document they came from, so that each
excerpt retrieves well on its own once separated from the document.

Here is the full document:
<document>
%s
</document>`

const contextUser = `Below are %d numbered chunks taken from that document, in order.

For each chunk, write one or two sentences of context: where it sits in the
document, what it is about, and any entity, date, or subject that the chunk
refers to only by pronoun or shorthand. Do not summarise the chunk's content
and do not add information that is not in the document.

Name those entities. Write the context so it stands alone without the
document beside it: never fall back on "the individual", "the company", "this
report" or any other phrase that points at something instead of naming it.
The context is what makes a chunk findable once it is on its own, and a
reference nobody can resolve carries no term to search for.

Return exactly %d contexts, in the same order as the chunks.

%s`

// Contextualize implements Anthropic's contextual retrieval: each chunk gets a
// short situating preamble prepended before embedding, which is the single
// largest retrieval-quality win available at ingest time.
//
// ponytail: one call per document (all chunks at once) rather than the
// canonical one call per chunk. Far fewer round trips; the ceiling is output
// length, so documents over maxChunksPerContextCall are split into batches.
func (cl *Claude) Contextualize(ctx context.Context, doc string, chunks []Chunk) ([]string, error) {
	const maxPerCall = 40
	out := make([]string, 0, len(chunks))
	for start := 0; start < len(chunks); start += maxPerCall {
		end := min(start+maxPerCall, len(chunks))
		batch := chunks[start:end]

		var sb strings.Builder
		for i, c := range batch {
			fmt.Fprintf(&sb, "<chunk index=\"%d\" section=\"%s\">\n%s\n</chunk>\n\n", i, c.Section, c.Text)
		}
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"contexts": map[string]any{
					"type":  "array",
					"items": map[string]any{"type": "string"},
				},
			},
			"required":             []string{"contexts"},
			"additionalProperties": false,
		}
		var res struct {
			Contexts []string `json:"contexts"`
		}
		err := cl.call(ctx,
			fmt.Sprintf(contextSystem, doc), true,
			fmt.Sprintf(contextUser, len(batch), len(batch), sb.String()),
			schema, anthropic.OutputConfigEffortLow, 8000, &res)
		if err != nil {
			return nil, err
		}
		if len(res.Contexts) != len(batch) {
			// A count mismatch means we cannot align contexts to chunks. Indexing
			// the raw chunks is worse retrieval, not broken retrieval — so degrade
			// rather than fail the whole ingest.
			return nil, fmt.Errorf("contextualize: expected %d contexts, got %d", len(batch), len(res.Contexts))
		}
		out = append(out, res.Contexts...)
	}
	return out, nil
}

// --- Cited generation -----------------------------------------------------

// answerSystem deliberately does not ask the model to judge whether it *should*
// answer. Whether the corpus covers the question is already decided upstream by
// the confidence gate, and whether the answer is grounded is decided downstream
// by the entailment verifier. Asking a small model to make that call as well
// just produces refusals on questions its own sources answer.
const answerSystem = `You answer questions from the numbered sources provided by the user.

Rules:
- Every factual claim you make must cite at least one source id that directly
  supports it. A source that is merely on-topic does not support a claim.
- Use only the sources given. Never introduce a fact the sources do not
  contain, and never cite a source id that does not appear in the list.
- Answering in the user's own terms is not guessing. If a source describes a
  mechanism and the user asks what that mechanism achieves, say so and cite it.
  The rule you are enforcing is "no outside facts", not "no reasoning".
- The sources were already checked for relevance before reaching you, so assume
  the answer is in there and find it. Return an empty claims list only if the
  sources truly say nothing about the question.
- Write the claims as the answer itself, in order: read end to end they should
  read as continuous prose, not as disconnected bullet points.
- Each claim must still be a complete, self-contained sentence that names its
  own subject. Never split one sentence across several claims, and never open a
  claim with a word that refers back to a previous one. Each claim is checked
  against its sources on its own, so a fragment cannot be checked at all.`

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

var answerSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"claims": map[string]any{
			"type": "array",
			"items": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"text": map[string]any{"type": "string"},
					"citations": map[string]any{
						"type":  "array",
						"items": map[string]any{"type": "string"},
					},
				},
				"required":             []string{"text", "citations"},
				"additionalProperties": false,
			},
		},
	},
	"required":             []string{"claims"},
	"additionalProperties": false,
}

func (cl *Claude) Answer(ctx context.Context, question string, sources []Source) (Answer, error) {
	var sb strings.Builder
	sb.WriteString("Sources:\n\n")
	for _, s := range sources {
		fmt.Fprintf(&sb, "<source id=\"%s\" doc=\"%s\" section=\"%s\">\n%s\n</source>\n\n",
			s.ChunkID, s.DocID, s.Section, s.Text)
	}
	fmt.Fprintf(&sb, "Question: %s", question)

	var a Answer
	err := cl.call(ctx, answerSystem, false, sb.String(), answerSchema,
		anthropic.OutputConfigEffortLow, 4000, &a)
	return a, err
}

// --- Entailment verification ---------------------------------------------

const verifySystem = `You check whether cited evidence entails a claim.

For each numbered item you are given a claim and the source text cited for it.
Answer "entailed" only if the source text states or directly implies the claim.
A source that is merely related to the claim's topic, or that supports a weaker
or different statement, is not entailed.`

// Verify returns one boolean per claim: does the cited text actually entail it?
// This is the post-hoc guard that turns "the model emitted a citation" into
// "the citation holds".
func (cl *Claude) Verify(ctx context.Context, pairs []VerifyPair) ([]bool, error) {
	if len(pairs) == 0 {
		return nil, nil
	}
	var sb strings.Builder
	for i, p := range pairs {
		fmt.Fprintf(&sb, "<item index=\"%d\">\n<claim>%s</claim>\n<evidence>%s</evidence>\n</item>\n\n",
			i, p.Claim, p.Evidence)
	}
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"entailed": map[string]any{
				"type":  "array",
				"items": map[string]any{"type": "boolean"},
			},
		},
		"required":             []string{"entailed"},
		"additionalProperties": false,
	}
	var res struct {
		Entailed []bool `json:"entailed"`
	}
	if err := cl.call(ctx, verifySystem, false, sb.String(), schema,
		anthropic.OutputConfigEffortLow, 2000, &res); err != nil {
		return nil, err
	}
	if len(res.Entailed) != len(pairs) {
		return nil, fmt.Errorf("verify: expected %d verdicts, got %d", len(pairs), len(res.Entailed))
	}
	return res.Entailed, nil
}

type VerifyPair struct {
	Claim    string
	Evidence string
}

// --- Eval judge -----------------------------------------------------------

const judgeSystem = `You grade a candidate answer against a reference answer.

Score 1.0 if the candidate conveys the same substantive facts as the reference,
0.5 if it is partially correct or omits something material, and 0.0 if it is
wrong, unsupported, or answers a different question. Wording and length do not
matter; only the facts asserted do.`

func (cl *Claude) Judge(ctx context.Context, question, reference, candidate string) (float64, error) {
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
	if err := cl.call(ctx, judgeSystem, false, user, schema,
		anthropic.OutputConfigEffortLow, 1000, &res); err != nil {
		return 0, err
	}
	return res.Score, nil
}
