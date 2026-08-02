# Ask My Docs

**Hybrid-retrieval RAG that refuses rather than guesses.** A single Go binary —
no Python, no orchestration framework — that indexes your documents and answers
questions from them with every claim tied to the chunk it came from.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Go 1.25+](https://img.shields.io/badge/Go-1.25%2B-00ADD8.svg)

Runs **fully local** (Qdrant + Ollama, no API keys, nothing leaves the machine),
fully hosted, or any mix — the embedding, reranking and generation stages are
independent knobs.

```
┌─ ingest ────────────────────────────────────────────────────────────┐
│  document → structure-aware chunks → LLM contextual preamble        │
│           → dense + sparse embeddings → Qdrant                      │
└─────────────────────────────────────────────────────────────────────┘
┌─ query ─────────────────────────────────────────────────────────────┐
│  semantic cache → query embed → dense ANN + sparse BM25             │
│                 → RRF fusion (server-side) → cross-encoder rerank   │
│                 → confidence gate → cited generation                │
│                 → citation validation → entailment check            │
└─────────────────────────────────────────────────────────────────────┘
```

## Why this exists

Most RAG demos will confidently answer a question their corpus does not cover.
This one has three independent guards, and **the failure mode is a refusal,
never an unsupported claim**:

1. **Confidence gate** — if the top reranked chunk scores below the threshold,
   the question is refused *without calling the model at all*. The threshold is
   adaptive: `max(MIN_RERANK_SCORE, RERANK_MARGIN × median candidate score)`,
   because a fixed sigmoid number does not transfer between corpora — the same
   reranker scores a prose answer `0.99` and an equally correct comma-separated
   list `0.008`.
2. **Citation validation** — citations are resolved against the chunks that
   were actually retrieved. A claim left with no valid citation refuses the
   whole answer.
3. **Entailment verification** — every claim is checked against the union of
   the chunks it cites. A claim its own sources do not entail refuses the whole
   answer.

Citations are enforced through **structured outputs** (strict `json_schema`,
downgrading once to `json_object` if the endpoint rejects it), not through
prompt begging.

## Requirements

| | Needed for | Notes |
|---|---|---|
| **Go 1.25+** | building | The only build dependency. |
| **Docker** | Qdrant + reranker | `docker compose up -d` starts both. |
| **Ollama** | the fully-local path | Optional if you use hosted providers for everything except reranking. |
| **~2GB VRAM or CPU** | the reranker | `bge-reranker-v2-m3` behind TEI. A CPU image is the default. |

No API key is required for the fully-local path.

## Install

### From a release (no Go toolchain needed)

Download the archive for your platform from
[Releases](https://github.com/ArekatlaNishanthchowdary/Ask-My-Docs/releases),
unpack it, and you have the binary **plus** the files it needs to run:
`.env.example`, `docker-compose.yml`, `README.md`, `LICENSE`.

```bash
tar -xzf ask-my-docs-v0.10.1-linux-amd64.tar.gz
cd ask-my-docs-v0.10.1-linux-amd64
sha256sum -c --ignore-missing SHA256SUMS.txt   # optional, verify the download
./ask-my-docs version
```

Then continue from step 1 below — skip the `git clone`, you already have
everything.

### From source

```bash
git clone https://github.com/ArekatlaNishanthchowdary/Ask-My-Docs.git
cd Ask-My-Docs
go build -o ask-my-docs .
```

## Quick start

### 1. Configure

```bash
cp .env.example .env
```

**Nothing is guessed for you.** Model choices in particular have no defaults —
the binary fails at startup naming the variable rather than indexing your
corpus with a model you did not pick.

### 2. Choose your models

```bash
ollama list                     # what you already have
ollama pull bge-m3              # good embedding default (1024-dim)
ollama pull qwen2.5:7b          # or bigger — see the note below
```

Then set them in `.env`:

```ini
PROVIDER=ollama
OLLAMA_EMBED_MODEL=bge-m3
OLLAMA_CHAT_MODEL=qwen2.5:7b
```

> The embedding model must actually support embeddings — a chat model returns
> `501` from Ollama. **Generation model size matters more than anything else
> here:** a 7B refused questions its own sources answered until several fixes
> landed, and still scores ~0.16 below a hosted 70B on the same retrieval.

### 3. Start the services

```bash
docker compose up -d            # Qdrant :6333, reranker :8081
```

The reranker defaults to a **CPU image** so it runs anywhere. For GPU, set
`TEI_IMAGE` to your compute capability and uncomment the `deploy:` block in
`docker-compose.yml`:

| GPU | `TEI_IMAGE` |
|---|---|
| RTX 40xx (Ada, 8.9) | `89-1.7` |
| RTX 30xx (Ampere, 8.6) | `86-1.7` |
| RTX 20xx (Turing, 7.5) | `turing-1.7` |
| CPU / anything else | `cpu-1.7` |

A mismatched tag fails loudly at startup rather than silently falling back.

### 4. Add documents and index

Drop files into `corpus/`, then check how they will be split **before**
indexing — this needs no services and no keys:

```bash
./ask-my-docs chunks -dir corpus     # ids, sizes, section paths
./ask-my-docs ingest -dir corpus
./ask-my-docs serve                  # UI + API on http://localhost:8080
```

Ingest is idempotent — re-running overwrites in place.

### 5. Build an eval set (this is what makes it maintainable)

Write items into `eval/golden.jsonl`; the file documents its own format.

```bash
./ask-my-docs calibrate               # derive MIN_RERANK_SCORE from your corpus
./ask-my-docs eval -update-baseline   # accept current numbers as the baseline
./ask-my-docs eval -verbose           # thereafter: gate, with per-item breakdown
```

Steps 1–4 give you a working system. Step 5 is what keeps it working — without
it, every later change is judged by eye, and the confidence threshold is a
guess.

## Choosing backends

Each stage is a separate knob because each has different constraints:
embedding is high-volume, reranking is latency-critical, generation is
quality-critical.

| Stage | Chosen by | Options |
|---|---|---|
| Embeddings | `PROVIDER` | `ollama` (local) · `cloud` (Voyage) · `nvidia` (NIM) |
| Reranking | `RERANKER_URL` | cross-encoder via TEI · LLM-scored fallback |
| Generation, verification | `LLM_PROVIDER` | `ollama` · `cloud` (Claude) · `openai` (any OpenAI-compatible) · `nvidia` |
| Eval judge | `JUDGE_PROVIDER` | any of the above |

Every backend satisfies the same two interfaces (`Embedder`, `LLM`), so the
pipeline, eval harness and CI gate are identical whichever mix you run — which
is what makes an A/B between them meaningful rather than anecdotal.

### Reranking is a separate service on purpose

**Ollama has no rerank endpoint** — `/api/rerank` 404s. Without `RERANKER_URL`
the pipeline falls back to an LLM scoring passages, which works but measured
**~23x slower and materially worse** (4460ms → 194ms p95 after the swap). Use
the cross-encoder.

### Hosted generation, local retrieval

The useful combination when you want a big model but lack the VRAM to host one
— only generation leaves the machine:

```bash
LLM_PROVIDER=openai \
OPENAI_API_KEY=... \
OPENAI_MODEL=llama-3.3-70b-versatile \
./ask-my-docs query "What does ERR-4030 mean?"
```

`OPENAI_BASE_URL` defaults to Groq. Point it at OpenRouter, Together,
Fireworks, vLLM or LM Studio — the wire format is identical, so one
implementation covers all of them. **None of them serve embeddings or
reranking**, so they cannot replace the retrieval half.

Free tiers rate-limit aggressively and this pipeline is token-heavy. The client
honours `retry-after` and backs off automatically, capped at
`OPENAI_MAX_BACKOFF_SEC` — per-*minute* limits are worth waiting out, per-*day*
limits are not, so past the cap it fails fast with the provider's own message
instead of sleeping for hours. Drop `OPENAI_CONCURRENCY` to 1 if ingest still
trips limits.

### NVIDIA NIM

`nvidia` is the exception: it serves embeddings as well as chat, so it can
drive retrieval too. Reranking still comes from `RERANKER_URL`, and the app
**refuses to start without it** rather than silently dropping the stage.

```bash
PROVIDER=nvidia LLM_PROVIDER=nvidia \
NVIDIA_API_KEY=nvapi-... \
NVIDIA_MODEL=nvidia/nemotron-3-super-120b-a12b \
QDRANT_COLLECTION=docs_nv \
./ask-my-docs ingest -dir corpus
```

`NVIDIA_EMBED_MODEL` defaults to `nvidia/nemotron-3-embed-1b` (2048-dim, 32k
context, multilingual). Two things to know:

- **Switching embedding model needs a new collection.** Widths differ (bge-m3
  is 1024, nemotron-3-embed-1b is 2048) and Qdrant rejects mismatched points
  rather than migrating them.
- **Queries and passages are encoded asymmetrically** (`input_type`). The same
  sentence embedded both ways sits **0.76 cosine apart**, so getting this wrong
  degrades retrieval silently rather than failing.

Two Nemotron-specific behaviours, both worked around in code:

- **It takes its reasoning switch in the system prompt**, not a request field.
  `NVIDIA_SYSTEM_PREFIX` (default `detailed thinking off`) supplies it. Left in
  default thinking mode, one generation spent 8192 output tokens reasoning and
  never reached the JSON (197s, truncated); with the directive the same call
  took **1.8s and 172 tokens**.
- **An unconstrained array schema gives it an exit.** Asked for one boolean per
  claim with `{"type":"array"}`, it reliably returned `{"entailed":[]}` — valid
  against the schema, and no work done. The verify and contextualize schemas
  pin `minItems`/`maxItems` to the expected length, which makes the shortcut
  unrepresentable.

Nemotron's generation latency is unpredictable on long documents; expect
occasional `max_tokens` truncation during ingest, where it degrades to indexing
raw chunks. Point `LLM_PROVIDER` at a cheap non-reasoning model for ingest if
you hit that.

### The eval judge

```bash
JUDGE_PROVIDER=openai OPENAI_JUDGE_MODEL=openai/gpt-oss-120b \
OPENAI_API_KEY=... ./ask-my-docs eval -verbose
```

⚠️ **Comparing two generators is only meaningful under one shared judge.** Left
unset, each model grades its own answers — which measured **0.15–0.19 too
generous**. Treat any correctness figure produced by the model that wrote the
answers as an upper bound, not a measurement.

## Commands

| Command | What it does |
|---|---|
| `serve` | API **and** UI on `:8080`. |
| `ingest -dir DIR [-acl a,b]` | Chunk, contextualize, embed and index. Idempotent. |
| `query [-acl a,b] [-json] Q` | One question, printed with sources and per-stage timings. |
| `eval [-verbose] [-update-baseline]` | Run the golden set, print metrics, fail on regression. |
| `calibrate` | Derive `MIN_RERANK_SCORE` from the golden set instead of guessing. |
| `chunks -dir DIR [-text]` | Print chunk boundaries and ids. Offline — no keys, no services. |
| `version` | Print the build version (`dev` for a local `go build`). |

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/query` | `{"question": "...", "acl": [], "no_cache": false}` → answer, claims, sources, timings, warnings |
| `GET` | `/documents` | Indexed documents with per-document chunk counts |
| `POST` | `/documents` | Multipart upload — writes into `corpus/` and indexes in the same request |
| `GET` | `/providers` | Current stage assignments, available backends with models, and why any are unavailable |
| `POST` | `/providers` | `{"stage": "llm", "provider": "nvidia", "model": "..."}` — switch at runtime |
| `GET` | `/healthz` | Live config: provider, model, reranker, gate threshold, chunk count |
| `GET` | `/` | The web UI |

A provider switch is validated against the provider's own model listing before
it is applied — a *listing lookup*, not a probe call, because a probe would burn
tokens from the budget people are escaping. It takes effect immediately for
in-flight requests without a restart.

## Web UI

`serve` hosts a UI at `http://localhost:8080/`. It is a single `ui.html`
compiled into the binary with `go:embed` — no npm, no build step, no separate
process to deploy.

It surfaces the things this pipeline works hard to produce and a plain chat box
would throw away:

- **Clickable citations** — clicking a claim's source id scrolls to that chunk
  and highlights it. The grounding is the product, so it is the most clickable
  thing on the page.
- **Per-stage timings** as a proportional bar — the same numbers that made
  every optimisation in this project legible.
- **Sources actually passed to the model**, with rerank scores. `TOP_K` is a
  ceiling and sub-threshold chunks are dropped, so this is usually fewer than 10.
- **Pipeline notes** — dropped citations, generator retries, unentailed claims.
  A refusal comes with the reason it refused.
- **Runtime provider/model switching** for the generate and verify stages,
  grouped by provider, with unavailable providers shown *alongside the reason*
  rather than hidden.

**Embeddings are deliberately not switchable at runtime.** The index holds
vectors from one specific model; changing it would not raise an error, it would
silently retrieve nonsense. The UI shows the fixed embedding model and says
that changing it requires re-indexing.

Refusals render as calm and italic rather than as errors, because they are a
designed outcome of the guards, not a fault.

## Document formats

Drag files onto the page, use *+ Add documents*, or drop them in `corpus/`.

| Type | Handling |
|---|---|
| `.md` `.markdown` `.txt` | indexed as-is |
| `.docx` | heading styles become markdown headings (`Title` parents `Heading 1`); tables keep their columns |
| `.pptx` | one section per slide, in numeric order |
| `.xlsx` | one section per sheet, rows as pipe-separated cells, shared strings resolved |
| `.doc` `.ppt` `.xls` | rejected with a message — the pre-2007 binary formats are not ZIP archives |

Office files are ZIP + XML, so this is `archive/zip` and `encoding/xml` from the
standard library — **no document-parsing dependency**. Extraction targets
markdown rather than flat text on purpose: the chunker splits on headings, so a
Word heading or slide boundary becoming a real `##` is what keeps chunks aligned
to the document's own structure.

Uploaded filenames are untrusted input, so they are **sanitised rather than
cleaned**: directory components stripped, extension must be one that is
actually indexed, content must be valid UTF-8, and the resolved path is
re-checked against the corpus root. `../../../escaped.md` lands as
`corpus/escaped.md`, and the response reports the name it was saved under.

Ingesting **clears the semantic cache**. New documents can change the correct
answer to a question someone already asked, and a cache hit skips retrieval
entirely — so a stale hit would keep new documents invisible to exactly the
questions most likely to be repeated.

## Configuration

Full annotated reference in [`.env.example`](.env.example). The binary reads
`.env` at startup; **real environment variables win over it**, so
`FOO=bar ./ask-my-docs …` still overrides for one-off experiments.

The knobs worth knowing:

| Variable | Default | Notes |
|---|---|---|
| `PROVIDER` | `ollama` | Embeddings, and generation unless overridden |
| `LLM_PROVIDER` | follows `PROVIDER` | Generation, verification, contextualization |
| `VERIFY_PROVIDER` / `JUDGE_PROVIDER` | follow above | Independent per-stage overrides |
| `RERANKER_URL` | `http://localhost:8081` | Unset falls back to LLM scoring (~23x slower) |
| `MIN_RERANK_SCORE` | per-backend | Derive with `calibrate`, don't guess |
| `RERANK_MARGIN` | `3.0` | Relative gate; `0` disables |
| `CANDIDATE_K` / `TOP_K` | `50` / `10` | Fused candidates → ceiling on chunks sent to the model |
| `MAX_CHUNK_CHARS` / `CHUNK_OVERLAP` | `1600` / `200` | Changing these requires re-index + re-baseline |
| `QDRANT_COLLECTION` | `docs` | Use a **new** one when changing embedding model |
| `REQUEST_TIMEOUT_SEC` | `240` | Per-query budget for `/query` |
| `CORPUS_DIR` / `MAX_UPLOAD_MB` | `corpus` / `32` | Upload target and size cap |
| `DEBUG_LLM` | unset | Logs every upstream request/response — how most bugs here were found |

## Evaluation and CI gating

`eval/golden.jsonl` holds `query → relevant_chunk_ids` (+ gold answers), each
tagged with the stage it exercises so a regression can be *attributed* rather
than just observed: exact-identifier lookups (the sparse leg), zero-overlap
paraphrases (the dense leg), a reranking discriminator, and a
negative-rejection case.

Metrics: Recall@10, nDCG@10, MRR, reranker nDCG lift over retrieval alone,
citation precision/recall, LLM-judged answer correctness, and p50/p95/p99
latency per stage.

`.github/workflows/eval.yml` runs this against a real Qdrant on every PR and
uploads `metrics.json` so runs are diffable. The gate uses **separate quality
and latency tolerances**: the five quality metrics may not drop by more than an
absolute tolerance, and `retrieve_p95_ms` / `rerank_p95_ms` may not rise by
more than a relative one. (A single shared tolerance flapped — it passed one
run and failed the next.)

**Chunk ids are the contract.** Change chunking and the ids in the golden set
change with it — regenerate with `chunks -dir` and re-baseline.

Baselines are config-specific and **not comparable across backends**. There is
none in this checkout; generate yours with `eval -update-baseline`.

### Reference measurements

Measured during development on a 15-item eval set over a small sample corpus
**not included here**. Kept because it is the evidence behind several design
decisions. It says nothing about how your corpus will score.

Setup: `bge-m3` + `qwen2.5:7b`, RTX 4060 (8GB).

| Metric | LLM reranker | Cross-encoder |
|---|---|---|
| `retrieve_p95_ms` | 44ms | **46ms** |
| `rerank_p95_ms` | 4460ms | **194ms** |
| end-to-end p95 | 8352ms | **4822ms** |
| citation precision | 1.00 | **1.00** |
| citation recall | 0.58 | **0.92** |
| answer correctness | 0.54 | **0.85** (self-graded) |

Three fixes produced that, in order of impact:

1. **A real cross-encoder.** The LLM scorer was both slow and miscalibrated —
   it scored a correct rank-1 chunk `0.00`, tripping the gate and refusing an
   answerable question.
2. **Filtering sources by the gate threshold, not just the top one.** `TOP_K`
   is a ceiling, not a quota. Padding the prompt to 10 chunks when one clears
   the bar buried the answer in noise, and the 7B responded by returning
   nothing at all — even with the right chunk at rank 1 scoring 0.996. Largest
   correctness win, and it cost *negative* lines of prompt.
3. **Taking the refusal decision away from the generator.** Whether the corpus
   covers a question is the gate's job; whether an answer is grounded is the
   verifier's. Asking a 7B to also judge sufficiency just produced refusals on
   questions its own sources answered.

### Local 7B vs hosted 70B

Same retrieval, same cross-encoder, same threshold, same independent judge:

| | local `qwen2.5:7b` | Groq `llama-3.3-70b` |
|---|---|---|
| answer correctness | 0.65 | **0.81 – 0.85** |
| citation recall | 0.833 | **0.917** |
| end-to-end p95 | 4860ms | **1189ms** |

Both better *and* ~3x faster, because only generation moves off-device.

**Read correctness as ±0.04, not to three decimals.** Only 13 items are judged
and each scores 0, 0.5 or 1.0, so one item flipping moves the mean by 0.038.
The 0.65 → 0.81 gap is several times that noise and is real; a 0.02 difference
between two runs is not.

**Recall@10 of 1.00 means nothing on a tiny corpus.** If the corpus has ≤
`TOP_K` chunks, "the top 10 contains the answer" is arithmetic, not quality.
The harness warns when this holds.

## Design notes

**Server-side RRF.** Both retrieval legs are Qdrant `prefetch` clauses fused by
Qdrant's own RRF, so hybrid search is one round trip rather than two concurrent
ones plus client-side fusion.

**Sparse vectors without corpus statistics.** The client sends term-hash → term
frequency; Qdrant's `idf` modifier applies the weighting server-side. That is
why `SparseEncode` is a pure function of one string with no index-wide state to
keep in sync.

**Contextual retrieval, batched.** Chunks are embedded with an LLM-written
preamble situating them in their document — one call per document, not per
chunk, with the body in the system prompt.

**Semantic cache in Qdrant.** A second collection, not Redis — the vector store
already running is the only thing that can answer "is this a near-duplicate of
a previous query". Bypassed whenever ACL tags are present, so a cached answer
can never cross a permission boundary.

**Verification against the union of cited chunks.** Checking each citation
individually looks stricter but is wrong: a claim drawn from two chunks cites
both, and neither alone entails it — so per-citation checking refuses perfectly
grounded answers.

**Tolerant citation resolution.** Models retype ids rather than copy them, and
space-vs-underscore is the slip they actually make. Citations resolve
case- and separator-insensitively, but **only against chunks that were actually
retrieved**, and an id matching more than one is dropped rather than guessed.
Invented ids still match nothing.

**ACLs.** Every chunk carries `acl` tags; `-acl` filters retrieval at the
Qdrant level, before anything reaches the model.

## Tests

```bash
go test ./...                                        # offline
QDRANT_TEST_URL=http://localhost:6333 go test ./...  # + wire format
```

Offline tests cover chunking, sparse encoding, Office extraction (OOXML
fixtures built in memory), ranking metrics, citation resolution and the CI
gate. With a Qdrant URL they also exercise the collection schema, sparse
vectors, RRF fusion and ACL filtering using synthetic vectors — no API keys.

## Project layout

```
main.go            config, wiring, HTTP server, CLI
query.go           the query pipeline and its three guards
ingest.go          chunking, contextualization, indexing
qdrant.go          collections, hybrid search, RRF
office.go          docx/pptx/xlsx extraction (stdlib only)
rerank_tei.go      cross-encoder client
ollama.go          local provider
openai_compat.go   any OpenAI-compatible endpoint
embed_compat.go    OpenAI-compatible embeddings
claude.go          Anthropic provider
providers.go       Voyage, tokenizer, shared types
eval.go            metrics, golden set, CI gate
ui.html            the entire frontend
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `OLLAMA_EMBED_MODEL is not set` | By design — no model defaults exist. Set it. |
| `501` from Ollama on embed | That model has no embedding head. Use `bge-m3` or similar. |
| `tei rerank: 413 batch size 50 > maximum 32` | Lower `RERANKER_BATCH`. |
| TEI exits at startup | `TEI_IMAGE` doesn't match your GPU's compute capability. |
| `Vector dimension error: expected 1024, got 2048` | Changed embedding model — use a new `QDRANT_COLLECTION` and re-ingest. |
| `response truncated at max_tokens` | Reasoning model. Raise `OPENAI_MAX_TOKENS`, or lower `OPENAI_CONTEXT_BATCH` for ingest. |
| Everything refuses | Gate too high for your corpus. Run `calibrate`. |
| Answers ignore new documents | Semantic cache. Ingest clears it; `no_cache: true` bypasses it. |

Set `DEBUG_LLM=1` to log every upstream request and response — a response that
parses into the right *shape* but the wrong *contents* is invisible from the
error alone, and that is how most bugs here were found.

## Contributing

Issues and PRs welcome.

- `go vet ./...` and `go test ./...` must pass.
- Non-trivial logic leaves one runnable check behind — the smallest test that
  fails if the logic breaks.
- If a change can affect retrieval or answer quality, include `eval` numbers
  before and after, under the same judge. Per-item output (`eval -verbose`)
  beats aggregates: guessing from aggregates got it wrong twice during
  development.
- Prefer the standard library. Office parsing is `archive/zip` + `encoding/xml`
  for a reason.

## Not built

Deliberately deferred:

- **Visual/ColPali page-image pipeline** — needs page rendering plus a model
  server; the text path is a working, measurable baseline underneath it.
- **Dedicated Tantivy shard** — add only if eval shows Qdrant's sparse vectors
  missing exact-match queries.
- **Kubernetes manifests, HPA, OpenTelemetry** — the binary is stateless and
  per-stage timings are already on every response, so wiring a tracer is
  additive.

## License

[MIT](LICENSE).
