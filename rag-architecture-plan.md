# "Ask My Docs" — Production RAG Architecture Plan

**Scope:** hybrid retrieval (BM25 + vector), cross-encoder reranking, citation enforcement, CI-gated eval, low latency at 1M+ docs / high QPS, cloud-deployed, no Python in the app layer.

---

## 1. Target SLOs (define these before writing code)

| Metric | Target |
|---|---|
| End-to-end p95 latency (retrieval + rerank, excl. LLM generation) | < 400ms |
| Retrieval-only p95 (hybrid search, 1M+ docs) | < 50ms |
| Rerank p95 (top-50 → top-10) | < 150ms |
| Citation coverage (every factual claim has a valid, entailing citation) | > 98% |
| Retrieval Recall@10 on golden eval set | > 90% |
| CI eval regression tolerance | fail build if any gated metric drops > 2% vs. baseline |

Write these down first — the eval pipeline in §7 is what enforces them, and you can't gate CI on numbers you haven't defined.

---

## 2. Why "skip Python" is actually easy here

The Python-heavy part of RAG stacks is almost always glue code (chunking loops, API orchestration, eval scripts) — not the actual hot path. The hot path (vector index, HNSW search, BM25) is already implemented in Rust or C++ under the hood in every serious engine (Qdrant, Tantivy, Lucene). So "skip Python" mostly means: **don't write your orchestration layer in Python** — use a compiled, concurrent language for services, and lean on managed inference APIs or gRPC-based model servers instead of hand-rolled Python ML glue.

**Recommendation given "no strong preference, AI-built":**

| Layer | Language | Why |
|---|---|---|
| Ingestion orchestrator, query API, eval harness | **Go** | Goroutines make concurrent I/O (calling embedding APIs, Qdrant, reranker, LLM) trivial and fast. Simple type system means agentic coding tools produce fewer subtle bugs than with Rust's ownership model, while still compiling to a single static binary with excellent gRPC/HTTP support. Since your latency is dominated by network hops (vector DB, model APIs) rather than CPU-bound app logic, Rust's raw speed edge over Go barely shows up here. |
| Storage / retrieval engine | **Qdrant (Rust, used as infra, not written by you)** | You get Rust-grade performance for free by using it as a service rather than writing Rust yourself. |
| CI eval scripts, GitHub Actions tooling | **TypeScript** | Best ecosystem fit for CI/test runners (Vitest) and for a lightweight admin/eval dashboard if you want one. |
| Model inference (if self-hosting embeddings/reranker instead of API) | **ONNX Runtime or Triton Inference Server, called via gRPC** | Language-agnostic serving; your Go services never touch Python even though the model itself may have been exported from a Python-trained checkpoint upstream. |

Only reach for Rust directly if you later need a custom fusion/scoring kernel faster than what Qdrant's client gives you — unlikely to be your bottleneck at first.

---

## 3. High-level architecture

```
                         ┌─────────────────────────────────────────┐
                         │              INGESTION                  │
  Source docs ──────────▶│  Classifier: text-clean vs visually-rich │
  (S3/GCS, upload API)   └──────────────┬───────────────┬──────────┘
                                         │               │
                          ┌──────────────▼───┐   ┌───────▼──────────────┐
                          │  TEXT PIPELINE    │   │  VISUAL PIPELINE      │
                          │  structure-aware  │   │  render page → image  │
                          │  extraction       │   │  ColPali/ColQwen2.5    │
                          │  → contextual     │   │  patch embeddings      │
                          │    chunking       │   │  (late-interaction)    │
                          │  → dense + sparse │   └───────┬────────────────┘
                          │    embeddings     │           │
                          └────────┬──────────┘           │
                                   └───────────┬───────────┘
                                               ▼
                         ┌───────────────────────────────────┐
                         │   Qdrant (sharded, quantized)      │
                         │   dense vectors + sparse (SPLADE)  │
                         │   + multi-vector (ColPali) fields  │
                         └──────────────┬──────────────────────┘
                                        │
        ┌───────────────────────────────▼───────────────────────────────┐
        │                        QUERY SERVICE (Go)                     │
        │  1. Query embed (dense) + sparse encode                       │
        │  2. Parallel retrieval: dense ANN + sparse BM25/SPLADE + (if  │
        │     query implies visual content) ColPali page search         │
        │  3. Reciprocal Rank Fusion → top-50 candidates                │
        │  4. Cross-encoder rerank → top-8-10                           │
        │  5. LLM generation, forced structured citations               │
        │  6. Post-hoc citation/entailment verifier                     │
        │  7. Confidence gate → "insufficient information" fallback     │
        └───────────────────────────────┬───────────────────────────────┘
                                        │
                              Redis semantic cache (pre-step 1, on hit skip to 5)
```

---

## 4. Ingestion: dual-path indexing

**Text pipeline** (clean PDFs, markdown, HTML, Office docs converted to text):
- Structure-aware extraction that preserves headings, tables, and section hierarchy (don't flatten to plain text first — chunk boundaries should respect document structure).
- **Contextual chunking**: before embedding each chunk, prepend a short LLM-generated summary of where the chunk sits in the document (Anthropic's Contextual Retrieval approach) — this is the single highest-leverage 2026-era technique for retrieval quality, and it's cheap if you batch it with prompt caching. Alternative if you want to skip the extra LLM call per chunk: **late chunking** (embed the full document first, then pool per-chunk from token-level embeddings) — cheaper, slightly lower quality than contextual retrieval on most benchmarks.
- Dual embedding: dense vector (for semantic recall) + sparse (BM25 or SPLADE) for exact-match precision on identifiers, codes, names.

**Visual pipeline** (scanned docs, tables, charts, slide decks — the part of your corpus that's "mixed"):
- Render each page as an image, embed with a ColPali-family model (ColQwen2.5 or ColSmolVLM are current-generation options) producing patch-level multi-vector embeddings, stored in Qdrant's multi-vector fields with late-interaction scoring.
- This avoids OCR entirely for these documents, which is exactly where OCR-based pipelines lose information (table structure, chart data, layout-dependent meaning).
- Route documents into this path automatically: image/scan ratio above a threshold, or a table/figure-density heuristic from the extraction step.

**Metadata for every chunk/page:** source doc id, page number, section path, doc version/timestamp, and any ACL tags — you'll need these for citation rendering and for permission-filtered retrieval.

---

## 5. Storage & indexing

- **Qdrant**, sharded and quantized, as the single retrieval engine — it natively supports dense vectors, sparse vectors (for SPLADE/BM25-style lexical search), and multi-vector late-interaction fields (for the ColPali path) in one system, so you're not stitching together a separate BM25 engine and vector DB with extra network hops. <cite index="24-1">Native hybrid support like this adds roughly 6ms of latency over vector-only search, in exchange for meaningfully better recall</cite> — worth it.
- If your domain leans heavily on exact-match lookups (SKUs, legal citations, error codes) and you find Qdrant's sparse-vector BM25 approximation isn't precise enough, add a dedicated **Tantivy** (Rust, Lucene-equivalent) shard for that subset and fuse at the RRF stage — but start without it and add only if eval shows a gap.
- Postgres for document metadata, ACLs, and versioning (not vectors) — keep this system boring and consistent.
- Object storage (S3/GCS) for source files and rendered page images, fronted by a CDN if page images are served to users for citation preview.

---

## 6. Query pipeline detail

1. **Query understanding** (optional but recommended at this scale): lightweight query rewrite/expansion via the LLM, and a classifier for "does this query likely need the visual/page-image path" (keywords like "chart," "table," "diagram," or query embedding similarity to known visual-heavy doc clusters).
2. **Parallel retrieval**: dense ANN + sparse lexical search fire concurrently (this is where Go's concurrency model pays off — two goroutines, one Qdrant client, negligible overhead), plus the ColPali path if triggered.
3. **Reciprocal Rank Fusion** to merge ranked lists into a single top-50 candidate set — simple, robust, and the field's default over learned fusion for a v1.
4. **Cross-encoder reranking** of the top-50 down to top-8–10. Model choice is a real tradeoff:

   | Option | Latency | Notes |
   |---|---|---|
   | Hosted API (Cohere Rerank 4, Voyage rerank-2.5) | ~600ms | <cite index="20-1">Voyage Rerank 2.5 and Cohere Rerank 3.5 post the fastest hosted response times at roughly 595-603ms average</cite> — too slow for your 150ms rerank budget as a hosted call; use only if you relax the latency target or run it async/pre-fetched. |
   | Self-hosted BGE-reranker-v2-m3 (GPU) | ~50-100ms | Apache 2.0, multilingual, matches API latency once on GPU. |
   | Self-hosted Jina Reranker v3 | ~188ms | <cite index="22-1">The only top-tier reranker to land under 200ms, at 81.33% Hit@1, with a 131k-token context and listwise scoring of 64 documents at once</cite>. |

   **Given your latency budget, self-host the reranker** (BGE-v2-m3 or Jina v3) behind an ONNX Runtime/Triton gRPC endpoint rather than calling a hosted API in the critical path. Benchmark both on your own eval set before committing — reranker quality is corpus-dependent.

5. **Generation with forced citations**: structured output/tool-calling schema where the model must emit `[doc_id:chunk_id]` markers, and those IDs are validated against the actual retrieved set (reject any citation to a chunk that wasn't retrieved — this is a hard constraint, not a suggestion to the model).
6. **Post-hoc citation verifier**: a lightweight entailment check (can reuse the reranker or a small NLI model) confirming each cited chunk actually supports the claim next to it. Low-entailment answers get regenerated once, then fall back to "insufficient information" rather than shipping an ungrounded claim.
7. **Confidence gate**: if the top rerank score is below a corpus-tuned threshold, skip generation and return "I don't have enough information" — this is your negative-rejection safeguard and it's cheap insurance against hallucination on out-of-corpus questions.
8. **Semantic cache** (Redis) in front of step 1 for repeat/near-duplicate queries — at high QPS this alone can cut average latency substantially.

---

## 7. Evaluation & CI gating

This is what makes the system "production" rather than a demo, and it needs to exist **before** you tune anything.

- **Golden eval set**: curated `query → relevant chunk IDs` (+ ideally gold answers), covering both text and visual-path documents, built from real or realistic queries in your domain. Aim for enough coverage to detect regressions in specific subsystems (retrieval-only failures vs. reranking failures vs. generation failures) — tag each eval item by which stage it exercises.
- **Metrics tracked per PR**:
  - Retrieval: Recall@10, nDCG@10, MRR
  - Reranking: NDCG lift over retrieval-only baseline
  - Generation: citation precision/recall, answer correctness (LLM-as-judge via structured API call, or semantic similarity to gold answer)
  - Latency: p50/p95/p99 at each stage, run against a staging index sized to represent production
- **CI pipeline** (GitHub Actions): on any PR touching chunking logic, embedding model, reranker, prompts, or fusion weights — run the full eval suite, compare to the last known-good baseline stored alongside the repo, and **fail the build** if any gated metric regresses past the tolerance in §1. Store eval run results as build artifacts so regressions are diffable.
- **Perf regression as a first-class CI gate**, not just quality — a load test (k6, or a small Go benchmark harness) against staging on every PR that touches the retrieval or rerank path, gated on the p95 targets.
- All of this in Go/TypeScript, no Python: the eval harness is just an HTTP/gRPC client hitting your own query service plus the Claude API for judging — no ML framework needed.

---

## 8. Deployment & scaling (1M+ docs, high QPS, cloud)

- **Qdrant**: managed (Qdrant Cloud) or self-managed on Kubernetes, sharded and replicated, with scalar/product quantization to keep memory bounded at this scale.
- **Go services** (ingestion workers, query API): containerized, horizontal pod autoscaling keyed on QPS/latency, deployed behind a load balancer with per-tenant rate limiting if multi-tenant.
- **GPU node pool** only if self-hosting embeddings/reranker/ColPali inference — otherwise these can start as managed API calls and move in-house once volume justifies the GPU cost.
- **Observability**: OpenTelemetry tracing through every pipeline stage (embed → retrieve → fuse → rerank → generate → verify) so you can see exactly where a slow or wrong answer came from, plus Prometheus/Grafana for the SLO dashboard from §1.
- **Ingestion throughput**: make ingestion idempotent and resumable (queue-based, e.g. via a managed queue) since re-indexing 1M+ docs after a chunking or embedding-model change is something you *will* need to do.

---

## 9. Phased build order

1. **Golden eval set first** — you cannot gate anything without it.
2. Text ingestion pipeline (MVP) → hybrid retrieval → basic generation with citations.
3. Reranking + citation verifier + confidence gate.
4. Visual/page-indexing pipeline for the visually-rich half of the corpus.
5. CI-gated eval pipeline wired into GitHub Actions, with perf gates.
6. Scale-out: sharding, semantic cache, autoscaling, full observability, production hardening.

Building it in this order means every later stage has a working, measurable baseline underneath it — including the visual pipeline, which is the highest-effort piece and easiest to defer without blocking everything else.

---

## 10. Open decisions to settle by benchmarking on your own corpus

- Hosted vs. self-hosted reranker (latency/cost/quality tradeoff — table in §6 is a starting point, not a verdict).
- Contextual retrieval vs. late chunking (quality vs. ingestion cost tradeoff).
- Whether you need a dedicated Tantivy BM25 shard or Qdrant's native sparse vectors are precise enough for your exact-match queries.
- Dense/sparse/visual fusion weights — RRF is a good default but should be tuned against the golden eval set, not guessed.

---

### Further reading (not reproduced, just linked)
- Qdrant vector DB benchmarks: digitalapplied.com/blog/vector-databases-for-ai-agents-pinecone-qdrant-2026
- ColPali paper: arxiv.org/abs/2407.01449
- Reranker comparison: particula.tech/blog/reranker-models-compared-cohere-voyage-jina-bge-latency-ndcg
- Hybrid search guide: supermemory.ai/blog/hybrid-search-guide
- RAG chunking strategies 2026: digitalapplied.com/blog/rag-chunking-strategies-2026-retrieval-quality-playbook
