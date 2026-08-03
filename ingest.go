package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"
)

type Chunk struct {
	Text    string
	Section string // heading path, e.g. "Billing > Refunds"
	Ordinal int
}

// Chunk splits markdown-ish text on heading boundaries first, then packs each
// section into size-bounded chunks with overlap. Structure before size: a chunk
// that straddles two sections retrieves badly for both.
func ChunkDoc(text string, maxChars, overlap int) []Chunk {
	var chunks []Chunk
	var path []string
	var buf []string
	bufLen := 0

	flush := func() {
		if bufLen == 0 {
			return
		}
		section := strings.Join(path, " > ")
		body := strings.Join(buf, "\n")
		for _, piece := range packText(body, maxChars, overlap) {
			chunks = append(chunks, Chunk{Text: piece, Section: section, Ordinal: len(chunks)})
		}
		buf, bufLen = nil, 0
	}

	for _, line := range strings.Split(text, "\n") {
		if lvl, title := heading(line); lvl > 0 {
			flush()
			for len(path) >= lvl {
				path = path[:len(path)-1]
			}
			for len(path) < lvl-1 {
				path = append(path, "")
			}
			path = append(path, title)
			continue
		}
		buf = append(buf, line)
		bufLen += len(line) + 1
	}
	flush()
	return chunks
}

func heading(line string) (int, string) {
	t := strings.TrimLeft(line, " ")
	if !strings.HasPrefix(t, "#") {
		return 0, ""
	}
	lvl := 0
	for lvl < len(t) && t[lvl] == '#' {
		lvl++
	}
	if lvl > 6 || lvl >= len(t) || t[lvl] != ' ' {
		return 0, ""
	}
	return lvl, strings.TrimSpace(t[lvl:])
}

// packText splits on paragraph boundaries, filling up to maxChars and carrying
// `overlap` characters of tail into the next chunk so a fact spanning a
// boundary is still retrievable from at least one side.
func packText(body string, maxChars, overlap int) []string {
	body = strings.TrimSpace(body)
	if body == "" {
		return nil
	}
	if len(body) <= maxChars {
		return []string{body}
	}
	var out []string
	var cur strings.Builder
	for _, para := range strings.Split(body, "\n\n") {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		// A single paragraph larger than the budget is hard-split. A CSV or
		// other table converted to one giant blob has no blank lines to pack
		// on, so this is not rare: it is the only path a 168-row CSV takes.
		// Cutting at a byte offset severs a word (or a table row) mid-cell,
		// which is exactly the garbled input that made a generator refuse to
		// answer from a chunk that actually held the answer. Prefer the last
		// line break, then the last space, within the budget.
		for len(para) > maxChars {
			if cur.Len() > 0 {
				out = append(out, cur.String())
				cur.Reset()
			}
			cut := maxChars
			if i := strings.LastIndexByte(para[:maxChars], '\n'); i > maxChars/2 {
				cut = i + 1
			} else if i := strings.LastIndexByte(para[:maxChars], ' '); i > maxChars/2 {
				cut = i + 1
			}
			out = append(out, para[:cut])
			para = para[max(cut-overlap, 0):]
		}
		if cur.Len()+len(para)+2 > maxChars && cur.Len() > 0 {
			s := cur.String()
			out = append(out, s)
			cur.Reset()
			if overlap > 0 && len(s) > overlap {
				cur.WriteString(s[len(s)-overlap:])
				cur.WriteString("\n\n")
			}
		}
		if cur.Len() > 0 {
			cur.WriteString("\n\n")
		}
		cur.WriteString(para)
	}
	if strings.TrimSpace(cur.String()) != "" {
		out = append(out, cur.String())
	}
	return out
}

// IngestDoc chunks, contextualizes, embeds, and upserts one document.
func (a *App) IngestDoc(ctx context.Context, docID, version string, acl []string, text string) (int, error) {
	chunks := ChunkDoc(text, a.Cfg.MaxChunkChars, a.Cfg.ChunkOverlap)
	if len(chunks) == 0 {
		return 0, nil
	}

	// The three context rungs of the ablation ladder — none, derived, written —
	// differ only in what fills this slice. They have to be chosen at ingest
	// time because the context is baked into the vector, so each rung needs its
	// own QDRANT_COLLECTION rather than a flag on the query.
	contexts := make([]string, len(chunks))
	if !a.off("llmcontext") {
		written, err := a.llm().Contextualize(ctx, docDigest(text, a.Cfg.ContextDocChars), chunks)
		if err != nil {
			// Retrieval quality degrades without the situating preamble, but the
			// document is still findable. Do not fail the whole ingest over it.
			fmt.Fprintf(os.Stderr, "warn: %s: contextualize failed, indexing raw chunks: %v\n", docID, err)
		} else {
			contexts = written
		}
	}

	// The embedded text carries the context; the payload keeps the raw chunk,
	// because that is what gets shown to the user and fed to the reranker.
	embedTexts := make([]string, len(chunks))
	for i, c := range chunks {
		if strings.TrimSpace(contexts[i]) == "" && !a.off("context") {
			contexts[i] = fallbackContext(docID, c.Section)
		}
		embedTexts[i] = strings.TrimSpace(contexts[i] + "\n\n" + c.Text)
	}

	pts := make([]Point, 0, len(chunks))
	for start := 0; start < len(embedTexts); start += a.Cfg.EmbedBatch {
		end := min(start+a.Cfg.EmbedBatch, len(embedTexts))
		vecs, err := a.Embedder.Embed(ctx, embedTexts[start:end], "document")
		if err != nil {
			return 0, fmt.Errorf("embed %s: %w", docID, err)
		}
		for i, vec := range vecs {
			c := chunks[start+i]
			chunkID := fmt.Sprintf("%s#%d", docID, c.Ordinal)
			pts = append(pts, Point{
				ID: pointID(chunkID),
				Vector: map[string]any{
					"dense":  vec,
					"sparse": SparseEncode(embedTexts[start+i]),
				},
				Payload: map[string]any{
					"doc_id":   docID,
					"chunk_id": chunkID,
					"text":     c.Text,
					"context":  contexts[start+i],
					"section":  c.Section,
					"ordinal":  c.Ordinal,
					"version":  version,
					"acl":      acl,
					"indexed":  time.Now().UTC().Format(time.RFC3339),
				},
			})
		}
	}
	return len(pts), a.Qdrant.Upsert(ctx, a.Cfg.Collection, pts)
}

// docDigest bounds what contextualization is allowed to call "the document".
//
// Every provider puts the document body in the system prompt of every batch, so
// a C-chunk document costs ceil(C/batch) x |document| prompt tokens — quadratic
// in its length. The 197-chunk CSV in this corpus costs ~1.9M tokens to
// contextualize; a thousand-chunk manual would cost ~50M for one document. That
// is the shape that makes a large corpus unaffordable rather than merely
// expensive, and no amount of concurrency tuning touches it.
//
// The full body was never what the model needed. It needs to know what the
// document is, which the opening lines and the heading outline carry; the text
// immediately around a chunk already arrives in the batch itself, because
// batches are contiguous runs of chunks. Bounding the body therefore makes the
// cost linear in C at a constant per batch, and leaves both kinds of context
// the model actually uses intact.
//
// budget <= 0 disables the bound and sends the whole document.
func docDigest(doc string, budget int) string {
	if budget <= 0 || len(doc) <= budget {
		return doc
	}
	// The opening is where a document says what it is: a title, a letterhead, a
	// name, a CSV's header row. Cut on a line boundary so it does not end
	// halfway through the sentence that names the subject.
	head := doc[:budget*2/3]
	if i := strings.LastIndexByte(head, '\n'); i > 0 {
		head = head[:i]
	}
	// A byte offset can land inside a multi-byte rune, and a document with no
	// newline in its head keeps that raw cut. Drop the broken tail rather than
	// send a replacement character.
	head = strings.ToValidUTF8(head, "")
	var sb strings.Builder
	sb.WriteString(head)
	sb.WriteString("\n\n[...document truncated...]\n")
	// The outline restores the global shape the truncation just removed, at a
	// fraction of the size — and it is what tells the model that a chunk from
	// the far end of a long document belongs to, say, a warranty appendix.
	// Documents with no headings (a CSV, a plain text file) simply have none.
	wroteHeader := false
	for _, line := range strings.Split(doc, "\n") {
		lvl, title := heading(line)
		if lvl == 0 {
			continue
		}
		if !wroteHeader {
			sb.WriteString("\nHeadings in the full document:\n")
			wroteHeader = true
		}
		if sb.Len()+len(title)+2*lvl > budget {
			sb.WriteString("...\n")
			break
		}
		sb.WriteString(strings.Repeat("  ", lvl-1))
		sb.WriteString(title)
		sb.WriteByte('\n')
	}
	return sb.String()
}

// fallbackContext derives a situating line from what ingest already knows —
// the filename and the heading path — with no model call and no tokens.
//
// It is measurably worse than an LLM-written context, but the comparison that
// matters is against *no* context. Measured on this corpus with the same
// cross-encoder the query path uses, chunks the reranker scored at 0.0001-0.0014
// raw scored 0.010-0.094 with this line prepended, against 0.13-0.39 for the
// LLM context. Filenames and headings usually name the entity a chunk only
// refers to by pronoun, which is the whole job of the context.
//
// So contextualization stays an enhancement rather than the thing standing
// between a corpus and working retrieval: when it is unaffordable, rate-limited
// or simply failed, chunks are still findable by name.
func fallbackContext(docID, section string) string {
	name := strings.TrimSuffix(filepath.Base(docID), filepath.Ext(docID))
	name = strings.NewReplacer("_", " ", "-", " ").Replace(name)
	if section == "" {
		return fmt.Sprintf("From the document %q.", name)
	}
	return fmt.Sprintf("From the document %q, section %q.", name, section)
}

// unchanged reports whether this document is already indexed at this exact
// version. Ingest has always written version (the file's mtime) into every
// payload and never read it back, so every run re-did every document.
//
// At corpus scale that is the difference between adding one document and
// re-contextualizing the whole corpus, and contextualization is the pipeline's
// single largest token consumer — it sends the document body once per batch of
// chunks. This is the check that makes a 1M-document index affordable to keep
// current rather than affordable only once.
//
// It cannot see a change that leaves mtime alone: a different embedding model,
// different chunk bounds, an edited prompt, a restored backup. Those change how
// the document should be indexed, not the document — hence -force.
func (a *App) unchanged(ctx context.Context, docID, version string) bool {
	n, err := a.Qdrant.CountFiltered(ctx, a.Cfg.Collection, DocVersionFilter(docID, version))
	// Not knowing is not the same as knowing it is current. Re-ingesting costs
	// tokens; skipping on a failed lookup costs a document that is silently
	// missing from the index, so the error case re-ingests.
	return err == nil && n > 0
}

// IngestDir walks a directory and ingests every .md/.txt file, concurrently.
// Ingest is idempotent — point IDs are derived from doc_id + ordinal — so a
// re-run after an embedding-model change overwrites in place rather than
// duplicating, which is what makes a full re-index safe to resume.
func (a *App) IngestDir(ctx context.Context, dir string, acl []string, force bool) (indexed, skipped, chunks int, err error) {
	var paths []string
	err = filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if isDoc(p) {
			paths = append(paths, p)
		}
		return nil
	})
	if err != nil {
		return 0, 0, 0, err
	}

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(a.Cfg.IngestConcurrency)
	counts := make([]int, len(paths))
	skips := make([]bool, len(paths))
	for i, p := range paths {
		i, p := i, p
		g.Go(func() error {
			rel, err := filepath.Rel(dir, p)
			if err != nil {
				rel = filepath.Base(p)
			}
			docID := filepath.ToSlash(rel)
			version := fmt.Sprintf("%d", fileModTime(p))
			// Checked before the file is even read: an unchanged document costs
			// one count query, not an extraction, a contextualization pass and
			// an embedding pass.
			if !force && a.unchanged(gctx, docID, version) {
				skips[i] = true
				fmt.Printf("  %s: unchanged, skipped\n", docID)
				return nil
			}
			raw, err := os.ReadFile(p)
			if err != nil {
				return err
			}
			// Office formats are extracted to markdown here so everything
			// downstream — chunking, contextualization, citations — works on
			// one representation.
			text, err := LoadDocumentText(docID, raw)
			if err != nil {
				return fmt.Errorf("%s: %w", docID, err)
			}
			n, err := a.IngestDoc(gctx, docID, version, acl, text)
			if err != nil {
				return fmt.Errorf("%s: %w", docID, err)
			}
			counts[i] = n
			fmt.Printf("  %s: %d chunks\n", docID, n)
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return 0, 0, 0, err
	}
	for i, c := range counts {
		chunks += c
		if skips[i] {
			skipped++
		}
	}
	return len(paths) - skipped, skipped, chunks, nil
}

// docExtensions is the single source of truth for what counts as a document,
// shared by the directory walk, the upload endpoint and the chunks command.
var docExtensions = map[string]bool{
	".md": true, ".markdown": true, ".txt": true,
	".docx": true, ".pptx": true, ".xlsx": true,
	".csv": true, ".tsv": true, ".tab": true,
	".pdf": true,
}

func isDoc(path string) bool { return docExtensions[strings.ToLower(filepath.Ext(path))] }

// acceptedTypes renders the supported list for error messages, so the message
// cannot drift from docExtensions.
func acceptedTypes() string {
	exts := make([]string, 0, len(docExtensions))
	for e := range docExtensions {
		exts = append(exts, e)
	}
	sort.Strings(exts)
	return strings.Join(exts, ", ")
}

// SafeCorpusPath resolves an untrusted, client-supplied filename to a path
// inside the corpus directory, or fails.
//
// This is a trust boundary: the name arrives from a browser upload, so it is
// treated as hostile. Directory components are stripped rather than cleaned,
// the extension must be one we actually index, and the resolved path is
// re-checked against the corpus root so a surviving traversal cannot escape.
func SafeCorpusPath(dir, name string) (string, error) {
	base := filepath.Base(filepath.FromSlash(strings.TrimSpace(name)))
	if base == "" || base == "." || base == ".." || strings.ContainsRune(base, 0) {
		return "", fmt.Errorf("invalid filename")
	}
	if strings.HasPrefix(base, ".") {
		return "", fmt.Errorf("dotfiles are not accepted")
	}
	if !isDoc(base) {
		ext := strings.ToLower(filepath.Ext(base))
		if want, legacy := legacyOffice[ext]; legacy {
			return "", fmt.Errorf("%s is the legacy binary format; re-save it as %s", ext, want)
		}
		return "", fmt.Errorf("unsupported type %q (accepted: %s)", filepath.Ext(base), acceptedTypes())
	}
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	full := filepath.Join(absDir, base)
	// Belt and braces: Join already cleans, but verify containment explicitly so
	// a future change to the sanitising above cannot silently open an escape.
	if full != filepath.Join(absDir, filepath.Base(full)) ||
		!strings.HasPrefix(full, absDir+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes the corpus directory")
	}
	return full, nil
}

// ClearCache drops every cached answer. Any ingest can change what the correct
// answer to an existing question is, and a semantic cache hit skips retrieval
// entirely — so without this, newly added documents stay invisible to exactly
// the questions users have already asked.
func (a *App) ClearCache(ctx context.Context) error {
	if err := a.Qdrant.DropCollection(ctx, a.Cfg.CacheCollection); err != nil {
		return err
	}
	return a.Qdrant.EnsureCollection(ctx, a.Cfg.CacheCollection, a.embedDim, 1)
}

func fileModTime(p string) int64 {
	fi, err := os.Stat(p)
	if err != nil {
		return 0
	}
	return fi.ModTime().Unix()
}
