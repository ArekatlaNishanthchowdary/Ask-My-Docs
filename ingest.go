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
		// A single paragraph larger than the budget is hard-split; nothing smarter
		// is available without a tokenizer, and it is rare in practice.
		for len(para) > maxChars {
			if cur.Len() > 0 {
				out = append(out, cur.String())
				cur.Reset()
			}
			out = append(out, para[:maxChars])
			para = para[maxChars-overlap:]
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

	contexts, err := a.llm().Contextualize(ctx, text, chunks)
	if err != nil {
		// Retrieval quality degrades without the situating preamble, but the
		// document is still findable. Do not fail the whole ingest over it.
		fmt.Fprintf(os.Stderr, "warn: %s: contextualize failed, indexing raw chunks: %v\n", docID, err)
		contexts = make([]string, len(chunks))
	}

	// The embedded text carries the context; the payload keeps the raw chunk,
	// because that is what gets shown to the user and fed to the reranker.
	embedTexts := make([]string, len(chunks))
	for i, c := range chunks {
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

// IngestDir walks a directory and ingests every .md/.txt file, concurrently.
// Ingest is idempotent — point IDs are derived from doc_id + ordinal — so a
// re-run after an embedding-model change overwrites in place rather than
// duplicating, which is what makes a full re-index safe to resume.
func (a *App) IngestDir(ctx context.Context, dir string, acl []string) (int, int, error) {
	var paths []string
	err := filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
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
		return 0, 0, err
	}

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(a.Cfg.IngestConcurrency)
	counts := make([]int, len(paths))
	for i, p := range paths {
		i, p := i, p
		g.Go(func() error {
			raw, err := os.ReadFile(p)
			if err != nil {
				return err
			}
			rel, err := filepath.Rel(dir, p)
			if err != nil {
				rel = filepath.Base(p)
			}
			docID := filepath.ToSlash(rel)
			// Office formats are extracted to markdown here so everything
			// downstream — chunking, contextualization, citations — works on
			// one representation.
			text, err := LoadDocumentText(docID, raw)
			if err != nil {
				return fmt.Errorf("%s: %w", docID, err)
			}
			version := fmt.Sprintf("%d", fileModTime(p))
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
		return 0, 0, err
	}
	total := 0
	for _, c := range counts {
		total += c
	}
	return len(paths), total, nil
}

// docExtensions is the single source of truth for what counts as a document,
// shared by the directory walk, the upload endpoint and the chunks command.
var docExtensions = map[string]bool{
	".md": true, ".markdown": true, ".txt": true,
	".docx": true, ".pptx": true, ".xlsx": true,
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
