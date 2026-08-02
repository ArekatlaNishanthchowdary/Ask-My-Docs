package main

import (
	"bytes"
	"fmt"
	"sort"
	"strings"

	"github.com/ledongthuc/pdf"
)

// PDF is the one format here that needs a dependency. Word, PowerPoint and
// Excel are ZIP archives of XML, which archive/zip and encoding/xml handle
// outright; PDF is a cross-referenced object graph with its own compression,
// font encodings and CMap tables, and a hand-rolled reader garbles real-world
// files rather than failing cleanly. A wrong answer from garbled text is worse
// than no answer, so this takes the library.
//
// What a PDF does NOT carry is chart data. OOXML caches the plotted numbers
// inside the file, which is why extractCharts can turn an Office chart into a
// table. A PDF chart is drawing operations or a raster image — its title, axis
// labels and legend are extractable text, its values are lines. See README.

// extractPDF emits one `## Page N` section per page.
//
// Page sections rather than one blob because the chunker splits on headings: a
// citation to a 60-page report is only useful if it narrows to a page, and page
// boundaries are the only structure a PDF reliably has.
func extractPDF(data []byte) (out string, err error) {
	// The parser panics on malformed xref tables and broken object streams
	// rather than returning an error. A corrupt upload must not take down the
	// ingest run or the server.
	defer func() {
		if r := recover(); r != nil {
			out, err = "", fmt.Errorf("unreadable PDF (corrupt or unsupported structure): %v", r)
		}
	}()

	r, err := pdf.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "encrypt") {
			return "", fmt.Errorf("PDF is encrypted; remove the password and re-save")
		}
		return "", fmt.Errorf("not a readable PDF: %w", err)
	}

	var blocks []string
	pages := r.NumPage()
	for i := 1; i <= pages; i++ {
		p := r.Page(i)
		if p.V.IsNull() {
			continue
		}
		text := pageText(p)
		if text == "" {
			continue
		}
		blocks = append(blocks, fmt.Sprintf("## Page %d", i), text)
	}

	if len(blocks) == 0 {
		// The single most common PDF failure, and it is silent otherwise: a
		// scan or an export of images has pages but no text layer. Say which
		// problem it is instead of indexing an empty document.
		return "", fmt.Errorf("no extractable text in %d page(s) — this is likely a scanned or image-only PDF, which needs OCR", pages)
	}
	return joinBlocks(blocks), nil
}

// pageText reconstructs a page's lines from positioned text runs.
//
// GetTextByRow groups runs by their Y coordinate, which is what keeps a line a
// line: PDF stores text as positioned fragments with no notion of lines, so
// concatenating runs in file order interleaves columns and turns a two-column
// page into alternating nonsense.
func pageText(p pdf.Page) string {
	rows, err := p.GetTextByRow()
	if err != nil {
		// Fall back to the flat extractor rather than dropping the page: worse
		// layout still beats no content.
		s, ferr := p.GetPlainText(nil)
		if ferr != nil {
			return ""
		}
		return strings.TrimSpace(s)
	}

	orderRows(p, rows)

	var lines []string
	for _, row := range rows {
		var b strings.Builder
		var prevEnd float64
		for i, t := range row.Content {
			// Runs on one line are separate objects with no spaces between
			// them; a visible gap is the only evidence a space belongs there.
			// Without this, "Total Revenue" arrives as "TotalRevenue".
			if i > 0 && t.X-prevEnd > t.FontSize*0.2 {
				b.WriteByte(' ')
			}
			b.WriteString(t.S)
			prevEnd = t.X + t.W
		}
		if line := strings.TrimSpace(b.String()); line != "" {
			lines = append(lines, line)
		}
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

// orderRows puts rows into reading order, top of page first.
//
// The direction cannot be assumed. Rows carry a transformed Y whose sign
// depends on the producer's text matrix: a Chrome-printed page reports the
// title at 37 and the footer at 224 (Y down), while the same file's untransformed
// user space has the title at 736 (Y up). Guessing gets one of those two
// backwards and silently indexes every page bottom-first — which reads as
// nonsense and buries a document's title, the strongest thing it has.
//
// So take the direction from the library's own stream-order extractor rather
// than from the geometry: whichever end of the sort the first line lands on is
// the top.
func orderRows(p pdf.Page, rows pdf.Rows) {
	if len(rows) < 2 {
		return
	}
	sort.SliceStable(rows, func(i, j int) bool { return rows[i].Position < rows[j].Position })

	plain, err := p.GetPlainText(nil)
	if err != nil {
		return
	}
	first := ""
	for _, l := range strings.Split(plain, "\n") {
		if first = strings.TrimSpace(l); first != "" {
			break
		}
	}
	if first == "" {
		return
	}
	// If the document's first text sits in the last row of the ascending sort,
	// ascending is backwards.
	if rowIndexOf(rows, first) > len(rows)/2 {
		sort.SliceStable(rows, func(i, j int) bool { return rows[i].Position > rows[j].Position })
	}
}

func rowIndexOf(rows pdf.Rows, text string) int {
	for i, row := range rows {
		var b strings.Builder
		for _, t := range row.Content {
			b.WriteString(t.S)
		}
		if strings.Contains(strings.Join(strings.Fields(b.String()), " "), text) {
			return i
		}
	}
	return -1
}
