package main

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Office documents are ZIP archives of XML, so archive/zip and encoding/xml
// handle them with no third-party dependency.
//
// Extraction targets markdown rather than a flat string on purpose: the
// chunker splits on headings, so a Word "Heading 2", a slide boundary, or a
// worksheet name becoming a real `##` is what keeps chunks aligned to the
// document's own structure. Flattening to plain text would quietly cost
// retrieval quality on exactly the documents people care most about.

// legacyOffice are the pre-2007 binary formats. They are not ZIP archives and
// need an entirely different parser, so say so rather than failing obscurely.
var legacyOffice = map[string]string{
	".doc": ".docx", ".ppt": ".pptx", ".xls": ".xlsx",
}

// LoadDocumentText turns raw file bytes into indexable text. It is the single
// entry point for both the directory walk and the upload endpoint, so the two
// can never disagree about what is supported.
func LoadDocumentText(name string, data []byte) (string, error) {
	ext := strings.ToLower(path.Ext(name))
	if want, legacy := legacyOffice[ext]; legacy {
		return "", fmt.Errorf("%s is the legacy binary format; re-save it as %s", ext, want)
	}
	switch ext {
	case ".docx":
		return extractDocx(data)
	case ".pptx":
		return extractPptx(data)
	case ".xlsx":
		return extractXlsx(data)
	case ".pdf":
		return extractPDF(data)
	case ".csv":
		return extractCSV(data, ',')
	case ".tsv", ".tab":
		return extractCSV(data, '\t')
	case ".md", ".markdown", ".txt":
		if !utf8.Valid(data) {
			return "", fmt.Errorf("not valid UTF-8 text")
		}
		return string(data), nil
	}
	return "", fmt.Errorf("unsupported type %q", ext)
}

func openZip(data []byte) (*zip.Reader, error) {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, fmt.Errorf("not a readable Office file (%w)", err)
	}
	return zr, nil
}

func zipEntry(zr *zip.Reader, name string) ([]byte, error) {
	for _, f := range zr.File {
		if f.Name == name {
			rc, err := f.Open()
			if err != nil {
				return nil, err
			}
			defer rc.Close()
			// Bound decompression: a zip bomb should not be able to exhaust
			// memory through an upload endpoint.
			return io.ReadAll(io.LimitReader(rc, 64<<20))
		}
	}
	return nil, fmt.Errorf("missing %s", name)
}

// --- .docx ----------------------------------------------------------------

var headingStyle = regexp.MustCompile(`^(?i)heading\s*([1-9])$`)

// extractDocx walks word/document.xml as a token stream rather than unmarshalling
// into structs, because paragraphs and tables interleave and their order is the
// document's reading order — a struct with separate slices would lose it.
func extractDocx(data []byte) (string, error) {
	zr, err := openZip(data)
	if err != nil {
		return "", err
	}
	body, err := zipEntry(zr, "word/document.xml")
	if err != nil {
		return "", fmt.Errorf("not a Word document: %w", err)
	}

	var out []string
	var para strings.Builder
	level := 0 // heading level of the current paragraph, 0 = body text
	inCell := false
	var row []string
	var tblRows [][]string

	// Word's "Title" style sits above "Heading 1" but both would map to level 1,
	// so a Heading 1 would replace the title in the section path instead of
	// nesting under it — and a title with no body text beneath it then produces
	// no chunk at all, losing the document's name from the index entirely.
	// Once a Title is seen, push the Heading levels down one.
	headingOffset := 0

	dec := xml.NewDecoder(bytes.NewReader(body))
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", fmt.Errorf("malformed Word XML: %w", err)
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "p":
				para.Reset()
				level = 0
			case "pStyle":
				if v := attr(t, "val"); v != "" {
					if m := headingStyle.FindStringSubmatch(strings.ReplaceAll(v, "-", "")); m != nil {
						n, _ := strconv.Atoi(m[1])
						level = min(n+headingOffset, 6)
					} else if strings.EqualFold(v, "Title") {
						level = 1
						headingOffset = 1
					}
				}
			case "tab":
				para.WriteString("\t")
			case "br", "cr":
				para.WriteString("\n")
			case "tbl":
				tblRows = nil
			case "tc":
				inCell = true
			case "tr":
				row = row[:0]
			case "t":
				// <w:t> holds the actual run text.
				var s string
				if err := dec.DecodeElement(&s, &t); err == nil {
					para.WriteString(s)
				}
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "p":
				text := strings.TrimSpace(para.String())
				if inCell {
					if text != "" {
						row = append(row, text)
					}
					break
				}
				if text == "" {
					break
				}
				if level > 0 {
					out = append(out, strings.Repeat("#", level)+" "+text)
				} else {
					out = append(out, text)
				}
			case "tc":
				inCell = false
			case "tr":
				if len(row) > 0 {
					// Buffered rather than emitted per row: the table is only
					// complete at </w:tbl>, and a header row is only a header
					// once the rows under it exist to be headed.
					tblRows = append(tblRows, append([]string(nil), row...))
					row = nil
				}
			case "tbl":
				// A row read back as a sentence is how table meaning gets lost;
				// markdown keeps each value under its column name.
				if md := markdownTable(tblRows); md != "" {
					out = append(out, md)
				}
				tblRows = nil
			}
		}
	}
	out = append(out, extractCharts(zr)...)
	return joinBlocks(out), nil
}

func attr(e xml.StartElement, name string) string {
	for _, a := range e.Attr {
		if a.Name.Local == name {
			return a.Value
		}
	}
	return ""
}

// --- .pptx ----------------------------------------------------------------

var slideNum = regexp.MustCompile(`slide(\d+)\.xml$`)

// extractPptx emits one `## Slide N` section per slide, which gives the chunker
// a natural boundary — a deck chunked without slide breaks mixes unrelated
// points into one chunk.
func extractPptx(data []byte) (string, error) {
	zr, err := openZip(data)
	if err != nil {
		return "", err
	}
	type slide struct {
		n    int
		name string
	}
	var slides []slide
	for _, f := range zr.File {
		if m := slideNum.FindStringSubmatch(f.Name); m != nil && strings.HasPrefix(f.Name, "ppt/slides/") {
			n, _ := strconv.Atoi(m[1])
			slides = append(slides, slide{n: n, name: f.Name})
		}
	}
	if len(slides) == 0 {
		return "", fmt.Errorf("not a PowerPoint document: no slides found")
	}
	// Numeric order, not lexical — otherwise slide10 sorts before slide2.
	sort.Slice(slides, func(i, j int) bool { return slides[i].n < slides[j].n })

	var out []string
	for _, s := range slides {
		body, err := zipEntry(zr, s.name)
		if err != nil {
			continue
		}
		lines, err := slideBlocks(body)
		if err != nil {
			return "", err
		}
		out = append(out, fmt.Sprintf("## Slide %d", s.n))
		out = append(out, lines...)
	}
	out = append(out, extractCharts(zr)...)
	return joinBlocks(out), nil
}

// slideBlocks pulls a slide's text in reading order, keeping tables as tables.
//
// Pulling every <a:t> flat — which is what this used to do — turns a 4x3 table
// into twelve loose strings: the row and column a number belonged to is gone,
// and "8" no longer means "8 of the ESC controller". Tables are where slides
// put their densest facts, so they are worth the walk.
func slideBlocks(body []byte) ([]string, error) {
	var out []string
	var rows [][]string
	var row []string
	var cell strings.Builder
	var inTable, inCell bool

	dec := xml.NewDecoder(bytes.NewReader(body))
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			return out, nil
		}
		if err != nil {
			return nil, fmt.Errorf("malformed Office XML: %w", err)
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "tbl":
				inTable, rows = true, nil
			case "tr":
				row = nil
			case "tc":
				inCell = true
				cell.Reset()
			case "t":
				var s string
				if dec.DecodeElement(&s, &t) != nil {
					continue
				}
				switch {
				case inCell:
					cell.WriteString(s)
				case !inTable:
					if s = strings.TrimSpace(s); s != "" {
						out = append(out, s)
					}
				}
			}
		case xml.EndElement:
			switch t.Name.Local {
			case "tc":
				inCell = false
				row = append(row, cell.String())
			case "tr":
				rows = append(rows, row)
			case "tbl":
				inTable = false
				if md := markdownTable(rows); md != "" {
					out = append(out, md)
				}
			}
		}
	}
}

// textRuns pulls the character data of every element with the given local name,
// one entry per element, skipping blanks.
func textRuns(body []byte, local string) ([]string, error) {
	var out []string
	dec := xml.NewDecoder(bytes.NewReader(body))
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			return out, nil
		}
		if err != nil {
			return nil, fmt.Errorf("malformed Office XML: %w", err)
		}
		if se, ok := tok.(xml.StartElement); ok && se.Name.Local == local {
			var s string
			if err := dec.DecodeElement(&s, &se); err == nil {
				if s = strings.TrimSpace(s); s != "" {
					out = append(out, s)
				}
			}
		}
	}
}

// --- .xlsx ----------------------------------------------------------------

type xlsxWorkbook struct {
	Sheets []struct {
		Name string `xml:"name,attr"`
		ID   string `xml:"id,attr"`
	} `xml:"sheets>sheet"`
}

type xlsxRels struct {
	Rels []struct {
		ID     string `xml:"Id,attr"`
		Target string `xml:"Target,attr"`
	} `xml:"Relationship"`
}

// extractXlsx emits one `## SheetName` section per worksheet and one line per
// row, cells pipe-separated. Sheets are resolved through the relationship file
// rather than assuming sheet1.xml is the first tab, because Excel does not
// guarantee that once sheets have been reordered or deleted.
func extractXlsx(data []byte) (string, error) {
	zr, err := openZip(data)
	if err != nil {
		return "", err
	}
	wbRaw, err := zipEntry(zr, "xl/workbook.xml")
	if err != nil {
		return "", fmt.Errorf("not an Excel document: %w", err)
	}
	var wb xlsxWorkbook
	if err := xml.Unmarshal(wbRaw, &wb); err != nil {
		return "", fmt.Errorf("malformed Excel workbook: %w", err)
	}

	target := map[string]string{}
	if relRaw, err := zipEntry(zr, "xl/_rels/workbook.xml.rels"); err == nil {
		var rels xlsxRels
		if xml.Unmarshal(relRaw, &rels) == nil {
			for _, r := range rels.Rels {
				target[r.ID] = r.Target
			}
		}
	}

	shared := sharedStrings(zr)

	var out []string
	for i, sh := range wb.Sheets {
		name := target[sh.ID]
		if name == "" {
			name = fmt.Sprintf("worksheets/sheet%d.xml", i+1) // pre-rels fallback
		}
		name = "xl/" + strings.TrimPrefix(strings.TrimPrefix(name, "/xl/"), "/")
		body, err := zipEntry(zr, name)
		if err != nil {
			continue
		}
		rows, err := sheetRows(body, shared)
		if err != nil {
			return "", err
		}
		if len(rows) == 0 {
			continue // an empty tab contributes nothing but a stray heading
		}
		out = append(out, "## "+sh.Name)
		out = append(out, rows...)
	}
	if len(out) == 0 {
		return "", fmt.Errorf("spreadsheet contains no readable cells")
	}
	out = append(out, extractCharts(zr)...)
	return joinBlocks(out), nil
}

func sharedStrings(zr *zip.Reader) []string {
	raw, err := zipEntry(zr, "xl/sharedStrings.xml")
	if err != nil {
		return nil
	}
	// Each <si> may hold several <t> runs (rich text); concatenate per <si>.
	var out []string
	dec := xml.NewDecoder(bytes.NewReader(raw))
	for {
		tok, err := dec.Token()
		if err != nil {
			return out
		}
		if se, ok := tok.(xml.StartElement); ok && se.Name.Local == "si" {
			var si struct {
				Runs []string `xml:"r>t"`
				Text string   `xml:"t"`
			}
			if dec.DecodeElement(&si, &se) == nil {
				if len(si.Runs) > 0 {
					out = append(out, strings.Join(si.Runs, ""))
				} else {
					out = append(out, si.Text)
				}
			}
		}
	}
}

func sheetRows(body []byte, shared []string) ([]string, error) {
	var out []string
	var cells []string
	var cellType string

	dec := xml.NewDecoder(bytes.NewReader(body))
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			return out, nil
		}
		if err != nil {
			return nil, fmt.Errorf("malformed worksheet XML: %w", err)
		}
		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "row":
				cells = cells[:0]
			case "c":
				cellType = attr(t, "t")
			case "v", "t":
				var s string
				if err := dec.DecodeElement(&s, &t); err != nil {
					continue
				}
				if cellType == "s" && t.Name.Local == "v" {
					// Shared-string cell: <v> is an index into the table.
					if i, err := strconv.Atoi(strings.TrimSpace(s)); err == nil && i >= 0 && i < len(shared) {
						s = shared[i]
					}
				}
				if s = strings.TrimSpace(s); s != "" {
					cells = append(cells, s)
				}
			}
		case xml.EndElement:
			if t.Name.Local == "row" && len(cells) > 0 {
				out = append(out, strings.Join(cells, " | "))
				cells = nil
			}
		}
	}
}

// joinBlocks separates blocks by blank lines so the chunker's paragraph packing
// has something to pack, and collapses runs of blanks.
func joinBlocks(blocks []string) string {
	var b strings.Builder
	for _, s := range blocks {
		if s = strings.TrimRight(s, " \t"); s == "" {
			continue
		}
		b.WriteString(s)
		b.WriteString("\n\n")
	}
	return strings.TrimSpace(b.String())
}
