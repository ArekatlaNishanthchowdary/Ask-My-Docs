package main

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"
)

// markdownTable renders rows as a GitHub-flavoured markdown table.
//
// Markdown rather than pipe-joined lines because the header survives: a bare
// "12 | 4 | FRAME" line means nothing once it is retrieved on its own, while a
// markdown table keeps the column names attached to the values for both the
// embedder and the model reading the chunk.
//
// Rows are padded to the widest row. Ragged input is normal in exported data,
// and a short row would otherwise shift every later cell under the wrong header.
func markdownTable(rows [][]string) string {
	if len(rows) == 0 {
		return ""
	}
	width := 0
	for _, r := range rows {
		if len(r) > width {
			width = len(r)
		}
	}
	if width == 0 {
		return ""
	}
	cell := func(r []string, i int) string {
		if i >= len(r) {
			return ""
		}
		// A literal pipe would terminate the cell and silently corrupt every
		// column to its right.
		return strings.TrimSpace(strings.NewReplacer("|", "\\|", "\n", " ", "\r", "").Replace(r[i]))
	}
	var b strings.Builder
	for ri, r := range rows {
		for i := 0; i < width; i++ {
			b.WriteString("| ")
			b.WriteString(cell(r, i))
			b.WriteByte(' ')
		}
		b.WriteString("|\n")
		if ri == 0 {
			for i := 0; i < width; i++ {
				b.WriteString("| --- ")
			}
			b.WriteString("|\n")
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// extractCSV turns delimiter-separated data into a markdown table.
//
// encoding/csv rather than strings.Split: quoted fields containing the
// delimiter, embedded newlines and doubled quotes are ordinary in real exports,
// and splitting on commas mangles all three.
//
// ponytail: the header is emitted once, so a CSV long enough to span several
// chunks leaves later chunks headerless. Repeat the header per chunk if that
// shows up in eval — it needs the chunker, not this function.
func extractCSV(data []byte, comma rune) (string, error) {
	// Excel writes UTF-8 with a BOM; left in place it becomes part of the first
	// header cell, so "id" silently stops matching "id".
	data = bytes.TrimPrefix(data, []byte{0xEF, 0xBB, 0xBF})
	if !utf8.Valid(data) {
		return "", fmt.Errorf("not valid UTF-8 text")
	}
	r := csv.NewReader(bytes.NewReader(data))
	r.Comma = comma
	// Ragged rows are common in hand-edited exports and are not worth refusing
	// the whole file over; markdownTable pads them.
	r.FieldsPerRecord = -1
	// Unescaped quotes inside unquoted fields (Measurement 6" pipe) are a hard
	// error otherwise.
	r.LazyQuotes = true

	var rows [][]string
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", fmt.Errorf("malformed delimited data: %w", err)
		}
		if len(rec) == 1 && strings.TrimSpace(rec[0]) == "" {
			continue // blank line
		}
		rows = append(rows, rec)
	}
	if len(rows) == 0 {
		return "", fmt.Errorf("no rows found")
	}
	return markdownTable(rows), nil
}
