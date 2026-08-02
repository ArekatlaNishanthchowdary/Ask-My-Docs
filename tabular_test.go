package main

import (
	"strings"
	"testing"
)

func TestExtractCSV(t *testing.T) {
	// Every trap a real export contains: a quoted comma, an embedded newline,
	// a doubled quote, a ragged row, a UTF-8 BOM, and a literal pipe.
	raw := "\xEF\xBB\xBFcomponent,qty,notes\n" +
		"\"Frame F450, with PCB\",4,FRAME\n" +
		"\"A2212 motor\",4,\"1000kv\nbrushless\"\n" +
		"\"ESC \"\"Simcon\"\"\",8,\n" +
		"Landing gear,8\n"

	got, err := extractCSV([]byte(raw), ',')
	if err != nil {
		t.Fatalf("extractCSV: %v", err)
	}
	for _, want := range []string{
		"| component | qty | notes |",
		"| --- | --- | --- |",
		"| Frame F450, with PCB | 4 | FRAME |", // quoted comma survives
		`| ESC "Simcon" | 8 |`,                 // doubled quote unescaped
		"| Landing gear | 8 |  |",              // ragged row padded
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	// The BOM must not end up glued to the first header cell.
	if strings.Contains(got, "\ufeff") {
		t.Errorf("BOM leaked into output:\n%s", got)
	}
	// An embedded newline must not break the row into two table rows.
	if n := strings.Count(got, "\n"); n != 5 {
		t.Errorf("got %d newlines, want 5 (6 lines: header+sep+4 rows):\n%s", n, got)
	}
}

func TestExtractCSVTabs(t *testing.T) {
	got, err := extractCSV([]byte("a\tb\n1\t2\n"), '\t')
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "| a | b |") || !strings.Contains(got, "| 1 | 2 |") {
		t.Errorf("tsv not parsed:\n%s", got)
	}
}

func TestMarkdownTableEscapesPipes(t *testing.T) {
	got := markdownTable([][]string{{"a|b", "c"}, {"1", "2"}})
	if !strings.Contains(got, `a\|b`) {
		t.Errorf("unescaped pipe would corrupt every later column:\n%s", got)
	}
}
