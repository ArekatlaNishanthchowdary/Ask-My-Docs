package main

import "testing"

// The .env.example convention is to document a setting's options inline, so
// uncommenting a line brings the comment with it. Before this was stripped,
// OPENAI_REASONING_EFFORT=low became "low   # reasoning models only (gpt-oss)"
// and every query failed 400 with a message naming a value the file appeared
// to contain — a parser bug wearing a configuration bug's clothes.
func TestDropInlineComment(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"low   # reasoning models only (gpt-oss)", "low"},
		{"low\t# tab-separated", "low"},
		{"low", "low"},
		{"", ""},
		// Multi-word values: the comment is still the last thing on the line.
		{"detailed thinking off  # nemotron", "detailed thinking off"},
		// A # that is part of the value has no whitespace before it.
		{"pa55#word", "pa55#word"},
		{"#000000", "#000000"},
		// Quoted values are verbatim — the escape hatch for a trailing " #".
		{`"low # literal"`, `"low # literal"`},
		{`'keep # this'`, `'keep # this'`},
		// Unterminated quote: return as-is rather than guessing.
		{`"unclosed # x`, `"unclosed # x`},
	} {
		if got := dropInlineComment(tc.in); got != tc.want {
			t.Errorf("dropInlineComment(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
