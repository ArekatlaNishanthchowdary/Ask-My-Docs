package main

import (
	"fmt"
	"strings"
	"testing"
)

// minimalPDF builds a one-page PDF with two text lines at given Y positions,
// using the standard PDF user space where Y increases upward. Offsets are
// computed as objects are appended, because a wrong xref makes the file
// unreadable and the test would be measuring the fixture, not the code.
func minimalPDF(topY, bottomY int, topText, bottomText string) []byte {
	content := fmt.Sprintf("BT /F1 12 Tf 20 %d Td (%s) Tj ET\nBT /F1 12 Tf 20 %d Td (%s) Tj ET\n",
		topY, topText, bottomY, bottomText)

	objs := []string{
		"<</Type/Catalog/Pages 2 0 R>>",
		"<</Type/Pages/Kids[3 0 R]/Count 1>>",
		"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 800]/Contents 4 0 R" +
			"/Resources<</Font<</F1 5 0 R>>>>>>",
		fmt.Sprintf("<</Length %d>>stream\n%sendstream", len(content), content),
		"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
	}

	var b strings.Builder
	b.WriteString("%PDF-1.4\n")
	offsets := make([]int, len(objs)+1)
	for i, o := range objs {
		offsets[i+1] = b.Len()
		fmt.Fprintf(&b, "%d 0 obj\n%s\nendobj\n", i+1, o)
	}
	xref := b.Len()
	fmt.Fprintf(&b, "xref\n0 %d\n0000000000 65535 f \n", len(objs)+1)
	for i := 1; i <= len(objs); i++ {
		fmt.Fprintf(&b, "%010d 00000 n \n", offsets[i])
	}
	fmt.Fprintf(&b, "trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n", len(objs)+1, xref)
	return []byte(b.String())
}

// In standard PDF user space the top of the page has the HIGHER Y. A
// Chrome-printed PDF reports the opposite (title 37, footer 224), so the
// extractor must not hardcode either direction — it derives it. This covers
// the Y-up half; the Y-down half was verified against a real Chrome export.
func TestPDFReadingOrderYUp(t *testing.T) {
	got, err := extractPDF(minimalPDF(700, 100, "TITLE LINE", "FOOTER LINE"))
	if err != nil {
		t.Fatalf("extractPDF: %v", err)
	}
	ti, fi := strings.Index(got, "TITLE LINE"), strings.Index(got, "FOOTER LINE")
	if ti < 0 || fi < 0 {
		t.Fatalf("missing text in:\n%s", got)
	}
	if ti > fi {
		t.Errorf("page extracted bottom-first:\n%s", got)
	}
	if !strings.Contains(got, "## Page 1") {
		t.Errorf("missing page heading:\n%s", got)
	}
}

// Failure has to be clean: these run on uploaded bytes, and the PDF parser
// panics on malformed structures rather than returning an error.
func TestPDFBadInputDoesNotPanic(t *testing.T) {
	for name, data := range map[string][]byte{
		"not a pdf":     []byte("# just markdown\n"),
		"truncated":     []byte("%PDF-1.4\n1 0 obj\n<</Type/Catalog"),
		"empty":         {},
		"header only":   []byte("%PDF-1.7\n%%EOF\n"),
		"random binary": {0x00, 0xFF, 0x10, 0x9A, 0x00, 0x01},
	} {
		t.Run(name, func(t *testing.T) {
			out, err := extractPDF(data)
			if err == nil {
				t.Errorf("want an error, got %q", out)
			}
		})
	}
}

// A PDF of scanned images has pages but no text layer. Indexing it as an empty
// document hides the real problem, so it must say which problem it is.
func TestPDFNoTextLayerExplainsItself(t *testing.T) {
	_, err := extractPDF(minimalPDF(700, 100, "", ""))
	if err == nil {
		t.Fatal("want an error for a page with no text")
	}
	if !strings.Contains(err.Error(), "OCR") {
		t.Errorf("error should point at OCR, got: %v", err)
	}
}
