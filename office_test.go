package main

import (
	"archive/zip"
	"bytes"
	"strings"
	"testing"
)

// Office files are ZIP+XML, so the fixtures are built in memory rather than
// checked in as binaries — the test stays readable and diffable.
func buildZip(t *testing.T, entries map[string]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for name, body := range entries {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatalf("zip create %s: %v", name, err)
		}
		if _, err := w.Write([]byte(body)); err != nil {
			t.Fatalf("zip write %s: %v", name, err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("zip close: %v", err)
	}
	return buf.Bytes()
}

func TestExtractDocxPreservesHeadingsAndTables(t *testing.T) {
	const doc = `<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
  <w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Employee Handbook</w:t></w:r></w:p>
  <w:p><w:r><w:t xml:space="preserve">Applies to </w:t></w:r><w:r><w:t>all staff.</w:t></w:r></w:p>
  <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Leave</w:t></w:r></w:p>
  <w:p><w:r><w:t>Annual leave is 25 days.</w:t></w:r></w:p>
  <w:tbl>
    <w:tr>
      <w:tc><w:p><w:r><w:t>Band</w:t></w:r></w:p></w:tc>
      <w:tc><w:p><w:r><w:t>Days</w:t></w:r></w:p></w:tc>
    </w:tr>
    <w:tr>
      <w:tc><w:p><w:r><w:t>Senior</w:t></w:r></w:p></w:tc>
      <w:tc><w:p><w:r><w:t>30</w:t></w:r></w:p></w:tc>
    </w:tr>
  </w:tbl>
</w:body></w:document>`

	got, err := extractDocx(buildZip(t, map[string]string{"word/document.xml": doc}))
	if err != nil {
		t.Fatalf("extractDocx: %v", err)
	}

	// Word heading styles must become real markdown headings, because that is
	// what the chunker splits on.
	for _, want := range []string{"# Employee Handbook", "## Leave"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing heading %q in:\n%s", want, got)
		}
	}
	// Runs inside one paragraph are a single line, not fragments.
	if !strings.Contains(got, "Applies to all staff.") {
		t.Errorf("adjacent runs were not joined:\n%s", got)
	}
	// Table rows keep their column structure.
	if !strings.Contains(got, "Band | Days") || !strings.Contains(got, "Senior | 30") {
		t.Errorf("table rows lost their shape:\n%s", got)
	}

	// The extracted markdown must actually chunk along those headings.
	chunks := ChunkDoc(got, 1600, 200)
	var sections []string
	for _, c := range chunks {
		sections = append(sections, c.Section)
	}
	joined := strings.Join(sections, ",")
	if !strings.Contains(joined, "Employee Handbook > Leave") {
		t.Errorf("headings did not produce a section path, got sections: %v", sections)
	}
}

func TestExtractPptxOrdersSlidesNumerically(t *testing.T) {
	slide := func(text string) string {
		return `<?xml version="1.0"?><p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">` +
			`<a:t>` + text + `</a:t></p:sld>`
	}
	got, err := extractPptx(buildZip(t, map[string]string{
		"ppt/slides/slide1.xml":  slide("Q3 Review"),
		"ppt/slides/slide2.xml":  slide("Revenue up 12 percent"),
		"ppt/slides/slide10.xml": slide("Closing remarks"),
	}))
	if err != nil {
		t.Fatalf("extractPptx: %v", err)
	}
	// slide10 must come last; lexical sorting would place it second.
	i1 := strings.Index(got, "Q3 Review")
	i2 := strings.Index(got, "Revenue up 12 percent")
	i10 := strings.Index(got, "Closing remarks")
	if !(i1 < i2 && i2 < i10) {
		t.Errorf("slides out of order (1:%d 2:%d 10:%d):\n%s", i1, i2, i10, got)
	}
	if !strings.Contains(got, "## Slide 10") {
		t.Errorf("slide boundaries missing:\n%s", got)
	}
}

func TestExtractXlsxResolvesSharedStringsAndSheetNames(t *testing.T) {
	files := map[string]string{
		"xl/workbook.xml": `<?xml version="1.0"?>
<workbook xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Pricing" sheetId="1" r:id="rId7"/></sheets></workbook>`,
		// rId7 -> sheet3.xml: the mapping must be followed, not guessed from
		// the filename, or reordered workbooks read the wrong tab.
		"xl/_rels/workbook.xml.rels": `<?xml version="1.0"?>
<Relationships><Relationship Id="rId7" Target="worksheets/sheet3.xml"/></Relationships>`,
		"xl/sharedStrings.xml": `<?xml version="1.0"?>
<sst><si><t>Widget</t></si><si><t>Gadget</t></si></sst>`,
		"xl/worksheets/sheet3.xml": `<?xml version="1.0"?>
<worksheet><sheetData>
<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1"><v>19.99</v></c></row>
<row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>24.50</v></c></row>
</sheetData></worksheet>`,
		"xl/worksheets/sheet1.xml": `<?xml version="1.0"?>
<worksheet><sheetData><row r="1"><c r="A1"><v>WRONG SHEET</v></c></row></sheetData></worksheet>`,
	}
	got, err := extractXlsx(buildZip(t, files))
	if err != nil {
		t.Fatalf("extractXlsx: %v", err)
	}
	if strings.Contains(got, "WRONG SHEET") {
		t.Errorf("resolved the sheet by filename instead of the relationship id:\n%s", got)
	}
	if !strings.Contains(got, "## Pricing") {
		t.Errorf("sheet name did not become a heading:\n%s", got)
	}
	// t="s" cells are indexes into sharedStrings, not literal numbers.
	if !strings.Contains(got, "Widget | 19.99") || !strings.Contains(got, "Gadget | 24.5") {
		t.Errorf("shared strings not resolved:\n%s", got)
	}
}

func TestLoadDocumentTextRejectsBadInput(t *testing.T) {
	cases := []struct {
		name string
		data []byte
		want string
	}{
		{"report.doc", []byte("anything"), "legacy binary format"},
		{"deck.ppt", []byte("anything"), "legacy binary format"},
		{"book.xls", []byte("anything"), "legacy binary format"},
		{"broken.docx", []byte("this is not a zip"), "not a readable Office file"},
		{"notes.md", []byte{0xff, 0xfe, 0x00}, "not valid UTF-8"},
		{"thing.rtf", []byte("x"), "unsupported type"},
	}
	for _, c := range cases {
		if _, err := LoadDocumentText(c.name, c.data); err == nil {
			t.Errorf("%s: expected an error", c.name)
		} else if !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error = %q, want it to mention %q", c.name, err, c.want)
		}
	}

	// A valid zip that is not a Word document should say so, rather than
	// producing empty text that silently indexes as nothing.
	empty := buildZip(t, map[string]string{"docProps/app.xml": "<app/>"})
	if _, err := LoadDocumentText("x.docx", empty); err == nil {
		t.Error("a zip without word/document.xml should be rejected")
	}

	// Plain text passes through untouched.
	if got, err := LoadDocumentText("a.md", []byte("# Hi\n\nthere")); err != nil || got != "# Hi\n\nthere" {
		t.Errorf("markdown passthrough = %q, %v", got, err)
	}
}

func TestSafeCorpusPathAcceptsOfficeAndRejectsLegacy(t *testing.T) {
	if _, err := SafeCorpusPath("corpus", "Report.docx"); err != nil {
		t.Errorf("Report.docx should be accepted: %v", err)
	}
	if _, err := SafeCorpusPath("corpus", "../../x.pptx"); err != nil {
		t.Errorf("traversal should be stripped and accepted, got: %v", err)
	}
	p, err := SafeCorpusPath("corpus", "../../x.pptx")
	if err == nil && !strings.HasSuffix(p, "x.pptx") {
		t.Errorf("path %q should end in the bare filename", p)
	}
	if _, err := SafeCorpusPath("corpus", "old.doc"); err == nil {
		t.Error("legacy .doc should be rejected with guidance")
	}
}
