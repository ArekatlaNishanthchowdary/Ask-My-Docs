package main

import (
	"archive/zip"
	"bytes"
	"strings"
	"testing"
)

// A minimal but structurally real chart part: title, bar plot, two series with
// cached categories and values. This is the shape Word/PowerPoint/Excel write.
const chartXML = `<?xml version="1.0"?>
<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
              xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
 <c:chart>
  <c:title><c:tx><c:rich><a:p><a:r><a:t>Revenue by Quarter</a:t></a:r></a:p></c:rich></c:tx></c:title>
  <c:plotArea>
   <c:barChart>
    <c:ser>
     <c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>2025</c:v></c:pt></c:strCache></c:strRef></c:tx>
     <c:cat><c:strRef><c:strCache>
       <c:pt idx="0"><c:v>Q1</c:v></c:pt><c:pt idx="1"><c:v>Q2</c:v></c:pt>
     </c:strCache></c:strRef></c:cat>
     <c:val><c:numRef><c:numCache>
       <c:pt idx="0"><c:v>120</c:v></c:pt><c:pt idx="1"><c:v>150</c:v></c:pt>
     </c:numCache></c:numRef></c:val>
    </c:ser>
    <c:ser>
     <c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>2026</c:v></c:pt></c:strCache></c:strRef></c:tx>
     <c:cat><c:strRef><c:strCache>
       <c:pt idx="0"><c:v>Q1</c:v></c:pt><c:pt idx="1"><c:v>Q2</c:v></c:pt>
     </c:strCache></c:strRef></c:cat>
     <c:val><c:numRef><c:numCache>
       <c:pt idx="0"><c:v>200</c:v></c:pt><c:pt idx="1"><c:v>260</c:v></c:pt>
     </c:numCache></c:numRef></c:val>
    </c:ser>
   </c:barChart>
  </c:plotArea>
 </c:chart>
</c:chartSpace>`

func TestParseChart(t *testing.T) {
	c, err := parseChart([]byte(chartXML))
	if err != nil {
		t.Fatalf("parseChart: %v", err)
	}
	if c.Title != "Revenue by Quarter" {
		t.Errorf("title = %q", c.Title)
	}
	if c.Kind != "bar" {
		t.Errorf("kind = %q, want bar", c.Kind)
	}
	if len(c.Series) != 2 {
		t.Fatalf("got %d series, want 2", len(c.Series))
	}
	// The series name comes from c:tx and must not be swallowed into values —
	// that is the mistake that turns "2025" into a data point.
	if c.Series[0].Name != "2025" || c.Series[1].Name != "2026" {
		t.Errorf("series names = %q, %q", c.Series[0].Name, c.Series[1].Name)
	}
	if got := strings.Join(c.Series[0].Vals, ","); got != "120,150" {
		t.Errorf("series 1 values = %q", got)
	}
	if got := strings.Join(c.Series[0].Cats, ","); got != "Q1,Q2" {
		t.Errorf("categories = %q", got)
	}

	md := c.render()
	for _, want := range []string{
		"### Revenue by Quarter (bar)",
		"| 2025 | 2026 |",
		"| Q1 | 120 | 200 |",
		"| Q2 | 150 | 260 |",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("missing %q in:\n%s", want, md)
		}
	}
}

// A chart reaches the indexed text of the document that contains it.
func TestChartsReachDocumentText(t *testing.T) {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	add := func(name, body string) {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		w.Write([]byte(body))
	}
	add("word/document.xml", `<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:body><w:p><w:r><w:t>See the chart below.</w:t></w:r></w:p></w:body>
</w:document>`)
	add("word/charts/chart1.xml", chartXML)
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}

	got, err := LoadDocumentText("report.docx", buf.Bytes())
	if err != nil {
		t.Fatalf("LoadDocumentText: %v", err)
	}
	for _, want := range []string{"See the chart below.", "## Charts", "Revenue by Quarter", "| Q2 | 150 | 260 |"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
}
