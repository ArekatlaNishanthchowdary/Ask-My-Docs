package main

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
)

// Charts are the one part of an Office document whose meaning is usually
// invisible to text extraction: the bars carry the numbers and the text layer
// has nothing. But a chart is not only pixels — OOXML stores a *cache* of the
// series it was plotted from (c:cat categories, c:val values) inside
// charts/chartN.xml, precisely so the chart can render without reopening its
// source workbook.
//
// That cache is the data, so a chart becomes a table with no image processing
// and no dependency. What this cannot do is read a chart that was pasted in as
// a picture — those are pixels and need a vision model. See README.
var chartPart = regexp.MustCompile(`^(word|ppt|xl)/charts/chart\d+\.xml$`)

type chartSeries struct {
	Name string
	Cats []string
	Vals []string
}

type chartData struct {
	Title  string
	Kind   string
	Series []chartSeries
}

// extractCharts renders every chart in the archive as a markdown table.
// Charts are appended as their own section rather than placed inline: the
// drawing-to-chart relationship chain differs per format, and a chart under the
// wrong heading is worse than one under an honest "Charts" heading.
func extractCharts(zr *zip.Reader) []string {
	var names []string
	for _, f := range zr.File {
		if chartPart.MatchString(f.Name) {
			names = append(names, f.Name)
		}
	}
	if len(names) == 0 {
		return nil
	}
	sort.Strings(names)

	var out []string
	for _, n := range names {
		body, err := zipEntry(zr, n)
		if err != nil {
			continue
		}
		c, err := parseChart(body)
		if err != nil || len(c.Series) == 0 {
			continue
		}
		if t := c.render(); t != "" {
			out = append(out, t)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return append([]string{"## Charts"}, out...)
}

func (c chartData) render() string {
	title := c.Title
	if title == "" {
		title = "Untitled chart"
	}
	head := "### " + title
	if c.Kind != "" {
		head += " (" + c.Kind + ")"
	}

	// Categories are shared across series, so they become the first column and
	// each series becomes one more — the shape someone would draw by hand.
	var cats []string
	for _, s := range c.Series {
		if len(s.Cats) > len(cats) {
			cats = s.Cats
		}
	}
	header := []string{""}
	for i, s := range c.Series {
		n := s.Name
		if n == "" {
			n = fmt.Sprintf("Series %d", i+1)
		}
		header = append(header, n)
	}
	rows := [][]string{header}

	n := len(cats)
	for _, s := range c.Series {
		if len(s.Vals) > n {
			n = len(s.Vals)
		}
	}
	for i := 0; i < n; i++ {
		label := ""
		if i < len(cats) {
			label = cats[i]
		}
		row := []string{label}
		for _, s := range c.Series {
			v := ""
			if i < len(s.Vals) {
				v = s.Vals[i]
			}
			row = append(row, v)
		}
		rows = append(rows, row)
	}
	return head + "\n\n" + markdownTable(rows)
}

// parseChart walks the chart part as a token stream. Series live under whichever
// plot element the chart uses (c:barChart, c:lineChart, c:pieChart …), and
// encoding/xml struct tags cannot express "any of those", so a walk it is —
// the same reason extractDocx walks rather than unmarshals.
func parseChart(body []byte) (chartData, error) {
	var c chartData
	dec := xml.NewDecoder(bytes.NewReader(body))

	var stack []string
	in := func(local string) bool {
		for _, s := range stack {
			if s == local {
				return true
			}
		}
		return false
	}

	var cur *chartSeries
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return c, fmt.Errorf("malformed chart XML: %w", err)
		}
		switch t := tok.(type) {
		case xml.StartElement:
			name := t.Name.Local
			stack = append(stack, name)

			switch {
			case name == "ser":
				c.Series = append(c.Series, chartSeries{})
				cur = &c.Series[len(c.Series)-1]

			case c.Kind == "" && strings.HasSuffix(name, "Chart") && in("plotArea"):
				// "barChart" -> "bar", "chart3D" variants keep their suffix.
				c.Kind = strings.TrimSuffix(name, "Chart")

			case name == "t" && in("title") && !in("plotArea"):
				// Title text is drawingml runs, not chart values.
				var s string
				if dec.DecodeElement(&s, &t) == nil {
					if s = strings.TrimSpace(s); s != "" {
						c.Title = strings.TrimSpace(c.Title + " " + s)
					}
				}
				stack = stack[:len(stack)-1]

			case name == "v" && cur != nil:
				var s string
				if dec.DecodeElement(&s, &t) != nil {
					stack = stack[:len(stack)-1]
					continue
				}
				s = strings.TrimSpace(s)
				switch {
				case in("tx"):
					if cur.Name == "" {
						cur.Name = s
					}
				case in("cat"):
					cur.Cats = append(cur.Cats, s)
				case in("val"):
					cur.Vals = append(cur.Vals, s)
				}
				stack = stack[:len(stack)-1]
			}
		case xml.EndElement:
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
			if t.Name.Local == "ser" {
				cur = nil
			}
		}
	}
	return c, nil
}
