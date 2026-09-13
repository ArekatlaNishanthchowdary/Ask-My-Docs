package rag

import (
	"hash/fnv"
	"math"
	"sort"
	"strings"
	"unicode"

	"askmydocs/internal/qdrant"
)

type Scored struct {
	Index int
	Score float32
}

// Tokenize is the lexical half of hybrid retrieval: lowercase, split on
// non-alphanumerics, drop single characters. Deliberately dumb — no stemming,
// no stopword list — because exact-match precision on identifiers, error codes
// and part numbers is the whole point of the sparse leg.
func Tokenize(s string) []string {
	fields := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	out := fields[:0]
	for _, f := range fields {
		if len(f) > 1 {
			out = append(out, f)
		}
	}
	return out
}

// SparseEncode builds a term-frequency sparse vector. Qdrant's "idf" modifier
// supplies the inverse-document-frequency weighting server-side, which is why
// this can be a pure function of a single string.
func SparseEncode(s string) qdrant.SparseVec {
	tf := map[uint32]float32{}
	for _, tok := range Tokenize(s) {
		h := fnv.New32a()
		h.Write([]byte(tok))
		tf[h.Sum32()]++
	}
	v := qdrant.SparseVec{Indices: make([]uint32, 0, len(tf)), Values: make([]float32, 0, len(tf))}
	for idx := range tf {
		v.Indices = append(v.Indices, idx)
	}
	sort.Slice(v.Indices, func(i, j int) bool { return v.Indices[i] < v.Indices[j] })
	for _, idx := range v.Indices {
		v.Values = append(v.Values, tf[idx])
	}
	return v
}

func cosine(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(float64(dot) / (math.Sqrt(float64(na)) * math.Sqrt(float64(nb))))
}

// truncate shortens a string for error messages. Duplicated in qdrant and
// providers too — three lines each, not worth a shared package.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
