package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTEIImageFor(t *testing.T) {
	for _, tc := range []struct {
		cap   float64
		image string
		exact bool
	}{
		{8.9, "89-1.7", true},     // RTX 40xx
		{8.6, "86-1.7", true},     // RTX 30xx
		{8.0, "1.7", true},        // A100
		{7.5, "turing-1.7", true}, // RTX 20xx
		{9.0, "hopper-1.7", true}, // H100
		{8.7, "86-1.7", true},     // between tags: round down, never up
		{7.0, cpuImage, false},    // V100 predates TEI's kernels
		{6.1, cpuImage, false},    // GTX 10xx
		{12.0, cpuImage, false},   // Blackwell, newer than any 1.7 tag
		{0, cpuImage, false},      // no GPU
	} {
		got, exact := teiImageFor(tc.cap)
		if got != tc.image || exact != tc.exact {
			t.Errorf("teiImageFor(%.1f) = (%q,%v), want (%q,%v)", tc.cap, got, exact, tc.image, tc.exact)
		}
	}
}

func TestUpsertEnv(t *testing.T) {
	p := filepath.Join(t.TempDir(), ".env")
	os.WriteFile(p, []byte("PROVIDER=ollama\nTEI_IMAGE=cpu-1.7\nADDR=:8080\n"), 0o644)

	if err := upsertEnv(p, "TEI_IMAGE", "89-1.7"); err != nil {
		t.Fatal(err)
	}
	if err := upsertEnv(p, "COMPOSE_FILE", "a.yml:b.yml"); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(p)
	s := string(got)

	// Replaced in place, not appended — a duplicate key would be shadowed by
	// the first occurrence and the change would silently do nothing.
	if strings.Count(s, "TEI_IMAGE=") != 1 {
		t.Errorf("TEI_IMAGE appears %d times, want 1:\n%s", strings.Count(s, "TEI_IMAGE="), s)
	}
	if !strings.Contains(s, "TEI_IMAGE=89-1.7") {
		t.Errorf("value not updated:\n%s", s)
	}
	if !strings.Contains(s, "COMPOSE_FILE=a.yml:b.yml") {
		t.Errorf("new key not appended:\n%s", s)
	}
	// Unrelated settings survive.
	for _, keep := range []string{"PROVIDER=ollama", "ADDR=:8080"} {
		if !strings.Contains(s, keep) {
			t.Errorf("clobbered %q:\n%s", keep, s)
		}
	}
}
