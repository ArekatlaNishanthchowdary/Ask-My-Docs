package qdrant

import (
	"strings"
	"testing"
)

func TestPointIDIsStableAndUUIDShaped(t *testing.T) {
	a, b := PointID("policy.md#3"), PointID("policy.md#3")
	if a != b {
		t.Fatal("point IDs must be deterministic or re-ingest duplicates chunks")
	}
	if a == PointID("policy.md#4") {
		t.Fatal("distinct chunks collided")
	}
	parts := strings.Split(a, "-")
	if len(parts) != 5 || len(parts[0]) != 8 || len(parts[4]) != 12 {
		t.Errorf("point ID %q is not UUID-shaped; Qdrant will reject it", a)
	}
}

// A restricted caller's document listing is an access check, so the filter has
// to carry their tags. Without the acl clause the count answers "how many
// chunks exist", which lists filenames they cannot read.
func TestDocIDFilterCarriesACL(t *testing.T) {
	must := DocIDFilter("secret.docx", []string{"hr"})["must"].([]map[string]any)
	if len(must) != 2 {
		t.Fatalf("filter has %d clauses, want doc_id and acl", len(must))
	}
	if must[1]["key"] != "acl" {
		t.Errorf("second clause keys on %v, want acl", must[1]["key"])
	}
	if got := DocIDFilter("x", nil)["must"].([]map[string]any); len(got) != 1 {
		t.Errorf("unrestricted filter has %d clauses, want just doc_id", len(got))
	}
}
