package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func hashOf(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}

func writeTokens(t *testing.T, ps []Principal) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokens.json")
	blob, err := json.Marshal(ps)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, blob, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// The bug this whole file exists for: ACL tags used to arrive in the request
// body, so a caller granted nothing could ask for anything by naming it.
// Whatever the body says must be discarded, not merged.
func TestGuardReplacesClientSuppliedACL(t *testing.T) {
	path := writeTokens(t, []Principal{
		{Name: "reader", Hash: hashOf("t-reader"), ACL: []string{"public"}},
	})
	store, err := LoadTokens(path)
	if err != nil {
		t.Fatal(err)
	}
	a := &App{Tokens: store}

	var got []string
	h := a.guard(false, func(w http.ResponseWriter, r *http.Request, p Principal) {
		var req QueryRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		req.ACL = p.Tags()
		got = req.ACL
		w.WriteHeader(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodPost, "/query",
		strings.NewReader(`{"question":"?","acl":["hr-confidential","finance"]}`))
	req.Header.Set("Authorization", "Bearer t-reader")
	rec := httptest.NewRecorder()
	h(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if len(got) != 1 || got[0] != "public" {
		t.Errorf("acl reaching retrieval = %v, want [public] — the body's tags must not survive", got)
	}
}

func TestGuardRejectsBadAndMissingTokens(t *testing.T) {
	path := writeTokens(t, []Principal{
		{Name: "reader", Hash: hashOf("t-reader"), ACL: []string{"*"}},
		{Name: "boss", Hash: hashOf("t-boss"), ACL: []string{"*"}, Admin: true},
	})
	store, err := LoadTokens(path)
	if err != nil {
		t.Fatal(err)
	}
	a := &App{Tokens: store}

	call := func(admin bool, header string) int {
		h := a.guard(admin, func(w http.ResponseWriter, r *http.Request, p Principal) {
			w.WriteHeader(http.StatusOK)
		})
		req := httptest.NewRequest(http.MethodGet, "/documents", nil)
		if header != "" {
			req.Header.Set("Authorization", header)
		}
		rec := httptest.NewRecorder()
		h(rec, req)
		return rec.Code
	}

	for _, tc := range []struct {
		name   string
		admin  bool
		header string
		want   int
	}{
		{"no header", false, "", http.StatusUnauthorized},
		{"unknown token", false, "Bearer nope", http.StatusUnauthorized},
		{"raw hash is not the token", false, "Bearer " + hashOf("t-reader"), http.StatusUnauthorized},
		{"wrong scheme", false, "Basic t-reader", http.StatusUnauthorized},
		{"lowercase scheme accepted", false, "bearer t-reader", http.StatusOK},
		{"valid reader", false, "Bearer t-reader", http.StatusOK},
		{"reader on admin route", true, "Bearer t-reader", http.StatusForbidden},
		{"admin on admin route", true, "Bearer t-boss", http.StatusOK},
	} {
		if got := call(tc.admin, tc.header); got != tc.want {
			t.Errorf("%s: status = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// Unrestricted has to be something a principal asked for. An empty list is the
// config slip that would otherwise read as "no filter" and hand over the corpus.
func TestLoadTokensFailsClosed(t *testing.T) {
	for _, tc := range []struct {
		name string
		ps   []Principal
		want string
	}{
		{"empty acl", []Principal{{Name: "x", Hash: hashOf("a")}}, "grants no access"},
		{"no principals", []Principal{}, "no principals"},
		{"short hash", []Principal{{Name: "x", Hash: "abc", ACL: []string{"*"}}}, "64 hex"},
		{"no name", []Principal{{Name: " ", Hash: hashOf("a"), ACL: []string{"*"}}}, "no name"},
		{"shared token", []Principal{
			{Name: "x", Hash: hashOf("a"), ACL: []string{"*"}},
			{Name: "y", Hash: hashOf("a"), ACL: []string{"*"}},
		}, "share a token"},
	} {
		_, err := LoadTokens(writeTokens(t, tc.ps))
		if err == nil {
			t.Errorf("%s: loaded without error; want a refusal", tc.name)
			continue
		}
		if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: error = %q, want it to mention %q", tc.name, err, tc.want)
		}
	}
}

func TestTagsOnlyNilForExplicitWildcard(t *testing.T) {
	if got := (Principal{ACL: []string{"*"}}).Tags(); got != nil {
		t.Errorf(`["*"].Tags() = %v, want nil (unrestricted)`, got)
	}
	if got := (Principal{ACL: []string{"public"}}).Tags(); len(got) != 1 || got[0] != "public" {
		t.Errorf("Tags() = %v, want [public]", got)
	}
	// The dangerous case: nil tags mean "no filter" downstream, so an empty ACL
	// must never produce them. LoadTokens rejects it, and this is the belt.
	if got := (Principal{}).Tags(); len(got) != 0 {
		t.Errorf("empty ACL Tags() = %v", got)
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

func TestMintedTokenResolvesToItsPrincipal(t *testing.T) {
	token, hash, err := mintToken()
	if err != nil {
		t.Fatal(err)
	}
	if len(token) != 64 || token == hash {
		t.Fatalf("mintToken returned token=%q hash=%q", token, hash)
	}
	store, err := LoadTokens(writeTokens(t, []Principal{
		{Name: "minted", Hash: hash, ACL: []string{"*"}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if p, ok := store.Lookup(token); !ok || p.Name != "minted" {
		t.Errorf("Lookup(minted token) = %+v, %v; want the minted principal", p, ok)
	}
	if _, ok := store.Lookup(""); ok {
		t.Error("empty token resolved to a principal")
	}
}
