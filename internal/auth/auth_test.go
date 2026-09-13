package auth

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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
