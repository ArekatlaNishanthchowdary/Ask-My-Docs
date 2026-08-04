package main

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"slices"
	"strings"
)

// Authentication exists here for one reason: before it, the ACL tags that
// decide which chunks a query may retrieve arrived in the request body. A
// caller asking for documents tagged "hr-confidential" simply said so and got
// them. That is not access control, it is a filter the client picks.
//
// So the tags now come from the token, and the request body's acl field is
// discarded rather than merged. Everything else here — the token file, the
// admin flag — exists to make that one substitution possible.
//
// ponytail: static bearer tokens, not OIDC. The security property that matters
// is "the server decides what you may see", and a token file delivers it with
// stdlib and no identity provider to stand up. Swapping this for OIDC later
// means replacing Lookup; the query path never learns the difference.

// Principal is who a request is on behalf of, and what it may see.
type Principal struct {
	Name string `json:"name"`
	// Hash is the SHA-256 of the bearer token, hex encoded. The token itself is
	// never stored: a leaked token file should not be a set of usable
	// credentials. `ask-my-docs token` mints the pair.
	Hash string `json:"token_sha256"`
	// ACL lists the tags this principal may retrieve. ["*"] means unrestricted.
	// An empty list is rejected at load time rather than treated as "no
	// restriction" — the one interpretation that turns a config slip into a
	// silent data leak.
	ACL   []string `json:"acl"`
	Admin bool     `json:"admin,omitempty"`
}

// Tags renders the principal's ACL in the form the retrieval filter expects,
// where nil means unrestricted. Only a principal that explicitly asked for "*"
// can produce that nil.
func (p Principal) Tags() []string {
	if slices.Contains(p.ACL, "*") {
		return nil
	}
	return p.ACL
}

// TokenStore maps a token's SHA-256 to its principal.
type TokenStore map[string]Principal

// LoadTokens reads the principal file, rejecting anything that would weaken the
// boundary rather than accepting it and hoping. Every check here is one that
// fails at startup instead of at 3am.
func LoadTokens(path string) (TokenStore, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("AUTH_TOKENS_FILE: %w", err)
	}
	var ps []Principal
	if err := json.Unmarshal(raw, &ps); err != nil {
		return nil, fmt.Errorf("%s: %w (expected a JSON array of principals)", path, err)
	}
	if len(ps) == 0 {
		return nil, fmt.Errorf("%s: no principals — every request would be rejected", path)
	}
	store := make(TokenStore, len(ps))
	for i, p := range ps {
		switch {
		case strings.TrimSpace(p.Name) == "":
			return nil, fmt.Errorf("%s: principal %d has no name", path, i)
		case len(p.Hash) != 64:
			return nil, fmt.Errorf("%s: principal %q: token_sha256 must be 64 hex characters, got %d — mint one with `ask-my-docs token`",
				path, p.Name, len(p.Hash))
		case len(p.ACL) == 0:
			return nil, fmt.Errorf("%s: principal %q grants no access; use [\"*\"] for unrestricted",
				path, p.Name)
		}
		p.Hash = strings.ToLower(p.Hash)
		if _, err := hex.DecodeString(p.Hash); err != nil {
			return nil, fmt.Errorf("%s: principal %q: token_sha256 is not hex", path, p.Name)
		}
		if prev, dup := store[p.Hash]; dup {
			return nil, fmt.Errorf("%s: principals %q and %q share a token", path, prev.Name, p.Name)
		}
		store[p.Hash] = p
	}
	return store, nil
}

// Lookup resolves a presented bearer token. Lookup is by hash, so the stored
// value is never compared against a secret and there is no string comparison to
// time.
func (ts TokenStore) Lookup(token string) (Principal, bool) {
	if token == "" {
		return Principal{}, false
	}
	sum := sha256.Sum256([]byte(token))
	p, ok := ts[hex.EncodeToString(sum[:])]
	return p, ok
}

// bearerToken pulls the credential out of an Authorization header.
func bearerToken(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if len(h) < 7 || !strings.EqualFold(h[:7], "bearer ") {
		return ""
	}
	return strings.TrimSpace(h[7:])
}

// anonymous is the principal used when ALLOW_ANONYMOUS is set. It is
// deliberately all-powerful: an unauthenticated deployment has no basis to
// pretend otherwise, and a half-restricted anonymous user would only make the
// logs harder to read.
var anonymous = Principal{Name: "anonymous", ACL: []string{"*"}, Admin: true}

// guard authenticates a request and hands the handler the principal it is
// acting for. Handlers never see the raw token.
func (a *App) guard(requireAdmin bool, h func(http.ResponseWriter, *http.Request, Principal)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if a.Tokens == nil {
			h(w, r, anonymous)
			return
		}
		p, ok := a.Tokens.Lookup(bearerToken(r))
		if !ok {
			// The challenge header is what makes a browser or an HTTP client
			// retry with credentials instead of just reporting a failure.
			w.Header().Set("WWW-Authenticate", `Bearer realm="ask-my-docs"`)
			httpErr(w, http.StatusUnauthorized, "missing or invalid bearer token")
			return
		}
		if requireAdmin && !p.Admin {
			httpErr(w, http.StatusForbidden, "this endpoint requires an admin token")
			return
		}
		h(w, r, p)
	}
}

// mintToken returns a fresh token and its stored hash.
func mintToken() (token, hash string, err error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", "", err
	}
	token = hex.EncodeToString(buf)
	sum := sha256.Sum256([]byte(token))
	return token, hex.EncodeToString(sum[:]), nil
}

func cmdToken(args []string) error {
	name := "someone"
	if len(args) > 0 && strings.TrimSpace(args[0]) != "" {
		name = args[0]
	}
	token, hash, err := mintToken()
	if err != nil {
		return err
	}
	// The token goes to stdout once and is never recoverable from the file,
	// which is the point of storing only the hash.
	fmt.Printf("token (give this to the caller, it is not stored anywhere):\n\n  %s\n\n", token)
	fmt.Printf("entry for AUTH_TOKENS_FILE:\n\n")
	entry, _ := json.MarshalIndent([]Principal{{
		Name: name, Hash: hash, ACL: []string{"*"},
	}}, "  ", "  ")
	fmt.Printf("  %s\n\n", entry)
	fmt.Print("Replace [\"*\"] with the tags this caller may retrieve; \"*\" means\n" +
		"unrestricted. Add \"admin\": true to allow uploading documents and\n" +
		"switching models.\n")
	return nil
}
