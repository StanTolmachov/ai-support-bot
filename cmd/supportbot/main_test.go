package main

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// End-to-end: the real binary wiring (config → KB index → agent → bot →
// dispatcher → Telegram polling) against fake Gemini and Telegram servers.
func TestEndToEndPollingWithFakes(t *testing.T) {
	gem := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, ":batchEmbedContents"):
			var body struct{ Requests []json.RawMessage }
			json.NewDecoder(r.Body).Decode(&body)
			embs := make([]map[string]any, len(body.Requests))
			for i := range embs {
				embs[i] = map[string]any{"values": []float32{1, 0.5, 0.25}}
			}
			json.NewEncoder(w).Encode(map[string]any{"embeddings": embs})
		case strings.HasSuffix(r.URL.Path, ":generateContent"):
			w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"An e-bike is 450 NOK per day [1]."}]}}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer gem.Close()

	var mu sync.Mutex
	var sentTexts []string
	gotReply := make(chan struct{}, 1)
	polls := 0
	tg := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		switch {
		case strings.HasSuffix(r.URL.Path, "/getUpdates"):
			mu.Lock()
			polls++
			first := polls == 1
			mu.Unlock()
			if first {
				w.Write([]byte(`{"ok":true,"result":[{"update_id":1,"message":{"message_id":1,
					"from":{"id":7,"first_name":"Ola"},"chat":{"id":7,"type":"private"},"text":"e-bike price?"}}]}`))
				return
			}
			select {
			case <-r.Context().Done():
			case <-time.After(200 * time.Millisecond):
			}
			w.Write([]byte(`{"ok":true,"result":[]}`))
		case strings.HasSuffix(r.URL.Path, "/sendMessage"):
			var p map[string]any
			json.Unmarshal(body, &p)
			mu.Lock()
			sentTexts = append(sentTexts, p["text"].(string))
			mu.Unlock()
			w.Write([]byte(`{"ok":true,"result":{"message_id":2,"chat":{"id":7,"type":"private"}}}`))
			select {
			case gotReply <- struct{}{}:
			default:
			}
		default:
			w.Write([]byte(`{"ok":true,"result":true}`))
		}
	}))
	defer tg.Close()

	dir := t.TempDir()
	t.Setenv("GEMINI_API_KEY", "k")
	t.Setenv("GEMINI_BASE_URL", gem.URL)
	t.Setenv("TELEGRAM_BOT_TOKEN", "T")
	t.Setenv("TELEGRAM_API_URL", tg.URL)
	t.Setenv("MODE", "polling")
	t.Setenv("KB_DIR", filepath.Join("..", "..", "kb"))
	t.Setenv("BOOKINGS_PATH", filepath.Join("..", "..", "data", "bookings.json"))
	t.Setenv("INDEX_PATH", filepath.Join(dir, "index.json"))

	done := make(chan error, 1)
	go func() { done <- run(slog.New(slog.NewTextHandler(io.Discard, nil)), nil) }()

	select {
	case <-gotReply:
	case err := <-done:
		t.Fatalf("run exited early: %v", err)
	case <-time.After(10 * time.Second):
		t.Fatal("no reply sent within 10s")
	}
	syscall.Kill(os.Getpid(), syscall.SIGINT) // graceful shutdown path
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("did not shut down")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(sentTexts) == 0 || !strings.HasPrefix(sentTexts[0], "An e-bike is 450 NOK per day [1].\n\nSources: ") {
		t.Fatalf("sent = %q", sentTexts)
	}
	if _, err := os.Stat(filepath.Join(dir, "index.json")); err != nil {
		t.Error("index was not persisted")
	}
}
