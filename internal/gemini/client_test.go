package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func testClient(t *testing.T, h http.HandlerFunc) (*Client, *[]time.Duration) {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	c := New("test-key", "chat-model", "embed-model")
	c.BaseURL = srv.URL
	var slept []time.Duration
	c.sleep = func(_ context.Context, d time.Duration) error { slept = append(slept, d); return nil }
	return c, &slept
}

func TestGenerateSendsKeyInHeaderAndParsesFunctionCall(t *testing.T) {
	c, _ := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models/chat-model:generateContent" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("x-goog-api-key") != "test-key" || r.URL.Query().Get("key") != "" {
			t.Error("API key must be sent in the header only")
		}
		var req GenerateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if len(req.Tools) != 1 || req.Tools[0].FunctionDeclarations[0].Name != "f" {
			t.Errorf("tools not sent: %+v", req.Tools)
		}
		w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[
			{"functionCall":{"name":"f","args":{"x":"1"}},"thoughtSignature":"sig=="}]}}]}`))
	})
	out, err := c.Generate(context.Background(), GenerateRequest{
		Contents: []Content{{Role: "user", Parts: []Part{{Text: "hi"}}}},
		Tools:    []Tool{{FunctionDeclarations: []FunctionDeclaration{{Name: "f", Description: "d"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	calls := FunctionCalls(out)
	if len(calls) != 1 || calls[0].Args["x"] != "1" {
		t.Fatalf("calls = %+v", calls)
	}
	// thought signature must survive a round trip so it can be echoed back
	data, _ := json.Marshal(out)
	if !strings.Contains(string(data), `"thoughtSignature":"sig=="`) {
		t.Errorf("thought signature lost: %s", data)
	}
}

func TestRetriesOn429ThenSucceeds(t *testing.T) {
	var n int32
	c, slept := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&n, 1) <= 2 {
			w.Header().Set("Retry-After", "7")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`))
	})
	out, err := c.Generate(context.Background(), GenerateRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if TextOf(out) != "ok" || out.Role != "model" {
		t.Errorf("out = %+v", out)
	}
	if len(*slept) != 2 || (*slept)[0] != 7*time.Second {
		t.Errorf("should honour Retry-After, slept %v", *slept)
	}
}

func TestBackoffOn5xxAndGivesUp(t *testing.T) {
	var n int32
	c, slept := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&n, 1)
		w.WriteHeader(http.StatusServiceUnavailable)
	})
	_, err := c.Generate(context.Background(), GenerateRequest{})
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != 503 {
		t.Fatalf("err = %v", err)
	}
	if n != 4 {
		t.Errorf("attempts = %d, want 4", n)
	}
	want := []time.Duration{time.Second, 2 * time.Second, 4 * time.Second}
	for i, d := range want {
		if (*slept)[i] != d {
			t.Errorf("backoff %d = %v, want %v", i, (*slept)[i], d)
		}
	}
}

func TestFailsFastOn400(t *testing.T) {
	var n int32
	c, _ := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&n, 1)
		http.Error(w, `{"error":"bad"}`, http.StatusBadRequest)
	})
	if _, err := c.Generate(context.Background(), GenerateRequest{}); err == nil {
		t.Fatal("expected error")
	}
	if n != 1 {
		t.Errorf("400 must not be retried, attempts = %d", n)
	}
}

func TestBlockedPromptIsAnError(t *testing.T) {
	c, _ := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"promptFeedback":{"blockReason":"SAFETY"}}`))
	})
	if _, err := c.Generate(context.Background(), GenerateRequest{}); err == nil || !strings.Contains(err.Error(), "SAFETY") {
		t.Fatalf("err = %v", err)
	}
}

func TestEmbedBatchesAndPrefixes(t *testing.T) {
	var calls int32
	c, _ := testClient(t, func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		if r.URL.Path != "/models/embed-model:batchEmbedContents" {
			t.Errorf("path = %s", r.URL.Path)
		}
		var body struct {
			Requests []embedRequest `json:"requests"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		resp := map[string]any{}
		var embs []map[string]any
		for _, rq := range body.Requests {
			if rq.Model != "models/embed-model" || rq.OutputDimensionality != 768 {
				t.Errorf("bad request item %+v", rq)
			}
			txt := rq.Content.Parts[0].Text
			if !strings.HasPrefix(txt, "title: ") && !strings.HasPrefix(txt, "task: search result | query: ") {
				t.Errorf("missing task prefix: %q", txt)
			}
			embs = append(embs, map[string]any{"values": []float32{float32(len(txt)), 1}})
		}
		resp["embeddings"] = embs
		json.NewEncoder(w).Encode(resp)
	})
	titles := make([]string, 150)
	texts := make([]string, 150)
	for i := range texts {
		texts[i] = "chunk"
	}
	titles[0] = "" // empty title must become "none"
	vecs, err := c.EmbedDocuments(context.Background(), titles, texts)
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 150 || calls != 2 {
		t.Fatalf("vectors=%d calls=%d, want 150 in 2 batches", len(vecs), calls)
	}
	if int(vecs[0][0]) != len("title: none | text: chunk") {
		t.Errorf("empty title not replaced with none")
	}
	if _, err := c.EmbedQuery(context.Background(), "price?"); err != nil {
		t.Fatal(err)
	}
}
