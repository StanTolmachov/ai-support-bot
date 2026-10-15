package telegram

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestWebhookHandlerSecurityAndBackPressure(t *testing.T) {
	var got []Update
	full := false
	h := WebhookHandler("s3cret", func(u Update) bool {
		if full {
			return false
		}
		got = append(got, u)
		return true
	})
	do := func(method, secret, body string) int {
		req := httptest.NewRequest(method, "/telegram/webhook", strings.NewReader(body))
		if secret != "" {
			req.Header.Set("X-Telegram-Bot-Api-Secret-Token", secret)
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec.Code
	}
	ok := `{"update_id":5,"message":{"message_id":1,"chat":{"id":7,"type":"private"},"text":"hi"}}`
	cases := []struct {
		name, method, secret, body string
		want                       int
	}{
		{"wrong method", http.MethodGet, "s3cret", ok, 405},
		{"no secret", http.MethodPost, "", ok, 403},
		{"wrong secret", http.MethodPost, "nope", ok, 403},
		{"bad json", http.MethodPost, "s3cret", "{", 400},
		{"ok", http.MethodPost, "s3cret", ok, 200},
	}
	for _, c := range cases {
		if code := do(c.method, c.secret, c.body); code != c.want {
			t.Errorf("%s: code %d, want %d", c.name, code, c.want)
		}
	}
	if len(got) != 1 || got[0].Message.Text != "hi" {
		t.Fatalf("enqueued %+v", got)
	}
	full = true
	if code := do(http.MethodPost, "s3cret", ok); code != 503 {
		t.Errorf("full queue: code %d, want 503 so Telegram retries", code)
	}
	noSecret := WebhookHandler("", func(Update) bool { return true })
	rec := httptest.NewRecorder()
	noSecret.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(ok)))
	if rec.Code != 403 {
		t.Error("empty configured secret must reject everything")
	}
}

func TestSendMessagePlainTextTruncatedAndNoTokenInErrors(t *testing.T) {
	var params map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/botTOKEN123/sendMessage") {
			t.Errorf("path %s", r.URL.Path)
		}
		json.NewDecoder(r.Body).Decode(&params)
		if params["chat_id"].(float64) == 13 {
			w.Write([]byte(`{"ok":false,"description":"Forbidden: bot was blocked by the user"}`))
			return
		}
		w.Write([]byte(`{"ok":true,"result":{"message_id":99,"chat":{"id":7,"type":"private"}}}`))
	}))
	defer srv.Close()
	c := New("TOKEN123")
	c.BaseURL = srv.URL

	m, err := c.SendMessage(context.Background(), 7, strings.Repeat("ø", 5000), 3)
	if err != nil || m.MessageID != 99 {
		t.Fatalf("m=%+v err=%v", m, err)
	}
	if _, hasParseMode := params["parse_mode"]; hasParseMode {
		t.Error("must send plain text")
	}
	if n := utf8.RuneCountInString(params["text"].(string)); n != 4096 {
		t.Errorf("text length %d runes, want 4096", n)
	}
	if params["reply_parameters"] == nil {
		t.Error("reply_parameters missing")
	}

	_, err = c.SendMessage(context.Background(), 13, "x", 0)
	if err == nil || !strings.Contains(err.Error(), "blocked") || strings.Contains(err.Error(), "TOKEN123") {
		t.Errorf("err = %v", err)
	}

	c.BaseURL = "http://127.0.0.1:1" // connection refused
	_, err = c.SendMessage(context.Background(), 7, "x", 0)
	if err == nil || strings.Contains(err.Error(), "TOKEN123") {
		t.Errorf("network error must not leak the token: %v", err)
	}
}

func TestPollAdvancesOffset(t *testing.T) {
	var offsets []float64
	ctx, cancel := context.WithCancel(context.Background())
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var p map[string]any
		json.NewDecoder(r.Body).Decode(&p)
		offsets = append(offsets, p["offset"].(float64))
		if len(offsets) == 1 {
			w.Write([]byte(`{"ok":true,"result":[{"update_id":10},{"update_id":11}]}`))
			return
		}
		cancel()
		w.Write([]byte(`{"ok":true,"result":[]}`))
	}))
	defer srv.Close()
	c := New("T")
	c.BaseURL = srv.URL
	var ids []int64
	c.Poll(ctx, func(u Update) { ids = append(ids, u.UpdateID) })
	if len(ids) != 2 || offsets[1] != 12 {
		t.Fatalf("ids=%v offsets=%v", ids, offsets)
	}
}
