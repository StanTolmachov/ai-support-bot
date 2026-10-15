// Package telegram is a minimal Telegram Bot API client: send messages,
// long polling and a secure webhook handler.
package telegram

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

type User struct {
	ID           int64  `json:"id"`
	FirstName    string `json:"first_name"`
	Username     string `json:"username,omitempty"`
	LanguageCode string `json:"language_code,omitempty"`
}

type Chat struct {
	ID   int64  `json:"id"`
	Type string `json:"type"`
}

type Message struct {
	MessageID      int64    `json:"message_id"`
	From           *User    `json:"from,omitempty"`
	Chat           Chat     `json:"chat"`
	Text           string   `json:"text,omitempty"`
	ReplyToMessage *Message `json:"reply_to_message,omitempty"`
}

type Update struct {
	UpdateID int64    `json:"update_id"`
	Message  *Message `json:"message,omitempty"`
}

type Client struct {
	Token   string
	BaseURL string
	HTTP    *http.Client
}

func New(token string) *Client {
	return &Client{Token: token, BaseURL: "https://api.telegram.org", HTTP: &http.Client{Timeout: 70 * time.Second}}
}

type apiResponse struct {
	OK          bool            `json:"ok"`
	Result      json.RawMessage `json:"result"`
	Description string          `json:"description"`
	Parameters  *struct {
		RetryAfter int `json:"retry_after"`
	} `json:"parameters"`
}

func (c *Client) call(ctx context.Context, method string, params, out any) error {
	body, err := json.Marshal(params)
	if err != nil {
		return err
	}
	url := fmt.Sprintf("%s/bot%s/%s", c.BaseURL, c.Token, method)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		// net/http errors include the URL, which contains the token: strip it.
		var ue interface{ Unwrap() error }
		if errors.As(err, &ue) {
			return fmt.Errorf("telegram %s: %w", method, ue.Unwrap())
		}
		return fmt.Errorf("telegram %s: request failed", method)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	var r apiResponse
	if err := json.Unmarshal(data, &r); err != nil {
		return fmt.Errorf("telegram %s: HTTP %d", method, resp.StatusCode)
	}
	if !r.OK {
		if r.Parameters != nil && r.Parameters.RetryAfter > 0 {
			return fmt.Errorf("telegram %s: rate limited, retry after %ds", method, r.Parameters.RetryAfter)
		}
		return fmt.Errorf("telegram %s: %s", method, r.Description)
	}
	if out != nil {
		return json.Unmarshal(r.Result, out)
	}
	return nil
}

const maxMessageLen = 4096

// SendMessage sends plain text (no parse mode, so model output can never
// break formatting). Text over Telegram's limit is truncated.
func (c *Client) SendMessage(ctx context.Context, chatID int64, text string, replyTo int64) (Message, error) {
	if r := []rune(text); len(r) > maxMessageLen {
		text = string(r[:maxMessageLen-1]) + "…"
	}
	params := map[string]any{"chat_id": chatID, "text": text, "link_preview_options": map[string]bool{"is_disabled": true}}
	if replyTo != 0 {
		params["reply_parameters"] = map[string]any{"message_id": replyTo, "allow_sending_without_reply": true}
	}
	var m Message
	err := c.call(ctx, "sendMessage", params, &m)
	return m, err
}

func (c *Client) SendTyping(ctx context.Context, chatID int64) error {
	return c.call(ctx, "sendChatAction", map[string]any{"chat_id": chatID, "action": "typing"}, nil)
}

func (c *Client) SetWebhook(ctx context.Context, url, secret string) error {
	return c.call(ctx, "setWebhook", map[string]any{
		"url": url, "secret_token": secret, "allowed_updates": []string{"message"}, "drop_pending_updates": false,
	}, nil)
}

func (c *Client) DeleteWebhook(ctx context.Context) error {
	return c.call(ctx, "deleteWebhook", map[string]any{}, nil)
}

func (c *Client) GetUpdates(ctx context.Context, offset int64, timeoutSec int) ([]Update, error) {
	var ups []Update
	err := c.call(ctx, "getUpdates", map[string]any{
		"offset": offset, "timeout": timeoutSec, "allowed_updates": []string{"message"},
	}, &ups)
	return ups, err
}

// Poll long-polls for updates until ctx is cancelled. Good for local
// development: no public URL needed.
func (c *Client) Poll(ctx context.Context, handle func(Update)) error {
	var offset int64
	backoff := time.Second
	for ctx.Err() == nil {
		ups, err := c.GetUpdates(ctx, offset, 50)
		if err != nil {
			if ctx.Err() != nil {
				break
			}
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
			}
			backoff = min(backoff*2, 30*time.Second)
			continue
		}
		backoff = time.Second
		for _, u := range ups {
			offset = u.UpdateID + 1
			handle(u)
		}
	}
	return ctx.Err()
}

// WebhookHandler verifies Telegram's secret header, then hands the update to
// enqueue. If enqueue reports the queue is full it answers 503, and Telegram
// retries later (natural back-pressure).
func WebhookHandler(secret string, enqueue func(Update) bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		got := r.Header.Get("X-Telegram-Bot-Api-Secret-Token")
		if secret == "" || subtle.ConstantTimeCompare([]byte(got), []byte(secret)) != 1 {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		var u Update
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&u); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if !enqueue(u) {
			http.Error(w, "busy", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
}
