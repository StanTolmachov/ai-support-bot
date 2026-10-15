// Package gemini is a small, dependency-free client for the Gemini REST API:
// text generation with function calling, and batch embeddings.
package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"
)

const DefaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

// Client talks to the Gemini API. The zero value is not usable; use New.
type Client struct {
	APIKey     string
	BaseURL    string
	ChatModel  string
	EmbedModel string
	EmbedDims  int
	MaxRetries int
	HTTP       *http.Client

	// sleep is replaceable in tests so retries don't slow them down.
	sleep func(ctx context.Context, d time.Duration) error
}

func New(apiKey, chatModel, embedModel string) *Client {
	return &Client{
		APIKey:     apiKey,
		BaseURL:    DefaultBaseURL,
		ChatModel:  chatModel,
		EmbedModel: embedModel,
		EmbedDims:  768,
		MaxRetries: 3,
		HTTP:       &http.Client{Timeout: 60 * time.Second},
		sleep:      sleepCtx,
	}
}

// ---------- wire types ----------

type Part struct {
	Text             string            `json:"text,omitempty"`
	FunctionCall     *FunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *FunctionResponse `json:"functionResponse,omitempty"`
	// Gemini 3 models return opaque thought signatures that must be sent back
	// unchanged in the next turn of a function-calling exchange.
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
	Thought          bool   `json:"thought,omitempty"`
}

type FunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

type FunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

type Schema struct {
	Type        string             `json:"type"`
	Description string             `json:"description,omitempty"`
	Properties  map[string]*Schema `json:"properties,omitempty"`
	Required    []string           `json:"required,omitempty"`
	Enum        []string           `json:"enum,omitempty"`
}

type FunctionDeclaration struct {
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Parameters  *Schema `json:"parameters,omitempty"`
}

type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations"`
}

type GenerationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
}

type GenerateRequest struct {
	SystemInstruction *Content          `json:"systemInstruction,omitempty"`
	Contents          []Content         `json:"contents"`
	Tools             []Tool            `json:"tools,omitempty"`
	GenerationConfig  *GenerationConfig `json:"generationConfig,omitempty"`
}

type generateResponse struct {
	Candidates []struct {
		Content      Content `json:"content"`
		FinishReason string  `json:"finishReason"`
	} `json:"candidates"`
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
}

// ---------- API ----------

// Generate returns the model's reply content (text and/or function calls).
func (c *Client) Generate(ctx context.Context, req GenerateRequest) (Content, error) {
	var out generateResponse
	url := fmt.Sprintf("%s/models/%s:generateContent", c.BaseURL, c.ChatModel)
	if err := c.post(ctx, url, req, &out); err != nil {
		return Content{}, err
	}
	if out.PromptFeedback != nil && out.PromptFeedback.BlockReason != "" {
		return Content{}, fmt.Errorf("gemini: prompt blocked: %s", out.PromptFeedback.BlockReason)
	}
	if len(out.Candidates) == 0 {
		return Content{}, errors.New("gemini: no candidates in response")
	}
	content := out.Candidates[0].Content
	if content.Role == "" {
		content.Role = "model"
	}
	return content, nil
}

type embedRequest struct {
	Model                string  `json:"model"`
	Content              Content `json:"content"`
	OutputDimensionality int     `json:"outputDimensionality,omitempty"`
}

type batchEmbedResponse struct {
	Embeddings []struct {
		Values []float32 `json:"values"`
	} `json:"embeddings"`
}

// EmbedDocuments embeds knowledge-base chunks. gemini-embedding-2 takes the
// task as a text prefix rather than a taskType field.
func (c *Client) EmbedDocuments(ctx context.Context, titles, texts []string) ([][]float32, error) {
	if len(titles) != len(texts) {
		return nil, errors.New("gemini: titles and texts length mismatch")
	}
	inputs := make([]string, len(texts))
	for i := range texts {
		title := titles[i]
		if title == "" {
			title = "none"
		}
		inputs[i] = "title: " + title + " | text: " + texts[i]
	}
	return c.embed(ctx, inputs)
}

// EmbedQuery embeds a user question for retrieval.
func (c *Client) EmbedQuery(ctx context.Context, q string) ([]float32, error) {
	v, err := c.embed(ctx, []string{"task: search result | query: " + q})
	if err != nil {
		return nil, err
	}
	return v[0], nil
}

const embedBatchSize = 100

func (c *Client) embed(ctx context.Context, inputs []string) ([][]float32, error) {
	url := fmt.Sprintf("%s/models/%s:batchEmbedContents", c.BaseURL, c.EmbedModel)
	all := make([][]float32, 0, len(inputs))
	for start := 0; start < len(inputs); start += embedBatchSize {
		end := min(start+embedBatchSize, len(inputs))
		reqs := make([]embedRequest, 0, end-start)
		for _, in := range inputs[start:end] {
			reqs = append(reqs, embedRequest{
				Model:                "models/" + c.EmbedModel,
				Content:              Content{Parts: []Part{{Text: in}}},
				OutputDimensionality: c.EmbedDims,
			})
		}
		var out batchEmbedResponse
		if err := c.post(ctx, url, map[string]any{"requests": reqs}, &out); err != nil {
			return nil, err
		}
		if len(out.Embeddings) != end-start {
			return nil, fmt.Errorf("gemini: expected %d embeddings, got %d", end-start, len(out.Embeddings))
		}
		for _, e := range out.Embeddings {
			all = append(all, e.Values)
		}
	}
	return all, nil
}

// ---------- transport ----------

// APIError is returned for non-2xx responses after retries are exhausted.
type APIError struct {
	Status int
	Body   string
}

func (e *APIError) Error() string { return fmt.Sprintf("gemini: HTTP %d: %s", e.Status, e.Body) }

func retryable(status int) bool {
	return status == http.StatusTooManyRequests || status >= 500
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	body, err := json.Marshal(in)
	if err != nil {
		return err
	}
	var lastErr error
	for attempt := 0; attempt <= c.MaxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-goog-api-key", c.APIKey) // header, never in the URL/logs

		resp, err := c.HTTP.Do(req)
		var wait time.Duration
		switch {
		case err != nil:
			if ctx.Err() != nil {
				return ctx.Err()
			}
			lastErr = err // network error: retry
		default:
			data, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
			resp.Body.Close()
			if resp.StatusCode/100 == 2 {
				return json.Unmarshal(data, out)
			}
			lastErr = &APIError{Status: resp.StatusCode, Body: truncate(string(data), 300)}
			if !retryable(resp.StatusCode) {
				return lastErr
			}
			if s, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil && s > 0 {
				wait = time.Duration(s) * time.Second
			}
		}
		if attempt == c.MaxRetries {
			break
		}
		if wait == 0 {
			wait = time.Second << attempt // 1s, 2s, 4s
		}
		if err := c.sleep(ctx, wait); err != nil {
			return err
		}
	}
	return lastErr
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// TextOf joins all non-thought text parts of a content.
func TextOf(c Content) string {
	var b bytes.Buffer
	for _, p := range c.Parts {
		if p.Text != "" && !p.Thought {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// FunctionCalls returns all function calls in a content.
func FunctionCalls(c Content) []FunctionCall {
	var calls []FunctionCall
	for _, p := range c.Parts {
		if p.FunctionCall != nil {
			calls = append(calls, *p.FunctionCall)
		}
	}
	return calls
}
