// Command eval runs the agent against eval/cases.json with the real Gemini
// API and prints a pass/fail table. Use it after every change to the prompt
// or the knowledge base:
//
//	GEMINI_API_KEY=... go run ./cmd/eval
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/StanTolmachov/ai-support-bot/internal/agent"
	"github.com/StanTolmachov/ai-support-bot/internal/bookings"
	"github.com/StanTolmachov/ai-support-bot/internal/gemini"
	"github.com/StanTolmachov/ai-support-bot/internal/kb"
)

type evalCase struct {
	Q           string   `json:"q"`
	Expect      string   `json:"expect"`       // answer | escalate | tool
	Contains    []string `json:"contains"`     // at least one must appear
	NotContains []string `json:"not_contains"` // none may appear
	Source      string   `json:"source"`       // substring of a cited source title
	Tool        string   `json:"tool"`
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "eval:", err)
		os.Exit(1)
	}
}

func run() error {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		return fmt.Errorf("GEMINI_API_KEY is required")
	}
	chatModel, embedModel := envOr("CHAT_MODEL", "gemini-3.8-flash"), envOr("EMBED_MODEL", "gemini-embedding-2")
	data, err := os.ReadFile("eval/cases.json")
	if err != nil {
		return err
	}
	var cases []evalCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return err
	}

	ctx := context.Background()
	llm := gemini.New(key, chatModel, embedModel)
	idx, _, err := kb.LoadOrBuild(ctx, llm, embedModel, "kb", "data/index.json")
	if err != nil {
		return err
	}
	store, err := bookings.Load("data/bookings.json")
	if err != nil {
		return err
	}
	ag := &agent.Agent{LLM: llm, Embedder: llm, Index: idx, Bookings: store,
		Business: "Polar Pedal Bike Rental, Tromsø", TopK: 4, MinScore: 0.35, MaxToolRounds: 3}

	passed := 0
	for i, c := range cases {
		start := time.Now()
		r, err := ag.Answer(ctx, nil, c.Q)
		problems := check(c, r, err)
		status := "PASS"
		if len(problems) > 0 {
			status = "FAIL"
		} else {
			passed++
		}
		fmt.Printf("%2d %s %5.1fs  %s\n", i+1, status, time.Since(start).Seconds(), c.Q)
		for _, p := range problems {
			fmt.Printf("        - %s\n", p)
		}
	}
	fmt.Printf("\n%d/%d passed (%.0f%%)\n", passed, len(cases), 100*float64(passed)/float64(len(cases)))
	if passed < len(cases) {
		os.Exit(2)
	}
	return nil
}

func check(c evalCase, r agent.Reply, err error) []string {
	if err != nil {
		return []string{"error: " + err.Error()}
	}
	var p []string
	switch c.Expect {
	case "escalate":
		if !r.Escalate {
			p = append(p, "expected hand-off, got answer: "+short(r.Text))
		}
		return p
	case "tool":
		if !slices.Contains(r.ToolsUsed, c.Tool) {
			p = append(p, fmt.Sprintf("expected tool %s, used %v", c.Tool, r.ToolsUsed))
		}
	}
	if r.Escalate {
		return append(p, "unexpected hand-off: "+r.EscalationReason)
	}
	if len(c.Contains) > 0 && !slices.ContainsFunc(c.Contains, func(s string) bool { return strings.Contains(r.Text, s) }) {
		p = append(p, fmt.Sprintf("answer lacks any of %q: %s", c.Contains, short(r.Text)))
	}
	for _, s := range c.NotContains {
		if strings.Contains(r.Text, s) {
			p = append(p, fmt.Sprintf("answer must not contain %q: %s", s, short(r.Text)))
		}
	}
	if c.Source != "" && !slices.ContainsFunc(r.Sources, func(s string) bool {
		return strings.Contains(strings.ToLower(s), strings.ToLower(c.Source))
	}) {
		p = append(p, fmt.Sprintf("expected a source matching %q, got %v", c.Source, r.Sources))
	}
	return p
}

func short(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) > 120 {
		return s[:120] + "…"
	}
	return s
}

func envOr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}
