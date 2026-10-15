// Command supportbot runs the AI support bot.
//
//	supportbot              run the bot (MODE=polling or MODE=webhook)
//	supportbot index        (re)build the knowledge-base index and exit
//	supportbot ask "..."    ask one question from the terminal, no Telegram needed
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/StanTolmachov/ai-support-bot/internal/agent"
	"github.com/StanTolmachov/ai-support-bot/internal/bookings"
	"github.com/StanTolmachov/ai-support-bot/internal/bot"
	"github.com/StanTolmachov/ai-support-bot/internal/gemini"
	"github.com/StanTolmachov/ai-support-bot/internal/kb"
	"github.com/StanTolmachov/ai-support-bot/internal/telegram"
)

type config struct {
	geminiKey, chatModel, embedModel   string
	telegramToken, mode, webhookSecret string
	publicURL, port                    string
	adminChatID                        int64
	business, fallbackEmail            string
	kbDir, indexPath, bookingsPath     string
}

func loadConfig() (config, error) {
	c := config{
		geminiKey:     os.Getenv("GEMINI_API_KEY"),
		chatModel:     env("CHAT_MODEL", "gemini-3.8-flash"),
		embedModel:    env("EMBED_MODEL", "gemini-embedding-2"),
		telegramToken: os.Getenv("TELEGRAM_BOT_TOKEN"),
		mode:          env("MODE", "polling"),
		webhookSecret: os.Getenv("WEBHOOK_SECRET"),
		publicURL:     strings.TrimRight(os.Getenv("PUBLIC_URL"), "/"),
		port:          env("PORT", "8080"),
		business:      env("BUSINESS_NAME", "Polar Pedal Bike Rental, Tromsø"),
		fallbackEmail: os.Getenv("FALLBACK_EMAIL"),
		kbDir:         env("KB_DIR", "kb"),
		indexPath:     env("INDEX_PATH", "data/index.json"),
		bookingsPath:  env("BOOKINGS_PATH", "data/bookings.json"),
	}
	if c.geminiKey == "" {
		return c, errors.New("GEMINI_API_KEY is required")
	}
	if v := os.Getenv("ADMIN_CHAT_ID"); v != "" {
		id, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			return c, fmt.Errorf("ADMIN_CHAT_ID: %w", err)
		}
		c.adminChatID = id
	}
	return c, nil
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if err := run(log, os.Args[1:]); err != nil {
		log.Error("fatal", "err", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger, args []string) error {
	cfg, err := loadConfig()
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	llm := gemini.New(cfg.geminiKey, cfg.chatModel, cfg.embedModel)
	if u := os.Getenv("GEMINI_BASE_URL"); u != "" {
		llm.BaseURL = u // proxies, or fakes in tests
	}
	idx, rebuilt, err := kb.LoadOrBuild(ctx, llm, cfg.embedModel, cfg.kbDir, cfg.indexPath)
	if err != nil {
		return fmt.Errorf("knowledge base: %w", err)
	}
	log.Info("knowledge base ready", "chunks", len(idx.Chunks), "rebuilt", rebuilt)

	if len(args) > 0 && args[0] == "index" {
		return nil
	}

	store, err := bookings.Load(cfg.bookingsPath)
	if err != nil {
		log.Warn("bookings not loaded; lookup tool will report unavailable", "err", err)
	}
	ag := &agent.Agent{
		LLM: llm, Embedder: llm, Index: idx, Bookings: store,
		Business: cfg.business, TopK: 4, MinScore: 0.35, MaxToolRounds: 3,
	}

	if len(args) > 0 && args[0] == "ask" {
		return ask(ctx, ag, strings.Join(args[1:], " "))
	}

	if cfg.telegramToken == "" {
		return errors.New("TELEGRAM_BOT_TOKEN is required")
	}
	tg := telegram.New(cfg.telegramToken)
	if u := os.Getenv("TELEGRAM_API_URL"); u != "" {
		tg.BaseURL = u // self-hosted Bot API server, or fakes in tests
	}
	b := bot.New(tg, ag, bot.Config{
		AdminChatID: cfg.adminChatID, FallbackEmail: cfg.fallbackEmail,
		HistoryTurns: 4, RatePerMinute: 10, HandoffTTL: 30 * time.Minute, AnswerTimeout: 45 * time.Second,
	}, log)

	d := bot.NewDispatcher(b.Handle, 8, 64)
	// Workers keep running on a non-cancelled context so in-flight updates
	// finish during graceful shutdown; each AI call still has its own timeout.
	d.Start(context.WithoutCancel(ctx))
	go func() {
		t := time.NewTicker(10 * time.Minute)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				b.Prune(24 * time.Hour)
			}
		}
	}()

	switch cfg.mode {
	case "polling":
		_ = tg.DeleteWebhook(ctx)
		log.Info("polling for updates")
		_ = tg.Poll(ctx, func(u telegram.Update) {
			for !d.Enqueue(u) && ctx.Err() == nil { // polling: wait instead of dropping
				time.Sleep(100 * time.Millisecond)
			}
		})
	case "webhook":
		if cfg.webhookSecret == "" || cfg.publicURL == "" {
			return errors.New("MODE=webhook needs WEBHOOK_SECRET and PUBLIC_URL")
		}
		if err := serveWebhook(ctx, log, tg, cfg, d); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unknown MODE %q", cfg.mode)
	}

	log.Info("shutting down")
	d.Stop()
	return nil
}

func serveWebhook(ctx context.Context, log *slog.Logger, tg *telegram.Client, cfg config, d *bot.Dispatcher) error {
	mux := http.NewServeMux()
	mux.Handle("/telegram/webhook", telegram.WebhookHandler(cfg.webhookSecret, d.Enqueue))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok")) })
	srv := &http.Server{Addr: ":" + cfg.port, Handler: mux, ReadHeaderTimeout: 10 * time.Second}

	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	if err := tg.SetWebhook(ctx, cfg.publicURL+"/telegram/webhook", cfg.webhookSecret); err != nil {
		return fmt.Errorf("setWebhook: %w", err)
	}
	log.Info("webhook listening", "port", cfg.port)

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return srv.Shutdown(shutdownCtx)
}

func ask(ctx context.Context, ag *agent.Agent, q string) error {
	if q == "" {
		return errors.New(`usage: supportbot ask "your question"`)
	}
	r, err := ag.Answer(ctx, nil, q)
	if err != nil {
		return err
	}
	if r.Escalate {
		fmt.Println("→ HAND-OFF to human:", r.EscalationReason)
		return nil
	}
	fmt.Println(r.Text)
	if len(r.Sources) > 0 {
		fmt.Println("\nSources:", strings.Join(r.Sources, "; "))
	}
	if len(r.ToolsUsed) > 0 {
		fmt.Println("Tools:", strings.Join(r.ToolsUsed, ", "))
	}
	return nil
}
