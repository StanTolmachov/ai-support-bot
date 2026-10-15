// Package bot holds the conversation logic: dedupe, rate limits, history,
// AI answers, and hand-off to a human operator in a Telegram admin chat.
package bot

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/StanTolmachov/ai-support-bot/internal/agent"
	"github.com/StanTolmachov/ai-support-bot/internal/telegram"
)

type Sender interface {
	SendMessage(ctx context.Context, chatID int64, text string, replyTo int64) (telegram.Message, error)
	SendTyping(ctx context.Context, chatID int64) error
}

type Answerer interface {
	Answer(ctx context.Context, history []agent.Turn, question string) (agent.Reply, error)
}

type Config struct {
	AdminChatID   int64         // operators' group or private chat; 0 disables hand-off
	FallbackEmail string        // shown when no operator chat is configured
	HistoryTurns  int           // user+model pairs kept per chat
	RatePerMinute int           // messages per user per minute
	HandoffTTL    time.Duration // how long a chat stays with a human without activity
	AnswerTimeout time.Duration
}

type session struct {
	history    []agent.Turn
	humanUntil time.Time
	window     []time.Time // timestamps of recent messages (rate limit)
	lastSeen   time.Time
}

type Bot struct {
	tg  Sender
	ai  Answerer
	cfg Config
	log *slog.Logger
	now func() time.Time

	mu       sync.Mutex
	sessions map[int64]*session
	handoffs map[int64]int64 // admin-chat message ID → customer chat ID
	seen     *ringSet        // recently processed update IDs
}

func New(tg Sender, ai Answerer, cfg Config, log *slog.Logger) *Bot {
	return &Bot{
		tg: tg, ai: ai, cfg: cfg, log: log, now: time.Now,
		sessions: map[int64]*session{},
		handoffs: map[int64]int64{},
		seen:     newRingSet(10000),
	}
}

// Text shown to customers. Kept in one place so a client can translate them.
var (
	msgWelcome   = "Hi! I'm the virtual assistant. Ask me about prices, opening hours, routes or your booking. Type /human any time to talk to a person."
	msgTextOnly  = "Sorry, I can only read text messages."
	msgSlowDown  = "You're sending messages very quickly. Please wait a minute and try again."
	msgHandedOff = "I've passed your question to a team member. They'll reply right here in this chat."
	msgBackToBot = "You're back with the virtual assistant. Ask me anything!"
	msgError     = "Sorry, something went wrong on my side."
)

// Handle processes one Telegram update. Safe for concurrent use; callers
// should keep updates of the same chat in order (see Dispatcher).
func (b *Bot) Handle(ctx context.Context, u telegram.Update) {
	if !b.seen.Add(u.UpdateID) { // Telegram re-delivers on webhook timeouts
		return
	}
	m := u.Message
	if m == nil || m.From == nil {
		return
	}
	if b.cfg.AdminChatID != 0 && m.Chat.ID == b.cfg.AdminChatID {
		b.handleOperator(ctx, m)
		return
	}
	if m.Chat.Type != "private" {
		return // the bot only talks to customers in private chats
	}
	if strings.TrimSpace(m.Text) == "" {
		b.send(ctx, m.Chat.ID, msgTextOnly, 0)
		return
	}
	b.handleCustomer(ctx, m)
}

func (b *Bot) handleCustomer(ctx context.Context, m *telegram.Message) {
	chatID, text := m.Chat.ID, strings.TrimSpace(m.Text)

	b.mu.Lock()
	s := b.sessionLocked(chatID)
	allowed := b.allowLocked(s)
	withHuman := b.now().Before(s.humanUntil)
	history := append([]agent.Turn(nil), s.history...)
	b.mu.Unlock()

	switch {
	case !allowed:
		b.send(ctx, chatID, msgSlowDown, 0)
		return
	case text == "/start":
		b.send(ctx, chatID, msgWelcome, 0)
		return
	case text == "/human":
		b.handOff(ctx, m, "customer_request: asked for a person via /human")
		return
	case withHuman:
		b.forwardToOperator(ctx, m)
		return
	}

	_ = b.tg.SendTyping(ctx, chatID)
	actx, cancel := context.WithTimeout(ctx, b.cfg.AnswerTimeout)
	reply, err := b.ai.Answer(actx, history, text)
	cancel()
	if err != nil {
		b.log.Error("answer failed", "chat", chatID, "err", err)
		reply = agent.Reply{Escalate: true, EscalationReason: "assistant error"}
	}
	if reply.Escalate {
		b.handOff(ctx, m, reply.EscalationReason)
		return
	}

	out := reply.Text
	if len(reply.Sources) > 0 {
		out += "\n\nSources: " + strings.Join(reply.Sources, "; ")
	}
	b.send(ctx, chatID, out, 0)
	b.log.Info("answered", "chat", chatID, "sources", len(reply.Sources), "tools", reply.ToolsUsed)

	b.mu.Lock()
	s.history = append(s.history, agent.Turn{Role: "user", Text: text}, agent.Turn{Role: "model", Text: reply.Text})
	if limit := b.cfg.HistoryTurns * 2; len(s.history) > limit {
		s.history = s.history[len(s.history)-limit:]
	}
	b.mu.Unlock()
}

// handOff moves the chat to a human operator.
func (b *Bot) handOff(ctx context.Context, m *telegram.Message, reason string) {
	if b.cfg.AdminChatID == 0 {
		msg := "I can't answer that one myself."
		if b.cfg.FallbackEmail != "" {
			msg += " Please email " + b.cfg.FallbackEmail + " and the team will help you."
		}
		b.send(ctx, m.Chat.ID, msg, 0)
		return
	}
	card := fmt.Sprintf("🙋 Hand-off: %s\nReason: %s\n\nCustomer wrote:\n%s\n\n↩️ Reply to this message to answer the customer. Reply /done to hand the chat back to the bot.",
		customerLabel(m), reason, m.Text)
	sent, err := b.tg.SendMessage(ctx, b.cfg.AdminChatID, card, 0)
	if err != nil {
		b.log.Error("hand-off failed", "err", err)
		b.send(ctx, m.Chat.ID, msgError+" Please try again in a few minutes.", 0)
		return
	}
	b.mu.Lock()
	b.handoffs[sent.MessageID] = m.Chat.ID
	b.sessionLocked(m.Chat.ID).humanUntil = b.now().Add(b.cfg.HandoffTTL)
	b.mu.Unlock()
	b.send(ctx, m.Chat.ID, msgHandedOff, 0)
	b.log.Info("handed off", "chat", m.Chat.ID, "reason", reason)
}

func (b *Bot) forwardToOperator(ctx context.Context, m *telegram.Message) {
	sent, err := b.tg.SendMessage(ctx, b.cfg.AdminChatID, customerLabel(m)+":\n"+m.Text, 0)
	if err != nil {
		b.log.Error("forward failed", "err", err)
		return
	}
	b.mu.Lock()
	b.handoffs[sent.MessageID] = m.Chat.ID
	b.sessionLocked(m.Chat.ID).humanUntil = b.now().Add(b.cfg.HandoffTTL)
	b.mu.Unlock()
}

// handleOperator relays operator replies (Telegram "reply" to a hand-off
// card or forwarded message) back to the right customer.
func (b *Bot) handleOperator(ctx context.Context, m *telegram.Message) {
	if m.ReplyToMessage == nil {
		return
	}
	b.mu.Lock()
	customer, ok := b.handoffs[m.ReplyToMessage.MessageID]
	b.mu.Unlock()
	if !ok {
		return
	}
	text := strings.TrimSpace(m.Text)
	if text == "/done" {
		b.mu.Lock()
		b.sessionLocked(customer).humanUntil = time.Time{}
		b.mu.Unlock()
		b.send(ctx, customer, msgBackToBot, 0)
		b.send(ctx, b.cfg.AdminChatID, "✅ Chat handed back to the bot.", m.MessageID)
		return
	}
	if text == "" {
		return
	}
	b.send(ctx, customer, "👤 "+text, 0)
	b.mu.Lock()
	b.sessionLocked(customer).humanUntil = b.now().Add(b.cfg.HandoffTTL)
	b.mu.Unlock()
}

func (b *Bot) send(ctx context.Context, chatID int64, text string, replyTo int64) {
	if _, err := b.tg.SendMessage(ctx, chatID, text, replyTo); err != nil {
		b.log.Error("send failed", "chat", chatID, "err", err)
	}
}

func (b *Bot) sessionLocked(chatID int64) *session {
	s, ok := b.sessions[chatID]
	if !ok {
		s = &session{}
		b.sessions[chatID] = s
	}
	s.lastSeen = b.now()
	return s
}

// allowLocked is a sliding-window rate limit per chat.
func (b *Bot) allowLocked(s *session) bool {
	cutoff := b.now().Add(-time.Minute)
	kept := s.window[:0]
	for _, t := range s.window {
		if t.After(cutoff) {
			kept = append(kept, t)
		}
	}
	s.window = kept
	if len(s.window) >= b.cfg.RatePerMinute {
		return false
	}
	s.window = append(s.window, b.now())
	return true
}

// Prune drops sessions idle longer than maxIdle and their hand-off links.
// Call periodically (main does it every 10 minutes).
func (b *Bot) Prune(maxIdle time.Duration) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	cutoff := b.now().Add(-maxIdle)
	n := 0
	for id, s := range b.sessions {
		if s.lastSeen.Before(cutoff) && b.now().After(s.humanUntil) {
			delete(b.sessions, id)
			n++
		}
	}
	for msgID, chat := range b.handoffs {
		if _, ok := b.sessions[chat]; !ok {
			delete(b.handoffs, msgID)
		}
	}
	return n
}

func customerLabel(m *telegram.Message) string {
	name := m.From.FirstName
	if m.From.Username != "" {
		name += " (@" + m.From.Username + ")"
	}
	return fmt.Sprintf("%s · chat %d", name, m.Chat.ID)
}

// ringSet remembers the last N int64 keys.
type ringSet struct {
	mu   sync.Mutex
	set  map[int64]struct{}
	ring []int64
	pos  int
}

func newRingSet(n int) *ringSet {
	return &ringSet{set: make(map[int64]struct{}, n), ring: make([]int64, 0, n)}
}

// Add returns false if the key was already present.
func (r *ringSet) Add(k int64) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.set[k]; ok {
		return false
	}
	if len(r.ring) < cap(r.ring) {
		r.ring = append(r.ring, k)
	} else {
		delete(r.set, r.ring[r.pos])
		r.ring[r.pos] = k
		r.pos = (r.pos + 1) % len(r.ring)
	}
	r.set[k] = struct{}{}
	return true
}
