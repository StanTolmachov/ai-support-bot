package bot

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/StanTolmachov/ai-support-bot/internal/agent"
	"github.com/StanTolmachov/ai-support-bot/internal/telegram"
)

const admin = int64(-100500)

type sent struct {
	chat int64
	text string
}

type fakeTG struct {
	mu     sync.Mutex
	nextID int64
	out    []sent
}

func (f *fakeTG) SendMessage(_ context.Context, chat int64, text string, _ int64) (telegram.Message, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.nextID++
	f.out = append(f.out, sent{chat, text})
	return telegram.Message{MessageID: f.nextID, Chat: telegram.Chat{ID: chat}}, nil
}
func (f *fakeTG) SendTyping(context.Context, int64) error { return nil }

func (f *fakeTG) last(chat int64) (string, int64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for i := len(f.out) - 1; i >= 0; i-- {
		if f.out[i].chat == chat {
			return f.out[i].text, int64(i + 1)
		}
	}
	return "", 0
}

type fakeAI struct {
	calls    int
	history  [][]agent.Turn
	reply    agent.Reply
	err      error
	escalate map[string]bool // questions that should escalate
}

func (f *fakeAI) Answer(_ context.Context, h []agent.Turn, q string) (agent.Reply, error) {
	f.calls++
	f.history = append(f.history, h)
	if f.err != nil {
		return agent.Reply{}, f.err
	}
	if f.escalate[q] {
		return agent.Reply{Escalate: true, EscalationReason: "complaint_or_refund: wants money back"}, nil
	}
	r := f.reply
	if r.Text == "" {
		r.Text = "answer to " + q
	}
	return r, nil
}

type harness struct {
	bot   *Bot
	tg    *fakeTG
	ai    *fakeAI
	clock time.Time
	upd   int64
}

func newHarness(adminChat int64) *harness {
	h := &harness{tg: &fakeTG{}, ai: &fakeAI{escalate: map[string]bool{}}, clock: time.Date(2026, 10, 15, 12, 0, 0, 0, time.UTC)}
	h.bot = New(h.tg, h.ai, Config{
		AdminChatID: adminChat, FallbackEmail: "help@example.com", HistoryTurns: 2,
		RatePerMinute: 5, HandoffTTL: 30 * time.Minute, AnswerTimeout: time.Second,
	}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	h.bot.now = func() time.Time { return h.clock }
	return h
}

func (h *harness) user(chat int64, text string) {
	h.upd++
	h.bot.Handle(context.Background(), telegram.Update{UpdateID: h.upd, Message: &telegram.Message{
		MessageID: h.upd, From: &telegram.User{ID: chat, FirstName: "Ola", Username: "ola"},
		Chat: telegram.Chat{ID: chat, Type: "private"}, Text: text,
	}})
}

func (h *harness) operator(replyTo int64, text string) {
	h.upd++
	h.bot.Handle(context.Background(), telegram.Update{UpdateID: h.upd, Message: &telegram.Message{
		MessageID: 9000 + h.upd, From: &telegram.User{ID: 1, FirstName: "Kari"},
		Chat: telegram.Chat{ID: admin, Type: "supergroup"}, Text: text,
		ReplyToMessage: &telegram.Message{MessageID: replyTo},
	}})
}

func TestStartAndDuplicateUpdates(t *testing.T) {
	h := newHarness(admin)
	u := telegram.Update{UpdateID: 1, Message: &telegram.Message{From: &telegram.User{ID: 7}, Chat: telegram.Chat{ID: 7, Type: "private"}, Text: "/start"}}
	h.bot.Handle(context.Background(), u)
	h.bot.Handle(context.Background(), u) // Telegram re-delivery
	if len(h.tg.out) != 1 || h.tg.out[0].text != msgWelcome {
		t.Fatalf("out = %+v", h.tg.out)
	}
}

func TestAnswerWithSourcesAndBoundedHistory(t *testing.T) {
	h := newHarness(admin)
	h.ai.reply = agent.Reply{Text: "450 NOK [1]", Sources: []string{"Prices › E-bikes"}}
	h.user(7, "price?")
	if got, _ := h.tg.last(7); got != "450 NOK [1]\n\nSources: Prices › E-bikes" {
		t.Fatalf("got %q", got)
	}
	h.ai.reply = agent.Reply{}
	for _, q := range []string{"a", "b", "c"} {
		h.user(7, q)
	}
	last := h.ai.history[len(h.ai.history)-1]
	if len(last) != 4 || last[0].Text != "a" || last[3].Text != "answer to b" {
		t.Fatalf("history not trimmed to 2 turns: %+v", last)
	}
}

func TestFullHandOffFlow(t *testing.T) {
	h := newHarness(admin)
	h.ai.escalate["I want a refund"] = true

	h.user(7, "I want a refund")
	card, cardID := h.tg.last(admin)
	if !strings.Contains(card, "Hand-off: Ola (@ola) · chat 7") || !strings.Contains(card, "complaint_or_refund") {
		t.Fatalf("admin card = %q", card)
	}
	if got, _ := h.tg.last(7); got != msgHandedOff {
		t.Fatalf("customer told %q", got)
	}

	// While with a human, customer messages go to the operator, not the AI.
	calls := h.ai.calls
	h.user(7, "order 1042, paid by card")
	fwd, fwdID := h.tg.last(admin)
	if h.ai.calls != calls || !strings.Contains(fwd, "order 1042") {
		t.Fatalf("message should be forwarded, got %q (ai calls %d→%d)", fwd, calls, h.ai.calls)
	}

	// Operator replies to the card and to the forwarded message.
	h.operator(cardID, "Hi Ola, refund is on its way.")
	if got, _ := h.tg.last(7); got != "👤 Hi Ola, refund is on its way." {
		t.Fatalf("customer got %q", got)
	}
	h.operator(fwdID, "Done, 3-5 days.")
	if got, _ := h.tg.last(7); got != "👤 Done, 3-5 days." {
		t.Fatalf("customer got %q", got)
	}

	// Unknown reply target is ignored.
	before := len(h.tg.out)
	h.operator(424242, "hello?")
	if len(h.tg.out) != before {
		t.Error("reply to unknown message must be ignored")
	}

	// /done hands back to the bot.
	h.operator(cardID, "/done")
	if got, _ := h.tg.last(7); got != msgBackToBot {
		t.Fatalf("customer got %q", got)
	}
	h.user(7, "opening hours?")
	if h.ai.calls != calls+1 {
		t.Error("after /done the AI must answer again")
	}
}

func TestHandOffExpiresAndHumanCommand(t *testing.T) {
	h := newHarness(admin)
	h.user(7, "/human")
	if got, _ := h.tg.last(admin); !strings.Contains(got, "customer_request") {
		t.Fatalf("admin card = %q", got)
	}
	h.clock = h.clock.Add(31 * time.Minute)
	h.user(7, "anyone there?")
	if h.ai.calls != 1 {
		t.Error("after the hand-off TTL the bot answers again")
	}
}

func TestNoOperatorChatFallsBackToEmail(t *testing.T) {
	h := newHarness(0)
	h.ai.escalate["refund"] = true
	h.user(7, "refund")
	if got, _ := h.tg.last(7); !strings.Contains(got, "help@example.com") {
		t.Fatalf("got %q", got)
	}
}

func TestAIErrorHandsOff(t *testing.T) {
	h := newHarness(admin)
	h.ai.err = errors.New("gemini down")
	h.user(7, "hi")
	if got, _ := h.tg.last(admin); !strings.Contains(got, "assistant error") {
		t.Fatalf("admin card = %q", got)
	}
}

func TestRateLimitSlidingWindow(t *testing.T) {
	h := newHarness(admin)
	for i := 0; i < 5; i++ {
		h.user(7, "q")
	}
	h.user(7, "q")
	if got, _ := h.tg.last(7); got != msgSlowDown || h.ai.calls != 5 {
		t.Fatalf("6th message should be limited: %q calls=%d", got, h.ai.calls)
	}
	h.user(8, "other user")
	if h.ai.calls != 6 {
		t.Error("limit is per chat")
	}
	h.clock = h.clock.Add(61 * time.Second)
	h.user(7, "q")
	if h.ai.calls != 7 {
		t.Error("window should slide after a minute")
	}
}

func TestIgnoresGroupsAndNonText(t *testing.T) {
	h := newHarness(admin)
	h.bot.Handle(context.Background(), telegram.Update{UpdateID: 50, Message: &telegram.Message{
		From: &telegram.User{ID: 1}, Chat: telegram.Chat{ID: -5, Type: "group"}, Text: "hi bot"}})
	if len(h.tg.out) != 0 {
		t.Error("group messages must be ignored")
	}
	h.user(7, "")
	if got, _ := h.tg.last(7); got != msgTextOnly {
		t.Errorf("got %q", got)
	}
}

func TestPruneRemovesIdleSessionsButKeepsActiveHandoffs(t *testing.T) {
	h := newHarness(admin)
	h.user(7, "hi")
	h.user(8, "/human")
	h.clock = h.clock.Add(20 * time.Minute)
	if n := h.bot.Prune(10 * time.Minute); n != 1 {
		t.Fatalf("pruned %d, want 1 (chat 8 is still with a human)", n)
	}
	if len(h.bot.handoffs) != 1 {
		t.Error("hand-off link of the active chat must stay")
	}
}

func TestRingSetEvictsOldest(t *testing.T) {
	r := newRingSet(2)
	r.Add(1)
	r.Add(2)
	r.Add(3) // evicts 1
	if !r.Add(1) || r.Add(3) {
		t.Fatal("ring set eviction broken")
	}
}

func TestDispatcherKeepsPerChatOrder(t *testing.T) {
	var mu sync.Mutex
	got := map[int64][]int64{}
	d := NewDispatcher(func(_ context.Context, u telegram.Update) {
		time.Sleep(time.Microsecond)
		mu.Lock()
		got[u.Message.Chat.ID] = append(got[u.Message.Chat.ID], u.UpdateID)
		mu.Unlock()
	}, 4, 1000)
	d.Start(context.Background())
	for i := int64(1); i <= 400; i++ {
		chat := i%7 - 3 // includes negative IDs (groups)
		if !d.Enqueue(telegram.Update{UpdateID: i, Message: &telegram.Message{Chat: telegram.Chat{ID: chat}}}) {
			t.Fatal("queue unexpectedly full")
		}
	}
	d.Stop()
	total := 0
	for chat, ids := range got {
		total += len(ids)
		for i := 1; i < len(ids); i++ {
			if ids[i] < ids[i-1] {
				t.Fatalf("chat %d out of order: %v", chat, ids)
			}
		}
	}
	if total != 400 {
		t.Fatalf("processed %d, want 400", total)
	}
}

func TestDispatcherReportsFullQueue(t *testing.T) {
	block := make(chan struct{})
	d := NewDispatcher(func(context.Context, telegram.Update) { <-block }, 1, 1)
	d.Start(context.Background())
	m := &telegram.Message{Chat: telegram.Chat{ID: 1}}
	d.Enqueue(telegram.Update{UpdateID: 1, Message: m}) // picked up by worker
	time.Sleep(10 * time.Millisecond)
	d.Enqueue(telegram.Update{UpdateID: 2, Message: m}) // fills buffer
	if d.Enqueue(telegram.Update{UpdateID: 3, Message: m}) {
		t.Fatal("expected back-pressure when the queue is full")
	}
	close(block)
	d.Stop()
}
