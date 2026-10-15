package bot

import (
	"context"
	"sync"

	"github.com/StanTolmachov/ai-support-bot/internal/telegram"
)

// Dispatcher runs N workers. Updates are sharded by chat ID, so messages from
// one chat are handled in order while different chats run in parallel.
type Dispatcher struct {
	handle func(context.Context, telegram.Update)
	queues []chan telegram.Update
	wg     sync.WaitGroup
}

func NewDispatcher(handle func(context.Context, telegram.Update), workers, queueSize int) *Dispatcher {
	d := &Dispatcher{handle: handle, queues: make([]chan telegram.Update, workers)}
	for i := range d.queues {
		d.queues[i] = make(chan telegram.Update, queueSize)
	}
	return d
}

func (d *Dispatcher) Start(ctx context.Context) {
	for _, q := range d.queues {
		d.wg.Add(1)
		go func(q chan telegram.Update) {
			defer d.wg.Done()
			for u := range q {
				d.handle(ctx, u)
			}
		}(q)
	}
}

// Enqueue never blocks; it returns false when that chat's worker is full.
func (d *Dispatcher) Enqueue(u telegram.Update) bool {
	var chat int64
	if u.Message != nil {
		chat = u.Message.Chat.ID
	}
	if chat < 0 {
		chat = -chat
	}
	select {
	case d.queues[chat%int64(len(d.queues))] <- u:
		return true
	default:
		return false
	}
}

// Stop closes the queues and waits for in-flight updates to finish.
// Call it only after the webhook server / poller has stopped enqueuing.
func (d *Dispatcher) Stop() {
	for _, q := range d.queues {
		close(q)
	}
	d.wg.Wait()
}
