// Package bookings is the demo "backend" the agent can query through a tool.
// In a real project this would call the client's booking system API.
package bookings

import (
	"encoding/json"
	"os"
	"strings"
)

type Booking struct {
	ID       string `json:"id"`
	LastName string `json:"last_name"`
	Bike     string `json:"bike"`
	Start    string `json:"start"` // YYYY-MM-DD HH:MM
	End      string `json:"end"`
	Pickup   string `json:"pickup"`
	Status   string `json:"status"` // confirmed | active | returned | cancelled
	Paid     bool   `json:"paid"`
}

type Store struct {
	byID map[string]Booking
}

func Load(path string) (*Store, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var list []Booking
	if err := json.Unmarshal(data, &list); err != nil {
		return nil, err
	}
	return NewStore(list), nil
}

func NewStore(list []Booking) *Store {
	s := &Store{byID: make(map[string]Booking, len(list))}
	for _, b := range list {
		s.byID[normID(b.ID)] = b
	}
	return s
}

// Lookup requires both the booking ID and the last name. A wrong last name
// looks exactly like a missing booking, so the bot can't be used to probe
// which booking IDs exist.
func (s *Store) Lookup(id, lastName string) (Booking, bool) {
	b, ok := s.byID[normID(id)]
	if !ok || !strings.EqualFold(strings.TrimSpace(b.LastName), strings.TrimSpace(lastName)) {
		return Booking{}, false
	}
	return b, true
}

func normID(id string) string {
	return strings.ToUpper(strings.NewReplacer(" ", "", "-", "", "#", "").Replace(strings.TrimSpace(id)))
}
