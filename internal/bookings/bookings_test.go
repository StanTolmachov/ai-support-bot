package bookings

import (
	"path/filepath"
	"testing"
)

func TestLookupNormalisesIDAndRequiresLastName(t *testing.T) {
	s := NewStore([]Booking{{ID: "PP-1042", LastName: "Hansen", Status: "confirmed"}})
	for _, id := range []string{"PP-1042", "pp1042", " pp 1042 ", "#PP-1042"} {
		if _, ok := s.Lookup(id, " hansen "); !ok {
			t.Errorf("id %q should match", id)
		}
	}
	if _, ok := s.Lookup("PP-1042", "Olsen"); ok {
		t.Error("wrong last name must not match")
	}
	if _, ok := s.Lookup("PP-1042", ""); ok {
		t.Error("empty last name must not match")
	}
}

func TestDemoDataLoads(t *testing.T) {
	s, err := Load(filepath.Join("..", "..", "data", "bookings.json"))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := s.Lookup("PP-1042", "Hansen"); !ok {
		t.Error("demo booking PP-1042 / Hansen missing")
	}
}
