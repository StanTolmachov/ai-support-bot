package kb

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const doc = `# Prices

Intro line.

## City bikes
City bike: 250 NOK per day.

## E-bikes
E-bike: 450 NOK per day.
`

func TestSplitBySectionsWithTitles(t *testing.T) {
	cs := Split("prices.md", doc)
	if len(cs) != 3 {
		t.Fatalf("chunks = %d, want 3 (intro + 2 sections)", len(cs))
	}
	if cs[1].Title != "Prices › City bikes" || cs[2].ID != "prices.md#3" {
		t.Errorf("unexpected chunk: %+v", cs[1:])
	}
	if !strings.Contains(cs[2].Text, "450") {
		t.Errorf("text lost: %q", cs[2].Text)
	}
}

func TestLongSectionsSplitOnParagraphs(t *testing.T) {
	para := strings.Repeat("word ", 100) // ~500 chars
	cs := Split("x.md", "## Big\n"+para+"\n\n"+para+"\n\n"+para+"\n")
	if len(cs) < 2 {
		t.Fatalf("long section not split: %d chunk", len(cs))
	}
	for _, c := range cs {
		if len(c.Text) > maxChunkChars {
			t.Errorf("chunk too long: %d", len(c.Text))
		}
	}
}

// fakeEmbedder maps keywords to axes so similarity is predictable.
type fakeEmbedder struct{ calls int }

func vec(s string) []float32 {
	s = strings.ToLower(s)
	v := make([]float32, 3)
	if strings.Contains(s, "city") {
		v[0] = 1
	}
	if strings.Contains(s, "e-bike") {
		v[1] = 1
	}
	v[2] = 0.1
	return v
}

func (f *fakeEmbedder) EmbedDocuments(_ context.Context, titles, texts []string) ([][]float32, error) {
	f.calls++
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = vec(titles[i] + " " + texts[i])
	}
	return out, nil
}

func TestBuildAndSearchRanksBySimilarity(t *testing.T) {
	idx, err := Build(context.Background(), &fakeEmbedder{}, "m", Split("prices.md", doc))
	if err != nil {
		t.Fatal(err)
	}
	hits := idx.Search([]float32{0, 5, 0}, 2) // unnormalised query on purpose
	if len(hits) != 2 || hits[0].Chunk.Title != "Prices › E-bikes" {
		t.Fatalf("top hit = %+v", hits[0].Chunk.Title)
	}
	if hits[0].Score < 0.99 || hits[0].Score > 1.0001 {
		t.Errorf("cosine of identical direction should be ~1, got %f", hits[0].Score)
	}
}

func TestLoadOrBuildReusesIndexUntilContentChanges(t *testing.T) {
	dir := t.TempDir()
	kbDir := filepath.Join(dir, "kb")
	os.Mkdir(kbDir, 0o755)
	os.WriteFile(filepath.Join(kbDir, "prices.md"), []byte(doc), 0o644)
	indexPath := filepath.Join(dir, "data", "index.json")
	emb := &fakeEmbedder{}

	_, rebuilt, err := LoadOrBuild(context.Background(), emb, "m", kbDir, indexPath)
	if err != nil || !rebuilt {
		t.Fatalf("first run should build: rebuilt=%v err=%v", rebuilt, err)
	}
	_, rebuilt, _ = LoadOrBuild(context.Background(), emb, "m", kbDir, indexPath)
	if rebuilt || emb.calls != 1 {
		t.Fatalf("unchanged KB must reuse the index (calls=%d)", emb.calls)
	}
	os.WriteFile(filepath.Join(kbDir, "prices.md"), []byte(strings.Replace(doc, "450", "500", 1)), 0o644)
	idx, rebuilt, _ := LoadOrBuild(context.Background(), emb, "m", kbDir, indexPath)
	if !rebuilt || emb.calls != 2 {
		t.Fatal("changed KB must trigger a rebuild")
	}
	if !strings.Contains(idx.Chunks[2].Text, "500") {
		t.Error("index has stale content")
	}
	_, rebuilt, _ = LoadOrBuild(context.Background(), emb, "other-model", kbDir, indexPath)
	if !rebuilt {
		t.Error("changing the embedding model must trigger a rebuild")
	}
}

func TestEmptyDirIsAnError(t *testing.T) {
	if _, err := LoadDir(t.TempDir()); err == nil {
		t.Fatal("expected error for empty KB")
	}
}

func TestDemoKnowledgeBaseSplitsCleanly(t *testing.T) {
	cs, err := LoadDir(filepath.Join("..", "..", "kb"))
	if err != nil {
		t.Fatal(err)
	}
	if len(cs) < 20 {
		t.Errorf("demo KB has only %d chunks", len(cs))
	}
	ids := map[string]bool{}
	for _, c := range cs {
		if ids[c.ID] {
			t.Errorf("duplicate chunk id %s", c.ID)
		}
		ids[c.ID] = true
		if strings.TrimSpace(c.Text) == "" || len(c.Text) > maxChunkChars || !strings.Contains(c.Title, " › ") {
			t.Errorf("bad chunk %s: title=%q len=%d", c.ID, c.Title, len(c.Text))
		}
	}
}
