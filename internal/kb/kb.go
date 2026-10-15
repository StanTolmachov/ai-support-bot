// Package kb loads a Markdown knowledge base, splits it into sections,
// embeds them and answers nearest-neighbour queries (cosine similarity).
//
// A few hundred chunks fit comfortably in memory, so there is no vector
// database: the index is a JSON file rebuilt only when the content changes.
package kb

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Chunk is one retrievable section of the knowledge base.
type Chunk struct {
	ID     string    `json:"id"`     // "prices.md#2"
	Source string    `json:"source"` // file name
	Title  string    `json:"title"`  // "Prices › E-bikes"
	Text   string    `json:"text"`
	Vector []float32 `json:"vector,omitempty"`
}

type Hit struct {
	Chunk Chunk
	Score float64
}

// Embedder is satisfied by *gemini.Client.
type Embedder interface {
	EmbedDocuments(ctx context.Context, titles, texts []string) ([][]float32, error)
}

type Index struct {
	Model  string  `json:"model"`
	Hash   string  `json:"hash"`
	Chunks []Chunk `json:"chunks"`
}

const maxChunkChars = 1200

// LoadDir reads every *.md file in dir and splits it into chunks.
func LoadDir(dir string) ([]Chunk, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.md"))
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	var chunks []Chunk
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, Split(filepath.Base(f), string(data))...)
	}
	if len(chunks) == 0 {
		return nil, fmt.Errorf("kb: no markdown content in %s", dir)
	}
	return chunks, nil
}

// Split turns one Markdown document into chunks: one per "## " section,
// with long sections split further on paragraph boundaries.
func Split(source, md string) []Chunk {
	docTitle := strings.TrimSuffix(source, filepath.Ext(source))
	type section struct{ heading, body string }
	var sections []section
	cur := section{}
	for _, line := range strings.Split(strings.ReplaceAll(md, "\r\n", "\n"), "\n") {
		switch {
		case strings.HasPrefix(line, "# "):
			docTitle = strings.TrimSpace(line[2:])
		case strings.HasPrefix(line, "## "):
			if strings.TrimSpace(cur.body) != "" {
				sections = append(sections, cur)
			}
			cur = section{heading: strings.TrimSpace(line[3:])}
		default:
			cur.body += line + "\n"
		}
	}
	if strings.TrimSpace(cur.body) != "" {
		sections = append(sections, cur)
	}

	var out []Chunk
	for _, s := range sections {
		title := docTitle
		if s.heading != "" {
			title += " › " + s.heading
		}
		for _, part := range splitLong(strings.TrimSpace(s.body), maxChunkChars) {
			out = append(out, Chunk{
				ID:     fmt.Sprintf("%s#%d", source, len(out)+1),
				Source: source,
				Title:  title,
				Text:   part,
			})
		}
	}
	return out
}

func splitLong(text string, limit int) []string {
	if len(text) <= limit {
		return []string{text}
	}
	var parts []string
	var cur strings.Builder
	for _, para := range strings.Split(text, "\n\n") {
		if cur.Len() > 0 && cur.Len()+len(para)+2 > limit {
			parts = append(parts, strings.TrimSpace(cur.String()))
			cur.Reset()
		}
		cur.WriteString(para + "\n\n")
	}
	if strings.TrimSpace(cur.String()) != "" {
		parts = append(parts, strings.TrimSpace(cur.String()))
	}
	return parts
}

// ContentHash identifies a knowledge-base version + embedding model.
func ContentHash(model string, chunks []Chunk) string {
	h := sha256.New()
	h.Write([]byte(model))
	for _, c := range chunks {
		h.Write([]byte{0})
		h.Write([]byte(c.ID + "\x00" + c.Title + "\x00" + c.Text))
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// Build embeds the chunks and returns a ready index.
func Build(ctx context.Context, emb Embedder, model string, chunks []Chunk) (*Index, error) {
	titles := make([]string, len(chunks))
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		titles[i], texts[i] = c.Title, c.Text
	}
	vecs, err := emb.EmbedDocuments(ctx, titles, texts)
	if err != nil {
		return nil, fmt.Errorf("kb: embedding: %w", err)
	}
	if len(vecs) != len(chunks) {
		return nil, errors.New("kb: embedder returned wrong number of vectors")
	}
	out := make([]Chunk, len(chunks))
	for i, c := range chunks {
		c.Vector = normalize(vecs[i])
		out[i] = c
	}
	return &Index{Model: model, Hash: ContentHash(model, chunks), Chunks: out}, nil
}

// LoadOrBuild reuses the index file if the content hash matches, otherwise
// rebuilds and saves it. Saves API calls and start-up time.
func LoadOrBuild(ctx context.Context, emb Embedder, model, kbDir, indexPath string) (*Index, bool, error) {
	chunks, err := LoadDir(kbDir)
	if err != nil {
		return nil, false, err
	}
	want := ContentHash(model, chunks)
	if idx, err := Load(indexPath); err == nil && idx.Hash == want {
		return idx, false, nil
	}
	idx, err := Build(ctx, emb, model, chunks)
	if err != nil {
		return nil, false, err
	}
	return idx, true, idx.Save(indexPath)
}

func Load(path string) (*Index, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var idx Index
	if err := json.Unmarshal(data, &idx); err != nil {
		return nil, err
	}
	return &idx, nil
}

// Save writes atomically (temp file + rename) so a crash never leaves a
// half-written index behind.
func (idx *Index) Save(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.Marshal(idx)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Search returns the k most similar chunks to the query vector.
func (idx *Index) Search(query []float32, k int) []Hit {
	q := normalize(query)
	hits := make([]Hit, 0, len(idx.Chunks))
	for _, c := range idx.Chunks {
		hits = append(hits, Hit{Chunk: c, Score: dot(q, c.Vector)})
	}
	sort.SliceStable(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if k < len(hits) {
		hits = hits[:k]
	}
	return hits
}

func normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	n := math.Sqrt(sum)
	out := make([]float32, len(v))
	if n == 0 {
		return out
	}
	for i, x := range v {
		out[i] = float32(float64(x) / n)
	}
	return out
}

func dot(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}
