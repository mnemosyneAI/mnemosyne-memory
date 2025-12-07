# Mnemosyne Architecture

## Design Principles

1. **Files over databases** - Everything is a file. Git-friendly, grep-friendly, human-readable.
2. **Simplicity over features** - Do one thing well. Memory, not a framework.
3. **Ownership over convenience** - Your data, your files, your control.
4. **Search over structure** - Semantic search finds what matters. Don't over-organize.

## Directory Structure

```
brain/
├── graph.tsv              # The knowledge graph
├── graph_semantics.tsv    # Pre-computed embeddings
├── knowledge/             # Your understanding
│   ├── stories/           # Narrative documents
│   │   ├── origins/       # How things began
│   │   ├── episodes/      # Significant events
│   │   ├── breakthroughs/ # Moments of change
│   │   ├── lessons/       # Hard-won wisdom
│   │   └── aspirations/   # Goals and why they matter
│   └── foundations/       # Core knowledge
└── reference/             # External knowledge ingested
```

## Data Flow

```
Input → Inscribe → graph.tsv → Embed → graph_semantics.tsv
                                            ↓
Query → Embed → Search (Semantic + Keyword) → RRF Fusion → Results
```

## Search: RRF Fusion

Mnemosyne combines two search methods:

1. **Semantic search** - Embedding similarity (understands meaning)
2. **Keyword search** - Term matching (catches exact phrases)

Results are combined using Reciprocal Rank Fusion:

```
RRF_score(item) = Σ (weight / (k + rank))
```

This gives better results than either method alone.

## The Graph

The graph is a simple TSV with entries that have:

- **Identity** - Unique ID, timestamp, domain
- **Stance** - What type of knowledge (fact, opinion, aspiration, etc.)
- **Content** - The actual information
- **Certainty** - How confident (0-1)
- **Links** - Optional connections to other entries

## Stories

Stories are YMJ files (YAML + Markdown + JSON) that:

- Link TO graph entries via `relates_to` in YAML header
- Contain narrative with context and meaning
- Have embeddings for search in JSON footer

Graph entries never link to stories (external data is brittle).

## Temporal Model

- `timestamp` - When the fact became true (can be backdated)
- `archived_date` - When the fact became invalid ("ACTIVE" if current)

No need for complex temporal logic. Archive date = invalidation date.
