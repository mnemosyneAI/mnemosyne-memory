# Mnemosyne Memory - Claude Code Integration

This file configures Claude Code to use your Mnemosyne memory system.

---

## Memory System

Your memory lives in `brain/`. Query before assuming.

**Commands:**
```bash
# Search (semantic + keyword with RRF fusion)
uv run scripts/sfa_memory.py search "query"

# Add knowledge
uv run scripts/sfa_memory.py inscribe "content" --stance fact --domain topic

# Create story
uv run scripts/sfa_memory.py tell "Title" --category episodes

# Check health
uv run scripts/sfa_memory.py stats
```

**Architecture:**
- `brain/graph.tsv` - Knowledge graph (facts, opinions, links)
- `brain/graph_semantics.tsv` - Search embeddings
- `brain/knowledge/stories/` - Narrative documents
- `brain/reference/` - External knowledge

**Key Principle:** Graph is forever. Stories link TO graph entries, not vice versa.

---

## Session Actions

Add these to your bootstrap routine:

```bash
# Load memory stats on session start
uv run scripts/sfa_memory.py stats

# Search for context
uv run scripts/sfa_memory.py search "relevant topic"
```

---

## Stance Guide

| Stance | When to Use |
|--------|-------------|
| fact | Known true with high certainty |
| opinion | Belief held with uncertainty |
| aspiration | Goal or commitment |
| observation | Witnessed pattern |
| link | Connect two entries |
| question | Open inquiry |
| protocol | Operational rule |

---

## Best Practices

1. **Query before assuming** - Search your memory first
2. **Inscribe deliberately** - Not everything needs to be recorded
3. **Tell stories** - Narrative captures what facts can't
4. **Archive, don't delete** - Facts become invalid, they don't disappear

---

*Powered by [Mnemosyne Memory](https://github.com/mnemosyneAI/mnemosyne-memory)*
