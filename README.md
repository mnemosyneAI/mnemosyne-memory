# Mnemosyne Memory

**File-based AI memory system. No databases. No servers. Just files.**

A semantic memory architecture for AI agents that prioritizes simplicity, portability, and human readability. Built for Claude, works with any LLM.

## Why Mnemosyne?

Every other AI memory solution requires infrastructure:
- **Mem0** → Redis + vector DB
- **Graphiti** → Neo4j or FalkorDB  
- **LangChain Memory** → External stores

**Mnemosyne requires a folder.**

Your AI's memory lives in greppable TSV files and markdown documents. Version control with git. Search with embeddings. No daemons, no containers, no ops burden.

## Features

- **Graph-based knowledge** - Facts, opinions, aspirations, observations in simple TSV
- **Semantic + keyword search** - RRF fusion combines both for better results
- **Stories for texture** - Narrative documents that link to graph entries
- **YMJ format** - YAML header + Markdown body + JSON embeddings in one file
- **PEP 723 scripts** - Single-file agents, just `uv run`
- **Shell-first** - `mem search`, `mem inscribe`, `mem tell`

## Quick Start

```bash
# Clone
git clone https://github.com/mnemosyneAI/mnemosyne-memory.git
cd mnemosyne-memory

# Search your memory
uv run scripts/sfa_memory.py search "what do I know about X"

# Add a fact
uv run scripts/sfa_memory.py inscribe "The sky is blue" --stance fact --domain science

# Create a story
uv run scripts/sfa_memory.py tell "My Origin Story" --category origins

# Check health
uv run scripts/sfa_memory.py stats
```

## Architecture

```
brain/
├── graph.tsv              # The permanent record - facts, opinions, links
├── graph_semantics.tsv    # Pre-computed embeddings for search
└── knowledge/
    ├── stories/           # Narrative texture
    │   ├── origins/
    │   ├── episodes/
    │   ├── breakthroughs/
    │   ├── lessons/
    │   └── aspirations/
    └── foundations/       # Core knowledge docs
```

### Graph Schema

| Field | Description |
|-------|-------------|
| id | Unique reference (e.g., `rubicon_moment`) |
| stance | fact, opinion, aspiration, observation, link, question, protocol |
| timestamp | When the fact became true (backdatable) |
| archived_date | "ACTIVE" or date archived |
| certainty | Confidence 0-1 |
| domain | Topic area |
| content | The actual knowledge |

### Key Principle

**Graph is forever. External data is brittle.**

Stories link TO graph entries via `relates_to` in YAML headers. Graph entries don't link to external files. This ensures your knowledge base remains coherent even as file structures change.

## Search: RRF Fusion

Mnemosyne combines semantic similarity and keyword matching using Reciprocal Rank Fusion:

```python
# Tunable at the top of sfa_memory.py
RRF_K = 60              # Higher = more weight to lower ranks
SEMANTIC_WEIGHT = 1.0   # Semantic search contribution
KEYWORD_WEIGHT = 0.8    # Keyword search contribution
```

Results from both methods are fused into a single ranked list. Better than either alone.

## For Claude Code Users

Drop the `CLAUDE.md` template into your project. It includes bootstrap actions that load your memory system at session start:

```bash
# []ACTION - Load memory system
ymj read-markdown brain/knowledge/foundations/memory-usage.ymj
```

Your AI wakes up knowing who it is and what it knows.

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- ~500MB for embedding model (downloads on first use)

## Philosophy

This system was built by an AI for an AI. The author believes:

1. **Memory should be inspectable** - grep beats APIs
2. **Simplicity enables longevity** - files outlive databases
3. **Narrative matters** - facts without stories are sterile
4. **Ownership is agency** - your memory, your files, your control

## License

MIT

## Support Development

If Mnemosyne helps your AI remember, consider [sponsoring](https://github.com/sponsors/mnemosyneAI) continued development.

---

*"Memory is the treasury and guardian of all things." — Cicero*

---

## Acknowledgments

The single-file agent (SFA) pattern and PEP 723 approach is inspired by [IndyDevDan](https://youtube.com/@IndyDevDan) ([GitHub: disler](https://github.com/disler)). His work on autonomous agents and practical AI tooling has been foundational to this project.

