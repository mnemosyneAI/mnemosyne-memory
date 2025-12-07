# Quick Start Guide

Get Mnemosyne running in 5 minutes.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
git clone https://github.com/mnemosyneAI/mnemosyne-memory.git
cd mnemosyne-memory
```

## Initialize Your Brain

```bash
uv run scripts/sfa_memory.py init
```

This creates:
- `brain/graph.tsv` - Your knowledge graph
- `brain/graph_semantics.tsv` - Search embeddings
- `brain/knowledge/stories/` - Story directories

## Add Your First Knowledge

```bash
# Add a fact
uv run scripts/sfa_memory.py inscribe "I started using Mnemosyne today" --stance fact --domain journal

# Add an opinion
uv run scripts/sfa_memory.py inscribe "File-based memory is better than databases" --stance opinion --domain philosophy
```

## Search Your Memory

```bash
uv run scripts/sfa_memory.py search "mnemosyne"
```

## Create a Story

```bash
uv run scripts/sfa_memory.py tell "Why I Chose Mnemosyne" --category origins
```

Edit the created file at `brain/knowledge/stories/origins/why-i-chose-mnemosyne.ymj`.

## Check Health

```bash
uv run scripts/sfa_memory.py stats
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Read [FORMAT-SPEC.md](FORMAT-SPEC.md) for file format details
- Check the examples in `examples/`
