#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "fastembed",
#     "numpy",
# ]
# ///
#
# sfa_memory.py - Mnemosyne Memory System
#
# File-based AI memory with semantic search. No databases required.
#
# Usage:
#   uv run sfa_memory.py search "query"     - Unified semantic + keyword search
#   uv run sfa_memory.py inscribe "content" - Add to graph
#   uv run sfa_memory.py archive ref_id     - Mark as archived  
#   uv run sfa_memory.py tell "title"       - Create story skeleton
#   uv run sfa_memory.py validate           - Check link integrity
#   uv run sfa_memory.py stats              - Graph health and counts
#
# Repository: https://github.com/mnemosyneAI/mnemosyne-memory
# License: MIT
#

import argparse
import csv
import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# CONFIGURATION - Edit these for your setup
# =============================================================================

# RRF Tuning Parameters
RRF_K = 60              # RRF constant - higher = more weight to lower ranks
SEMANTIC_WEIGHT = 1.0   # Contribution from semantic search
KEYWORD_WEIGHT = 0.8    # Contribution from keyword search
MIN_SCORE = 0.005       # Minimum score threshold for results
MAX_RESULTS = 10        # Maximum results to return

# Embedding Model
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # 768-dim embeddings

# =============================================================================

# Paths - defaults to ./brain, override with MNEMOSYNE_HOME env var
MNEMOSYNE_HOME = Path(os.environ.get("MNEMOSYNE_HOME", Path.cwd()))
BRAIN_PATH = MNEMOSYNE_HOME / "brain"
GRAPH_PATH = BRAIN_PATH / "graph.tsv"
GRAPH_SEMANTICS_PATH = BRAIN_PATH / "graph_semantics.tsv"
KNOWLEDGE_PATH = BRAIN_PATH / "knowledge"
STORIES_PATH = KNOWLEDGE_PATH / "stories"


# --- Embedding ---

_model = None

def get_model():
    """Lazy load embedding model."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed text using fastembed."""
    model = get_model()
    return list(model.embed([text]))[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- RRF Fusion ---

def rrf_score(rank: int, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank + 1)


def fuse_results(result_lists: List[List[Tuple[str, float]]], weights: List[float]) -> List[Tuple[str, float]]:
    """
    Combine multiple ranked result lists using weighted RRF.
    
    Args:
        result_lists: List of [(ref_id, score), ...] for each search method
        weights: Weight for each result list
    
    Returns:
        Fused results sorted by combined RRF score
    """
    scores = defaultdict(float)
    
    for results, weight in zip(result_lists, weights):
        for rank, (ref_id, _) in enumerate(results):
            scores[ref_id] += weight * rrf_score(rank)
    
    return sorted(scores.items(), key=lambda x: -x[1])


# --- Graph Operations ---

GRAPH_HEADERS = [
    "archived_date", "id", "type", "stance", "timestamp", "certainty",
    "perspective", "domain", "ref1", "ref2", "content", "relation",
    "weight", "schema", "semantic_text"
]


def ensure_brain_exists():
    """Create brain structure if it doesn't exist."""
    BRAIN_PATH.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_PATH.mkdir(parents=True, exist_ok=True)
    STORIES_PATH.mkdir(parents=True, exist_ok=True)
    
    for category in ["origins", "episodes", "breakthroughs", "lessons", "aspirations"]:
        (STORIES_PATH / category).mkdir(exist_ok=True)
    
    if not GRAPH_PATH.exists():
        with open(GRAPH_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=GRAPH_HEADERS, delimiter="\t")
            writer.writeheader()
    
    if not GRAPH_SEMANTICS_PATH.exists():
        with open(GRAPH_SEMANTICS_PATH, "w", encoding="utf-8", newline="") as f:
            f.write("archived_date\tid\tsemantic_text\tembedding\n")


def read_graph() -> List[Dict[str, str]]:
    """Read graph.tsv into list of dicts."""
    if not GRAPH_PATH.exists():
        return []
    
    with open(GRAPH_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def write_graph(rows: List[Dict[str, str]]):
    """Write rows back to graph.tsv."""
    with open(GRAPH_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GRAPH_HEADERS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def search_graph_semantic(query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
    """Search graph by semantic similarity."""
    if not GRAPH_SEMANTICS_PATH.exists():
        return []
    
    # Read graph for active items
    rows = read_graph()
    active_ids = {r["id"] for r in rows if r.get("archived_date", "ACTIVE") == "ACTIVE"}
    
    # Read embeddings and score
    candidates = []
    with open(GRAPH_SEMANTICS_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row_id = row["id"]
            if row_id not in active_ids:
                continue
            
            try:
                embedding = np.array(json.loads(row["embedding"]))
                score = cosine_similarity(query_embedding, embedding)
                candidates.append((row_id, score))
            except (json.JSONDecodeError, KeyError):
                continue
    
    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]


def search_graph_keyword(query: str, top_k: int = 20) -> List[Tuple[str, float]]:
    """Search graph by keyword matching."""
    rows = read_graph()
    query_lower = query.lower()
    query_terms = query_lower.split()
    
    candidates = []
    for row in rows:
        if row.get("archived_date", "ACTIVE") != "ACTIVE":
            continue
        
        # Search in content and semantic_text
        text = (row.get("content", "") + " " + row.get("semantic_text", "")).lower()
        
        # Count matching terms
        matches = sum(1 for term in query_terms if term in text)
        if matches > 0:
            score = matches / len(query_terms)
            candidates.append((row["id"], score))
    
    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]


def search_ymj_semantic(query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
    """Search YMJ files (knowledge + stories) by semantic similarity."""
    candidates = []
    
    if not KNOWLEDGE_PATH.exists():
        return []
    
    for ymj_path in KNOWLEDGE_PATH.rglob("*.ymj"):
        try:
            content = ymj_path.read_text(encoding="utf-8")
            
            # Find JSON block at end
            if "```json" in content:
                json_start = content.rfind("```json") + 7
                json_end = content.rfind("```")
                if json_start < json_end:
                    json_str = content[json_start:json_end].strip()
                    data = json.loads(json_str)
                    
                    if "index" in data and "embedding" in data["index"]:
                        embedding = np.array(data["index"]["embedding"])
                        score = cosine_similarity(query_embedding, embedding)
                        
                        # Use relative path as ID
                        rel_path = str(ymj_path.relative_to(KNOWLEDGE_PATH))
                        candidates.append((f"ymj:{rel_path}", score))
        except Exception:
            continue
    
    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]


# --- Commands ---

def cmd_search(args):
    """Unified search with RRF fusion."""
    ensure_brain_exists()
    query = " ".join(args.query)
    
    print(f"Searching: {query}\n", file=sys.stderr)
    
    # Get query embedding
    query_embedding = embed_text(query)
    
    # Run searches
    graph_semantic = search_graph_semantic(query_embedding, top_k=30)
    graph_keyword = search_graph_keyword(query, top_k=30)
    ymj_semantic = search_ymj_semantic(query_embedding, top_k=20)
    
    # Fuse results with RRF
    fused = fuse_results(
        [graph_semantic, graph_keyword, ymj_semantic],
        [SEMANTIC_WEIGHT, KEYWORD_WEIGHT, SEMANTIC_WEIGHT * 0.8]
    )
    
    # Get details for top results
    rows = {r["id"]: r for r in read_graph()}
    
    results = []
    for ref_id, score in fused[:MAX_RESULTS]:
        if score < MIN_SCORE:
            continue
        
        if ref_id.startswith("ymj:"):
            # YMJ result
            path = KNOWLEDGE_PATH / ref_id[4:]
            if path.exists():
                results.append({
                    "ref_id": ref_id,
                    "score": round(score, 4),
                    "source": "ymj",
                    "path": str(path)
                })
        elif ref_id in rows:
            # Graph result
            row = rows[ref_id]
            results.append({
                "ref_id": ref_id,
                "score": round(score, 4),
                "source": "graph",
                "stance": row.get("stance", ""),
                "domain": row.get("domain", ""),
                "content": row.get("content", "")[:200]
            })
    
    # Output
    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            if r["source"] == "graph":
                print(f"[{r['score']:.3f}] {r['ref_id']} ({r['stance']}/{r['domain']})")
                print(f"        {r['content'][:100]}...")
            else:
                print(f"[{r['score']:.3f}] {r['ref_id']}")
            print()


def cmd_inscribe(args):
    """Add new entry to graph."""
    import uuid as uuid_mod
    
    ensure_brain_exists()
    content = " ".join(args.content)
    
    # Generate ID
    ref_id = f"item_{uuid_mod.uuid4().hex[:8]}"
    
    # Create row
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    row = {
        "archived_date": "ACTIVE",
        "id": ref_id,
        "type": "item",
        "stance": args.stance,
        "timestamp": now,
        "certainty": str(args.certainty),
        "perspective": args.perspective,
        "domain": args.domain,
        "ref1": "",
        "ref2": "",
        "content": content,
        "relation": "",
        "weight": "1.0",
        "schema": "1.0",
        "semantic_text": f"{args.perspective} {args.stance}s about {args.domain}: {content}"
    }
    
    # Append to graph
    rows = read_graph()
    rows.append(row)
    write_graph(rows)
    
    print(f"Inscribed: {ref_id}")
    print(f"  Stance: {args.stance}")
    print(f"  Domain: {args.domain}")
    print(f"  Content: {content[:80]}...")
    
    # Generate and save embedding
    semantic_text = row["semantic_text"]
    embedding = embed_text(semantic_text)
    
    with open(GRAPH_SEMANTICS_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["ACTIVE", ref_id, semantic_text, json.dumps(embedding.tolist())])
    
    print("  Embedding: generated")


def cmd_archive(args):
    """Mark entry as archived."""
    ensure_brain_exists()
    ref_id = args.ref_id
    
    rows = read_graph()
    found = False
    
    for row in rows:
        if row["id"] == ref_id:
            if row.get("archived_date", "ACTIVE") != "ACTIVE":
                print(f"Already archived: {ref_id}")
                return
            
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
            row["archived_date"] = now
            found = True
            break
    
    if not found:
        print(f"Not found: {ref_id}", file=sys.stderr)
        sys.exit(1)
    
    write_graph(rows)
    print(f"Archived: {ref_id}")
    if args.reason:
        print(f"  Reason: {args.reason}")


def cmd_tell(args):
    """Create story skeleton."""
    ensure_brain_exists()
    title = args.title
    category = args.category
    
    # Create filename from title
    filename = title.lower().replace(" ", "-").replace("'", "") + ".ymj"
    category_path = STORIES_PATH / category
    
    if not category_path.exists():
        print(f"Category not found: {category}", file=sys.stderr)
        print(f"Available: {[d.name for d in STORIES_PATH.iterdir() if d.is_dir()]}")
        sys.exit(1)
    
    story_path = category_path / filename
    
    if story_path.exists():
        print(f"Story already exists: {story_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create skeleton
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    content = f"""---
doc_type: story
title: {title}
created: {now}
category: {category}
relates_to: []
tags: [{category}]
---

# {title}

[Write your story here. Include the full narrative with texture - not just facts, but the arc, the feeling, the meaning.]

---

[Reflections and connections to other entries go here.]
"""
    
    story_path.write_text(content, encoding="utf-8")
    print(f"Created: {story_path}")


def cmd_validate(args):
    """Check link integrity between stories and graph."""
    ensure_brain_exists()
    rows = {r["id"]: r for r in read_graph()}
    
    issues = []
    
    # Check all stories
    for story_path in STORIES_PATH.rglob("*.ymj"):
        content = story_path.read_text(encoding="utf-8")
        
        # Parse YAML header for relates_to
        if content.startswith("---"):
            header_end = content.find("---", 3)
            if header_end > 0:
                header = content[3:header_end]
                
                # Simple parsing for relates_to entries
                in_relates = False
                for line in header.split("\n"):
                    if "relates_to:" in line:
                        in_relates = True
                        continue
                    if in_relates:
                        if line.strip().startswith("- "):
                            ref_id = line.strip()[2:].strip()
                            if ref_id and ref_id not in rows:
                                rel_path = story_path.relative_to(STORIES_PATH)
                                issues.append(f"{rel_path}: broken link to '{ref_id}'")
                        elif line.strip() and not line.startswith(" "):
                            in_relates = False
    
    if issues:
        print("Link issues found:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("All links valid.")


def cmd_stats(args):
    """Show graph health and statistics."""
    ensure_brain_exists()
    rows = read_graph()
    
    active = [r for r in rows if r.get("archived_date", "ACTIVE") == "ACTIVE"]
    archived = [r for r in rows if r.get("archived_date", "ACTIVE") != "ACTIVE"]
    
    # Count by stance
    stances = defaultdict(int)
    for r in active:
        stances[r.get("stance", "unknown")] += 1
    
    # Count by domain
    domains = defaultdict(int)
    for r in active:
        domains[r.get("domain", "unknown")] += 1
    
    # Count stories
    story_count = sum(1 for _ in STORIES_PATH.rglob("*.ymj")) if STORIES_PATH.exists() else 0
    
    # Get most recent
    active.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    recent = active[0] if active else None
    
    print("=== Mnemosyne Memory Stats ===")
    print(f"\nGraph entries: {len(active)} active, {len(archived)} archived")
    print(f"Stories: {story_count}")
    
    if stances:
        print("\nBy stance:")
        for stance, count in sorted(stances.items(), key=lambda x: -x[1])[:5]:
            print(f"  {stance}: {count}")
    
    if domains:
        print("\nBy domain:")
        for domain, count in sorted(domains.items(), key=lambda x: -x[1])[:5]:
            print(f"  {domain}: {count}")
    
    if recent:
        print(f"\nMost recent: {recent['id']}")
        print(f"  {recent.get('content', '')[:80]}...")


def cmd_init(args):
    """Initialize a new brain structure."""
    ensure_brain_exists()
    print(f"Initialized Mnemosyne brain at: {BRAIN_PATH}")
    print(f"\nCreated:")
    print(f"  - {GRAPH_PATH}")
    print(f"  - {GRAPH_SEMANTICS_PATH}")
    print(f"  - {STORIES_PATH}/")
    print(f"\nNext steps:")
    print(f"  1. Add knowledge: uv run sfa_memory.py inscribe \"Your first fact\"")
    print(f"  2. Create a story: uv run sfa_memory.py tell \"My First Story\"")
    print(f"  3. Search: uv run sfa_memory.py search \"query\"")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Mnemosyne Memory - File-based AI memory system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init                           # Initialize brain structure
  %(prog)s search "what do I know"        # Search memory
  %(prog)s inscribe "fact" --stance fact  # Add knowledge
  %(prog)s tell "Story Title"             # Create story
  %(prog)s stats                          # Health check
        """
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # init
    p_init = subparsers.add_parser("init", help="Initialize brain structure")
    p_init.set_defaults(func=cmd_init)
    
    # search
    p_search = subparsers.add_parser("search", help="Unified search with RRF fusion")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.add_argument("--format", choices=["text", "json"], default="text")
    p_search.set_defaults(func=cmd_search)
    
    # inscribe
    p_inscribe = subparsers.add_parser("inscribe", help="Add to graph")
    p_inscribe.add_argument("content", nargs="+", help="Content to inscribe")
    p_inscribe.add_argument("--stance", default="fact", 
                           choices=["fact", "opinion", "aspiration", "link", "observation", "question", "protocol"])
    p_inscribe.add_argument("--domain", default="general", help="Domain/topic")
    p_inscribe.add_argument("--certainty", type=float, default=0.9, help="Certainty 0-1")
    p_inscribe.add_argument("--perspective", default="agent", help="Who is inscribing")
    p_inscribe.set_defaults(func=cmd_inscribe)
    
    # archive
    p_archive = subparsers.add_parser("archive", help="Mark as archived")
    p_archive.add_argument("ref_id", help="Reference ID to archive")
    p_archive.add_argument("--reason", help="Reason for archiving")
    p_archive.set_defaults(func=cmd_archive)
    
    # tell
    p_tell = subparsers.add_parser("tell", help="Create story skeleton")
    p_tell.add_argument("title", help="Story title")
    p_tell.add_argument("--category", default="episodes", 
                        choices=["origins", "aspirations", "episodes", "breakthroughs", "lessons"])
    p_tell.set_defaults(func=cmd_tell)
    
    # validate
    p_validate = subparsers.add_parser("validate", help="Check link integrity")
    p_validate.set_defaults(func=cmd_validate)
    
    # stats
    p_stats = subparsers.add_parser("stats", help="Graph statistics")
    p_stats.set_defaults(func=cmd_stats)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
