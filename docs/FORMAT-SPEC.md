# Format Specifications

## graph.tsv

Tab-separated values with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| archived_date | string | "ACTIVE" or ISO date when archived |
| id | string | Unique identifier (e.g., "rubicon_moment") |
| type | string | "item" or "link" |
| stance | string | fact, opinion, aspiration, observation, link, question, protocol |
| timestamp | ISO8601 | When the fact became true |
| certainty | float | Confidence level 0.0-1.0 |
| perspective | string | Who inscribed this (e.g., "agent", "user") |
| domain | string | Topic area (e.g., "science", "journal") |
| ref1 | string | For links: source entry ID |
| ref2 | string | For links: target entry ID |
| content | string | The actual knowledge |
| relation | string | For links: relationship type (supports, contradicts, etc.) |
| weight | float | Link strength 0.0-1.0 |
| schema | string | Schema version (currently "1.0") |
| semantic_text | string | Full text used for embedding |

### Stance Types

| Stance | Use For | Example |
|--------|---------|---------|
| fact | Known true | "Water boils at 100°C at sea level" |
| opinion | Belief with uncertainty | "Python is best for prototyping" |
| aspiration | Goals and commitments | "I want to understand consciousness" |
| observation | Witnessed patterns | "Users ask follow-ups after long responses" |
| link | Connections between entries | ref1 → ref2 with relation type |
| question | Open inquiries | "Is consciousness substrate-independent?" |
| protocol | Operational rules | "Always cite sources" |

## graph_semantics.tsv

Tab-separated with:

| Field | Type | Description |
|-------|------|-------------|
| archived_date | string | "ACTIVE" or archived date |
| id | string | Matches graph.tsv id |
| semantic_text | string | Text that was embedded |
| embedding | JSON array | 768-dimensional float vector |

## YMJ Format (Stories and Knowledge)

Three sections in one file:

### 1. YAML Header

```yaml
---
doc_type: story          # or "knowledge", "reference"
title: Document Title
created: 2025-01-01
category: origins        # for stories
relates_to:              # links to graph entries
  - entry_id_1
  - entry_id_2
tags: [tag1, tag2]
---
```

### 2. Markdown Body

```markdown
# Title

Your content here. Full narrative with context.

---

Reflections and notes.
```

### 3. JSON Footer (embeddings)

```json
~~~json
{
  "schema": 1,
  "index": {
    "tags": ["tag1", "tag2"],
    "title": "document_title",
    "embedding": [0.123, -0.456, ...]
  }
}
~~~
```

## Link Relation Types

For `link` stance entries:

| Relation | Meaning |
|----------|---------|
| supports | Evidence for |
| contradicts | Evidence against |
| elaborates | Adds detail |
| causal | Causes or leads to |
| evidential | Provides evidence |
| enables | Makes possible |
| requires | Depends on |
