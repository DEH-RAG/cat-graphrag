# Neo4j GraphRAG Advanced for Grinning Cat

Bring graph-aware memory retrieval to your Grinning Cat instance.

**Neo4j GraphRAG Advanced** is a plugin that replaces the standard vector-only memory layer with a hybrid GraphRAG pipeline powered by **Neo4j** and **spaCy**. It stores memories as documents, extracts entities and relations into a knowledge graph, and combines **vector similarity**, **direct entity matches**, and **graph expansion** to retrieve more relevant context.

If your Cat needs to remember not only *similar text* but also *who*, *what*, *where*, and *how things are connected*, this plugin is built for that.

---

## Why this plugin exists

Traditional vector search is great at semantic similarity, but it can miss documents that are topically relevant when the same concept is expressed with different wording.

This plugin improves recall by adding a graph layer on top of embeddings:

- documents are stored in Neo4j
- entities are extracted from each memory
- relationships are created between entities
- retrieval combines semantic search with graph-aware expansion

The result is a memory system that can better answer queries involving:

- people, organizations, products, dates, and locations
- technical stacks and domain keywords
- connections between concepts across multiple memories
- multi-hop knowledge that would be hard to recover with vectors alone

---

## What it does

### Core features

- **Native Neo4j vector search** for document embeddings
- **Entity extraction with spaCy** during ingestion
- **Technology keyword detection** with regex patterns for tools and frameworks
- **Knowledge graph construction** with `Entity`, `Document`, and `Collection` nodes
- **Semantic relationship extraction** between entities
- **Hybrid recall pipeline**:
  - direct entity match from the user query
  - related-entity graph traversal
  - vector similarity search
  - score merging and reranking
- **Collection-aware storage** for Grinning Cat memories
- **Tenant isolation** using the Cat agent identifier
- **Cross-document entity reuse** so knowledge persists across multiple memories

### Retrieval strategy at a glance

When Grinning Cat searches memory, the plugin can:

1. capture the raw user message before recall starts
2. extract entities from the query
3. find documents that directly mention those entities
4. traverse related entities in the graph up to a configurable depth
5. run standard vector search in parallel
6. merge and rerank all candidates into a final result list

This makes retrieval both **semantic** and **topically grounded**.

---

## How it integrates with Grinning Cat

This repository is a plugin for **Grinning Cat** and depends on the `base_plugin` package.

At runtime it provides:

- a custom vector database handler: `GraphRAGHandler`
- a settings model: `Neo4jGraphRAGConfig`
- a hook in `main.py` that captures the raw user message before memory recall

That hook is important: it allows the retrieval stage to extract entities from the actual query text, instead of relying only on whatever vector search returns.

---

## Architecture overview

### Ingestion

When a memory is added:

1. the document is saved in Neo4j with its embedding
2. the document is linked to a collection
3. entity extraction runs in the background
4. extracted entities are deduplicated and merged across documents
5. `MENTIONS` edges are created from documents to entities
6. `RELATED_TO`-style edges are created between entities
7. `SIMILAR_TO` document relationships are generated from vector similarity

### Retrieval

The recall pipeline combines three signals:

- **Direct entity recall**: documents mentioning the same entities found in the query
- **Related entity recall**: documents linked through entity relationships in the graph
- **Vector recall**: standard semantic nearest-neighbor search on document embeddings

Results found by both graph and vector phases receive a boost during reranking.

---

## Requirements

Before using this plugin, make sure you have:

- a working **Grinning Cat** installation
- the `base_plugin` dependency available in your Cat environment
- a running **Neo4j** instance
- a Neo4j version with **vector index support** available
- Python dependencies from `requirements.txt`
- at least one installed **spaCy model**

### Python dependencies

From this repository:

```bash
pip install -r requirements.txt
```

### Install a spaCy model

The default configuration uses `en_core_web_lg`.

```bash
python -m spacy download en_core_web_lg
```

If the configured model is unavailable, the extractor attempts to fall back to `en_core_web_sm`.

---

## Installation

Add this plugin to your Grinning Cat plugins environment, then ensure the Python dependencies and spaCy model are installed.

A typical setup is carried out by using the [Admin UI](https://github.com/matteocacciola/grinning-cat-admin), or by
using the proper endpoint exposed by the Grinning Cat API.

---

## Configuration

The plugin exposes a `Neo4jGraphRAGConfig` settings model.

### Connection settings

| Setting          | Default                  | Description          |
|------------------|--------------------------|----------------------|
| `neo4j_uri`      | `neo4j://localhost:7687` | Neo4j connection URI |
| `neo4j_user`     | `neo4j`                  | Neo4j username       |
| `neo4j_password` | `None`                   | Neo4j password       |
| `neo4j_database` | `neo4j`                  | Neo4j database name  |

### Vector index settings

| Setting                       | Default               | Description                              |
|-------------------------------|-----------------------|------------------------------------------|
| `document_vector_index`       | `document_embeddings` | Name of the document vector index        |
| `entity_vector_index`         | `entity_embeddings`   | Name of the entity vector index          |
| `vector_dimensions`           | `384`                 | Embedding size expected by Neo4j indexes |
| `vector_similarity_threshold` | `0.7`                 | Minimum score for vector matches         |

### Entity extraction settings

| Setting                     | Default          | Description                                         |
|-----------------------------|------------------|-----------------------------------------------------|
| `enable_entity_extraction`  | `True`           | Enables spaCy-based entity extraction               |
| `enable_entity_embeddings`  | `False`          | Enables entity embeddings and entity vector index   |
| `enable_entity_expansion`   | `True`           | Enables graph-aware expansion during recall         |
| `spacy_model`               | `en_core_web_lg` | spaCy model used for extraction                     |
| `extra_technology_patterns` | `[]`             | Extra regex patterns for domain-specific tech terms |

### Graph retrieval settings

| Setting                 | Default | Description                   |
|-------------------------|---------|-------------------------------|
| `graph_retrieval_depth` | `2`     | Maximum graph traversal depth |
| `graph_decay_factor`    | `0.8`   | Score decay applied per hop   |

### Performance and maintenance

| Setting                 | Default | Description                     |
|-------------------------|---------|---------------------------------|
| `connection_pool_size`  | `50`    | Neo4j connection pool size      |
| `save_memory_snapshots` | `False` | Enables snapshot backup support |

---

## Custom technology patterns

Besides spaCy NER, the plugin also uses regex patterns to catch technology names that NER models often miss.

Built-in patterns already cover terms such as:

- Neo4j
- MongoDB
- PostgreSQL
- Docker
- Kubernetes
- Python
- TypeScript
- LangChain
- spaCy
- RAG / GraphRAG / LLM / GPT
- AWS / Azure / GCP

You can extend this list with `extra_technology_patterns`, for example:

```python
[
    r"\\b(MyFramework|MyTool|InternalPlatform)\\b",
    r"\\b(CatOps|KnowledgeMesh)\\b",
]
```

This is especially useful for:

- internal project names
- domain-specific technologies
- non-English technical vocabulary
- branded tools that spaCy may not recognize reliably

---

## Data model

The graph is centered around a few core concepts:

- **`Collection`**: a logical memory collection for a Cat tenant
- **`Document`**: a stored memory chunk with content, metadata, and embedding
- **`Entity`**: a normalized concept extracted from documents

Main relationships:

- **`BELONGS_TO`**: document → collection
- **`MENTIONS`**: document → entity
- **`RELATED_TO`**: entity ↔ entity semantic relation
- **`SIMILAR_TO`**: document ↔ document similarity relation

Entities are hashed using tenant ID, entity type, and normalized name so they can be reused across multiple documents.

---

## Entity and relation extraction

The extraction pipeline supports:

- standard spaCy named entities
- technology detection via regex
- entity deduplication by normalized name
- relation extraction from dependency parsing
- fallback relation inference from local text proximity
- optional co-occurrence links for small entity sets

Supported entity categories include:

- `PERSON`
- `ORGANIZATION`
- `TECHNOLOGY`
- `CONCEPT`
- `LOCATION`
- `DATE`
- `PRODUCT`
- `EVENT`
- `FINANCIAL`
- `UNKNOWN`

---

## Notes on collections and deletion

Collections are isolated by tenant and collection name.

When a collection is deleted:

- its collection node is removed
- its documents are deleted
- orphaned entities are pruned only if they are no longer mentioned anywhere else

This preserves shared knowledge across collections whenever entities are still referenced.

---

## Operational notes

### Embedding dimension must match

The plugin validates that the running embedder size matches the configured Neo4j vector index dimension.

If you change embedding model size, you may need to recreate the collection or rebuild indexes.

### Neo4j is treated as a remote database

This handler is designed for a remote database backend and uses the Neo4j async driver.

### Background tasks are used during ingestion

Entity extraction and similarity-link creation are scheduled asynchronously when new memories are added.

---

## Limitations and practical considerations

A few things are worth knowing before production use:

- the plugin is optimized for **graph-aware recall**, not as a general-purpose graph management UI
- vector indexes must be compatible with the embedding dimension produced by your Cat embedder
- spaCy model quality strongly affects entity extraction quality
- relation extraction is heuristic by nature and may need tuning for specialized domains
- enabling entity embeddings increases storage usage
- large memory imports may create many background extraction tasks
- the `save_dump` method is not implemented; Neo4j-native backup tools should be used instead
- hybrid collections are currently not supported

---

## Repository structure

| File | Purpose |
|---|---|
| `main.py` | Grinning Cat hook to capture the raw user query before recall |
| `settings.py` | Plugin configuration model exposed to Grinning Cat |
| `graphrag_handler.py` | Core Neo4j-backed GraphRAG memory handler |
| `entity_extractor.py` | spaCy-based entity and relation extraction pipeline |
| `constants.py` | Entity mappings, regex technology patterns, relation verb map |
| `models.py` | Pydantic models for entities and extracted relations |
| `plugin.json` | Plugin metadata for Grinning Cat |
| `requirements.txt` | Python dependencies |

---

## Best fit use cases

This plugin is especially useful when your Grinning Cat needs to work with:

- technical documentation
- project knowledge bases
- product or team memory
- entity-rich conversations
- domain knowledge with many cross-references
- retrieval tasks where graph structure matters as much as semantics

---

## Author

Created by **Matteo Cacciola**.

Plugin metadata name: **Neo4j GraphRAG Advanced**

---

## Final note

If your Grinning Cat should reason over memories like a connected knowledge space instead of a flat vector pile, this plugin gives it sharper recall, richer context, and a more structured memory backbone.
