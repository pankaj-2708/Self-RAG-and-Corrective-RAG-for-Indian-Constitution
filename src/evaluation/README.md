# evaluation

End-to-end [Ragas](https://docs.ragas.io/) evaluation pipeline for the Self-RAG system. Builds a synthetic test set from the legal corpus and scores the pipeline across four RAG quality metrics.

## Files

| File | Description |
|---|---|
| [`knowledge_graph.py`](knowledge_graph.py) | Builds a Ragas `KnowledgeGraph` over the Constitution + IPC corpus, capturing entity and relationship structure. Output saved to `data/knowledge_graph.json`. |
| [`clustering.py`](clustering.py) | Embeds all document chunks with `all-mpnet-base-v2`, sweeps K-means (k=4–24), selects optimal k via silhouette score, and stratified-samples 100 representative documents for diverse test set coverage. |
| [`test_set_generation.py`](test_set_generation.py) | Uses the Ragas `TestsetGenerator` with the knowledge graph and sampled documents to synthesise diverse evaluation queries (single-hop, multi-hop) with reference answers and contexts. Output saved to `data/test_set.csv`. |
| [`evaluate.py`](evaluate.py) | Runs the full LangGraph pipeline on each test query, scores results with Ragas, and saves per-query metrics to `data/evaluation_progress_results/`. Supports **resumable runs** — if interrupted, it picks up where it left off. |

## Pipeline

```
knowledge_graph.py  →  clustering.py  →  test_set_generation.py  →  evaluate.py
```

This pipeline is tracked by DVC (`dvc.yaml` in the project root). Run it reproducibly with:

```bash
dvc repro
```

Or run individual stages directly:

```bash
python src/evaluation/knowledge_graph.py
python src/evaluation/clustering.py
python src/evaluation/test_set_generation.py
python src/evaluation/evaluate.py
```

## Metrics

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer factually consistent with the retrieved context? |
| **Answer Relevancy** | Does the answer address the user's question? |
| **Context Precision** | Are the retrieved documents relevant to the question? (Non-LLM, reference-based) |
| **Context Recall** | Were all necessary reference documents retrieved? |

## Results (50-row test set)

| Metric | Score |
|---|---|
| Faithfulness | 0.96 |
| Answer Relevancy | 0.75 |
| Context Precision | 0.95 |
| Context Recall | 0.90 |

> See [`ragas_results.ipynb`](../../notebooks/ragas_results.ipynb) for detailed per-query analysis and visualisations.
