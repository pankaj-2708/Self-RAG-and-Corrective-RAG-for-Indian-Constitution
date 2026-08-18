# notebooks

Jupyter notebooks used for data collection, prototyping, and evaluation analysis. These are exploratory/research notebooks — production code lives in `src/`.

## Notebooks

| Notebook | Purpose |
|---|---|
| [`data_collector.ipynb`](data_collector.ipynb) | Web scrapes and structures the legal corpus — Constitution of India articles from the Legislative Department and IPC sections from India Code — into `data/articles.json` and `data/penal_code_sections.json`. |
| [`srag.ipynb`](srag.ipynb) | Original Self-RAG prototyping and experiments. Used to design and iterate on the LangGraph pipeline before it was refactored into `src/workflow/`. |
| [`ragas.ipynb`](ragas.ipynb) | Ragas evaluation experiments — knowledge graph construction, test set generation, and metric scoring. Results and design decisions from here were productionised into `src/evaluation/`. |
| [`ragas_results.ipynb`](ragas_results.ipynb) | Analysis and visualisation of evaluation results — score distributions, latency percentiles, token usage breakdown, and per-query inspection. |

## Usage

```bash
# from the project root, with the virtual environment activated
jupyter notebook notebooks/
```
