# ChaTU RAG Thesis

Implementation for *Retrieval Optimization for Local RAG Systems over Unstructured Data*.

- Original ChaTU retrieval baseline and a modular RAG retrieval pipeline.
- Sparse, dense and hybrid retrieval with multilingual search, reranking and page/group routing.
- Experiments on chunking, language strategies, retrieval architecture and Priority retrieval.
- Synthetic query generation, manual navigation benchmarks, relevance judgments and judge calibration.
- Evaluation on synthetic and user-log queries, including ranking metrics, latency and paired comparisons.
- Thesis figure/table export, a local RAG API and optional ChaTU/Rasa integration.

## Structure

- `thesis_rag/` — retrieval, dataset construction, evaluation and RAG API.
- `configs/thesis/` — experiments, prompts and the publication source catalog.
- `modules/`, `bots/`, `templates/` — original baseline and chat integration.
- `scripts/thesis/`, `services/` — evaluation workers, model service and publication export.
- `tests/` — framework, baseline and adapter tests.
- `requirements/legacy/` — dependency lists for the original ChaTU components.

## Setup

Linux, Python 3.13 and uv:

```bash
uv sync --locked --group analysis --group runtime --group local-llm
uv run --locked thesis-rag --help
```

The original baseline uses the `rasa3.x` Conda environment. Scraping uses `scraper`.
The lists in `requirements/legacy/` contain shared dependencies (`requirements.txt`),
Rasa/scraper additions and a separate list for the old LLM server.

## Reproduce the thesis results

The commands below require the original thesis datasets, corpus and experiment
outputs at the paths recorded in the configurations.

Regenerate the thesis figures and tables from the stored experiment outputs:

```bash
uv run --locked --group analysis python scripts/thesis/export_publication.py --prepare-statistics
```

Output: `outputs/thesis/publication/`. The export requires both synthetic
and private log experiment outputs.

Validate the corpus and prepare the final synthetic experiment:

```bash
uv run --locked thesis-rag corpus verify data/thesis/corpus/snapshot_manifest.json
uv run --locked thesis-rag eval suite plan --config configs/thesis/experiments/final_synthetic_test.yaml
uv run --locked thesis-rag eval suite --help
```

Running retrieval also requires the configured model services and baseline runtime.
Use the archived Qrels to compare results with the thesis.

## Tests

```bash
uv lock --check
uv run --locked --group test pytest -W error tests/thesis_rag
conda run -n scraper python -m unittest discover -s tests/legacy
conda run -n rasa3.x python -m unittest discover -s tests/rasa_rag
```

Checks requiring restored datasets are opt-in: add `--run-artifact-tests` to pytest.
