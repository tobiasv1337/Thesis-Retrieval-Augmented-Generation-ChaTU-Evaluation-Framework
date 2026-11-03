# ChaTU RAG Thesis

Implementation for *Retrieval Optimization for Local RAG Systems over Unstructured Data*.

- Original ChaTU retrieval baseline and a modular RAG retrieval pipeline.
- Sparse, dense and hybrid retrieval with multilingual search, reranking and page/group routing.
- Experiments on chunking, language strategies, retrieval architecture and Priority retrieval.
- Synthetic query generation, manual navigation benchmarks, relevance judgments and judge calibration.
- Evaluation on synthetic and user-log queries, including ranking metrics, latency and paired comparisons.
- Thesis figure/table export, a local RAG API and optional ChaTU/Rasa integration.

## Reproduce the thesis results

### 1. Install

Requirements: Linux, Python 3.13 and uv. Run the following commands from the repository root:

```bash
uv sync --locked --group analysis
```

### 2. Download the data

Choose a profile. Each level includes the data from the level above it.

| Profile | Data | Results |
|---|---|---|
| `synthetic` | Synthetic/manual queries, judgments and experiment outputs | Synthetic thesis figures and tables |
| `synthetic-corpus` | Also the website corpus and baseline/routing artifacts | Also corpus statistics and navigation-depth figures |
| `full` | Also the restricted log datasets and their experiments | Complete thesis export |

Download the files for your profile from these links:

| Download link | Files | Needed for |
|---|---|---|
| `<SYNTHETIC_ARCHIVES_URL>` | `synthetic-queries.tar.gz`, `synthetic-experiments.tar.gz` | All profiles |
| `<CORPUS_ARCHIVES_URL>` | `corpus.tar.gz`, `retrieval-support.tar.gz` | `synthetic-corpus`, `full` |
| `<LOG_ARCHIVES_URL>` | `private-queries.tar.gz`, `private-experiments.tar.gz` | `full` |

Open the links in your browser. For restricted shares, use the access details supplied with the thesis or by the data owner.
Download the individual `.tar.gz` files and keep their filenames unchanged.

Create a folder in the repository and move the downloaded files there:

```bash
mkdir -p archives
```

Keep the archives packed. The next command extracts them into the required directories.
The unpacked data takes about 52 GB for `synthetic`, 56 GB for `synthetic-corpus` and 82 GB for `full`.
You also need space for the downloaded archives and temporary extraction files.

### 3. Restore and evaluate

Example for the synthetic results:

```bash
uv run --locked python scripts/release/artifacts.py restore --profile synthetic

uv run --locked --group analysis python scripts/thesis/export_publication.py \
  --scope synthetic --prepare-statistics
```

The archive tool uses `archives/` in the repository by default.
For files stored elsewhere, add `--archives-dir /path/to/archives`.
Use the same profile name for `--profile` and `--scope` to reproduce another level.
The restore command verifies SHA-256 checksums and restores the original artifact paths.
The exporter uses archived evaluation outputs and computes missing statistical summaries from per-query metrics.

The exported files are grouped by topic in these directories:

- `synthetic`: `outputs/thesis/publication-synthetic/`
- `synthetic-corpus`: `outputs/thesis/publication-synthetic-corpus/`
- `full`: `outputs/thesis/publication/`

Each topic contains `figures/` with PDF, SVG and PNG files, `tables/` with LaTeX tables,
and `data/` with CSV values.
An optional `--preview` adds an HTML gallery, PDF catalogues and a ZIP reading copy. It requires `pdflatex`.

## Rerun retrieval experiments

Restore `synthetic-corpus` first. Experiments using user logs require `full`.
Install the runtime dependencies and configure the model services specified in `configs/thesis/`.
The legacy baseline also needs the `rasa3.x` Conda environment. Its dependency lists are in `requirements/legacy/`.

```bash
uv sync --locked --group analysis --group runtime --group local-llm
uv run --locked thesis-rag corpus verify data/thesis/corpus/snapshot_manifest.json

uv run --locked thesis-rag eval suite retrieve \
  --config configs/thesis/experiments/final_synthetic_test.yaml \
  --suite-output-root outputs/reproduction/suites \
  --retrieval-output-root outputs/reproduction/runs
```

This example runs the final synthetic comparison. Other suites are defined in `configs/thesis/experiments/`.
Model weights and running inference services must be supplied separately.
Use the frozen corpus and judgments for thesis comparisons, and separate output directories for new runs.

## Repository structure

- `thesis_rag/` — retrieval pipeline, dataset construction, evaluation and RAG API.
- `configs/thesis/` — experiment suites, system configurations, prompts and thesis result sources.
- `modules/`, `bots/`, `templates/` — original ChaTU baseline and chat integration.
- `scripts/thesis/`, `services/` — evaluation workers, model service and figure/table export.
- `scripts/release/` — archive creation and restoration.
- `tests/` — framework, baseline and adapter tests.

## Tests

```bash
uv lock --check
uv run --locked --group test pytest -W error tests/thesis_rag tests/release
```

Legacy tests use their respective Conda environments:

```bash
conda run -n scraper python -m unittest discover -s tests/legacy
conda run -n rasa3.x python -m unittest discover -s tests/rasa_rag
```
