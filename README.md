# Evaluation Framework for Retrieval-Augmented Generation Systems

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

The synthetic datasets and experiment outputs are available from the public download below.
The complete TU Berlin website corpus is not publicly distributed because of copyright restrictions.
For access to the corpus and the accompanying retrieval artifacts, contact
[Tobias Veselsky](mailto:veselsky@tu-berlin.de).
User-log queries and their associated experiment data cannot be shared for privacy reasons.

Each profile includes the data from the preceding profile.

| Profile | Contents | Access |
|---|---|---|
| `synthetic` | Synthetic and curated queries, reference answers, retrieval results, judgments and metrics | Public download |
| `synthetic-corpus` | Also the frozen website corpus and baseline/routing artifacts for retrieval reruns | On request |
| `full` | Also user-log queries and results from log-based and mixed experiments | Internal only |

| Download | Files | Needed for |
|---|---|---|
| [Synthetic archives](https://tubcloud.tu-berlin.de/s/E6Xs6Ex66maTCxa) | `synthetic-queries.tar.gz`, `synthetic-experiments.tar.gz` | `synthetic`, `synthetic-corpus` |
| [Corpus archives (password required)](https://tubcloud.tu-berlin.de/s/fnSPwjS8G7fMKQ5) | `corpus.tar.gz`, `retrieval-support.tar.gz` | Additional files for `synthetic-corpus` |

Download the individual `.tar.gz` files in your browser and keep their filenames unchanged.

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

Restore `synthetic-corpus` to rerun retrieval. The `synthetic` profile reproduces figures
and tables from saved evaluation outputs.
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
For pooling, judging, metrics and reports, follow the [suite workflow below](#3-define-and-run-a-retrieval-suite)
with this config and the same output roots.
Model identities and service settings are listed [below](#models-and-inference-settings).
Use the frozen corpus and judgments for thesis comparisons, and separate output directories for new runs.

## Run your own experiments

### 1. Scrape and build a new corpus

Use a **separate clone without restored archives** for a new crawl. The scraper replaces
`data/az/scraper/topics/` and `data/az/mapping.csv`. The commands below use the default
paths in that clone, leaving your thesis reproduction checkout intact.

Install the scraper's dependencies in its own Conda environment:

```bash
conda create -n scraper python=3.10 pip
conda run -n scraper python -m pip install \
  requests beautifulsoup4 unidecode regex 'pandas<3' 'numpy<2' deep-translator
uv sync --locked --group analysis --group runtime --group local-llm
```

Review the crawl seeds in `modules/scrape_az.py` and the checked-in
`data/az/blacklist.csv`, `overwrite_topics.csv` and `manual_utterances.csv`.
This calls the scraper and metadata translator directly:

```bash
conda run --no-capture-output -n scraper python -c '
from pathlib import Path
from modules.scrape_az import Scraper, Translator
cache = Path("data/az/translations.json")
if not cache.exists():
    cache.write_text("{}", encoding="utf-8")
Scraper("data/az/", "scraper", "topics").run()
Translator("data/az/", "scraper", "topics").run()
'

uv run --locked thesis-rag corpus snapshot --config configs/thesis/corpus.yaml
uv run --locked thesis-rag corpus verify data/thesis/corpus/snapshot_manifest.json
```

The crawl collects German pages and their available English counterparts. The translator
uses Google Translate for missing public topic/title metadata. This minimal setup does
not build the legacy FAISS index or train Rasa models.
The snapshot command creates pages, sections, context groups, relationships, metadata
and the configured chunk variants. Check `data/thesis/corpus/corpus_quality_report.json`
before generating questions. A new crawl changes source content and hashes, so its
queries, indexes and judgments must be built against the new snapshot. It will not
reproduce the thesis numbers exactly.

### 2. Create and freeze a query set

Start the model services described below. Before generating data, configure languages,
model endpoints, prompts and split sizes in `configs/thesis/questions.yaml`.
The thesis defaults produce 1,000 development, 500 validation and 1,000 test questions.
Keep `splits.target_counts` consistent with `splits.large_benchmark` when changing sizes.
Commit your configuration changes in the separate clone so the question set records a
clean code revision.

```bash
uv run --locked thesis-rag eval questions families
uv run --locked thesis-rag eval questions generation-units
uv run --locked thesis-rag eval questions build
uv run --locked thesis-rag eval questions status
uv run --locked thesis-rag eval questions seal --kind base
uv run --locked thesis-rag eval questions verify
```

`build` generates candidates, filters and reviews them, removes duplicates and generates
replacement candidates until the configured targets are met or generation capacity is
exhausted. Continue to `seal` only after `target_reached`. If deduplication reports
unresolved clusters, run `eval questions dedup-resolve --reviewer-id YOUR_ID` through
`uv run --locked thesis-rag`, then rerun `build`.
The base seal writes the split datasets and `data/thesis/eval/base_question_set_manifest.json`.
Optional human query review uses `human-plan`, `human-review`, `human-materialize` and
`seal --kind human` under `eval questions`. Each command has `--help`.

### 3. Define and run a retrieval suite

[The example suite](configs/examples/retrieval.yaml) compares
[BM25](configs/thesis/systems/modern_sparse.yaml),
[BGE-M3 dense retrieval](configs/thesis/systems/modern_dense.yaml) and their
[hybrid](configs/thesis/systems/modern_hybrid.yaml) on the full development split.
It works with a newly built query set or restored `synthetic-corpus` data.
For a quick setup check, add `query_limit: 10` under `defaults` to limit all three
variants equally. Reserve validation/test queries for held-out comparisons.

The system YAMLs define the pipelines. The suite YAML selects systems, queries and
comparisons. Its optional `overrides` change individual system settings for a variant.
The [thesis suites below](#thesis-experiment-suites) show more extensive comparisons.

Start the BGE-M3 embedding and Gemma judge services from the
[model setup below](#models-and-inference-settings). This example needs no Rasa runtime
or reranker. The [judge experiment](configs/thesis/judge_experiments/retrieval_primary.yaml)
selects Gemma, and [relevance.yaml](configs/thesis/relevance.yaml) defines its prompt,
judgment cache and metric cutoffs.

Run these steps in order from the repository root:

```bash
suite=configs/examples/retrieval.yaml
judges=configs/thesis/judge_experiments/retrieval_primary.yaml
suite_root=outputs/experiments/suites

uv run --locked thesis-rag eval suite plan \
  --config "$suite" --suite-output-root "$suite_root"
uv run --locked thesis-rag eval suite index \
  --config "$suite" --suite-output-root "$suite_root"
uv run --locked thesis-rag eval suite retrieve \
  --config "$suite" --suite-output-root "$suite_root" \
  --retrieval-output-root outputs/experiments/runs
uv run --locked thesis-rag eval suite pool \
  --config "$suite" --suite-output-root "$suite_root"
uv run --locked thesis-rag eval suite judge \
  --config "$suite" --suite-output-root "$suite_root" --judge-experiment "$judges"
uv run --locked thesis-rag eval suite metrics \
  --config "$suite" --suite-output-root "$suite_root" --judge-experiment "$judges" --no-manual
uv run --locked --group analysis thesis-rag eval suite report \
  --config "$suite" --suite-output-root "$suite_root" --judge-experiment "$judges" \
  --no-manual --analysis
```

`plan` validates the inputs, `index` builds the required indexes and `retrieve` runs each
system. `pool` merges retrieved candidates for judging. `judge` writes relevance labels,
`metrics` computes ranking metrics and latency, and `report --analysis` creates comparison
reports, tables and figures. The commands print their output paths under the suite's
run directory in `outputs/experiments/suites/`. Retrieval records are stored separately
in `outputs/experiments/runs/`. Index and judgment caches use the paths in the system
and relevance configurations.

The example uses model judgments. For human relevance labels, use `eval suite qrel-review`
and `qrel-materialize`, then omit `--no-manual` once that view is complete.
Rerun interrupted steps with the same configuration to reuse valid completed artifacts.
Missing judgments or mismatched corpus hashes must be resolved before metrics can be computed.
For new experiments, use these suite reports. The thesis publication exporter selects
specific archived runs through `configs/thesis/publication/catalog.json`.

### Thesis experiment suites

These suites provide examples for larger studies. Use `synthetic-corpus` for the synthetic
comparisons. All suites need the embedding and judge services above.
Additional requirements are listed below.

| Suite configuration | Comparison | Additional requirements |
|---|---|---|
| [Chunking](configs/thesis/experiments/retrieval_chunking_hybrid.yaml) | Fixed, recursive and section-based chunks, plus chunk lengths | Derived 600/1,000-character chunks from [chunk-set config](configs/thesis/chunk_sets/section_preferred_lengths.yaml) |
| [Language strategies](configs/thesis/experiments/retrieval_language_policy.yaml) | Same-language, cross-lingual, query-language completion and counterpart deduplication | None |
| [RRF fusion](configs/thesis/experiments/retrieval_rrf_constant_dedup_top20.yaml) | Fusion constants with counterpart deduplication | None |
| [Rerankers](configs/thesis/experiments/retrieval_rerankers.yaml) | Models, instructions and metadata inputs | Matching services for the [reranker profiles](configs/thesis/rerankers.json) |
| [Page/group routing](configs/thesis/experiments/retrieval_parent_routing.yaml) | Global, page, group and hierarchical retrieval, with/without summaries | Routing tables with metadata/summaries and Nemotron |
| [Priority retrieval](configs/thesis/experiments/retrieval_priority_generalization_synthetic.yaml) | Priority slots, descendant depth, NLU routing and reranker choice | Manual page benchmark, Rasa routing models, Nemotron and zerank |
| [Final synthetic comparison](configs/thesis/experiments/final_synthetic_test.yaml) | Legacy baseline, global hybrid and Priority variants | Baseline FAISS/Rasa artifacts, manual page benchmark, Nemotron and zerank |
| [Final log comparison](configs/thesis/experiments/final_user_logs_test_1000.yaml) | The final systems on 1,000 user-log queries | Same as the synthetic comparison, plus restricted log data |

Set `suite` in the commands above to the linked YAML path. Each suite references its
system configurations. For reranker sweeps, use `retrieve --experiment EXPERIMENT_ID`
with the matching service running, then pool once every experiment has completed.

For a new corpus, regenerate the required artifacts before running these suites.
Use `corpus chunk-set` for additional chunk lengths, and `corpus summarize` plus
`corpus routing-tables` for summary-based routing. NLU/Priority and legacy comparisons
also need corpus-matched Rasa models, manual page benchmarks and baseline artifacts.
The archived versions belong to the original corpus.

## Models and inference settings

These are the pinned configurations for query construction and the final retrieval
comparison. SHA-256 values identify Ollama artifacts, while Git revisions identify
Hugging Face model repositories.

| Role | Model / request name | Digest or revision | Settings |
|---|---|---|---|
| Query generation | Qwen/Qwen3.6-35B-A3B (`qwen3.6:35b`) | SHA-256 `07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522` | Q4_K_M, temperature 0.2, max. 8,000 output tokens, JSON schema |
| Query review, duplicate adjudication, relevance judge | google/gemma-4-31B-it (`gemma4:31b-review-32k`) | SHA-256 `36f5ca871cd0cecd37bc056f276caf9c984c5fb6827135dbe164a9103c25935c` | Q4_K_M, 32,768 context, temperature 0. Review: reasoning `medium`, max. 8,000 tokens. Dedup: `none`, 8,000. Judge: `none`, 2,000 |
| Query deduplication embeddings | `bge-m3:latest` | SHA-256 `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab` | Batch size 32, thresholds in `questions.yaml` |
| Dense retrieval embeddings | `bge-m3:567m-fp16` | SHA-256 `daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c` | FP16, 1,024 dimensions, normalized, batch size 32 |
| Final reranker | `nvidia/llama-nemotron-rerank-1b-v2` | Git `828765652b05bd439c9789d2a6d093db1caa1443` | BF16, max. length 2,048, title/heading/text input |
| Alternative final reranker | `zeroentropy/zerank-2-reranker` | Git `5eae30d5ee3c6b2df2ef6d723bde45172d761c4c` | BF16, max. length 2,048, title/heading/text input |
| Legacy embeddings | `sentence-transformers/all-MiniLM-L6-v2` | No model revision pinned in the system config | Use the archived FAISS and Rasa snapshot for the original baseline |

[questions.yaml](configs/thesis/questions.yaml) contains the generation/review prompts,
seeds, retry limits and deduplication thresholds.
[relevance.yaml](configs/thesis/relevance.yaml) lists every calibration judge with its
full digest, request alias and endpoint checks.
[rerankers.json](configs/thesis/rerankers.json) lists all reranker models, revisions,
scoring methods and instruction variants.
Routing-summary variants are pinned in `configs/thesis/summaries_routing*.yaml`.

Install Ollama separately, start it with `ollama serve` and load the listed model artifacts
on port 11434. Generation and judging use its
OpenAI-compatible `/v1` endpoint, while dense retrieval uses the native API.
Check the full installed digests with `curl -s http://localhost:11434/api/tags`.
Tags alone do not establish model identity. The Gemma review alias is a custom model
whose stored parameters are `num_ctx=32768`, `temperature=1`, `top_k=64`, `top_p=0.95`.
Requests override temperature as shown above. A newly created alias must still match
the recorded digest for a strict thesis rerun. For a different model in a new experiment,
record its actual identity in the configuration before generating data.

For suites with a reranker, start its service in another terminal, for example:

```bash
uv run --locked --project services/reranker python services/reranker/serve.py \
  --profile nemotron_1b --gpu 0 --port 8010
```

Use the matching profile and endpoint for each reranker experiment. The service requires
an NVIDIA GPU and its own locked environment. LLM, embedding and reranker weights
are separate from the data archives.

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
