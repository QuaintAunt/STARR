# Experiments and output reference

Run `python experiments/run.py EXPERIMENT --data-root data` from the repository
root. Replace `EXPERIMENT` with a name below. Scene IDs, selected pairs, parameter
grids and the 16 matrix models are defined in
[experiments/config.json](../experiments/config.json). `--scene ID` selects one
eligible scene, except for `matrix` and `direct-reuse`.

## Data preparation

Use the same downloaded corpus file for every text encoder and keep the CC3M
image directory fixed across encoders. Text generation takes the first one million
corpus rows; image generation uses the first one million sorted filenames and
preserves positions for unreadable images. Embeddings are L2-normalized, stored
as FP16 NPY and converted to headerless row-major little-endian FP32 BIN.
Base text encoders without a Sentence Transformers configuration use its default
mean-pooling wrapper. Text generation and BIN conversion refuse to overwrite
existing files.

Fine-tuning uses FP16 and TF32 on a compatible CUDA GPU. Main scenes 1 and 6 need
the BGE-base checkpoint; the full matrix also needs BGE-large. Optional retrieval
evaluation is in [dataset/finetune/evaluate.py](../dataset/finetune/evaluate.py).
Peak RAM and end-to-end runtime have not been measured for this release.

The [preparation scripts](../scripts/) accept `--help` and support selecting one
corpus, model size or text scene. Set `PYTHON` to an interpreter command on PATH
or an absolute path. See [the model and scene configuration](../experiments/config.json)
for text model IDs, dimensions, filename stems and E5 prefixes.

## Experiment-to-paper mapping

Paths below are relative to the work directory (`outputs/` by default).
When `--scene ID` is supplied, the summary becomes `EXPERIMENT-sceneID.jsonl`.

| Experiment | Paper analysis | Output and fields |
| --- | --- | --- |
| `main` | Overall Performance: the 15-transition comparison | `main.jsonl`; `scene`, `method`, `fresh_build_s`, `migration_s`, `recall_at_10_percent`, `query_time_ms` |
| `recall-qps` | Query Performance: the six Recall--QPS curves | `recall-qps.jsonl` contains validation summaries. Measurements are in `query-SXX-METHOD.jsonl`: `ef_search`, `round`, `qps`, `recall_at_10_percent` |
| `matrix` | Performance Across Model Transitions: the 240-transition heatmaps | `matrix.jsonl`; `old_embedding`, `new_embedding`, `fresh_build_s`, `migration_s`, `fresh_recall_percent`, `migration_recall_percent` |
| `ablation` | Ablation Study: performance and unreachable-vertex tables | `ablation.jsonl` contains variant records and per-scene completion summaries. Variant files are `ablation-SXX_VARIANT.raw.json`: `status`, `variant`, `migration_s`, `recall_at_10_percent`, `unreachable` and `stats` |
| `starr-sweep` | Parameter Sensitivity: STARR construction--recall tradeoffs | `starr-sweep.jsonl`; `scene`, `tau`, `B`, `migration_s`, `recall_at_10_percent` |
| `dgm-sweep` | Additional DGM parameter study | `dgm-sweep.jsonl`; `scene`, `method`, `C`, `L`, `H`, `alpha`, `migration_s`, `recall_at_10_percent` |
| `direct-reuse` | Additional baseline: retain the source graph and substitute target vectors | `direct-reuse.jsonl`; `source`, `target`, `construction_time_s`, `recall_at_10_percent`, `query_time_ms` |

`dgm-sweep` and `direct-reuse` are additional measurement entry points; they
are not assigned to a dedicated figure in the manuscript accompanying this release.
Figure rendering scripts are not included.

## Reading the measurements

- JSONL files contain one JSON object per line. STARR is called `Ours` in
  `method`; ablations identify the complete method with `variant: "full"`.
- `recall_at_10_percent`, `fresh_recall_percent` and `migration_recall_percent`
  use a 0--100 scale. Times ending in `_s` are seconds; `query_time_ms` is the
  mean time per query in milliseconds. QPS can be computed as
  `1000 / query_time_ms` where that field is present.
- In `main`, Fresh construction time is `fresh_build_s`; its `migration_s`
  is zero. Migration methods use `migration_s`. Construction speedup is
  `fresh_build_s / migration_s`. For the main comparison, Fresh time includes
  graph allocation and insertion; vector loading and index saving are outside
  the timer.
- In `matrix`, `fresh_build_s` is insertion-only Fresh construction time.
  Speedup is `fresh_build_s / migration_s`; the heatmap recall difference is
  `migration_recall_percent - fresh_recall_percent`, in percentage points.
  These records retain CSV-derived string values; convert fields to numbers
  before arithmetic.
- For `recall-qps`, group the measurement records by scene, method and
  `ef_search`; use median `qps` across the five timed rounds. The summary JSONL
  confirms the query count and consistency but does not hold the curve points.
- For ablations, inspect records with a `variant`. A `status: "failed"` record
  gives a reason; it is not a numeric measurement. Per-scene `completed`/`failed`
  summaries contain no recall or migration time. The `no_connect` and
  `no_share_no_connect` variants intentionally disable reachability repair.
- Migration timing includes target graph allocation and migration. Corpus
  download, embedding generation, source-index construction, ground-truth
  computation and disk I/O are outside this timing. The `direct-reuse`
  construction time measures graph allocation/copying and vector substitution.

The runner caches indexes under `indexes/` and exact ground truth under `truth/`
inside the work directory. Use a fresh work directory when changing vector files
or query settings. Existing result files are not overwritten; interrupted runs
may leave partial measurements, so use a new `--work-dir` for a clean rerun.

## Query protocols

All experiments evaluate the same 1,000 indexed objects as queries, loaded from
[experiments/query_ids.bin](../experiments/query_ids.bin). Self-matches are
included, and ground truth consists of exact target-space top-10 neighbors.
The shared query loader validates the count, uniqueness and bounds of the IDs.

Each evaluation warms up with all 1,000 queries before timing. Recall--QPS curves
use five timed rounds per search width and report median QPS; other experiments
use one timed evaluation. Matrix and Direct Reuse truth caches store the ordered
query IDs and reject mismatches.

Measured times depend on hardware and execution conditions. Model and corpus
revisions are not pinned; record revisions and object order when preparing data.
