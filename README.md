# STARR: Standardized Tolerance for Adaptive Reuse and Repair

STARR reuses and repairs an HNSW index when the embedding model changes.

## Quick check

Requires Linux, GCC with C++17/OpenMP, Make and Python 3.10+.
C++ dependencies are [included](src/third_party/README.md). From the repository root:

```bash
make -j2
make check
```

The synthetic checks need no datasets, model weights, GPU or Python ML packages.
They test basic behavior and script routing, not the full paper measurements.

## Try one scene

Scene 2 migrates MS MARCO from BGE-base-en to BGE-base-en-v1.5, without fine-tuning.
It uses one million objects; the NPY/BIN vectors occupy about 8.6 GiB, plus corpus,
model and index storage.

```bash
pip install -r dataset/requirements.txt
bash scripts/download_data.sh MsMarco
bash scripts/embed_text.sh --scene 2
python experiments/run.py main --data-root data --scene 2
```

The runner builds indexes and exact ground truth, then compares Fresh, STARR,
DGM-Local and DGM-Search. Results are in `outputs/main-scene2.jsonl`; STARR is
named `Ours`. See [the result fields](docs/experiments.md#reading-the-measurements).

Use `--dry-run` to inspect experiment commands, `--cpu` to pin execution to an
available CPU, and `--work-dir` to choose an output directory. Existing results
are not overwritten.

## Full experiments

For a fresh full preparation, use this path instead of the single-scene setup.
It requires a compatible CUDA GPU for fine-tuning and about 132.3 GiB for vectors
alone, plus datasets, models and indexes. Text generation and BIN conversion
refuse to overwrite existing outputs.

```bash
pip install -r dataset/requirements.txt
bash scripts/download_data.sh all
bash scripts/finetune.sh all
bash scripts/embed_text.sh all
bash scripts/embed_cc3m.sh
```

Run `python experiments/run.py EXPERIMENT --data-root data` with an experiment
from [the experiment guide](docs/experiments.md). The guide maps each experiment
to the paper analysis, output files, metrics and query protocol.

Preparation scripts accept `--help` and resolve paths from the repository root.
Set `PYTHON` to select an interpreter; the default is `python3`.
Model IDs, dimensions, filenames and scene settings are in
[experiments/config.json](experiments/config.json).

## Reading the source

| Order | Entry point | Purpose |
| --- | --- | --- |
| 1 | [src/starr.h](src/starr.h) | API, options, statistics and ownership contract |
| 2 | [src/starr.cpp](src/starr.cpp) | Start at `migrateHnswIndexStable()` and follow its helpers |
| 3 | [src/benchmark.cpp](src/benchmark.cpp) | DGM adapter, validation, ground truth, timing and recall |
| 4 | [experiments/run.py](experiments/run.py) | Scene selection and experiment dispatch |
| 5 | [src/main.cpp](src/main.cpp) | Build/migration CLI; see `build/starr --help` |

Main scenes 1, 2, 3 and 5 use [src/matrix/starr.cpp](src/matrix/starr.cpp), linked
with `src/benchmark.cpp` as `build/benchmark_matrix`. The separate
[src/matrix/benchmark.cpp](src/matrix/benchmark.cpp) runs the 240-transition matrix.
Other STARR experiments use `src/starr.cpp`; [Makefile](Makefile) shows the linkage.

## Reproducibility

All experiments use the same 1,000 indexed queries, including self-matches.
See [data preparation](docs/experiments.md#data-preparation) and
[query protocols](docs/experiments.md#query-protocols) for details.
Corpora, model weights and indexes are downloaded or generated separately.
The runners output measurements; figure rendering scripts are not included.

## License

Apache-2.0. See [LICENSE](LICENSE) and the
[third-party source inventory](src/third_party/README.md) for licenses, commits and patches.
