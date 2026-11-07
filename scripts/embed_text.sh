#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"

"$PYTHON" - "$@" <<'PY'
import argparse
import json
from pathlib import Path
import subprocess
import sys

cfg = json.loads(Path("experiments/config.json").read_text())
text_scenes = [s for s in cfg["scenes"] if s["dataset"] != "CC3M"]
p = argparse.ArgumentParser(prog="scripts/embed_text.sh", description="Generate text embeddings.")
p.add_argument("dataset", nargs="?", choices=["MsMarco", "Wikipedia_dpr", "all"],
               help="dataset to generate (default: all)")
p.add_argument("--scene", type=int, choices=[s["paper_scene"] for s in text_scenes],
               help="generate only the source and target files for one text scene")
args = p.parse_args()
if args.scene is not None and args.dataset is not None:
    p.error("choose a dataset or --scene, not both")
selected = None
if args.scene is not None:
    scene = next(s for s in text_scenes if s["paper_scene"] == args.scene)
    selected = {scene["source_vectors"], scene["target_vectors"]}
models = {m["name"].removeprefix("BAAI_"): m for m in cfg["matrix_models"]}
jobs = {f'MsMarco/msmarco_{m["name"]}.bin': m for m in cfg["matrix_models"]}
for scene in cfg["scenes"]:
    if scene["dataset"] == "Wikipedia_dpr":
        for side in ("source", "target"):
            jobs[scene[f"{side}_vectors"]] = models[scene[f"{side}_model"]]
for relative, model in jobs.items():
    dataset = relative.split("/")[0]
    if selected is not None and relative not in selected:
        continue
    if args.dataset not in (None, "all", dataset):
        continue
    corpus = "msmarco_passages.txt" if dataset == "MsMarco" else "wikipedia_dpr_passages.txt"
    output = Path("data") / relative
    subprocess.run([
        sys.executable, "dataset/embed_text.py", "--input",
        str(Path("data/corpora") / dataset / corpus), "--model", model["model_id"],
        "--prefix", model["prefix"], "--output", str(output.with_suffix(".npy"))
    ], check=True)
    subprocess.run([
        sys.executable, "dataset/npy_to_bin.py", str(output.with_suffix(".npy")), str(output)
    ], check=True)
PY
