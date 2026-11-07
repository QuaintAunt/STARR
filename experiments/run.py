#!/usr/bin/env python3
"""Run STARR experiments with 1,000 fixed queries and save new measurements."""

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "experiments/config.json").read_text())
METHODS = ["Fresh", "Ours", "DGM-Local", "DGM-Search"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "experiment",
        choices=[
            "main",
            "recall-qps",
            "ablation",
            "starr-sweep",
            "dgm-sweep",
            "matrix",
            "direct-reuse",
        ],
    )
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="directory containing MsMarco/, Wikipedia_dpr/, CC3M/ FP32 BIN files",
    )
    p.add_argument("--work-dir", type=Path, default=ROOT / "outputs")
    p.add_argument(
        "--scene",
        type=int,
        choices=range(1, 16),
        help="one scene ID from config.json; omit for the full experiment",
    )
    p.add_argument("--cpu", type=int, help="optional single allowed CPU for measured processes")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    data = args.data_root.resolve()
    work = args.work_dir.resolve()
    if args.cpu is not None and args.cpu not in os.sched_getaffinity(0):
        p.error("CPU unavailable")
    result = work / (args.experiment + (f"-scene{args.scene}" if args.scene else "") + ".jsonl")
    if result.exists():
        p.error("result already exists; choose a new work directory")
    # All experiments load the same saved query IDs; all methods use one thread.
    env = dict(
        os.environ,
        QUERY_IDS_BINARY=str(ROOT / "experiments/query_ids.bin"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        DGM_THREADS="1",
    )
    if args.cpu is not None:
        env["EXPECTED_CPU"] = str(args.cpu)
    prefix = [] if args.cpu is None else ["taskset", "-c", str(args.cpu)]
    if not args.dry_run:
        work.mkdir(parents=True, exist_ok=True)
        subprocess.run(["make"], cwd=ROOT, check=True)

    def command(binary, arguments, extra=None, cwd=None):
        cmd = prefix + [str(ROOT / "build" / binary), *map(str, arguments)]
        print(json.dumps({"command": cmd}), flush=True)
        if args.dry_run:
            return []
        proc = subprocess.run(
            cmd,
            cwd=cwd or ROOT,
            env=env | dict(extra or {}),
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        )
        # Compiler output and progress stay on the terminal. No log files are made.
        return [
            json.loads(line.partition(" ")[2])
            for line in proc.stdout.splitlines()
            if line.startswith(("RESULT ", "VARIANT "))
        ]

    def save(records, **metadata):
        if args.dry_run:
            return
        with result.open("a") as f:
            for r in records:
                f.write(json.dumps(dict(r, **metadata)) + "\n")
                f.flush()

    def vectors(relative, dim):
        path = data / relative
        if not args.dry_run and (not path.is_file() or path.stat().st_size != CFG["n"] * dim * 4):
            p.error("missing or wrong-size vector BIN: " + str(path))
        return path

    def index(relative, dim):
        vec = vectors(relative, dim)
        dest = work / "indexes" / Path(relative).parent / (Path(relative).stem + "_hnsw.bin")
        meta = dest.with_suffix(".json")
        if not dest.exists():
            if not args.dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
            raw = command("starr", ["build", vec, dim, CFG["n"], dest])
            if raw:
                meta.write_text(json.dumps(raw[-1]) + "\n")
        return dest, (json.loads(meta.read_text()) if meta.exists() else {})

    def truth(relative, dim):
        # Exact target-space ground truth is prepared outside measured experiment loops.
        dest = work / "truth" / Path(relative).parent / (Path(relative).stem + ".gt")
        if not dest.exists():
            if not args.dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
            command("starr", ["truth", vectors(relative, dim), dim, CFG["n"], dest])
        return dest

    def matches(s, pairs):
        return [s["dataset"], s["source_model"], s["target_model"]] in pairs

    scenes = CFG["scenes"]
    if args.experiment == "recall-qps":
        scenes = [s for s in scenes if s["paper_scene"] in CFG["recall_qps_scenes"]]
    if args.experiment == "ablation":
        scenes = [s for s in scenes if matches(s, CFG["ablation_pairs"])]
    if args.experiment in ["starr-sweep", "dgm-sweep"]:
        scenes = [s for s in scenes if matches(s, CFG["sweep_pairs"])]
    if args.scene:
        scenes = [s for s in scenes if s["paper_scene"] == args.scene]
    if not scenes:
        p.error("scene is not part of this experiment")
    if args.experiment in ["matrix", "direct-reuse"]:
        if args.scene:
            p.error("--scene applies to the other experiments")
        models = []
        for model in sorted(CFG["matrix_models"], key=lambda m: m["name"]):
            relative = f"MsMarco/msmarco_{model['name']}.bin"
            vectors(relative, model["dim"])
        for model in sorted(CFG["matrix_models"], key=lambda m: m["name"]):
            relative = f"MsMarco/msmarco_{model['name']}.bin"
            idx, meta = index(relative, model["dim"])
            models.append((model["name"], model["dim"], idx, meta))
        # The matrix binary resolves vectors and indexes relative to its work directory.
        data_link = work / "data/MsMarco"
        if not args.dry_run:
            data_link.parent.mkdir(parents=True, exist_ok=True)
            if not data_link.exists():
                data_link.symlink_to(data / "MsMarco", target_is_directory=True)
        if args.experiment == "matrix":
            for name, dim, idx, meta in models:
                arguments = [name, dim, meta.get("insertion_s", 0)]
                for old, olddim, _, oldmeta in models:
                    if old != name:
                        arguments.extend([old, olddim, oldmeta.get("insertion_s", 0)])
                if args.dry_run:
                    command("matrix", arguments, cwd=work)
                    continue
                run = subprocess.run(
                    prefix + [str(ROOT / "build/matrix"), *map(str, arguments)],
                    cwd=work,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    check=True,
                )
                header = (
                    "deviation_thresh,dataset,old_embedding,new_embedding,old_build_s,"
                    "fresh_build_s,migration_s,fresh_query_ms,migration_query_ms,"
                    "fresh_recall_percent,migration_recall_percent,overlap_percent,reinsert_percent"
                ).split(",")
                for row in csv.reader(run.stdout.splitlines()):
                    if len(row) == len(header):
                        save([dict(zip(header, row))], query_count=CFG["query_count"])
        else:
            # Direct Reuse keeps the full original graph and substitutes vectors.
            target = next(m for m in models if m[0] == "gte-large")
            for source in ["BAAI_bge-large-en-v1.5", "contriever", "mpnet-base", "MiniLM-L6-H384"]:
                old = next(m for m in models if m[0] == source)
                raw = command(
                    "direct_reuse",
                    [
                        data / "MsMarco" / f"msmarco_{source}.bin",
                        old[1],
                        data / "MsMarco/msmarco_gte-large.bin",
                        target[1],
                        old[2],
                        target[2],
                        work / "truth/MsMarco/gte-large_1000_queries.bin",
                    ],
                )
                save(raw, source=source, target="gte-large")
        return
    for s in scenes:
        sid = s["paper_scene"]
        old = vectors(s["source_vectors"], s["source_dim"])
        target = vectors(s["target_vectors"], s["target_dim"])
        src, _ = index(s["source_vectors"], s["source_dim"])
        gt = truth(s["target_vectors"], s["target_dim"])
        common = [old, s["source_dim"], target, s["target_dim"], src, gt, CFG["n"]]
        if args.experiment == "ablation":
            out = work / f"ablation-S{sid:02d}"
            save(command("ablation", ["batch", *common, out]), scene=sid)
            continue
        if args.experiment == "dgm-sweep":
            for C, alpha in itertools.product(
                CFG["dgm_local_grid"]["C"], CFG["dgm_local_grid"]["alpha"]
            ):
                save(command("dgm_sweep", ["run", "DGM-Local", *common, C, 0, 0, alpha]), scene=sid)
            for L, H, alpha in itertools.product(*CFG["dgm_search_grid"].values()):
                save(
                    command("dgm_sweep", ["run", "DGM-Search", *common, 0, L, H, alpha]), scene=sid
                )
            continue
        if args.experiment == "starr-sweep":
            for tau, B in itertools.product(CFG["tau_grid"], CFG["B_grid"]):
                save(
                    command(
                        "benchmark",
                        ["run", "Ours", *common],
                        {"STARR_TAU": str(tau), "STARR_B": str(B)},
                    ),
                    scene=sid,
                    tau=tau,
                    B=B,
                )
            continue
        fresh, build = index(s["target_vectors"], s["target_dim"])
        for method in METHODS:
            inputs = [
                old,
                s["source_dim"],
                target,
                s["target_dim"],
                fresh if method == "Fresh" else src,
                gt,
                CFG["n"],
            ]
            if args.experiment == "recall-qps":
                rawpath = work / f"query-S{sid:02d}-{method}.jsonl"
                save(command("recall_qps", ["curve", method, *inputs, rawpath]), scene=sid)
            else:
                # Selected MS MARCO scenes retain the implementation used in the matrix.
                binary = (
                    "benchmark_matrix"
                    if method == "Ours" and s["implementation"] == "matrix"
                    else "benchmark"
                )
                save(
                    command(binary, ["run", method, *inputs]),
                    scene=sid,
                    fresh_build_s=build.get("build_s"),
                )


if __name__ == "__main__":
    main()
