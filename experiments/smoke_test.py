"""Small synthetic checks; does not load or run the million-vector datasets."""

import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
env = dict(os.environ, OMP_NUM_THREADS="1", DGM_THREADS="1")
for key in list(env):
    if key.startswith("EXPECTED_") or key in ["QUERY_IDS_BINARY", "STARR_TAU", "STARR_B"]:
        env.pop(key)


def run(binary, *args, success=True):
    proc = subprocess.run(
        [str(ROOT / "build" / binary), *map(str, args)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert (proc.returncode == 0) == success, proc.stdout[-3000:]
    records = [json.loads(s[7:]) for s in proc.stdout.splitlines() if s.startswith("RESULT ")]
    return records[-1] if records else None


with tempfile.TemporaryDirectory(prefix="starr-runner-test-") as folder:
    data = Path(folder) / "data"
    work = Path(folder) / "work"
    for experiment, binary, expected_count in [
        ("main", "benchmark", 56),
        ("recall-qps", "recall_qps", 24),
        ("ablation", "ablation", 6),
        ("starr-sweep", "benchmark", 84),
        ("dgm-sweep", "dgm_sweep", 189),
        ("matrix", "matrix", 16),
        ("direct-reuse", "direct_reuse", 4),
    ]:
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments/run.py"),
                experiment,
                "--data-root",
                str(data),
                "--work-dir",
                str(work),
                "--dry-run",
            ],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        commands = [json.loads(line)["command"] for line in proc.stdout.splitlines()]
        selected = [c for c in commands if Path(c[0]).name == binary]
        assert len(selected) == expected_count, (experiment, len(selected))
        if experiment == "matrix":
            assert len({c[1] for c in selected}) == 16
            assert all(len(c) == 49 for c in selected), "each target needs 15 source models"
    assert not data.exists() and not work.exists(), "dry-run created data or work files"


for binary in ["benchmark", "benchmark_matrix", "recall_qps", "ablation", "dgm_sweep"]:
    run(binary, "self-test")
run("direct_reuse", "--self-test")
with tempfile.TemporaryDirectory(prefix="starr-test-") as folder:
    folder = Path(folder)
    fixture = folder / "vectors"
    run("benchmark", "fixture", fixture)
    truth = folder / "truth.gt"
    run("starr", "truth", fixture / "target.bin", 7, 1500, truth)
    for method in ["Fresh", "Ours", "DGM-Local", "DGM-Search"]:
        index = fixture / ("target.hnsw" if method == "Fresh" else "source.hnsw")
        record = run(
            "benchmark",
            "run",
            method,
            fixture / "source.bin",
            5,
            fixture / "target.bin",
            7,
            index,
            truth,
            1500,
        )
        assert record["source_unchanged"] and record["structure_valid"]
        assert record["query_count"] == 1000 and 0 <= record["total_hits"] <= 10000
    record = run("matrix", "self-test", fixture)
    assert record["query_count"] == record["warmup_queries"] == 1000
    ids = list(reversed(range(1000)))
    query_file = folder / "query_ids.bin"
    query_file.write_bytes(struct.pack("<1000Q", *ids))
    env["QUERY_IDS_BINARY"] = str(query_file)
    record = run("matrix", "self-test", fixture)
    assert record["query_ids"] == ids
    run("starr", "truth", fixture / "target.bin", 7, 1500, folder / "fixed.gt")
    raw = (folder / "fixed.gt").read_bytes()
    assert list(struct.unpack("<1000Q", raw[40:8040])) == record["query_ids"]
    for invalid in [ids[:-1], ids + [1001], [0] * 1000, ids[:-1] + [1500]]:
        query_file.write_bytes(struct.pack(f"<{len(invalid)}Q", *invalid))
        run("matrix", "self-test", fixture, success=False)
    env.pop("QUERY_IDS_BINARY")
    source = folder / "source.hnsw"
    run("starr", "build", fixture / "source.bin", 5, 1500, source)
    target = folder / "target.hnsw"
    record = run("starr", "migrate", source, 5, fixture / "target.bin", 7, 1500, target, 2, 16)
    assert record["audit"]["reachable"] == 1500
    run("starr", "migrate", source, 5, fixture / "target.bin", 7, 1500, target, success=False)
    run(
        "starr",
        "migrate",
        source,
        5,
        fixture / "target.bin",
        7,
        1500,
        folder / "invalid.hnsw",
        -1,
        16,
        success=False,
    )
print(
    "PASS: seven experiment dry-runs, native self-tests, four-method evaluation, "
    "1,000 shared queries, bound truth caches, 5→7 dimensions, "
    "index save, source immutability and invalid-input checks"
)
