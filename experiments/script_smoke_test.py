"""Check preparation scripts without downloading data or loading models."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

with tempfile.TemporaryDirectory(prefix="starr-script-test-") as folder:
    folder = Path(folder)
    repo = folder / "repository with spaces"
    shutil.copytree(ROOT / "scripts", repo / "scripts")
    (repo / "experiments").mkdir()
    shutil.copy2(ROOT / "experiments/config.json", repo / "experiments/config.json")
    log = folder / "commands.jsonl"
    python = folder / "record-python"
    python.write_text(f"#!{sys.executable}\n" """import json, os, subprocess, sys
from pathlib import Path

def record(args, **kwargs):
    with open(os.environ["STARR_TEST_LOG"], "a") as f:
        f.write(json.dumps({"args": args, "cwd": os.getcwd(),
                            "env": {k: v for k, v in os.environ.items()
                                    if k.startswith("STARR_") and not k.startswith("STARR_TEST_")}}) + "\\n")
    if os.environ.get("STARR_TEST_FAIL_ON") == args[0]:
        raise subprocess.CalledProcessError(1, args)

if sys.argv[1] == "-":
    subprocess.run = lambda cmd, **kwargs: record(cmd[1:], **kwargs)
    sys.argv = sys.argv[1:]
    exec(compile(sys.stdin.read(), "embed_text.sh", "exec"))
else:
    record(sys.argv[1:])
""")
    python.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("STARR_")}
    env.update(PYTHON=str(python), STARR_TEST_LOG=str(log))

    def run(name, *args, fail_on=None, success=True):
        log.unlink(missing_ok=True)
        proc = subprocess.run(
            ["bash", str(repo / "scripts" / name), *args],
            cwd=folder,
            env=env | ({"STARR_TEST_FAIL_ON": fail_on} if fail_on else {}),
            capture_output=True,
            text=True,
        )
        assert (proc.returncode == 0) == success, proc.stderr
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    for script in (repo / "scripts").glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
        assert run(script.name, "--help") == []
        assert run(script.name, "invalid", success=False) == []
        assert run(script.name, "all", "extra", success=False) == []
    assert not (repo / "data").exists(), "help/invalid arguments created data files"

    for dataset in ("MsMarco", "Wikipedia_dpr"):
        events = run("download_data.sh", dataset)
        assert len(events) == 1
        event = events[0]
        assert (Path(event["cwd"]) / event["args"][0]).resolve() == (
            repo / "dataset" / dataset / "download.py"
        )

    events = run("download_data.sh", "all")
    assert len(events) == 3
    for event, relative in zip(
        events,
        [
            "dataset/MsMarco/download.py",
            "dataset/Wikipedia_dpr/download.py",
            "dataset/CC3M/download.py",
        ],
    ):
        assert (Path(event["cwd"]) / event["args"][0]).resolve() == repo / relative

    events = run("finetune.sh", "all")
    assert len(events) == 3 and events[0]["args"] == ["dataset/finetune/prepare_data.py"]
    assert [e["env"]["STARR_MODEL_NAME"] for e in events[1:]] == [
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
    ]
    assert [e["env"]["STARR_TRAIN_OUTPUT"] for e in events[1:]] == [
        "checkpoints/bge-base",
        "checkpoints/bge-large",
    ]

    for selection, count in [("MsMarco", 16), ("Wikipedia_dpr", 8), ("all", 24)]:
        events = run("embed_text.sh", selection)
        assert len(events) == 2 * count, (selection, len(events))
        outputs = {e["args"][2] for e in events[1::2]}
        assert len(outputs) == count
        if selection != "all":
            assert all(p.startswith(f"data/{selection}/") for p in outputs)
        for event in events[::2]:
            args = event["args"]
            prefix = args[args.index("--prefix") + 1]
            assert prefix == ("passage: " if "intfloat/e5-large-v2" in args else "")

    events = run("embed_cc3m.sh")
    assert len(events) == 12
    generated = {e["env"]["STARR_OUTPUT_FILE"] for e in events[:6]}
    assert generated == {e["args"][1] for e in events[6:]}
    cfg = json.loads((repo / "experiments/config.json").read_text())
    for scene in cfg["scenes"]:
        if scene["dataset"] == "CC3M":
            continue
        events = run("embed_text.sh", "--scene", str(scene["paper_scene"]))
        assert len(events) == 4
        assert {e["args"][2] for e in events[1::2]} == {
            "data/" + scene["source_vectors"],
            "data/" + scene["target_vectors"],
        }
        assert all(e["args"][0] == "dataset/embed_text.py" for e in events[::2])
    assert run("embed_text.sh", "--scene", "11", success=False) == []
    assert run("embed_text.sh", "--scene", "0", success=False) == []
    assert run("embed_text.sh", "MsMarco", "--scene", "2", success=False) == []

    events = run("embed_cc3m.sh")
    required = {
        "data/" + s[f"{side}_vectors"]
        for s in cfg["scenes"]
        if s["dataset"] == "CC3M"
        for side in ("source", "target")
    }
    assert required == {e["args"][2] for e in events[6:]}

    for script, args, first in [
        ("download_data.sh", ["text"], "../../../dataset/MsMarco/download.py"),
        ("finetune.sh", ["all"], "dataset/finetune/prepare_data.py"),
        ("embed_text.sh", ["all"], "dataset/embed_text.py"),
        ("embed_cc3m.sh", [], "dataset/CC3M/embed_dinov1.py"),
    ]:
        assert len(run(script, *args, fail_on=first, success=False)) == 1

print(
    "PASS: preparation script paths, single-corpus downloads, ten text scenes, "
    "model selection, filenames, E5 prefixes and failure handling"
)
