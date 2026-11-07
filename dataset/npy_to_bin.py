#!/usr/bin/env python3
"""Convert a two-dimensional NPY embedding file into the paper's raw FP32 BIN."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--rows", type=int, default=1000000)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing output overwrite")
    if args.rows <= 0:
        p.error("rows must be positive")
    arr = np.load(args.input, mmap_mode="r", allow_pickle=False)
    if (
        arr.ndim != 2
        or arr.shape[0] < args.rows
        or arr.shape[1] == 0
        or not np.issubdtype(arr.dtype, np.floating)
    ):
        p.error("expected at least rows floating point vectors")
    # Preserve row order and float16 quantization if present; do not renormalize.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    try:
        with args.output.open("xb") as f:
            for first in range(0, args.rows, 4096):
                chunk = np.asarray(
                    arr[first : min(first + 4096, args.rows)], dtype="<f4", order="C"
                )
                if not np.isfinite(chunk).all():
                    raise ValueError("nonfinite embedding")
                data = chunk.tobytes()
                f.write(data)
                digest.update(data)
    except Exception:
        args.output.unlink(missing_ok=True)
        raise
    metadata = {
        "rows": args.rows,
        "dim": arr.shape[1],
        "source_dtype": str(arr.dtype),
        "output_dtype": "little-endian float32, row-major, no header",
        "sha256": digest.hexdigest(),
        "source": args.input.name,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
