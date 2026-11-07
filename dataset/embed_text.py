"""Generate normalized text embeddings with an explicit model and input order."""

import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--model", required=True, help="Hugging Face model ID or local fine-tuned checkpoint"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, default=1000000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--prefix", default="", help='e.g. "passage: " for E5')
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing output overwrite")
    if args.rows <= 0 or args.batch_size <= 0:
        p.error("rows and batch size must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model)
    count = 0
    output = None
    try:
        with args.input.open(encoding="utf-8") as f:
            while count < args.rows:
                batch = []
                for _ in range(min(args.batch_size, args.rows - count)):
                    line = f.readline()
                    # Reject missing rows rather than shifting object IDs across encoders.
                    if not line or not line.strip():
                        raise ValueError(
                            "short or empty corpus row; preserve object correspondence"
                        )
                    batch.append(args.prefix + line.strip())
                encoded = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                if output is None:
                    output = np.lib.format.open_memmap(
                        args.output,
                        mode="w+",
                        dtype=np.float16,
                        shape=(args.rows, encoded.shape[1]),
                    )
                if not np.isfinite(encoded).all():
                    raise ValueError("nonfinite embedding")
                output[count : count + len(batch)] = encoded.astype(np.float16)
                count += len(batch)
                print(f"{count}/{args.rows}", flush=True)
        output.flush()
    except Exception:
        del output
        args.output.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
