from datasets import load_dataset

# Stream the corpus rather than loading it into memory.
dataset = load_dataset(
    "sentence-transformers/msmarco-corpus",
    "passage",
    split="train",
    streaming=True,  # Stream records.
)

output_file = "msmarco_passages.txt"
count = 0

with open(output_file, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(item["text"] + "\n")
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} records...")

print(f"Done: wrote {count} records")
