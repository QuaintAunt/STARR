from datasets import load_dataset
from sentence_transformers import InputExample
import random
import pickle

random.seed(42)


def prepare_data(dataset, split, max_samples):
    examples = []
    skipped = 0

    for item in dataset[split]:
        query = item["query"]
        texts = item["passages"]["passage_text"]
        labels = item["passages"]["is_selected"]

        # Collect relevant passages (is_selected == 1).
        positives = [t for t, l in zip(texts, labels) if l == 1]
        # Collect non-relevant passages (is_selected == 0).
        negatives = [t for t, l in zip(texts, labels) if l == 0]

        # Require both positive and negative passages for a triplet.
        if not positives or not negatives:
            skipped += 1
            continue

        examples.append(
            InputExample(
                texts=[
                    query,
                    random.choice(positives),
                    random.choice(negatives),
                ]
            )
        )

        if len(examples) >= max_samples:
            break

    print(
        f"{split}: {len(examples)} triplets, skipped {skipped} records without both passage types"
    )
    return examples


print("Loading the dataset...")
dataset = load_dataset("ms_marco", "v2.1")

train_examples = prepare_data(dataset, "train", max_samples=500000)
val_examples = prepare_data(dataset, "validation", max_samples=5000)

# Save prepared triplets for the training script.
with open("train_examples.pkl", "wb") as f:
    pickle.dump(train_examples, f)
with open("val_examples.pkl", "wb") as f:
    pickle.dump(val_examples, f)

print("\nExample triplet:")
ex = train_examples[0]
print("Query:    ", ex.texts[0][:80])
print("Positive: ", ex.texts[1][:80])
print("Negative: ", ex.texts[2][:80])
print("\nPrepared data saved.")
