import os
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# Load the original and fine-tuned models.
original_model = SentenceTransformer(os.environ.get("STARR_MODEL_NAME", "BAAI/bge-base-en-v1.5"))
finetuned_model = SentenceTransformer(
    os.environ.get("STARR_FINETUNED_MODEL", "./checkpoints/bge-base/best")
)

# Load the MS MARCO v1.1 validation split for evaluation.
print("Loading the evaluation split...")
dataset = load_dataset("ms_marco", "v1.1", split="validation")

# Construct query, passage and relevance mappings.
queries, corpus, relevant_docs = {}, {}, {}

for item in dataset:
    qid = str(item["query_id"])
    queries[qid] = item["query"]
    relevant_docs[qid] = set()

    for pid, (text, selected) in enumerate(
        zip(item["passages"]["passage_text"], item["passages"]["is_selected"])
    ):
        doc_id = f"{qid}_{pid}"
        corpus[doc_id] = text
        if selected == 1:
            relevant_docs[qid].add(doc_id)

# Remove queries without relevant passages.
queries = {k: v for k, v in queries.items() if relevant_docs.get(k)}
relevant_docs = {k: v for k, v in relevant_docs.items() if v}
print(f"Queries: {len(queries)}, corpus passages: {len(corpus)}")

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="msmarco-official-dev",
    show_progress_bar=True,
)

print("\nOriginal model")
original_results = evaluator(original_model)

print("\nFine-tuned model")
finetuned_results = evaluator(finetuned_model)

# Compare retrieval metrics.
metrics = [
    "msmarco-official-dev_cosine_ndcg@10",
    "msmarco-official-dev_cosine_mrr@10",
    "msmarco-official-dev_cosine_recall@10",
]

print(f"\n{'Metric':<35} {'Original':>8} {'Tuned':>8} {'Change':>8}")
print("-" * 55)
for m in metrics:
    orig = original_results.get(m, 0)
    fine = finetuned_results.get(m, 0)
    diff = fine - orig
    arrow = "↑" if diff > 0 else "↓"
    short = m.replace("msmarco-official-dev_cosine_", "")
    print(f"{short:<35} {orig:>8.4f} {fine:>8.4f} {arrow}{abs(diff):.4f}")
