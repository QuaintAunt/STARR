import pickle
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from datasets import Dataset
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True

# ── 1. Load prepared triplets ──────────────────────────────────────────
with open("train_examples.pkl", "rb") as f:
    train_examples = pickle.load(f)
with open("val_examples.pkl", "rb") as f:
    val_examples = pickle.load(f)

print(f"Training triplets: {len(train_examples)}, validation triplets: {len(val_examples)}")


# ── 2. Convert InputExample triplets to Hugging Face datasets ──────────
def examples_to_dataset(examples):
    return Dataset.from_dict(
        {
            "anchor": [e.texts[0] for e in examples],
            "positive": [e.texts[1] for e in examples],
            "negative": [e.texts[2] for e in examples],
        }
    )


train_dataset = examples_to_dataset(train_examples)
val_dataset = examples_to_dataset(val_examples)

# ── 3. Model and loss ───────────────────────────────────
model = SentenceTransformer(os.environ.get("STARR_MODEL_NAME", "BAAI/bge-base-en-v1.5"))
train_loss = MultipleNegativesRankingLoss(model)

# ── 4. Retrieval evaluator ─────────────────────────────────────────────
val_queries = {str(i): ex.texts[0] for i, ex in enumerate(val_examples)}
val_corpus = {str(i): ex.texts[1] for i, ex in enumerate(val_examples)}
val_relevant = {str(i): {str(i)} for i in range(len(val_examples))}

evaluator = InformationRetrievalEvaluator(
    queries=val_queries,
    corpus=val_corpus,
    relevant_docs=val_relevant,
    name="msmarco-val",
    show_progress_bar=True,
)

# ── 5. Training arguments ───────────────────────────────────────────
args = SentenceTransformerTrainingArguments(
    output_dir=os.environ.get("STARR_TRAIN_OUTPUT", "./checkpoints/bge-base"),
    # Training hyperparameters
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=16,  # Effective batch size: 256 on one device.
    learning_rate=1e-5,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    # Precision and performance
    fp16=True,  # FP16 mixed precision.
    tf32=True,  # Enable TF32 on supported GPUs.
    # Evaluation and checkpointing
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="msmarco-val_cosine_ndcg@10",
    # Logging
    logging_steps=100,
    report_to="none",  # Disable external experiment trackers.
)

# ── 6. Trainer ────────────────────────────────────────────
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    loss=train_loss,
    evaluator=evaluator,
)

print(
    f"\nEffective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size}"
)
print("Starting training...\n")

trainer.train()
model.save_pretrained(args.output_dir + "/best")
print("\nTraining complete!")
