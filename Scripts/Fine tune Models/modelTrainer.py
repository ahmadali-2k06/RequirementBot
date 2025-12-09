import os
import torch
import pandas as pd
import json
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import shutil

# --------------------------
# CONFIGURATION
# --------------------------
# Path to the base generic model
BASE_MODEL_PATH = r"C:\Users\ytbro\Downloads\DeBerta v3-Base"

# Path where your NEW Quality Attribute model will be saved
OUTPUT_DIR = r"quality_attribute_classifier"

# INPUT DATA
# 1. For the first run: Point this to your big dataset
# 2. For the "50 reqs" update: Point this to the small CSV with just the new errors/reqs
CSV_PATH = r"cleaned_requirements_attributes.csv    "

TEXT_COLUMN = "requirement"
LABEL_COLUMN = "quality_attribute"  # e.g., Security, Performance, Usability

# RESUME LOGIC
# Set False for the very first training run.
# Set True if you want to load the model from OUTPUT_DIR and train it further on new data.
RESUME_FROM_PREVIOUS_MODEL = False

# P1000 SETTINGS
TEST_SPLIT = 0.15
SEED = 42
EPOCHS = 3  # You might reduce this to 1 or 2 when training on just 50 small items
BATCH_SIZE = 1
GRAD_ACCUMULATION = 8
LR = 2e-5
MAX_LENGTH = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------
# 1. LOAD & PREPARE DATA
# --------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

# Clean text
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

# --- LABEL MAPPING LOGIC ---
label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")

if RESUME_FROM_PREVIOUS_MODEL and os.path.exists(label_map_path):
    # If resuming, we MUST use the old ID mapping to keep the model consistent
    print(f"Loading existing label map from {label_map_path}")
    with open(label_map_path, 'r') as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
else:
    # Create new mapping
    unique_labels = sorted(df[LABEL_COLUMN].unique().tolist())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Create output dir if not exists to save the map
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(label_map_path, 'w') as f:
        json.dump(label2id, f)
    print(f"Created new label map with {len(unique_labels)} classes: {unique_labels}")

# Map labels to integers
df["label"] = df[LABEL_COLUMN].map(label2id)

# Filter out rows where label might not exist in the map (safety check for resume mode)
if df["label"].isnull().any():
    print("Warning: Some labels in the new CSV were not found in the previous model's training.")
    print("Dropping unknown labels...")
    df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)

# Create Dataset
dataset = Dataset.from_pandas(df[[TEXT_COLUMN, "label"]])

# If dataset is tiny (the 50 reqs scenario), don't split, just train on all
if len(df) < 50:
    print("Dataset is small (Correction Mode). Using all data for training, no validation split.")
    dataset_dict = DatasetDict({"train": dataset, "validation": dataset})
else:
    dataset = dataset.train_test_split(test_size=TEST_SPLIT, seed=SEED)
    dataset_dict = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

# --------------------------
# 2. TOKENIZER
# --------------------------
# Always load tokenizer from base path to ensure vocabulary consistency
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)


def tokenize(batch):
    return tokenizer(
        batch[TEXT_COLUMN],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


dataset_encoded = dataset_dict.map(tokenize, batched=True)

# --------------------------
# 3. MODEL LOADING
# --------------------------
num_labels = len(label2id)

if RESUME_FROM_PREVIOUS_MODEL and os.path.exists(os.path.join(OUTPUT_DIR, "final_model")):
    load_path = os.path.join(OUTPUT_DIR, "final_model")
    print(f"--- RESUMING: Loading fine-tuned model from {load_path} ---")
else:
    load_path = BASE_MODEL_PATH
    print(f"--- STARTING FRESH: Loading base model from {load_path} ---")

model = AutoModelForSequenceClassification.from_pretrained(
    load_path,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # Safety for slight config variances
    device_map="auto"
)

# Fix for DeBERTa Gradient Checkpointing bug
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
if hasattr(model, "get_input_embeddings"):
    model.get_input_embeddings().requires_grad_(True)

# --------------------------
# 4. TRAINER SETUP
# --------------------------
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    warmup_ratio=0.1,
    logging_steps=5,  # Log more often for small datasets
    save_steps=100,  # Save less frequently
    save_total_limit=2,
    fp16=False,
    seed=SEED,
    optim="adamw_torch",
    report_to="none"
)

accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return accuracy_metric.compute(predictions=preds, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_encoded["train"],
    eval_dataset=dataset_encoded["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# --------------------------
# 5. EXECUTION
# --------------------------
print("Starting Training...")
trainer.train()

# --------------------------
# 6. SAVING
# --------------------------
final_save_path = os.path.join(OUTPUT_DIR, "final_model")
print(f"Saving updated model to: {final_save_path}")

trainer.save_model(final_save_path)
tokenizer.save_pretrained(final_save_path)

# Ensure label map is in the final folder too for easy inference later
with open(os.path.join(final_save_path, "config.json"), 'r') as f:
    config_data = json.load(f)
    # Double check labels are saved in config
    if 'id2label' not in config_data:
        print("Manually injecting labels into config...")
        config_data['id2label'] = id2label
        config_data['label2id'] = label2id
        with open(os.path.join(final_save_path, "config.json"), 'w') as out_f:
            json.dump(config_data, out_f, indent=2)

print("Process Complete.")