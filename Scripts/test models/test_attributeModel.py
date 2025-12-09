import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from tqdm import tqdm

# --------------------------
# CONFIGURATION
# --------------------------
# Path to your fine-tuned model (the folder containing config.json and pytorch_model.bin)
MODEL_PATH = r"../../models/quality_attribute_classifier/final_model"

# Path to your test data
TEST_FILE_PATH = r"../../data/datasets/attrbiutes_dataset/quality_test.txt"

# DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --------------------------
# 1. LABEL MAPPING
# --------------------------
# Maps the short codes in your .txt file to the Full Names the model knows.
# Ensure these match the spelling in your training data EXACTLY.
CODE_TO_LABEL = {
    "__label__US": "Usability",
    "__label__SE": "Security",
    "__label__PE": "Performance",
    "__label__RA": "Reliability",  # or Availability, depending on your training
    "__label__SC": "Scalability",
    "__label__PO": "Portability",
    "__label__LE": "Legal and Compliance",
    "__label__MA": "Maintainability",
    "__label__CO": "Compatibility",
    "__label__AV": "Availability"
}

# --------------------------
# 2. LOAD MODEL
# --------------------------
print(f"Loading model from {MODEL_PATH}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()  # Set to evaluation mode
except Exception as e:
    print(f"Error loading model: {e}")
    print("Did you run the training script yet? Make sure the path is correct.")
    exit()

# Get the model's internal label map (ID -> Label Name)
id2label = model.config.id2label
label2id = model.config.label2id

print(f"Model knows {len(id2label)} classes: {list(id2label.values())}")

# --------------------------
# 3. READ & PREPARE TEST DATA
# --------------------------
texts = []
true_labels = []

print(f"Reading test file: {TEST_FILE_PATH}")
with open(TEST_FILE_PATH, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line: continue

        # Split on first space: "__label__XX The text starts here..."
        parts = line.split(' ', 1)

        if len(parts) == 2:
            code = parts[0]
            text = parts[1]

            # Convert Code -> Full Name
            full_label = CODE_TO_LABEL.get(code)

            if full_label:
                # IMPORTANT CHECK: Does the model actually know this label?
                if full_label in label2id:
                    texts.append(text)
                    true_labels.append(full_label)
                else:
                    # If model wasn't trained on "Legal", we can't test "Legal"
                    print(f"Skipping line {line_num}: Model doesn't know label '{full_label}' (Code: {code})")
            else:
                print(f"Warning: Unknown code '{code}' on line {line_num}")

print(f"\nLoaded {len(texts)} valid test samples.")

if len(texts) == 0:
    print("No valid samples found to test. Check your mapping and file format.")
    exit()

# --------------------------
# 4. RUN PREDICTIONS
# --------------------------
predicted_labels = []

print("Running predictions...")
batch_size = 16

# Process in batches for speed
for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i: i + batch_size]

    # Tokenize
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get ID of highest score
    pred_ids = torch.argmax(logits, dim=1).cpu().numpy()

    # Convert ID -> Label Name
    for pid in pred_ids:
        predicted_labels.append(id2label[pid])

# --------------------------
# 5. REPORT RESULTS
# --------------------------
print("\n" + "=" * 30)
print("       EVALUATION REPORT       ")
print("=" * 30)

acc = accuracy_score(true_labels, predicted_labels)
print(f"Overall Accuracy: {acc:.2%}")
print("-" * 30)

# Detailed Report
report = classification_report(true_labels, predicted_labels, zero_division=0)
print(report)

# --------------------------
# 6. OPTIONAL: SHOW ERRORS
# --------------------------
# Print first 5 mistakes to help you analyze
print("\n--- First 5 Mistakes ---")
mistake_count = 0
for text, true, pred in zip(texts, true_labels, predicted_labels):
    if true != pred:
        print(f"Text:  {text[:100]}...")
        print(f"True:  {true}")
        print(f"Pred:  {pred}")
        print("-" * 20)
        mistake_count += 1
        if mistake_count >= 5:
            break