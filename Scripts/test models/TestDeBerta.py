import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------
# CONFIG
# --------------------------
# Point this to where the training just finished
MODEL_PATH = r"../../models/fr_nfr_classifier/final_model"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model from {MODEL_PATH}...")

try:
    # Load the model you just trained
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


def predict(text):
    # 1. Prepare text for the model
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)

    # 2. Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Convert raw scores to probabilities (0-100%)
        probs = F.softmax(logits, dim=-1)

    # 3. Interpret result
    # 0 = Non-Functional, 1 = Functional (Based on your training logic)
    pred_label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_label].item()

    label_text = "FUNCTIONAL (Yes)" if pred_label == 1 else "NON-FUNCTIONAL (No)"

    return label_text, confidence


# --------------------------
# INTERACTIVE LOOP
# --------------------------
print("\n" + "=" * 50)
print(" REQUIREMENT CLASSIFIER TESTER")
print("=" * 50)
print("Type a requirement to test (or 'q' to quit)")

while True:
    user_input = input("\nEnter Requirement: ")
    if user_input.lower() in ['q', 'exit']:
        break

    if not user_input.strip():
        continue

    label, conf = predict(user_input)

    # Print with visual clarity
    print("-" * 30)
    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.2%}")  # Shows percentage (e.g., 98.5%)
    print("-" * 30)