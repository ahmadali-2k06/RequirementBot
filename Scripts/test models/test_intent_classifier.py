import os
from transformers import pipeline

# --- CONFIGURATION ---
# This must match the OUTPUT_DIR from your training script
# If you ran the training script from 'src', this path looks for the model
# two folders up. Adjust if your folder is empty.
MODEL_PATH = "../../models/intent_classifier"


def test_interactive():
    print(f"--- Loading Model from: {os.path.abspath(MODEL_PATH)} ---")

    # Check if path exists first
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model folder not found!")
        print(f"Checked path: {os.path.abspath(MODEL_PATH)}")
        print("Tip: Check where 'train_model.py' saved the model.")
        return

    try:
        # Load the pipeline
        # "return_all_scores=True" gives confidence for all labels (optional)
        classifier = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print("\n" + "=" * 50)
    print("🤖 INTENT CLASSIFIER TESTER")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50 + "\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        if not user_input.strip():
            continue

        # Get Prediction
        # The pipeline returns a list of dicts: [{'label': 'requirement', 'score': 0.99}]
        result = classifier(user_input)[0]

        label = result['label']
        score = result['score']

        # Formatting the output nicely
        print(f"Bot: [{label.upper()}] (Confidence: {score:.2%})")
        print("-" * 30)


if __name__ == "__main__":
    test_interactive()