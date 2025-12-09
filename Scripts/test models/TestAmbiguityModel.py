from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# ---------------------------------------------------------
# 1. LOAD YOUR SAVED MODEL
# ---------------------------------------------------------
model_path = r"../../models/ambiguity_detector"

print(f"📂 Loading model from: {model_path}...")
try:
    tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Did you run the training script to the end?")
    exit()


# ---------------------------------------------------------
# 2. DEFINE ANALYSIS & PARSING FUNCTION
# ---------------------------------------------------------
def analyze_requirement(text):
    input_text = "analyze requirement: " + text

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=128,
        truncation=True
    )

    # Generate Output
    outputs = model.generate(
        inputs.input_ids,
        max_length=256,
        num_beams=5,
        early_stopping=True
    )

    # Decode
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- PARSING LOGIC (Make it look nice) ---
    result = {"original": text, "raw": raw_output}

    if "Ambiguous: NO" in raw_output:
        result["status"] = "🟢 CLEAR"
        result["reason"] = "N/A"
        result["correction"] = "N/A"
    else:
        result["status"] = "🔴 AMBIGUOUS"

        # Extract Reason
        if "Reason:" in raw_output:
            try:
                # Split by "Reason:" and take the part after it, then split by "|" to stop before Correction
                part1 = raw_output.split("Reason:")[1]
                reason_text = part1.split("|")[0].strip()
                result["reason"] = reason_text
            except:
                result["reason"] = "Could not parse reason."

        # Extract Correction
        if "Correction:" in raw_output:
            try:
                correction_text = raw_output.split("Correction:")[1].strip()
                result["correction"] = correction_text
            except:
                result["correction"] = "Could not parse correction."

    return result


# ---------------------------------------------------------
# 3. RUN TEST CASES
# ---------------------------------------------------------
test_cases = [
    # 1. Simple Clear (Should be NO)
    "The system shall allow the user to upload a PDF file.",

    # 2. Vague Word (Should be YES + Reason + Template)
    "The system must handle heavy traffic efficiently.",

    # 3. Negative Constraint (Should be YES + Retention Policy)
    "The application must never delete user data.",

    # 4. Complex/Fluff (Should be YES + Simplified List)
    "The solution needs to be a comprehensive, world-class platform that integrates seamlessly with all existing tools.",

    # 5. Auto-Scaling Logic (Should be YES + Trigger/Action)
    "The platform shall automatically scale resources as needed to maintain optimal responsiveness."
]

print("\n" + "=" * 60)
print("🧪 RUNNING DIAGNOSTIC TESTS")
print("=" * 60)

for req in test_cases:
    analysis = analyze_requirement(req)

    print(f"\n📝 Input: {analysis['original']}")
    print(f"   Status: {analysis['status']}")

    if analysis['status'] == "🔴 AMBIGUOUS":
        print(f"   🧐 Why:   {analysis.get('reason', 'N/A')}")
        print(f"   ✨ Fix:   {analysis.get('correction', 'N/A')}")

    print("-" * 60)

# ---------------------------------------------------------
# 4. INTERACTIVE MODE
# ---------------------------------------------------------
print("\nType your own requirement below (or type 'exit'):")
while True:
    user_input = input("\n> ")
    if user_input.lower() in ["exit", "quit"]:
        break

    analysis = analyze_requirement(user_input)
    print(f"Status: {analysis['status']}")
    if analysis['status'] == "🔴 AMBIGUOUS":
        print(f"Reason: {analysis.get('reason')}")
        print(f"Fix:    {analysis.get('correction')}")