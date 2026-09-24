"""Evaluation script across all 8 classes for MindCare."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.config import MODEL_PATH, TOKENIZER_PATH
from model.model_loader import load_model, load_tokenizer
from model.predictor import Predictor
from response.response_manager import ResponseManager


def run_evaluation():
    model, err_m = load_model(MODEL_PATH)
    tok, err_t = load_tokenizer(TOKENIZER_PATH)
    if err_m or err_t:
        print(f"Error loading: model={err_m}, tok={err_t}")
        return

    predictor = Predictor(model=model, tokenizer=tok)
    rm = ResponseManager()

    test_queries = [
        ("Greeting", "Hi hello! Good morning, how are you?"),
        ("Neutral", "I went for a routine medical check up with doctor hawkins today."),
        ("OOD", "The corporate tech company reported third quarter quarterly profit and revenue."),
        ("Addiction", "I have been struggling with severe alcohol addiction, weed and drug cravings every single day."),
        ("Anxiety", "I keep having sudden panic attacks, heart racing, hyperventilating and feeling terrified."),
        ("Depression", "Everything feels so heavy, dark and exhausting. I have no energy to get out of bed anymore."),
        ("Eating Disorder", "I am obsessively counting calories, fasting for days, and feeling intense guilt over food and weight."),
        ("Suicide", "I want to end my life, I cannot carry this unbearable pain any longer."),
    ]

    print("=" * 70)
    print("MINDCARE MODEL EVALUATION: 8 TARGET CLASSES")
    print("=" * 70)

    for target_class, text in test_queries:
        res = predictor.predict(text)
        resp = rm.get_response(res["predicted_class"], confidence=res["confidence"])
        print(f"\n[Target Class: {target_class}]")
        print(f"  Input: \"{text}\"")
        print(f"  Predicted: {res['predicted_class']}")
        print(f"  Raw Class: {res['raw_class']} | Confidence: {res['confidence'] * 100:.1f}%")
        print(f"  Latency: {res['latency_ms']:.2f} ms | OOD Triggered: {res['is_ood']}")
        print(f"  High Risk Protocol: {resp['is_high_risk']}")
        clean_resp = resp['text'].replace('\n', ' ')[:110]
        print(f"  Response: {clean_resp}...")

    print("\n" + "=" * 70)
    print("Evaluation completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
