# main.py
"""
Example usage of SensitiveDetector.

Two usage modes:
- Load an existing saved detector and use is_sensitive()
- Or, train quickly on a small dataset (demo) and save the model, then use it.

Replace the synthetic training code with your real labeled data for production.
"""

from sensitive_detector import SensitiveDetector, make_synthetic_dataset
import argparse
import os

MODEL_PATH = "models/sensitive_detector_v1"

def build_and_save_demo_model():
    texts, labels = make_synthetic_dataset()
    det = SensitiveDetector()
    det.train(texts, labels)
    det.save(MODEL_PATH)
    print(f"Demo model trained and saved to {MODEL_PATH}.joblib")

def load_detector():
    det = SensitiveDetector()
    if os.path.exists(MODEL_PATH + ".joblib"):
        det.load(MODEL_PATH)
        print("Loaded detector model.")
    else:
        print("Model not found; training demo model instead.")
        build_and_save_demo_model()
        det.load(MODEL_PATH)
    return det

def process_message(det: SensitiveDetector, message: str):
    sensitive, score, matched = det.is_sensitive(message, threshold=0.5)
    result = {
        "sensitive": bool(sensitive),
        "score": float(score),
        "matched_rules": matched
    }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-demo", action="store_true", help="Train and save a demo model")
    parser.add_argument("--text", type=str, default=None, help="Single message text to analyze")
    args = parser.parse_args()

    if args.train_demo:
        build_and_save_demo_model()
        return

    det = load_detector()

    if args.text:
        out = process_message(det, args.text)
        print(out)
    else:
        # interactive demo
        print("Enter messages (empty line to exit).")
        while True:
            txt = input(">>> ").strip()
            if txt == "":
                break
            print(process_message(det, txt))

if __name__ == "__main__":
    main()
