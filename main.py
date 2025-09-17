import os
from sensitive_detector import SensitiveDetector, make_synthetic_dataset

MODEL_PATH = "models/sensitive_detector_v1"

def load_detector():
    det = SensitiveDetector()
    if os.path.exists(MODEL_PATH + ".pt") and \
       os.path.exists(MODEL_PATH + ".tokenizer.joblib") and \
       os.path.exists(MODEL_PATH + ".meta.json"):
        det.load(MODEL_PATH)
        print("DL detector loaded from disk.")
    else:
        print("Model not found; training demo model...")
        texts, labels = make_synthetic_dataset()
        det.train(texts, labels, epochs=1, batch_size=2)  # small batch & epoch for demo
        det.save(MODEL_PATH)
        print("Demo model trained and saved.")
    return det

def main():
    det = load_detector()
    print("Enter messages to check for sensitive content (empty line to exit).")
    while True:
        msg = input("\nMessage: ").strip()
        if not msg:
            break
        sensitive, score, matched = det.is_sensitive(msg)
        print(f"Sensitive: {sensitive}")
        print(f"Score: {score:.3f}")
        print(f"Matched rules: {matched if matched else 'None'}")

if __name__ == "__main__":
    main()
