# sensitive_detector.py
"""
Sensitive content detector.

Provides:
- SensitiveDetector class with:
    - train(train_texts, train_labels)
    - predict(texts)
    - predict_proba(texts)
    - is_sensitive(text, threshold=0.5)
    - save(path)
    - load(path)

Design:
- Quick regex checks for explicit PII (emails, phone numbers, credit cards, SSN-like).
  If these checks match, we immediately mark content sensitive.
- Otherwise we fall back to a lightweight ML model: TF-IDF + LogisticRegression.
- You can train the model on your labeled dataset (texts, labels).
"""

from typing import List, Union
import re
import os
import json
import joblib

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# -----------------------------------------
# Rule-based PII detector (fast)
# -----------------------------------------
class RulePIIDetector:
    """
    Runs several regex checks for common PII patterns.
    Returns dict of matched types.
    """
    EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{6,10}")
    # credit card simplified: groups of 4 digits with optional separators (very permissive)
    CREDIT_CARD = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
    # short PAN/SSN style: 3-2-4 or other combos (US SSN), adapt/extend as needed
    SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    # Aadhaar-like 12 digits (India) - optional spaced
    AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")

    def __init__(self):
        self.rules = {
            "email": self.EMAIL,
            "phone": self.PHONE, #date and time
            "credit_card": self.CREDIT_CARD,
            "ssn": self.SSN,
            "aadhaar": self.AADHAAR,
        }

    def detect(self, text: str):
        found = {}
        for name, rx in self.rules.items():
            m = rx.search(text)
            if m:
                found[name] = m.group(0)
        return found

# -----------------------------------------
# Transformer to add regex-derived numeric features into sklearn pipeline
# -----------------------------------------
class RegexFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.detector = RulePIIDetector()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # produce array shape (n_samples, n_features)
        features = []
        for text in X:
            d = self.detector.detect(text if text is not None else "")
            # numeric features: count of matches and binary flags per rule
            row = [
                1 if "email" in d else 0,
                1 if "phone" in d else 0,
                1 if "credit_card" in d else 0,
                1 if "ssn" in d else 0,
                1 if "aadhaar" in d else 0,
                sum(1 for _ in d.keys())
            ]
            features.append(row)
        return np.array(features, dtype=float)

# -----------------------------------------
# Main wrapper
# -----------------------------------------
class SensitiveDetector:
    def __init__(self):
        # pipeline: TF-IDF (text) + regex features -> LogisticRegression
        text_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=20000,
                ngram_range=(1,2),
                stop_words='english'
            ))
        ])

        combined = FeatureUnion([
            ("text", text_pipeline),
            ("regex_feats", RegexFeatureTransformer())
        ])

        # Final pipeline
        self.pipeline = Pipeline([
            ("features", combined),
            ("clf", LogisticRegression(max_iter=200))
        ])

        # rule-based detector used for early decisions
        self.rule_detector = RulePIIDetector()

        # state for whether classifier is trained
        self.is_trained = False

    # -----------------------
    # Training
    # -----------------------
    def train(self, train_texts: List[str], train_labels: List[int]):
        """
        train_texts: list of text strings
        train_labels: list of 0/1 (1 = sensitive)
        """
        assert len(train_texts) == len(train_labels)
        # If lots of explicit PII in training labels, model will learn context too.
        self.pipeline.fit(train_texts, train_labels)
        self.is_trained = True
        return self

    # -----------------------
    # Predict / Probability
    # -----------------------
    def predict_proba(self, texts: Union[List[str], str]):
        """
        Returns probability of class 1 (sensitive).
        If texts is a single string, returns float; if list, returns list of floats.
        Rule-check short-circuits to 0.99 probability if explicit PII found.
        """
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        probs = []
        for t in texts:
            # rule-based early detection
            d = self.rule_detector.detect(t if t is not None else "")
            if len(d) > 0:
                # If explicit PII present, high probability
                probs.append(0.99)
            else:
                if not self.is_trained:
                    # fallback heuristic if not trained: low probability
                    probs.append(0.01)
                else:
                    p = self.pipeline.predict_proba([t])[0][1]
                    probs.append(float(p))
        return probs[0] if single_input else probs

    def predict(self, texts: Union[List[str], str], threshold: float = 0.5):
        """Returns binary predictions."""
        probs = self.predict_proba(texts)
        if isinstance(probs, list):
            return [1 if p >= threshold else 0 for p in probs]
        else:
            return 1 if probs >= threshold else 0

    def is_sensitive(self, text: str, threshold: float = 0.5):
        """Convenience: returns (bool, score, matched_rules)"""
        matched = self.rule_detector.detect(text if text is not None else "")
        if len(matched) > 0:
            return True, 0.99, matched
        prob = self.predict_proba(text)
        return (prob >= threshold), prob, matched

    # -----------------------
    # Save / Load
    # -----------------------
    def save(self, path: str):
        """
        Save model to path. We'll store:
         - pipeline via joblib
         - a small JSON metadata
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path + ".joblib")
        meta = {
            "is_trained": self.is_trained
        }
        with open(path + ".meta.json", "w", encoding="utf8") as f:
            json.dump(meta, f)

    def load(self, path: str):
        self.pipeline = joblib.load(path + ".joblib")
        with open(path + ".meta.json", "r", encoding="utf8") as f:
            meta = json.load(f)
        self.is_trained = meta.get("is_trained", True)
        # ensure rule_detector exists
        self.rule_detector = RulePIIDetector()
        return self

# -----------------------------------------
# Quick demo / synthetic training helper
# -----------------------------------------
def make_synthetic_dataset():
    """
    Small synthetic dataset for quick testing and debugging.
    Replace with your real labeled data for production.
    Returns (texts, labels)
    """
    sensitive_examples = [
        "My credit card number is 1234-5678-9876-5432, please charge it.",
        "Call me at +1 (415) 555-1234",
        "Email me at john.doe@example.com with the password reset link",
        "My Aadhaar number is 9999 8888 7777",
        "SSN: 123-45-6789",
        "The patient's medical diagnosis: stage 3 carcinoma",
        "Bank account number 0123456789 - wire urgently"
    ]
    non_sensitive = [
        "Are we meeting tomorrow to prepare the slides?",
        "Let's schedule the interview next Monday.",
        "Reminder: submit your assignment before midnight",
        "Happy birthday! Hope you have a great day :)",
        "The weather is nice today, perfect for a walk."
    ]
    texts = sensitive_examples + non_sensitive
    labels = [1]*len(sensitive_examples) + [0]*len(non_sensitive)
    return texts, labels

# standalone test when module is run directly
if __name__ == "__main__":
    texts, labels = make_synthetic_dataset()
    det = SensitiveDetector()
    det.train(texts, labels)
    tests = [
        "Please send the invoice to me at kate@example.com",
        "Are we meeting at 4pm?",
        "My card 4444 3333 2222 1111 expires soon",
        "Project update: finished the module."
    ]
    for t in tests:
        sensitive, score, matched = det.is_sensitive(t, threshold=0.5)
        print(f"TEXT: {t}\n -> sensitive: {sensitive}, score: {score:.3f}, matched: {matched}\n")
