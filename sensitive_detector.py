from typing import List, Union
import re
import os
import json
import joblib
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np

# -----------------------
# Rule-based detector
# -----------------------
class RulePIIDetector:
    EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{6,10}")
    CREDIT_CARD = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
    SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")

    def __init__(self):
        self.rules = {
            "email": self.EMAIL,
            "phone": self.PHONE,
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

# -----------------------
# DL Dataset
# -----------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# -----------------------
# Main DL detector
# -----------------------
class SensitiveDetector:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        ).to(self.device)
        self.rule_detector = RulePIIDetector()
        self.is_trained = False

    def train(self, texts: List[str], labels: List[int], epochs=3, batch_size=8):
        dataset = TextDataset(texts, labels, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_tensor = batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_tensor)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, loss={total_loss/len(loader):.4f}")
        self.is_trained = True
        return self

    def predict_proba(self, texts: Union[str, List[str]]):
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        probs = []
        self.model.eval()
        with torch.no_grad():
            for t in texts:
                d = self.rule_detector.detect(t)
                if len(d) > 0:
                    probs.append(0.99)
                else:
                    if not self.is_trained:
                        probs.append(0.01)
                    else:
                        enc = self.tokenizer(t, truncation=True, padding='max_length', max_length=128, return_tensors="pt").to(self.device)
                        out = self.model(**enc)
                        p = torch.softmax(out.logits, dim=1)[0,1].item()
                        probs.append(p)
        return probs[0] if single_input else probs

    def predict(self, texts: Union[str, List[str]], threshold=0.5):
        probs = self.predict_proba(texts)
        if isinstance(probs, list):
            return [1 if p >= threshold else 0 for p in probs]
        return 1 if probs >= threshold else 0

    def is_sensitive(self, text: str, threshold=0.5):
        matched = self.rule_detector.detect(text)
        if len(matched) > 0:
            return True, 0.99, matched
        prob = self.predict_proba(text)
        return (prob >= threshold), prob, matched

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path + ".pt")
        joblib.dump(self.tokenizer, path + ".tokenizer.joblib")
        meta = {"is_trained": self.is_trained}
        with open(path + ".meta.json", "w") as f:
            json.dump(meta, f)

    def load(self, path):
        self.tokenizer = joblib.load(path + ".tokenizer.joblib")
        self.model.load_state_dict(torch.load(path + ".pt", map_location=self.device))
        with open(path + ".meta.json") as f:
            meta = json.load(f)
        self.is_trained = meta.get("is_trained", True)
        self.rule_detector = RulePIIDetector()
        return self

# -----------------------
# Synthetic dataset for demo
# -----------------------
def make_synthetic_dataset():
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

# -----------------------
# Demo run
# -----------------------
if __name__ == "__main__":
    texts, labels = make_synthetic_dataset()
    det = SensitiveDetector()
    det.train(texts, labels, epochs=1)
    for t in ["Please send the invoice to me at kate@example.com", "Are we meeting at 4pm?"]:
        sensitive, score, matched = det.is_sensitive(t)
        print(f"TEXT: {t}\n -> sensitive: {sensitive}, score: {score:.3f}, matched: {matched}\n")
