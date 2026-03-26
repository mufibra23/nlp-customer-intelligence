"""
Sentiment Classifier - Fine-tunes DistilBERT for 3-class sentiment.
  Train:   python src/sentiment_model.py --train
  Predict: python src/sentiment_model.py --predict "This product is terrible"
"""
import os, argparse, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
MODEL_SAVE_DIR = "models/sentiment"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts, self.labels, self.tokenizer = texts, labels, tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), max_length=MAX_LENGTH,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(loader, desc="Training", leave=False):
        ids, mask, labels = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["label"].to(DEVICE)
        optimizer.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total_loss += out.loss.item()
        correct += (torch.argmax(out.logits, dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            ids, mask, labels = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["label"].to(DEVICE)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            total_loss += out.loss.item()
            all_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    preds, labels = np.array(all_preds), np.array(all_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"loss": total_loss/len(loader), "accuracy": accuracy_score(labels, preds),
            "f1": f1, "predictions": preds, "labels": labels}


def train(num_epochs=NUM_EPOCHS):
    print(f"Training on {DEVICE} | {num_epochs} epochs")
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    print(f"Data: {len(train_df):,} train / {len(val_df):,} val / {len(test_df):,} test")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID).to(DEVICE)
    def make_loader(df, shuffle=False):
        return DataLoader(SentimentDataset(df["clean_text"].tolist(), df["label"].map(LABEL2ID).tolist(), tokenizer), batch_size=BATCH_SIZE, shuffle=shuffle)
    train_loader, val_loader, test_loader = make_loader(train_df, True), make_loader(val_df), make_loader(test_df)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    best_f1, patience_counter, history = 0, 0, []
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"  Train Loss: {t_loss:.4f} | Acc: {t_acc:.4f}")
        val_r = evaluate(model, val_loader)
        print(f"  Val Loss: {val_r['loss']:.4f} | Acc: {val_r['accuracy']:.4f} | F1: {val_r['f1']:.4f}")
        history.append({"epoch": epoch+1, "train_loss": t_loss, "val_f1": val_r["f1"]})
        if val_r["f1"] > best_f1:
            best_f1 = val_r["f1"]; patience_counter = 0
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            model.save_pretrained(MODEL_SAVE_DIR); tokenizer.save_pretrained(MODEL_SAVE_DIR)
            print(f"  Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 2: print("  Early stopping"); break
    print("\n--- Final Test ---")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_DIR).to(DEVICE)
    test_r = evaluate(model, test_loader)
    print(f"  Accuracy: {test_r['accuracy']:.4f} | F1: {test_r['f1']:.4f}")
    print(classification_report(test_r["labels"], test_r["predictions"], target_names=list(LABEL2ID.keys())))
    with open(os.path.join(MODEL_SAVE_DIR, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": float(test_r["accuracy"]), "test_f1": float(test_r["f1"]), "history": history}, f, indent=2)
    print(f"Model saved to {MODEL_SAVE_DIR}/")


class SentimentClassifier:
    def __init__(self, model_dir=MODEL_SAVE_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
    def predict(self, text):
        enc = self.tokenizer(text, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=1).squeeze().cpu().numpy()
        pid = int(np.argmax(probs))
        return {"label": ID2LABEL[pid], "confidence": float(probs[pid]),
                "probabilities": {ID2LABEL[i]: float(probs[i]) for i in range(NUM_LABELS)}}
    def predict_batch(self, texts, batch_size=32):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=1).cpu().numpy()
            for j, p in enumerate(probs):
                pid = int(np.argmax(p))
                results.append({"text": batch[j][:100], "label": ID2LABEL[pid], "confidence": float(p[pid])})
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", type=str)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    if args.train: train(args.epochs)
    elif args.predict:
        c = SentimentClassifier(); r = c.predict(args.predict)
        print(f"Sentiment: {r['label']} ({r['confidence']:.3f})")
    else: parser.print_help()
