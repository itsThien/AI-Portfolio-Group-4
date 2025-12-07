import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train(args):
    # Load dataset
    df = pd.read_csv(args.csv_file)
    print(f"Dataset shape: {df.shape}")

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(df[args.label_column].values)
    num_classes = len(np.unique(labels_encoded))
    print(f"Number of classes: {num_classes}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df[args.text_column].values, labels_encoded, test_size=0.2, random_state=42
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Datasets and loaders
    train_dataset = TextDataset(X_train, y_train, tokenizer, args.max_length)
    test_dataset = TextDataset(X_test, y_test, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_classes
    )
    model = model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")

    # Save model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Text Classifier")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--text_column", type=str, default="text", help="Text column name")
    parser.add_argument("--label_column", type=str, default="label", help="Label column name")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained model name")
    parser.add_argument("--max_length", type=int, default=128, help="Max token length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./transformer_model", help="Directory to save model")

    args = parser.parse_args()
    train(args)
