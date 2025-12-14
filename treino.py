import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, default_data_collator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import numpy as np
import evaluate
import random
from transformers import AutoTokenizer
from transformers import RobertaConfig
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.manual_seed_all(42)

train_df = pd.read_csv("train_final.csv")
test_df = pd.read_csv("test_clean.csv")
train_df = train_df.dropna(subset=["clean_tweet", "label"])


train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["clean_tweet"].tolist(),
    train_df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=train_df["label"]
)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

model_name = "cardiffnlp/twitter-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

tokenized_datasets = dataset.map(tokenize, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

config = RobertaConfig.from_pretrained(
    model_name,
    num_labels=2,
    hidden_dropout_prob=0.5,
    classifier_dropout=0.6
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config
).to(device)

num_epochs = 15
batch_size = 32
num_training_steps = len(tokenized_datasets["train"]) // batch_size * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

unique, counts = np.unique(train_df["label"], return_counts=True)
num_classes = len(unique)
total = sum(counts)
class_counts = dict(zip(unique.tolist(), counts.tolist()))

class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Class counts:", class_counts)
print("Class weights:", class_weights.cpu().numpy())

criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)

train_loader = DataLoader(
    tokenized_datasets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=default_data_collator
)
val_loader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=default_data_collator
)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "f1": []}

best_val_loss = float("inf")
best_model_path = "best_model.pt"

for epoch in range(num_epochs):
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1:03}/{num_epochs} - Train", leave=False)
    for batch in loop:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        preds = torch.argmax(outputs.logits, dim=-1)
        labels = batch["labels"]

        train_loss += loss.item()
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=correct_train / total_train)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = correct_train / total_train

    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            preds = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]

            val_loss += loss.item()
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct_val / total_val
    f1 = f1_metric.compute(predictions=all_preds, references=all_labels)["f1"]

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["f1"].append(f1)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        tqdm.write("Novo melhor modelo salvo!")

    tqdm.write(f"Epoch {epoch+1:03}/{num_epochs} | "
               f"TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | "
               f"TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f} | F1={f1:.4f}")

model.load_state_dict(torch.load(best_model_path))
model.eval()

torch.save(model.state_dict(), "modelo_final.pth")
tokenizer.save_pretrained("tokenizer/")
print("Modelo final e tokenizer guardados!")

final_preds = []
final_labels = []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        preds = torch.argmax(outputs.logits, dim=-1)
        labels = batch["labels"]

        final_preds.extend(preds.cpu().numpy())
        final_labels.extend(labels.cpu().numpy())

print("\n=== CLASSIFICATION REPORT FINAL ===")
print(classification_report(final_labels, final_preds, digits=4))

cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão (Melhor Modelo)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()