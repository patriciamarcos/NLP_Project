import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# CARREGAR TOKENIZER E MODELO (BEST MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "cardiffnlp/twitter-roberta-base"
tokenizer = AutoTokenizer.from_pretrained("tokenizer/")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

state = torch.load("best_model.pt", map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# CARREGAR TEST SET
test_df = pd.read_csv("test_clean.csv")
test_sentences = test_df["clean_tweet"].tolist()

# 4) FUNÇÕES
def classify_text(text):
    encoded = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred = torch.argmax(probs).item()
    return pred, float(probs[pred]), probs.cpu().numpy()


def forward_wrapper(input_ids):
    outputs = model(input_ids=input_ids)
    return outputs.logits


lig = LayerIntegratedGradients(forward_wrapper, model.roberta.embeddings)


def explain_text(text, target_class):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    input_ids = encoded["input_ids"]
    baseline = torch.zeros_like(input_ids).to(device)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline,
        target=target_class,
        return_convergence_delta=True
    )

    token_scores = attributions.sum(dim=-1).squeeze().cpu().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return tokens, token_scores


# GRÁFICO 1 — BARRAS
def plot_explanation(tokens, scores, top_k=10):
    tokens_clean = [t.replace("Ġ", "") for t in tokens if t not in ["<pad>", "<s>", "</s>"]]
    scores_clean = [s for t, s in zip(tokens, scores) if t not in ["<pad>", "<s>", "</s>"]]

    top_idx = sorted(range(len(scores_clean)), key=lambda i: abs(scores_clean[i]), reverse=True)[:top_k]
    top_tokens = [tokens_clean[i] for i in top_idx]
    top_scores = [scores_clean[i] for i in top_idx]
    colors = ["green" if s > 0 else "red" for s in top_scores]

    fig = plt.figure(figsize=(8, 4))
    sns.barplot(x=top_scores, y=top_tokens, palette=colors)
    plt.xlabel("Influência na previsão")
    plt.ylabel("Tokens")
    plt.title(f"Explicabilidade (Top {top_k})")

    return fig
