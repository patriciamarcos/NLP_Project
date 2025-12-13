import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

train_df = pd.read_csv("train_clean.csv")
print(train_df["clean_tweet"].isna().sum())
# DATA AUGMENTATION — BATCH BACK-TRANSLATION (NLLB)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo ativo: {device}")

# ---- Modelo único multilingue NLLB ----
model_name = "facebook/nllb-200-distilled-600M"

nllb_tok = AutoTokenizer.from_pretrained(model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

gen_cfg = GenerationConfig.from_pretrained(model_name)
nllb_model.generation_config = gen_cfg
nllb_model.config.use_cache = False

# ========= FUNÇÃO BASE: TRADUZIR EM BATCH (CORRIGIDA) =========
def translate_batch(text_list, tokenizer, model, src_lang, tgt_lang, batch_size=16):
    tokenizer.src_lang = src_lang

    outputs = []
    forced_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]

        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        generated = model.generate(
            **tokens,
            forced_bos_token_id=forced_id,
            generation_config=gen_cfg,
            max_length=128
        )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )

        outputs.extend(decoded)

    return outputs


# ========= BACKTRANSLATION: EN → ES → EN =========
def back_translate_es_batch(text_list):
    # Traduzir Inglês -> Espanhol
    es_batch = translate_batch(text_list, nllb_tok, nllb_model, src_lang="eng_Latn", tgt_lang="spa_Latn")
    # Traduzir Espanhol -> Inglês
    en_batch = translate_batch(es_batch, nllb_tok, nllb_model, src_lang="spa_Latn", tgt_lang="eng_Latn")
    return en_batch


# ========= BACKTRANSLATION: EN → FR → EN =========
def back_translate_fr_batch(text_list):
    # Traduzir Inglês -> Francês
    fr_batch = translate_batch(text_list, nllb_tok, nllb_model, src_lang="eng_Latn", tgt_lang="fra_Latn")
    # Traduzir Francês -> Inglês
    en_batch = translate_batch(fr_batch, nllb_tok, nllb_model, src_lang="fra_Latn", tgt_lang="eng_Latn")
    return en_batch


# ---------- APLICAR APENAS À CLASSE MINORITÁRIA ----------
hate_df = train_df[train_df["label"] == 1]
hate_df = hate_df[~hate_df["clean_tweet"].isna()]
texts = hate_df["clean_tweet"].astype(str).tolist()
print("Número de frases minoritárias (limpas):", len(texts))
print("Iniciando Back-Translation ES...")
bt_es = back_translate_es_batch(texts)
print("Iniciando Back-Translation FR...")
bt_fr = back_translate_fr_batch(texts)

# ---------- CONSTRUIR AUGMENTATION ----------
augmented = []
for a, b in zip(bt_es, bt_fr):
    if a and a.strip():
        augmented.append((" ".join(a.split()), 1))
    if b and b.strip():
        augmented.append((" ".join(b.split()), 1))

aug_df = pd.DataFrame(augmented, columns=["clean_tweet", "label"])

# ---------- CONCATENAR ----------
train_df = pd.concat([train_df, aug_df], ignore_index=True)

print("\nNova distribuição após augmentation:")
print(train_df["label"].value_counts(normalize=True) * 100)
train_df.to_csv("train_final.csv", index=False)
print("\ntrain_final.csv guardado com sucesso!")