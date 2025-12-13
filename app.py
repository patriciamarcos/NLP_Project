from interacao import classify_text, explain_text, plot_explanation, test_sentences
import streamlit as st

# 1) CONFIG STREAMLIT
st.set_page_config(page_title="Classificador de Tweets", layout="wide")

st.title("Classificador de Tweets com Explicabilidade")
st.write("Modelo baseado em **Twitter-RoBERTa** com Integrated Gradients (Captum).")

# 2) INTERFACE
st.subheader("Escolhe um tweet ou escreve o teu próprio")

col1, col2 = st.columns(2)

with col1:
    chosen = st.selectbox(
        "Exemplos do test set:",
        ["(Escrever manualmente)"] + test_sentences[:600]
    )

with col2:
    manual_text = st.text_area("Texto manual:", height=130)

if chosen != "(Escrever manualmente)":
    text = chosen
else:
    text = manual_text.strip()

if st.button("Classificar texto", type="primary"):
    if len(text) == 0:
        st.warning("Insere um texto primeiro.")
    else:
        pred, conf, probs = classify_text(text)

        st.write("### Resultado")
        st.write(f"**Texto:** {text}")
        st.write(f"**Classe prevista:** `{pred}`")
        st.write(f"**Confiança:** `{conf:.4f}`")
        st.write(f"**Probabilidades:** {probs}")

        st.write("### Explicabilidade por token")
        tokens, scores = explain_text(text, pred)

        explanation_text = " ".join(
            [f"{t.replace('Ġ', '')}({s:.2f})" for t, s in zip(tokens, scores)]
        )
        st.code(explanation_text)

        # Gráfico
        st.write("### Gráfico de Barras")
        fig1 = plot_explanation(tokens, scores)
        st.pyplot(fig1)
