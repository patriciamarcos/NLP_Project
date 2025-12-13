# Hate Speech Detection with Explainable NLP

Este repositório contém um pipeline completo para deteção automática de discurso de ódio em tweets, utilizando Twitter-RoBERTa, data augmentation por back-translation, e explicabilidade com Integrated Gradients, bem como uma aplicação interativa desenvolvida em Streamlit.

## Estrutura do Projeto

A execução do projeto segue a seguinte ordem lógica:

- processamento.py que gera os seguintes ficheiros:
    - train_clean.csv
    - test_clean.csv

- augmentation.py que gera o ficheiro:
    - test_clean.csv

- treino.py que guarda:
    - best_model.pt
    - modelo_final.pth
    - pasta tokenizer/

- interacao.py

- app.py

## Executar a Aplicação

Após treinar o modelo e garantir que os ficheiros best_model.pt e tokenizer/ existem, a aplicação pode ser iniciada com o seguinte comando:

``` 
streamlit run app.py 
```
