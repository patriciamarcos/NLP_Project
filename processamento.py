import zipfile
import pandas as pd
from io import TextIOWrapper
import re
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

zip_path = "twitter.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    file_names = zip_ref.namelist()
    print("Ficheiros no ZIP:", file_names)

    train_name = [f for f in file_names if 'train' in f.lower()][0]
    test_name = [f for f in file_names if 'test' in f.lower()][0]

    with zip_ref.open(train_name) as train_file:
        train_df = pd.read_csv(TextIOWrapper(train_file, 'utf-8'))
    with zip_ref.open(test_name) as test_file:
        test_df = pd.read_csv(TextIOWrapper(test_file, 'utf-8'))

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\n", train_df.info())
print("\n", test_df.info())


print("\n Distribuição das classes no treino:")
print(train_df['label'].value_counts(normalize=True).round(3) * 100)

tweet_length = train_df['tweet'].apply(lambda x: len(str(x).split()))
print("\nComprimento médio dos tweets:", tweet_length.mean())

print("\nExemplo de tweet NORMAL (label 0):")
print(train_df[train_df['label'] == 0]['tweet'].iloc[0])

print("\nExemplo de tweet com HATE SPEECH (label 1):")
print(train_df[train_df['label'] == 1]['tweet'].iloc[0])


stop_words = set(stopwords.words('english'))

def fix_mojibake(text):
    try:
        return text.encode('cp1252').decode('utf-8')
    except:
        return text


def clean_tweet(text):
    text = fix_mojibake(text)
    text = re.sub(r'@[\w_]+', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

train_df['clean_tweet'] = train_df['tweet'].apply(clean_tweet)
test_df['clean_tweet'] = test_df['tweet'].apply(clean_tweet)

print("Pré-processamento concluído com sucesso!")
print("\nExemplo antes/depois:\n")
print("Original:", train_df['tweet'].iloc[0])
print("Limpo:", train_df['clean_tweet'].iloc[0])

train_df_to_save = train_df.drop(columns=['tweet'])
test_df_to_save = test_df.drop(columns=['tweet'])
test_df_to_save.to_csv('test_clean.csv', index=False)
print("Saved to test_clean.csv (apenas tweet limpo)")
train_df_to_save.to_csv('train_clean.csv', index=False)
print("Saved to train_clean.csv (apenas tweet limpo)")


hate_text_clean = " ".join(train_df[train_df['label'] == 1]['clean_tweet'].astype(str))
normal_text_clean = " ".join(train_df[train_df['label'] == 0]['clean_tweet'].astype(str))

hate_wc_clean = WordCloud(width=800, height=500, background_color='black', colormap='Reds').generate(hate_text_clean)
normal_wc_clean = WordCloud(width=800, height=500, background_color='black', colormap='Blues').generate(normal_text_clean)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(normal_wc_clean, interpolation='bilinear')
plt.axis('off')
plt.title("Tweets Normais", fontsize=16)

plt.subplot(1, 2, 2)
plt.imshow(hate_wc_clean, interpolation='bilinear')
plt.axis('off')
plt.title("Hate Speech", fontsize=16)

plt.show()