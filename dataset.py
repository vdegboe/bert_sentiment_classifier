import torch
import gdown
import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import BertTokenizer


def load_dataset():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_id = "118sgCDikvMsYTKBtkmHJwXfm1hDBI35b"
    output = os.path.join(data_dir, "dataset.csv")

    if not os.path.exists(output):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

    df = pd.read_csv(output, encoding='latin-1')
    df = df[['OriginalTweet', 'Sentiment']]
    df = df.rename(columns={'OriginalTweet': 'Texte'})

    df['Sentiment'] = df['Sentiment'].map({
        'Extremely Negative': 0,
        'Negative': 1,
        'Neutral': 2,
        'Positive': 3,
        'Extremely Positive': 4
    })

    return df


def analyze_dataset(df):
    print(f"Nombre d'exemples: {len(df)}")
    print(f"Nombre de classes: {len(df['Sentiment'].unique())}")

    sentiment_counts = df['Sentiment'].value_counts()
    total = sentiment_counts.sum()

    print("\nDistribution des sentiments:")
    for sentiment, count in sentiment_counts.items():
        proportion = count / total
        percentage = proportion * 100
        print(f"  {sentiment}: {count} ({percentage:.2f}%)")

    max_count = sentiment_counts.max()
    min_count = sentiment_counts.min()
    ratio = max_count / min_count
    print(f"\nRatio de déséquilibre: {ratio:.2f}:1")

    return sentiment_counts


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer_name="bert-base-uncased", max_length=128, label_mapping=None):
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.data = data.reset_index(drop=True)
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data["Texte"][idx]
        label = self.data["Sentiment"][idx]

        if self.label_mapping:
            label = self.label_mapping[label]

        ids = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": ids["input_ids"].squeeze(),
            "attention_mask": ids["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }
