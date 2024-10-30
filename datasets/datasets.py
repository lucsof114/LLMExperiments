import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import torch
import re

class PresidentialDataset(Dataset):
    """
    A PyTorch Dataset class for handling and processing Donald Trump's speeches.

    Attributes:
        DATASET_DIR (str): Directory where the dataset is stored.
        tokenizer (AutoTokenizer): Tokenizer for encoding text data.
        dataset (DataFrame): DataFrame containing the speeches and their tokenized forms.
        seq_len (int): Sequence length for tokenized data.
    """
    DATASET_DIR = "/Users/lucassoffer/Documents/Develop/cursor_ai/datasets/trump_speeches"

    def __init__(self, tokenizer_name="meta-llama/Llama-3.2-1B", seq_len=512):
        if not os.path.exists(os.path.join(self.DATASET_DIR, "data.csv")):
            self.download_dataset()

        self.dataset = pd.read_csv(os.path.join(self.DATASET_DIR, "data.csv"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset['tokens'] = self.dataset['text'].apply(lambda text: self.tokenizer.encode(text, add_special_tokens=False))
        self.token_indexer = self.dataset['tokens'].apply(len).cumsum().values
        self.seq_len = seq_len
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})



    @classmethod
    def download_dataset(cls, dataset_path=DATASET_DIR):
        import kagglehub
        import hashlib
        print("Downloading dataset...") 
        path = kagglehub.dataset_download("christianlillelund/donald-trumps-rallies")
        # save the dataset to the dataset_path and convert to csv
        out = {}
        for file_name in os.listdir(path):
            if file_name.endswith(".txt"):
                with open(os.path.join(path, file_name), 'r', encoding='utf-8') as file:
                    text = file.read()
                    hash_index = hashlib.md5(text.encode()).hexdigest()[:8]
                    out[hash_index] = text
        pd.Series(out).to_csv(os.path.join(dataset_path, "clean/donald-trumps-rallies.csv"))

        data = pd.read_csv(os.path.join(dataset_path, 'raw/MrTrumpSpeeches.csv'))
        def clean_row(row):
            row = row.dropna()
            row = ' '.join(row.values.astype(str))
            row = re.sub(r'\[.*?\]', '', row)
            row = re.sub(r'^(.*? {2,}){2}', '', row)
            matches = list(re.finditer(r' {2,}', row))
            if len(matches) > 1:
                second_to_last_match = matches[-2]
                row = row[:second_to_last_match.end()]
            row = re.sub(r' {2,}', ' ', row)
            return row
        
        cleaned_data = data.apply(clean_row, axis=1)
        out = {}
        for text in cleaned_data:
            hash_index = hashlib.md5(text.encode()).hexdigest()[:8]
            out[hash_index] = text
        pd.Series(out).to_csv(os.path.join(dataset_path, "clean/trump_youtube_transcripts.csv"))

        dataset_files = []
        for file in os.listdir(os.path.join(cls.DATASET_DIR, "clean")):
            if file.endswith(".csv"):
                dataset_files.append(file)
        dataset = pd.concat([pd.read_csv(os.path.join(cls.DATASET_DIR, "clean", file), index_col=0).rename(columns={'0': 'text'}) for file in dataset_files]) 
        dataset.to_csv(os.path.join(cls.DATASET_DIR, "data.csv"))

        print("Dataset downloaded and moved to", dataset_path)

    def __len__(self):
        return self.token_indexer[-1]

    def __getitem__(self, idx):
        text_idx = next((i for i, val in enumerate(self.token_indexer) if val >= idx), None)
        if text_idx is None:
            text_idx = len(self.token_indexer) - 1
        elif self.token_indexer[text_idx] > idx:
            text_idx = 0

        token_idx = idx - self.token_indexer[text_idx] + 1
        text_tokens = self.dataset.iloc[text_idx]['tokens']
        seq = text_tokens[max(0, token_idx-self.seq_len):token_idx]
        if len(seq) < self.seq_len:
            seq = torch.cat([torch.tensor([self.tokenizer.pad_token_id] * (self.seq_len - len(seq))), torch.tensor(seq)])
        return seq


if __name__ == "__main__":
    # Instantiate the dataset
    trump_speeches = TrumpSpeeches()

    # Print the length of the dataset
    print(f"Number of speeches in the dataset: {len(trump_speeches)}")

    # Print a sample speech
    sample_idx = 0
    sample_speech = trump_speeches[sample_idx]
    print(f"Sample speech at index {sample_idx}:")
    print(sample_speech)
    total_words = sum(len(speech['text'].split()) for speech in trump_speeches)
    print(f"Total number of words in the dataset: {total_words}")
