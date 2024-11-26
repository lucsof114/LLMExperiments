import pandas as pd
from torch.utils.data import Dataset
import os
import torch
import re

class PresidentialDataset(Dataset):
    """
    An IterableDataset for next token prediction training.
    Generates (input_sequence, next_token) pairs infinitely.
    """
    DATASET_DIR = "datasets/speeches"

    def __init__(self, tokenizer, dataset, seq_len=512, mask_prob=0.1):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        

        # Tokenize all texts
        self.dataset['tokens'] = self.dataset['text'].apply(
            lambda text: self.tokenizer.encode(text)
        )

        self.dataset['num_samples'] = self.dataset['tokens'].apply(
            lambda t: len(t) - self.seq_len
        )
        

    def __len__(self):
        # Return the number of samples in the dataset
        return self.dataset['num_samples'].sum()


    def __getitem__(self, idx):
        # Find the text index and position within the text
        cumulative_tokens = self.dataset['num_samples'].cumsum()
        text_idx = (cumulative_tokens > idx).idxmax()
        token_idx = idx - (cumulative_tokens[text_idx - 1] if text_idx > 0 else 0)
        
        # Get the tokens for the selected text
        text_tokens = self.dataset.iloc[text_idx]['tokens']

        assert token_idx < len(text_tokens) - self.seq_len, "Index out of range for token sequence."
        
        # Create input-output pair
        input_sequence = text_tokens[token_idx:token_idx + self.seq_len]
        next_tokens = text_tokens[(token_idx + 1):(token_idx + self.seq_len + 1)]        
        return torch.tensor(input_sequence), torch.tensor(next_tokens, dtype=torch.long)

    @classmethod
    def get_text(cls):
        if not os.path.exists(os.path.join(cls.DATASET_DIR, "data.csv")):
            cls.download_dataset()
        dataset = pd.read_csv(os.path.join(cls.DATASET_DIR, "data.csv"))
        return dataset


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
