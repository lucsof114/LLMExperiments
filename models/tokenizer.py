from transformers import AutoTokenizer, AutoModel
import torch

class CustomTokenizer:
    def __init__(self, pretrained_model_name, dataset):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.used_tokens = self._find_used_tokens(dataset)
        self.token_id_map = {original_id: new_id for new_id, original_id in enumerate(self.used_tokens)}
        self.reverse_token_id_map = {v: k for k, v in self.token_id_map.items()}

    def _find_used_tokens(self, dataset):
        used_tokens = set()
        for text in dataset:
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            used_tokens.update(token_ids)
        return sorted(used_tokens)

    def encode(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        reduced_token_ids = [self.token_id_map[token_id] for token_id in token_ids if token_id in self.token_id_map]
        return reduced_token_ids

    def decode(self, reduced_token_ids):
        original_token_ids = [self.reverse_token_id_map[token_id] for token_id in reduced_token_ids]
        tokens = self.tokenizer.convert_ids_to_tokens(original_token_ids)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def get_embeddings(self, text):
        model = AutoModel.from_pretrained(self.tokenizer.name_or_path)
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = model(**inputs)
        return outputs.last_hidden_state
