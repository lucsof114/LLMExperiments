from transformers import AutoTokenizer, AutoModel
import torch
import json
import torch.nn as nn

class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(TokenEmbeddings, self).__init__()
        self.embed_dim = hidden_size
        self.model = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x):
        return self.model(x)

class TokenMapper:
    """
    A class that maps tokens to new ids and vice versa. 
    Used to reduce the number of tokens in the vocabulary from a pretrained model.
    """
    def __init__(self, tokenizer, token_id_map, reverse_token_id_map, embeddings):
        self.tokenizer = tokenizer
        self.token_id_map = token_id_map
        self.reverse_token_id_map = reverse_token_id_map
        self.embeddings = embeddings

    @classmethod
    def init_from_dataset(cls, dataset, pretrained_model_name="meta-llama/Llama-3.2-1B"):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        used_tokens = cls.find_used_tokens(dataset, tokenizer)
        token_id_map = {original_id: new_id for new_id, original_id in enumerate(used_tokens)}
        reverse_token_id_map = {v: k for k, v in token_id_map.items()}

        model = AutoModel.from_pretrained(pretrained_model_name)
        embeddings = TokenEmbeddings(tokenizer.vocab_size, model.config.hidden_size)
        original_embeddings =  model.get_input_embeddings().weight.data
        for original_id, new_id in  token_id_map.items():
            if original_id == tokenizer.pad_token_id:
                continue
            embeddings.model.weight.data[new_id] = original_embeddings[original_id]

        return cls(tokenizer, token_id_map, reverse_token_id_map, embeddings)

    @property
    def embed_dim(self):
        return self.embeddings.embed_dim

    @classmethod
    def load(cls, path):
        # Load the entire dictionary from the .pth file
        checkpoint = torch.load(path)
        
        # Extract the configuration and state dictionary
        token_id_map = checkpoint['token_id_map']
        reverse_token_id_map = checkpoint['reverse_token_id_map']
        pretrained_model_name = checkpoint['pretrained_model_name']
        
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        embeddings = TokenEmbeddings(tokenizer.vocab_size, checkpoint['embed_dim'])
        
        # Load the model's state dictionary
        embeddings.load_state_dict(checkpoint['state_dict'])
        
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return cls(tokenizer, token_id_map, reverse_token_id_map, embeddings)

    def save(self, path):
        checkpoint = {
            'state_dict': self.embeddings.state_dict(),
            'token_id_map': self.token_id_map,
            'reverse_token_id_map': self.reverse_token_id_map,
            'embed_dim': self.embed_dim,
            'pretrained_model_name': self.tokenizer.name_or_path
        }
        
        torch.save(checkpoint, path)

    @property
    def pad_token_id(self):
        return self.token_id_map[self.tokenizer.pad_token_id]

    @property
    def vocab_size(self):
        """Returns the size of the reduced vocabulary."""
        return len(self.token_id_map)

    @property
    def pretrained_vocab_size(self):
        """Returns the size of the original pretrained model's vocabulary."""
        return self.tokenizer.vocab_size

    @classmethod
    def find_used_tokens(cls, dataset, tokenizer):
        used_tokens = set(tokenizer.all_special_ids)
        for text in dataset['text'].values:
            tokens =  tokenizer.tokenize(text)
            token_ids =   tokenizer.convert_tokens_to_ids(tokens)
            used_tokens.update(token_ids)
        return used_tokens

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
        reduced_token_ids = self.encode(text)
        reduced_token_ids_tensor = torch.tensor(reduced_token_ids, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embeddings = self.embeddings(reduced_token_ids_tensor)
        return embeddings
