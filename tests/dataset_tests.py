import unittest
import pandas as pd
import torch
from datasets.datasets import PresidentialDataset
from models.tokenizer import TokenMapper
import numpy as np

class TestPresidentialDataset(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset
        self.text = PresidentialDataset.get_text()
        # Initialize TokenMapper
        self.token_mapper = TokenMapper(self.text.values)
        
        # Default parameters
        self.seq_len = 512
        self.mask_prob = 0.1
        
        # Create dataset
        self.dataset = PresidentialDataset(
            tokenizer=self.token_mapper,
            dataset=self.text,
            seq_len=self.seq_len,
            mask_prob=self.mask_prob
        )

    def test_sample_datapoints(self):
        """Test individual samples from the dataset"""
        iterator = iter(self.dataset)
        
        # Get a few samples
        for _ in range(5):
            input_seq, label = next(iterator)
            
            # Check shapes and types
            self.assertIsInstance(input_seq, torch.Tensor)
            self.assertIsInstance(label, torch.Tensor)
            self.assertEqual(input_seq.shape[0], self.seq_len)
            self.assertEqual(label.shape, torch.Size([]))
            
            # Check if values are within valid range
            self.assertTrue(torch.all(input_seq >= 0))
            self.assertTrue(torch.all(input_seq < self.token_mapper.vocab_size))
            self.assertTrue(label >= 0 and label < self.token_mapper.vocab_size)

    def test_sequence_length_distribution(self):
        """Test if the sequence length distribution matches expected values"""
        iterator = iter(self.dataset)
        seq_lengths = []
        n_samples = 1000
        
        # Collect sequence lengths
        for _ in range(n_samples):
            input_seq, _ = next(iterator)
            # Count non-padding tokens
            seq_lengths.append(torch.sum(input_seq != self.token_mapper.tokenizer.pad_token_id).item())
        
        # Calculate expected value
        expected_length = (1 - self.mask_prob) * self.seq_len + self.mask_prob * (self.seq_len + 1) / 2
        mean_length = np.mean(seq_lengths)
        
        # Check if mean length is within 5% of expected value
        self.assertLess(abs(mean_length - expected_length) / expected_length, 0.05,
                       f"Expected length: {expected_length}, Got: {mean_length}")

    def test_padding_consistency(self):
        """Test that padding is always applied at the end of sequences"""
        iterator = iter(self.dataset)
        
        for _ in range(10):
            input_seq, _ = next(iterator)
            non_pad_indices = (input_seq != self.token_mapper.tokenizer.pad_token_id).nonzero()
            
            if len(non_pad_indices) > 0:
                # Check if all padding tokens are at the end
                last_non_pad_idx = non_pad_indices[-1]
                self.assertTrue(torch.all(input_seq[:last_non_pad_idx + 1] != self.token_mapper.tokenizer.pad_token_id))
                self.assertTrue(torch.all(input_seq[last_non_pad_idx + 1:] == self.token_mapper.tokenizer.pad_token_id))

if __name__ == '__main__':
    unittest.main()
