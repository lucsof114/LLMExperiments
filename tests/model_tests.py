import unittest
import torch
import numpy as np
from models import PresidentialAttention

class TestPresidentialAttention(unittest.TestCase):
    def setUp(self):
        self.config = {
            'n_heads': 4,
            'hidden_size': 64,
            'attn_dropout': 0.1
        }
        self.attention = PresidentialAttention(self.config)
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = self.config['hidden_size']

    def test_output_shape(self):
        query = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        key = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        value = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        output = self.attention(query, key, value)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))

    # def test_attention_mask(self):
    #     query = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
    #     key = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
    #     value = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
    #     attention_mask = torch.zeros(self.batch_size, 1, self.seq_len, self.seq_len)
    #     attention_mask[:, :, :, 5:] = 1  # Mask out the second half of the sequence

    #     output_with_mask = self.attention(query, key, value, attention_mask)
    #     output_without_mask = self.attention(query, key, value)

    #     # Check that the outputs are different when a mask is applied
    #     self.assertFalse(torch.allclose(output_with_mask, output_without_mask))

    # def test_no_attention_mask(self):
    #     query = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
    #     key = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
    #     value = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
    #     output = self.attention(query, key, value)
    #     self.assertIsNotNone(output)

if __name__ == '__main__':
    unittest.main()
