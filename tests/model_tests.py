import unittest
import torch
import numpy as np
from models import PresidentialAttention, PresidentialModel

class TestPresidentialAttention(unittest.TestCase):
    def setUp(self):
        self.config = {
            'n_heads': 4,
            'hidden_size': 64,
            'attn_dropout': 0.0,
            'batch_size': 2,
            'seq_len': 10
        }
        self.attention = PresidentialAttention(self.config)
        self.batch_size = self.config['batch_size']
        self.seq_len = self.config['seq_len']
        self.hidden_size = self.config['hidden_size']

    def test_output_shape(self):
        query = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        key = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        value = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        output = self.attention(query, key, value)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))

    def test_attention_mask(self):
        query = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        key = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        value = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        attention_mask = torch.zeros(self.batch_size, 1, self.seq_len, self.seq_len)
        attention_mask[:, :, :, 5:] = 1  # Mask out the second half of the sequence

        output_with_mask = self.attention(query, key, value, attention_mask)
        output_without_mask = self.attention(query, key, value)

        # Check that the outputs are different when a mask is applied
        self.assertFalse(torch.allclose(output_with_mask, output_without_mask))

    def test_no_attention_mask(self):
        query = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        key = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        value = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        output = self.attention(query, key, value)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))

    def test_masked_values_unchanged(self):
        query = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        key = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        value = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        attention_mask = torch.zeros(self.batch_size, 1, self.seq_len, self.seq_len)
        attention_mask[:, :, :, 5:] = 1  # Mask out the second half of the sequence

        # Run attention with the mask
        output_with_mask = self.attention(query, key, value, attention_mask)

        # Modify the masked values
        query[:, 5:] = torch.rand(self.batch_size, self.seq_len - 5, self.hidden_size)
        key[:, 5:] = torch.rand(self.batch_size, self.seq_len - 5, self.hidden_size)
        value[:, 5:] = torch.rand(self.batch_size, self.seq_len - 5, self.hidden_size)

        # Run attention again with the modified masked values
        output_with_modified_masked_values = self.attention(query, key, value, attention_mask)

        # Assert that the output is unchanged
        self.assertTrue(torch.allclose(output_with_mask[:, :5], output_with_modified_masked_values[:, :5]))

if __name__ == '__main__':
    unittest.main()
