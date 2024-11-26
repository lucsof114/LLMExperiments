import unittest
from models.tokenizer import TokenMapper

class TestTokenMapper(unittest.TestCase):
    def setUp(self):
        self.char_dataset = ["a", "b", "c", "d", "e", "f", "g", "abcd"]  # Individual characters as input
        self.token_mapper = TokenMapper(self.char_dataset)
        # For character-level tokenization, use a simple character dataset

    def test_char_vocab_size(self):
        """Test that vocab_size equals the number of unique characters in the dataset"""
        # When dataset is individual chars, vocab_size should equal number of unique chars
        expected_size = len(set(self.char_dataset))  # 7 unique characters
        self.assertEqual(self.token_mapper.vocab_size, expected_size)

    def test_text_to_token_ids(self):
        """Test encoding text to reduced token IDs"""
        # Using characters from our dataset
        text = "abcd"
        token_ids = self.token_mapper.encode(text)
        
        # Check that token IDs are within expected range
        self.assertTrue(all(0 <= tid < self.token_mapper.vocab_size for tid in token_ids))
        # Check length matches input
        self.assertEqual(len(token_ids), 1)

    def test_token_ids_to_text(self):
        """Test decoding token IDs back to text"""
        original_text = "abcd"
        # Encode and then decode
        token_ids = self.token_mapper.encode(original_text)
        decoded_text = self.token_mapper.decode(token_ids)
        
        # Check if we get back the original text
        self.assertEqual(decoded_text, original_text)


if __name__ == '__main__':
    unittest.main()
