class SimpleTokenizer:
    def __init__(self, vocabulary):
        self.vocab = {'0': 0}  # Special token for padding
        self.vocab.update({word: i+1 for i, word in enumerate(vocabulary)})
        self.inverse_vocab = {i: word for word, i in self.vocab.items()}
    
    def tokenize(self, text):
        """Converts text to a sequence of token IDs."""
        return [self.vocab.get(token, self.vocab['0']) for token in text.split()]
    
    def pad_sequence(self, sequence, max_length):
        """Pads sequences to the max_length with the padding token ID (0)."""
        return sequence + [self.vocab['0']] * (max_length - len(sequence))
    
    def decode(self, token_ids):
        """Converts a sequence of token IDs back to text."""
        return ' '.join(self.inverse_vocab.get(id, '0') for id in token_ids)

# Example usage
vocabulary_size = 50000  # As defined earlier
vocabulary = ["token_{}".format(i) for i in range(1, vocabulary_size + 1)]  # Generate vocabulary
tokenizer = SimpleTokenizer(vocabulary)

# Tokenize and pad a sample sequence
sample_sequence = "token_1 token_2 token_9999"
token_ids = tokenizer.tokenize(sample_sequence)
padded_token_ids = tokenizer.pad_sequence(token_ids, context_window_length)  # Assuming context_window_length is 512

# Decode back to text (just for demonstration)
decoded_sequence = tokenizer.decode(padded_token_ids[:len(token_ids)])  # Decode without padding for clarity

print(f"Original: {sample_sequence}")
print(f"Token IDs: {token_ids[:10]}")
print(f"Padded Token IDs: {padded_token_ids[:10]} + Padding")
print(f"Decoded: {decoded_sequence}")

