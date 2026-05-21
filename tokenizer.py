class CratonTokenizer:
    """
    A custom, lightweight tokenizer built from scratch.
    Neural Networks cannot read English letters. This script converts text strings 
    into arrays of mathematical integer IDs that Craton's brain can process.
    We are starting with a character-level tokenizer for extreme speed and low memory on the tablet.
    """
    def __init__(self):
        # We start with basic ASCII characters for our zero-cost proof of concept
        chars = sorted(list(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-:;'\"()[]{}<>@#$%^&*_=+\\|/\n\t`~")))
        
        self.vocab_size = len(chars) + 1 # +1 for unknown characters
        
        # Lookup tables
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.unk_id = self.vocab_size - 1

    def encode(self, text):
        """String -> List of Integers"""
        return [self.stoi.get(c, self.unk_id) for c in text]

    def decode(self, ids):
        """List of Integers -> String"""
        return ''.join([self.itos.get(i, "?") for i in ids])
