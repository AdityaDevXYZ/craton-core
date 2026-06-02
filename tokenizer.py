import tiktoken

class CratonTokenizer:
    """
    Advanced Byte-Pair Encoding (BPE) Tokenizer.
    Upgraded from Phase 1 character-level to Phase 6 sub-word level.
    This gives Craton the exact same vocabulary efficiency as GPT-4.
    """
    def __init__(self):
        # We use the cl100k_base tokenizer (GPT-4 standard)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.encoder.n_vocab

    def encode(self, text):
        """String -> List of Integers"""
        return self.encoder.encode(text, allowed_special="all")

    def decode(self, ids):
        """List of Integers -> String"""
        return self.encoder.decode(ids)
