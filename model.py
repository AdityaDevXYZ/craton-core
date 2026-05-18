import numpy as np
from transformer import TransformerBlock
from config import CratonConfig

class CratonModel:
    """
    The complete Craton Neural Network Architecture.
    This brings together the Embeddings, the Transformer Blocks, and the Language Modeling Head.
    """
    def __init__(self, config):
        self.config = config
        
        # 1. Token Embedding: Translates word IDs into math vectors
        # Using 0.02 standard deviation for stability
        self.token_embedding = np.random.randn(config.vocab_size, config.d_model) * 0.02
        
        # 2. Positional Embedding: Gives the model a sense of sequence/time
        # Since it processes all words at once, it needs to know the order of words
        self.position_embedding = np.random.randn(config.max_seq_len, config.d_model) * 0.02
        
        # 3. Stack of Transformer Blocks (The deep reasoning layers)
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layers)]
        
        # 4. Final linear projection head (translates brain state back into vocabulary probabilities)
        self.lm_head_W = np.random.randn(config.d_model, config.vocab_size) * 0.02
        self.lm_head_b = np.zeros(config.vocab_size)

    def forward(self, idx, mask=None):
        """
        Pushes a sequence of word IDs forward through the brain to predict the next word.
        idx shape: (batch_size, sequence_length)
        """
        batch_size, seq_len = idx.shape
        
        # Extract embeddings for the input tokens
        tok_emb = self.token_embedding[idx] # Shape: (batch_size, seq_len, d_model)
        
        # Get positional embeddings
        pos = np.arange(0, seq_len)
        pos_emb = self.position_embedding[pos] # Shape: (seq_len, d_model)
        
        # Combine them (Word Meaning + Word Position)
        x = tok_emb + pos_emb
        
        # Pass through the deep layers of the brain
        for block in self.blocks:
            x = block.forward(x, mask)
            
        # Project back to vocabulary to get next-word predictions
        logits = np.dot(x, self.lm_head_W) + self.lm_head_b
        return logits
