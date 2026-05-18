# Craton Core Configuration
# We start with a micro-architecture to test on the tablet locally.
# These parameters will scale up massively for the cloud training phase.

class CratonConfig:
    def __init__(self):
        self.vocab_size = 50000 
        self.d_model = 1024      # Exploded from 256 to 1024 (Massive brain width)
        self.n_heads = 16        # Exploded from 8 to 16
        self.n_layers = 12       # Exploded from 4 to 12 (Deep reasoning capability)
        self.d_ff = 4096         # Exploded from 1024 to 4096 (Massive memory)
        self.dropout = 0.1
        self.max_seq_len = 1024  # Doubled the context window length
