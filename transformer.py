import numpy as np
from attention import MultiHeadAttention

class FeedForward:
    """
    The 'memory' and non-linear processing part of the neural network block.
    """
    def __init__(self, config):
        scale1 = np.sqrt(2.0 / (config.d_model + config.d_ff))
        self.W1 = np.random.randn(config.d_model, config.d_ff) * scale1
        self.b1 = np.zeros(config.d_ff)
        
        scale2 = np.sqrt(2.0 / (config.d_ff + config.d_model))
        self.W2 = np.random.randn(config.d_ff, config.d_model) * scale2
        self.b2 = np.zeros(config.d_model)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        hidden = self.relu(np.dot(x, self.W1) + self.b1)
        return np.dot(hidden, self.W2) + self.b2

class LayerNorm:
    """
    Normalizes the data flowing through the network to keep gradients stable.
    """
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        norm_x = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta

class TransformerBlock:
    """
    A single full block of Craton's brain.
    Contains Self-Attention, Layer Normalization, and Feed Forward layers.
    Craton will stack multiple of these blocks to form its deep neural network.
    """
    def __init__(self, config):
        self.attention = MultiHeadAttention(config)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x, mask=None):
        # Sublayer 1: Multi-Head Attention + Add & Norm (Residual Connection)
        attn_out, _ = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_out)
        
        # Sublayer 2: Feed Forward + Add & Norm (Residual Connection)
        ff_out = self.ff.forward(x)
        x = self.norm2.forward(x + ff_out)
        
        return x
