import numpy as np

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class MultiHeadAttention:
    """
    The core reasoning mechanism of Craton.
    Allows the model to focus on different parts of the input sequence simultaneously.
    Written entirely in NumPy to run natively on the Lenovo tablet at zero cost.
    """
    def __init__(self, config):
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = self.d_model // self.n_heads
        
        # Initialize weights
        # Using Xavier/Glorot initialization for better mathematical stability
        scale = np.sqrt(2.0 / (self.d_model + self.d_model))
        self.W_q = np.random.randn(self.d_model, self.d_model) * scale
        self.W_k = np.random.randn(self.d_model, self.d_model) * scale
        self.W_v = np.random.randn(self.d_model, self.d_model) * scale
        self.W_o = np.random.randn(self.d_model, self.d_model) * scale

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape
        
        # 1. Linear projections
        Q = np.dot(q, self.W_q)
        K = np.dot(k, self.W_k)
        V = np.dot(v, self.W_v)
        
        # 2. Split into multiple heads
        # Shape: (batch_size, n_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 3. Scaled Dot-Product Attention
        # scores = (Q * K^T) / sqrt(d_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        if mask is not None:
            # Apply mask (so the model can't look into the future during training)
            scores = np.where(mask == 0, -1e9, scores)
            
        attention_weights = softmax(scores)
        
        # context = weights * V
        context = np.matmul(attention_weights, V)
        
        # 4. Concatenate heads and project output
        # Shape back to: (batch_size, seq_len, d_model)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = np.dot(context, self.W_o)
        
        return output, attention_weights
