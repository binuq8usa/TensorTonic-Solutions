import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    context_size = seq_len
    embed_size = d_model

    pos_vector = np.arange(0, context_size).reshape(-1,1) # column vector 
    dim_vector = np.arange(0, (embed_size + 1)//2).reshape(1,-1)
    freq_vector = base ** (2 * dim_vector / embed_size)

    freq_table = pos_vector / freq_vector

    pos_embeds = np.zeros((context_size, embed_size))
    pos_embeds[:, 0::2] = np.sin(freq_table)
    pos_embeds[:, 1::2] = np.cos(freq_table[:, : embed_size // 2])

    return pos_embeds    