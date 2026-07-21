import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    if isinstance(anchor, list):
        anchor = np.array(anchor)
    if isinstance(positive, list):
        positive = np.array(positive)
    if isinstance(negative, list):
        negative = np.array(negative)

    assert (anchor.ndim == positive.ndim)
    assert (positive.ndim == negative.ndim)

    if anchor.ndim == 1:
        anchor = np.reshape(anchor, (1,-1)) # batch x dim
        positive = np.reshape(positive, (1, -1)) # batch x dim
        negative = np.reshape(negative, (1,-1)) # batch x dim
    
    # apply batch wise distance
    d_a_p = np.sum((anchor - positive) * (anchor - positive), axis=1) # batch x dim -> # batch x 1
    d_a_n = np.sum((anchor - negative) * (anchor - negative), axis = 1) # batch x dim ->. #batch x 1
    mat_val = d_a_p - d_a_n + margin # batch x dim 

    idxs = (mat_val < 0)
    mat_val[idxs] = 0

    loss = np.mean(mat_val)
    return loss