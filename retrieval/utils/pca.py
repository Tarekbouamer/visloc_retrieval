import numpy as np


def PCA(M, s=1.0):
    """ compute principal components analysis (PCA) of matrix M
    Args:
        M: matrix of shape (N, D)
        s: scaling factor
    Returns:
        m: mean of M
        P: principal components of M
    """

    # N number of samples
    N = M.shape[0]

    # mean of M
    m = M.mean(axis=0, keepdims=True)
    Xc = M - m

    # covariance matrix of M
    Xcov = np.dot(Xc.T, Xc)
    Xcov = (Xcov + Xcov.T) / (2*N)

    # eigval eigvec
    eigval, eigvec = np.linalg.eig(Xcov)

    # sort eigval and eigvec
    order = eigval.argsort()[::-1]

    eigval = eigval[order]
    eigvec = eigvec[:, order]
    
    # principal components of M 
    P = np.dot(np.linalg.inv(np.diag(np.power(eigval, 0.5*s))), eigvec.T)

    return m, P.T
