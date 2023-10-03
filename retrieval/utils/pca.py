import numpy as np


def PCA(X, s=1.0):
    """Learn PCA whitening with shrinkage from given descriptors"""
    N = X.shape[0]

    # Learning PCA w/o annotations
    m = X.mean(axis=0, keepdims=True)
    Xc = X - m

    Xcov = np.dot(Xc.T, Xc)
    Xcov = (Xcov + Xcov.T) / (2*N)

    eigval, eigvec = np.linalg.eig(Xcov)

    order = eigval.argsort()[::-1]

    eigval = eigval[order]

    eigvec = eigvec[:, order]

    P = np.dot(np.linalg.inv(np.diag(np.power(eigval, 0.5*s))), eigvec.T)

    return m, P.T
