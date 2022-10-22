import numpy as np
from scipy.sparse.linalg import eigs

def PCA_whitenlearn_shrinkage(X, s=1.0):
    
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


# def PCA_whitenlearn_shrinkage(X, s=1.0):
#     """""
#         Learn PCA whitening with shrinkage from given descriptors
    
#     """""
  
#     N = X.shape[1]

#     # Learning PCA w/o annotations
#     m = X.mean(axis=1, keepdims=True)
#     Xc = X - m
    
#     Xcov = np.dot(Xc, Xc.T)
#     Xcov = (Xcov + Xcov.T) / (2*N)
#     eigval, eigvec = np.linalg.eig(Xcov)
    
#     order = eigval.argsort()[::-1]
    
#     eigval = eigval[order]
#     eigvec = eigvec[:, order]

#     P = np.dot(np.linalg.inv(np.sqrt(np.diag(eigval))), eigvec.T)

#     return m, P


def PCA(x: np.ndarray, out_dim=None, subtract_mean=True, log_debug=None):
    # translated from MATLAB:
    # - https://github.com/Relja/relja_matlab/blob/master/relja_PCA.m
    # - https://github.com/Relja/netvlad/blob/master/addPCA.m


    inp_dims    = x.shape[0]
    n_points    = x.shape[1]
    
    if log_debug:
      log_debug('PCA for {%s} points of dimension {%s} to PCA dimension {%s}', n_points, inp_dims, out_dim)

    if subtract_mean:
        # Subtract mean
        mu = np.mean(x, axis=1)
        
        x = (x.T - mu).T
    else:
        mu = np.zeros(inp_dims)


    assert out_dim <= inp_dims


    if inp_dims <= n_points:
        do_dual = False
        # x2 = dims * dims
        x2 = np.matmul(x, x.T) / (n_points - 1)
    else:
        do_dual = True
        # x2 = vectors * vectors
        x2 = np.matmul(x.T, x) / (n_points - 1)


    if out_dim < x2.shape[0]:
        log_debug('Compute {%s} eigenvectors', out_dim)
        lams, u = eigs(x2, out_dim)
    else:
        log_debug('Compute eigenvectors')
        lams, u = np.linalg.eig(x2)

    assert np.all(np.isreal(lams)) and np.all(np.isreal(u))
    lams = np.real(lams)
    u = np.real(u)

    sort_indices = np.argsort(lams)[::-1]
    
    lams = lams[sort_indices]
    
    u = u[:, sort_indices]

    if do_dual:
        # U = x * ( U * diag(1./sqrt(max(lams,1e-9))) / sqrt(nPoints-1) );
        diag        = np.diag(1. / np.sqrt(np.maximum(lams, 1e-9)))
        utimesdiag  = np.matmul(u, diag)
        u           = np.matmul(x, utimesdiag / np.sqrt(n_points - 1))

    return u, lams, mu