import numpy as np
import scipy.sparse as sp


def gcpca(Xt, Xb, St, Sb, n_components=50, yita=0.1, mu=0.2):
    
    # --- Defensive conversion ---
    if not sp.issparse(St):
        St = sp.csr_matrix(St)
    if not sp.issparse(Sb):
        Sb = sp.csr_matrix(Sb)
        
    # --- Normalized Laplacian construction ---
    normL1 = normalized_laplacian(St)
    normL2 = normalized_laplacian(Sb)
    
    # --- Adjust Laplacians ---
    I1 = sp.identity(St.shape[0])
    I2 = sp.identity(Sb.shape[0])
    L11 = I1 - mu * normL1
    L22 = I2 - mu * normL2
    
    # --- Compute covariance matrices ---
    cov1 = Xt.T @ (L11 @ Xt)
    cov2 = Xb.T @ (L22 @ Xb)
    
    RR = cov1 - yita * cov2
    
    eigvals, eigvecs = np.linalg.eigh(RR)
    idx = np.argsort(eigvals)[::-1][:n_components]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    fg_comp = Xt @ eigvecs
    bg_comp = Xb @ eigvecs
    return eigvecs, fg_comp.A, bg_comp.A
    
    
def normalized_laplacian(A):
    d = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L = sp.diags(d) - A
    normL = D_inv_sqrt @ L @ D_inv_sqrt
    return normL