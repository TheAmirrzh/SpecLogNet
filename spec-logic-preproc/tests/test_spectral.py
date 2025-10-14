# tests/test_spectral.py
import numpy as np
from src.spectral import adjacency_from_edges, compute_symmetric_laplacian, topk_eig
import scipy.sparse as sp

def test_topk_eig_small_graph():
    # simple chain graph of 4 nodes
    n = 4
    rows = [0,1,2]
    cols = [1,2,3]
    data = [1,1,1]
    A = sp.coo_matrix((data, (rows, cols)), shape=(n,n)).tocsr()
    L = compute_symmetric_laplacian(A)
    vals, vecs = topk_eig(L, k=2)
    assert vals.shape[0] == 2
    assert vecs.shape[0] == n
    assert np.all(np.isfinite(vals))