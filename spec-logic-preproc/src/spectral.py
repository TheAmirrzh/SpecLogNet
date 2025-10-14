# src/spectral.py
"""
Compute Laplacian and top-k eigenvectors; cache results to disk.
Uses scipy.sparse for efficiency.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import os
import hashlib
import json
from typing import Tuple


def adjacency_from_edges(n_nodes: int, edges: list) -> sp.csr_matrix:
    """
    Convert a list of edges to a sparse adjacency matrix.
    """
    rows = []
    cols = []
    data = []
    for e in edges:
        rows.append(e["src"])
        cols.append(e["dst"])
        data.append(1)
        # Make symmetric (undirected)
        rows.append(e["dst"])
        cols.append(e["src"])
        data.append(1)
    
    return sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()


def compute_symmetric_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    """
    Compute the symmetric normalized Laplacian L = I - D^(-1/2) A D^(-1/2).
    """
    n = A.shape[0]
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees_inv_sqrt = np.power(degrees, -0.5, where=(degrees != 0))
    D_inv_sqrt = sp.diags(degrees_inv_sqrt)
    return sp.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt


def topk_eig(L: sp.csr_matrix, k: int = 8, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k eigenvalues and eigenvectors of the Laplacian.
    """
    try:
        vals, vecs = eigsh(L, k=min(k, L.shape[0]-1), which='SM', tol=tol)
        return vals, vecs
    except:
        # Fallback to dense computation if sparse fails
        L_dense = L.toarray()
        vals, vecs = np.linalg.eigh(L_dense)
        return vals[:k], vecs[:, :k]


def cache_eigenvectors(instance: dict, cache_dir: str = "data_processed/eigen_cache") -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute and cache eigenvectors for an instance.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hash of the instance
    instance_str = json.dumps(instance, sort_keys=True)
    instance_hash = hashlib.md5(instance_str.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{instance_hash}.json")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached = json.load(f)
            return np.array(cached['eigenvalues']), np.array(cached['eigenvectors'])
    
    # Compute eigenvectors
    n_nodes = len(instance['nodes'])
    A = adjacency_from_edges(n_nodes, instance['edges'])
    L = compute_symmetric_laplacian(A)
    vals, vecs = topk_eig(L, k=min(16, n_nodes-1))
    
    # Cache results
    cache_data = {
        'eigenvalues': vals.tolist(),
        'eigenvectors': vecs.tolist(),
        'n_nodes': n_nodes
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    return vals, vecs