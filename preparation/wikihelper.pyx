import pyximport
pyximport.install() #https://github.com/andersbll/cudarray/issues/25#issuecomment-146217359
import numpy as np
import scipy as sp
cimport numpy as np
from cython cimport floating,boundscheck,wraparound
from cython.parallel import prange
import sklearn.preprocessing as skp
import sklearn.metrics.pairwise as smp
from libc.math cimport fabs

np.import_array()

@boundscheck(False)  # Deactivate bounds checking
@wraparound(False)
def cython_manhattan(floating[::1] X_data, int[:] X_indices, int[:] X_indptr,
                     floating[::1] Y_data, int[:] Y_indices, int[:] Y_indptr,
                     double[:, ::1] D):
    """Pairwise L1 distances for CSR matrices.
    Usage:
    >>> D = np.zeros(X.shape[0], Y.shape[0])
    >>> cython_manhattan(X.data, X.indices, X.indptr,
    ...                  Y.data, Y.indices, Y.indptr,
    ...                  D)
    """
    cdef np.npy_intp px, py, i, j, ix, iy
    cdef double d = 0.0
    
    cdef int m = D.shape[0]
    cdef int n = D.shape[1]
    
    with nogil:                          
        for px in prange(m):
            for py in range(n):
                i = X_indptr[px]
                j = Y_indptr[py]
                d = 0.0
                while i < X_indptr[px+1] and j < Y_indptr[py+1]:
                    if i < X_indptr[px+1]: ix = X_indices[i]
                    if j < Y_indptr[py+1]: iy = Y_indices[j]
                    
                    if ix==iy:
                        d = d+fabs(X_data[i]-Y_data[j])
                        i = i+1
                        j = j+1
                    
                    elif ix<iy:
                        d = d+fabs(X_data[i])
                        i = i+1
                    else:
                        d = d+fabs(Y_data[j])
                        j = j+1
                
                if i== X_indptr[px+1]:
                    while j < Y_indptr[py+1]:
                        iy = Y_indices[j]
                        d = d+fabs(Y_data[j])
                        j = j+1                                            
                else:
                    while i < X_indptr[px+1]:
                        ix = X_indices[i]
                        d = d+fabs(X_data[i])
                        i = i+1
                D[px,py] = d

def sparse_manhattan(X,Y=None):
    X, Y = smp.check_pairwise_arrays(X, Y)
    X = sp.sparse.csr_matrix(X, copy=False)
    Y = sp.sparse.csr_matrix(Y, copy=False)
    res = np.empty(shape=(X.shape[0],Y.shape[0]))
    cython_manhattan(X.data,X.indices,X.indptr,
                     Y.data,Y.indices,Y.indptr,
                             res)
    return res

def year_diffs(graph):
    return [graph.nodes[node]['year'] - graph.nodes[neighbor]['year']
            for node in graph.nodes
            for neighbor in list(graph.successors(node))]

def word_diffs(graph, tfidf):
    dists = sparse_manhattan(X=skp.binarize(tfidf).transpose())
    nodes = list(graph.nodes)
    return [dists[nodes.index(node), nodes.index(neighbor)]
            for node in nodes
            for neighbor in list(graph.successors(node))]

def sum_weight_diffs(graph, tfidf):
    nodes = list(graph.nodes)
    diff = []
    for node in nodes:
        for neighbor in graph.successors(node):
            v1 = tfidf[:,nodes.index(node)]
            v2 = tfidf[:,nodes.index(neighbor)]
            idx = np.concatenate([v1.indices, v2.indices])
            diff.append( np.sum(np.absolute(v1[idx]-v2[idx])) )
    return diff

def weight_diffs(graph, tfidf):
    nodes = list(graph.nodes)
    diff = np.empty(shape=0)
    for node in nodes:
        for neighbor in graph.successors(node):
            v1 = tfidf[:,nodes.index(node)]
            v2 = tfidf[:,nodes.index(neighbor)]
            idx = np.concatenate([v1.indices, v2.indices])
            diff = np.append(diff, np.absolute(v1[idx]-v2[idx]).data)
    return diff

def neighbor_sim(graph, tfidf):
    nodes = list(graph.nodes)
    return [smp.cosine_similarity(tfidf[:,nodes.index(node)].transpose(),
                                  tfidf[:,nodes.index(neighbor)].transpose())[0,0]
            for node in nodes
            for neighbor in list(graph.successors(node))]

def cos_sim(a, b):
     return smp.cosine_similarity(a.transpose(), b.transpose())[0,0]
