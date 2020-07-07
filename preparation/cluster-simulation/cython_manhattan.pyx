import numpy as np
cimport numpy as np
from cython cimport floating,boundscheck,wraparound
from cython.parallel import prange

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