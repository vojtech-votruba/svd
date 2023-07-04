'''With this implementation I decided to make SVD much faster. 
The theoretical basis for this version was taken from this wikipedia article: https://en.wikipedia.org/wiki/Singular_value_decomposition'''
from copy import deepcopy
import numpy as np
from scipy import linalg #For this version, I use scipy because it has LU decomp. function

def svd(A_matrix, N: int):
    U = np.identity(A_matrix.shape[0])
    V_hermit = np.identity(A_matrix.shape[1])
    M = A_matrix
    for _ in range(N): # Paul Godfrey (2023). Simple SVD (https://www.mathworks.com/matlabcentral/fileexchange/12674-simple-svd), MATLAB Central File Exchange. 검색됨 2023/7/3. 
        Q, R = linalg.qr(M)
        Q_,R_ = linalg.qr(R.T)
        L = R_.T
        P = Q_

        U = U @ Q
        V_hermit =  P.T @ V_hermit
        M = L

    return [U,L,V_hermit]