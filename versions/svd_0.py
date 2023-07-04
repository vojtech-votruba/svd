'''This implementation is the one we used in our LA class (MFF UK, 2022/2023 LS) to manualy calculate SVD.
It is extremly slow, but I wanted to put it here as it is a good demonstration of how the decomposition works'''

import numpy as np
from sympy import Matrix #We are using Sympy because it doesn't use svd to calculte nullspace https://github.com/sympy/sympy/blob/master/sympy/matrices/subspaces.py

epsilon = 1E-7
#Simple algorithm to return SVD terms of A = U \Sigma V*
def svd(A_matrix):

    A_star_A = A_matrix.T @ A_matrix #Matrix of a form A*A
    eigenvalues, V_matrix = np.linalg.eigh(A_star_A) 

    ind = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[~ind]
    V_matrix = V_matrix.T[~ind].T #Sorting the eigenvalues from largest to lowest (with coresponding eigenvectors)

    nonzero_eigen = []
    for lambda_value in eigenvalues:
        if lambda_value > epsilon:
            nonzero_eigen.append(lambda_value)
    singl_values = np.sqrt(nonzero_eigen) #Calculating singular values

    U_matrix = np.column_stack([A_matrix @ V_matrix[:,seq]/value for seq,value in enumerate(singl_values)])  
    U_sympy = Matrix(U_matrix.T)
    complement =  np.array(U_sympy.nullspace())
    for i in range(complement.shape[0]):
        U_matrix = np.column_stack((U_matrix, complement[i]))
    Sigma_matrix = np.diagflat(singl_values)
    while Sigma_matrix.shape[1] < A_matrix.shape[1]:
        Sigma_matrix = np.column_stack((Sigma_matrix, np.zeros((Sigma_matrix.shape[0],1))))

    while Sigma_matrix.shape[0] < A_matrix.shape[0]:
        Sigma_matrix = np.row_stack((Sigma_matrix, np.zeros((Sigma_matrix.shape[1],1)).T))

    return [U_matrix, Sigma_matrix,V_matrix.T]