import numpy as np
from numpy import dot, outer

def householder(x):
    a = np.sign(x[0])*np.linalg.norm(x)
    sigma = (x[0] + a)/a
    r = np.copy(x)
    r[0] = x[0] + a
    r /= x[0] + a
    return r, sigma
    
def bidiagonalize(A):
    M, N = A.shape
    
    V = np.zeros((M, M))
    U = np.zeros((M, N))
    
    K = 1
    i = 0
    
    Y = np.zeros((M, K))
    X = np.zeros((N, K))
    
    #Householder of column
    V[i:,i], tv = householder(A[0:,0])
    Y[i:,i] = tv*dot(A.T, V[i:,i])

    #Householder of rows
    U[i+1:,i], tu = householder(A[0,1:] - V[i:,i][0]*Y[i+1:,i])
    X[i:,i] = tu*dot(A[:,1:],  U[i+1:,i]) - tu*dot(Y[i+1:,i],  U[i+1:,i])*V[i:,i]
    
    A -= dot(V[:,:K], Y.T)
    A[:,1:] -= dot(X,  U[i+1:,:K].T)
    
    return U, A, V 
    
np.random.seed(0)
A = np.random.rand(4, 4)
U, A, V = bidiagonalize(A)
print A
