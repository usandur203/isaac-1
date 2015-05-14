import numpy as np
from numpy import dot, outer

def dlarfg(x):
    a = -np.sign(x[0])*np.linalg.norm(x)
    sigma = (a - x[0])/a
    r = np.copy(x)
    r[0] = x[0] - a
    r /= x[0] - a
    return r, sigma, a

def dlabrd(A, NB):
    M, N = A.shape
    X = np.zeros((M, NB))
    Y = np.zeros((N, NB))
    tauq = np.zeros(NB)
    taup = np.zeros(NB)
    e = np.zeros(NB)
    d = np.zeros(NB)
    for i in range(NB):
        #Update A[i:, i]
        A[i:, i]   -= dot(A[i:, :i]     , Y[i, :i])
        A[i:, i]   -= dot(X[i:, :i]     , A[:i, i])
        #Householder A[i:,i]
        A[i:, i], tauq[i], d[i] = dlarfg(A[i:, i])
        if i < NB - 1:
            #Compute Y[i+1:,i]
            Y[i+1:, i]  = dot(A[i:,i+1:].T  , A[i:, i])
            Y[:i, i]    = dot(A[i:,:i].T    , A[i:, i])
            Y[i+1:, i] -= dot(Y[i+1:, :i]   , Y[:i, i])
            Y[:i, i]    = dot(X[i:, :i].T   , A[i:, i])
            Y[i+1:, i] -= dot(A[:i,i+1:].T  , Y[:i, i])
            Y[i+1:, i] *= tauq[i]
            #Update A[i, i+1:]
            A[i, i+1:] -= dot(Y[i+1:,:i+1], A[i,:i+1])
            A[i, i+1:] -= dot(A[:i, i+1:].T, X[i,:i]) 
            #Householder of A[i, i+1:]
            A[i, i+1:], taup[i], e[i+1] = dlarfg(A[i,i+1:])
            #Compute X[i+1:,i]
            X[i+1:,i]  = dot(A[i+1:,i+1:]   , A[i,i+1:])
            X[:i+1,i]  = dot(Y[i+1:,:i+1].T , A[i,i+1:])
            X[i+1:,i] -= dot(A[i+1:,:i+1]   , X[:i+1,i])
            X[:i,i]    = dot(A[:i, i+1:]    , A[i, i+1:])
            X[i+1:,i] -= dot(X[i+1:,:i]     , X[:i,i])
            X[i+1:,i] *= taup[i]
    return A, e, d
    

np.random.seed(0)
np.set_printoptions(precision=2, suppress=True)
A = np.random.rand(4,4)
A, e, d = dlabrd(A.copy(), 4)
