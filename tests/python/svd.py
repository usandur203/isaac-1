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
    
    NK = 1
    V = np.zeros((M, M))
    U = np.zeros((M, N))
    
    i = 0
    
    for K in range(N-1):
        Y = np.zeros((M, NK))
        X = np.zeros((N, NK))
        for i in range(NK):
            ki = K*NK + i
            #Householder of column
            V[ki:,ki], tv = householder(A[ki:,ki])
            Y[ki:,i] = tv*dot(A[ki:,ki:].T, V[ki:,ki])
            #Householder of rows
            U[ki+1:,ki], tu = householder(A[ki,ki+1:] - V[ki,ki]*Y[ki+1:,i])
            X[ki:,i] = tu*dot(A[ki:,ki+1:],  U[ki+1:,ki]) - tu*dot(Y[ki+1:,i],  U[ki+1:,ki])*V[ki:,ki]
        A[:,:] -= dot(V[:,K:K+NK], Y.T)
        A[:,K+1:] -= dot(X,  U[K+1:,K:K+NK].T)
    
    return U, A, V
    
    
    
#~ def bidiagonalize(A):
    #~ M, N = A.shape
    #~ 
    #~ V = np.zeros((M, M))
    #~ U = np.zeros((M, N))
    #~ 
    #~ K = 1
    #~ i = 0
    #~ 
    #~ Y = np.zeros((M, K))
    #~ X = np.zeros((N, K))
    #~ 
    #~ #Householder of column
    #~ V[i:,i], tv = householder(A[i:,i])
    #~ Y[i:,i] = tv*dot(A.T, V[i:,i])
    #~ #Householder of rows
    #~ U[i+1:,i], tu = householder(A[i,i+1:] - V[i:,i][i]*Y[i+1:,i])
    #~ X[i:,i] = tu*dot(A[:,1:],  U[i+1:,i]) - tu*dot(Y[i+1:,i],  U[i+1:,i])*V[i:,i]
    #~ 
    #~ A[i:,i:] -= dot(V[:,:K], Y.T)
    #~ A[i:,i+1:] -= dot(X,  U[i+1:,:K].T)
    #~ 
    #~ return U, A, V 
    
np.random.seed(0)
np.set_printoptions(precision=2, suppress=True)
A = np.random.rand(6, 6)
print A
U, A, V = bidiagonalize(A)
print A
