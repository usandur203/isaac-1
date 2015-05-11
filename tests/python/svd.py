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
    #~ k = 1
    #~ i = 0
    #~ M, N = A.shape
    #~ 
    #~ V = np.zeros(M, M)
    #~ U = np.zeros(M, N)
    
    #Householder of column
    v, tv = householder(A[0:,0])
    y = tv*dot(A.T, v)

    #Householder of rows
    u, tu = householder(A[0,1:] - v[0]*y[1:])
    x = tu*dot(A[:,1:], u) - tu*dot(y[1:], u)*v
    
    A -= outer(v, y)
    A[:,1:] -= outer(x, u)
    
#~ def bidiagonalize(A):
    #~ k = 1
    #~ i = 0
    #~ 
    #~ #Householder of column
    #~ v, tv = householder(A[i:,i])
    #~ y = tv*dot(A.T, v)
    #~ A -= outer(v,y)
    #~ 
    #~ #Householder of rows
    #~ u, tu = householder(A[i,i+1:])
    #~ x = tu*dot(A[:,i+1:], u)
    #~ A[:,i+1:] -= np.outer(x, u)
    #~ 
    #~ print A
    
np.random.seed(0)
A = np.random.rand(4, 4)
bidiagonalize(A)
