import numpy as np
from numpy import dot, outer, sqrt, cos, sin
from math import atan2

#Bidiagonalization
def sign(x):
	return 1 if x>=0 else -1
	
def larfg(x):
    a = -np.sign(x[0])*np.linalg.norm(x)
    sigma = (a - x[0])/a
    r = np.copy(x)
    r[0] = x[0] - a
    r /= x[0] - a
    return r, sigma, a

def labrd(A, NB):
    if(A.dtype==np.int32):
        A = A.astype(np.float32)
    M, N = A.shape
    X = np.zeros((M, NB))
    Y = np.zeros((N, NB))
    tauq = np.zeros(NB)
    taup = np.zeros(NB)
    s = np.zeros(NB)
    d = np.zeros(NB)
    for i in range(NB):
        #Update A[i:, i]
        A[i:, i]   -= dot(A[i:, :i]     , Y[i, :i])
        A[i:, i]   -= dot(X[i:, :i]     , A[:i, i])
        #Householder A[i:,i]
        A[i:, i], tauq[i], d[i] = larfg(A[i:, i])
        #print i, A
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
            A[i, i+1:], taup[i], s[i+1] = larfg(A[i,i+1:])
            #Compute X[i+1:,i]
            X[i+1:,i]  = dot(A[i+1:,i+1:]   , A[i,i+1:])
            X[:i+1,i]  = dot(Y[i+1:,:i+1].T , A[i,i+1:])
            X[i+1:,i] -= dot(A[i+1:,:i+1]   , X[:i+1,i])
            X[:i,i]    = dot(A[:i, i+1:]    , A[i, i+1:])
            X[i+1:,i] -= dot(X[i+1:,:i]     , X[:i,i])
            X[i+1:,i] *= taup[i]
            
    return A, d, s, tauq, taup

def gebd2(A):
    M, N = A.shape
    tauq = np.zeros(N)
    taup = np.zeros(N)
    s = np.zeros(N)
    d = np.zeros(N)
    for i in range(N):
        # Householder vector
        A[i:, i], tauq[i], d[i] = larfg(A[i:, i])
        #  Apply H(i) to A(i:m,i+1:n) from the left
        x  = dot(A[i:,i+1:].T  , A[i:, i])
        A[i:,i+1:] -= tauq[i]*np.outer(A[i:,i], x)
        if i < N - 1:
            # Householder vector
            A[i,i+1:], taup[i], s[i+1] = larfg(A[i,i+1:])
            # Apply G(i) to A(i+1:m,i+1:n) from the right 
            x = dot(A[i+1:,i+1:],A[i,i+1:])
            A[i+1:, i+1:] -= taup[i]*np.outer(x, A[i,i+1:])
        else:
            taup[i] = 0
    return A, d, s, tauq, taup
    
#Diagonalization
def rot(f, g):
	if f==0:
		return  0, 1, g
	elif abs(f) > abs(g):
		t = g/f
		tt = sqrt(1 + t**2)
		return 1/tt, t/tt, f*tt
	else:
		t = f/g
		tt = sqrt(1 + t**2)
		return t/tt, 1/tt, g*tt
		
def isqr_forward(d, s):
	oldcs = 1
	cs = 1
	M = d.size
	for i in range(M - 1):
		cs, sn, r = rot(d[i], s[i])
		if i > 0:
			s[i-1] = oldsn*r
		oldcs, oldn, d[i] = rot(oldcs*r, d[i+1])
	h = sn*cs
	s[N - 2] = h*oldsn
	d[N - 1] = h*oldcs
	
def svd22(a, b, c, d):
	'''Closed Form SVD of A = [a b ; c d]'''
	#Phi
	SU = [a*a + b*b, a*c + b*d, c*c + d*d]
	phi = .5 * atan2(2*SU[1], SU[0] - SU[2])
	Cphi = cos(phi)
	Sphi = sin(phi)
	U = [Cphi, -Sphi, Sphi, Cphi]
	#Theta
	SW = [a*a + c*c, a*b + c*d, b*b + d*d]
	theta = .5 * atan2(2*SW[1], SW[0] - SW[2])
	Ctheta = cos(theta)
	Stheta = sin(theta)
	W = [Ctheta, -Stheta, Stheta, Ctheta]
	#Singular values from U
	dsum = SU[0] + SU[2]
	diff = sqrt((SU[0] - SU[2])**2 + 4*SU[1]**2)
	sig = [sqrt((dsum + diff)/2), sqrt((dsum - diff)/2)]
	#Correction matrix
	AW = [ a*W[0] + b*W[2], a*W[1] + b*W[3], c*W[0] + d*W[2], c*W[1] + d*W[3]]
	C = [sign(U[0] * AW[0] + U[2] * AW[2]), sign(U[1]*AW[1] + U[3]*AW[3])]
	V = [W[0]*C[0], W[1]*C[1], W[2]*C[0], W[3]*C[1]]
	return U, sig, V
	

def bdsqr(s, e):
	N = d.size
	khi = N
	ii = 0
	maxitr = 1
	thres = 1e-4
	for i in range(maxitr):
		if khi<=1:
			break
		if ii > maxitr:
			break
		
def orgbr(vect, A, tau):
    M = A.shape[0]
    N = A.shape[1]
    if vect=='Q':
        Q = np.eye(M)
        for i in reversed(range(M)):
            x = np.dot(Q[i:,i:].T,A[i:,i])
            Q[i:,i:] -= np.outer(tau[i]*A[i:,i], x)
        return Q
    if vect=='P':
        PT = np.eye(N)
        for i in reversed(range(N-1)):
            x = np.dot(PT[i:,i+1:],A[i,i+1:])
            PT[i:,i+1:] -= np.outer(x,tau[i]*A[i,i+1:])
        return PT


np.random.seed(0)
np.set_printoptions(precision=2, suppress=True)
A = np.random.rand(4, 4).astype(np.float32)
mindim = min(A.shape)
T = np.copy(A)

A, d, s, tauq, taup = labrd(A, A.shape[0])
#A, d, s, tauq, taup = gebd2(A)

Q = orgbr('Q', A, tauq)
PT = orgbr('P', A, taup) 
B = np.zeros(A.shape)
B[:mindim, :mindim] = np.diag(d) + np.diag(s[1:], 1 if A.shape[0]>=A.shape[1] else -1)
print np.dot(Q, np.dot(B, PT)) - T
