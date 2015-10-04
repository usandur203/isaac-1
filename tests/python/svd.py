import numpy as np
from numpy import dot, outer, sqrt, cos, sin
from math import atan2

#Bidiagonalization
def sign(x):
	return 1 if x>=0 else -1
	
def larfg(alpha, x):
    xnorm = np.linalg.norm(x)
    beta = -np.sign(alpha)*np.sqrt(alpha**2 + xnorm**2)
    sigma = (beta - alpha)/beta
    x /= alpha - beta
    return x, sigma, beta

def labrd(A, tauq, taup, d, e, X, Y, NB):
    M, N = A.shape
    for i in range(NB):
        #Update A[i:, i]
        A[i:, i]   -= dot(A[i:, :i]     , Y[i, :i])
        A[i:, i]   -= dot(X[i:, :i]     , A[:i, i])
        #Householder A[i:,i]
        A[i+1:, i], tauq[i], d[i] = larfg(A[i,i], A[i+1:, i])
        A[i, i] = 1
        #print i, A
        if i < N - 1:
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
            A[i, i+2:], taup[i], e[i] = larfg(A[i,i+1], A[i,i+2:])
            A[i, i+1] = 1
            #Compute X[i+1:,i]
            X[i+1:,i]  = dot(A[i+1:,i+1:]   , A[i,i+1:])
            X[:i+1,i]  = dot(Y[i+1:,:i+1].T , A[i,i+1:])
            X[i+1:,i] -= dot(A[i+1:,:i+1]   , X[:i+1,i])
            X[:i,i]    = dot(A[:i, i+1:]    , A[i, i+1:])
            X[i+1:,i] -= dot(X[i+1:,:i]     , X[:i,i])
            X[i+1:,i] *= taup[i]
            
def gebd2(A, tauq, taup, d, e):
    M, N = A.shape
    for i in range(N):
        # Householder vector
        A[i+1:, i], tauq[i], d[i] = larfg(A[i,i], A[i+1:, i])
        A[i, i] = 1
        #  Apply H(i) to A(i:m,i+1:n) from the left
        x  = dot(A[i:,i+1:].T  , A[i:, i])
        A[i:,i+1:] -= tauq[i]*np.outer(A[i:,i], x)
        if i < N - 1:
            # Householder vector
            A[i,i+2:], taup[i], e[i] = larfg(A[i,i+1], A[i,i+2:])
            A[i,i+1] = 1
            # Apply G(i) to A(i+1:m,i+1:n) from the right 
            x = dot(A[i+1:,i+1:],A[i,i+1:])
            A[i+1:, i+1:] -= taup[i]*np.outer(x, A[i,i+1:])
        else:
            taup[i] = 0

def gebrd(A, tauq, taup, d, e, nb):
    M, N = A.shape
    X = np.zeros((M, nb))
    Y = np.zeros((N, nb))
    i = 0
    while N - i >= nb:
        labrd(A[i:,i:], tauq[i:], taup[i:], d[i:], e[i:], X[i:,:], Y[i:,:], nb)
        i += nb
        A[i:,i:] -= np.dot(A[i:,i-nb:i], Y[i:,:].T)
        A[i:,i:] -= np.dot(X[i:,:], A[i-nb:i,i:])
    gebd2(A[i:,i:], tauq[i:], taup[i:], d[i:], e[i:])
    
    
def orgbr(vect, A, K, tau):
    M = A.shape[0]
    N = A.shape[1]
    if vect=='Q':
        Q = np.eye(M)
        for i in reversed(range(K)):
            x = np.dot(Q[i:,i:].T,A[i:,i])
            Q[i:,i:] -= np.outer(tau[i]*A[i:,i], x)
        return Q
    if vect=='P':
        PT = np.eye(N)
        for i in reversed(range(K-1)):
            x = np.dot(PT[i:,i+1:],A[i,i+1:])
            PT[i:,i+1:] -= np.outer(x,tau[i]*A[i,i+1:])
        return PT    
        
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
		
def isqr(shift, direction, d, e):
    oldcs = 1
    cs = 1
    M = d.size
    if shift==0:
        if direction=='forward':
            for i in range(M - 1):
                cs, sn, r = rot(d[i]*cs, e[i])
                if i > 0:
                    e[i-1] = oldsn*r
                oldcs, oldsn, d[i] = rot(oldcs*r, d[i+1]*sn)
            h = d[M-1]*cs
            e[M - 2] = h*oldsn
            d[M - 1] = h*oldcs
        if direction=='backward':
            for i in range(M-1, 0, -1):
                cs, sn, r = rot(d[i]*cs, e[i-1])
                if i < M - 1:
                    e[i] = oldsn*r
                oldcs, oldsn, d[i] = rot(oldcs*r, d[i-1]*sn)
            h = d[0]*cs
            e[0] = h*oldsn
            d[0] = h*oldcs
    else:
        if direction=='forward':
            sign = 1 if d[0] > 0 else -1
            f = (abs(d[0]) - shift) * (sign + shift/d[0])
            g = e[0]
            for i in range(M - 1):
                cs, sn, r = rot(f, g)
                if i > 0:
                    e[i-1] = r
                f = cs * d[i] + sn * e[i]
                e[i] = cs*e[i] - sn*d[i]
                g = sn * d[i+1]
                d[i+1] = cs*d[i+1]
                
                cs, sn, r = rot(f, g)
                d[i] = r
                f = cs*e[i] + sn*d[i+1]
                d[i+1] = cs*d[i+1] - sn*e[i]
                if i < M - 2:
                    g = sn * e[i+1]
                    e[i+1] = cs*e[i+1]
            e[M-2] = f
            
        if direction=='backward':
            sign = 1 if d[M-1] > 0 else -1
            f = (abs(d[M-1]) - shift)*(sign + shift/d[M-1])
            g = e[M-2]
            for i in range(M-1, 0, -1):
                cs, sn, r = rot(f, g)
                if i < M-1:
                    e[i] = r
                f = cs*d[i] + sn*e[i-1]
                e[i-1] = cs*e[i-1] - sn*d[i]
                g = sn*d[i-1]
                d[i-1] = cs*d[i-1]
                
                cs, sn, r = rot(f, g)
                d[i] = r
                f = cs*e[i-1] + sn*d[i-1]
                d[i-1] = cs*d[i-1] - sn*e[i-1]
                if i > 1:
                    g = sn*e[i-2]
                    e[i-2] = cs*e[i-2]
            e[0] = f
    
    
    
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
	

def bdsqr(d, em, tol = 1e-4, maxit = 6):
    N = d.size
    maxit = maxit*N**2
    xmin = np.finfo(d.dtype).xmin
    eps = np.finfo(d.dtype).epps
    #Set threshold
    if tol >= 0:
        sminoa = abs(d[0])
        mu = sminoa
        for i in range(1, N):
            if sminoa==0:
                break
            mu = abs(d[i])*mu/(mu + abs(e[i-1]))
            sminoa = min(sminoa, mu)
        sminoa /= np.sqrt(N)
        thresh = max(tol*sminoa, maxit * xmin)
    else:
        smax = max(max(abs(d)),max(abs(e)))
        thresh = max(abs(tol)*smax, maxit*xmin)
    #Main iteration
    M = N
    for i in range(maxit):
        if M <= 1:
            break
        #Find diagonal block to work on
        if tol < 0 and abs(d[M-1]) <= thresh:
            d[M-1] = 0
        smax = abs(D[M-1])
        smin = smax
        for ll in reversed(range(M-1)):
            abss = abs(d[ll])
            abse = abs(e[ll])
            if tol < 0 and abss <= thresh:
                d[ll] = 0
            if abse <= thresh:
                e[ll] = 0
                break
            smin = min(smin, abss)
            smax = max(max(smax,abss),abse)
        #Convergence of bottom value
        if e[M-2]==0:
            M -= 1
            continue
        #Handles 2 x 2 block
        ll = ll + 1
        if ll==M-2:
            d[ll], d[ll+1] = svd22(d[ll], e[ll], 0, d[ll+1])
            e[ll] = 0
            M -= 2
            continue
        isqr(0,'forward',d, e)
		

    
np.random.seed(0)
np.set_printoptions(precision=2, suppress=True)
A = np.random.rand(6,6).astype(np.float32)
mindim = min(A.shape)
T = np.copy(A)

tauq = np.zeros(mindim)
taup = np.zeros(mindim)
e = np.zeros(mindim-1)
d = np.zeros(mindim)

gebrd(A, tauq, taup, d, e, 4)

Q = orgbr('Q', A, mindim, tauq)
PT = orgbr('P', A, mindim, taup) 
B = np.zeros(A.shape)
B[:mindim, :mindim] = np.diag(d) + np.diag(e, 1 if A.shape[0]>=A.shape[1] else -1)
print np.dot(Q, np.dot(B, PT)) - T

print bdsqr(d, e)

print np.linalg.svd(T)[1]
