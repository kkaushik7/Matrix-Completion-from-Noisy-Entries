# This code is based on the MATLAB implementation available at:https://github.com/airoldilab/SBA/tree/master/OptSpace_matlab

import numpy as np
import sys
from scipy.sparse.linalg import svds
import math
from numpy.linalg import norm
from numpy.matlib import repmat
from scipy.sparse import csr_matrix

# Usage: python optSpace.py [input_file_name.txt] [rank] [max_iter]
# Example usage: python optSpace.py inputMat.txt 1 100

def readMatrix(fileName):
	'''
	Load the Matrix from a text file
	'''
	A = np.loadtxt(fileName,delimiter=',')
	return A

def replaceElements(M,num):
	'''
	Convert all non-zero elements in a matrix to 'num'
	'''
	ind = M
	ind[ind != 0] = num
	return ind

def trim(M,E):
	'''
	Perform Matrix Trim as described by Keshavan et. al. (2009)
	'''
	(m,n) = M.shape
	M_Et = M
	# Trim the Columns first
	colSums = np.sum(E,axis=0)
	colMean = np.mean(colSums)
	for col in range(n):
		if sum(E[:,col] >= 2*colMean):
			M_Et = replaceElements(M_Et,0)

	# Trim the Rows next
	rowSums = np.sum(E,axis=1)
	rowMean = np.mean(rowSums)
	for row in range(m):
		if sum(E[row,: ] >= 2*rowMean):
			M_Et = replaceElements(M_Et,0)
	return M_Et


def G(X,m0,r):
	'''
	A helper function
	'''
	z = np.transpose(np.sum(np.power(X,2),axis=1))/(2*m0*r)
	y = np.power(math.e,(z-1)**2) - 1 
	y[ (z < 1).nonzero() ] = 0 
	out = np.sum(y)
	return out
	

def F_t(X,Y,S,M_E,E,m0,rho):
	'''
	Objective Function F(t)
	'''
	(n,r) = X.shape 
	out1 = np.sum(np.power((np.dot(np.dot(X,S),np.transpose(Y)) - M_E)*E,2))/2  
	#out2 =  rho*G(Y,m0,r) ;
	#out3 =  rho*G(X,m0,r) ;
	out = out1#+out2+out3 ;
	return out


def Gp(X,m0,r):
	'''
	A helper function
	'''
	z = np.transpose(np.sum(np.power(X,2),axis=1))/(2*m0*r)
	z = 2*(np.power(math.e,(z-1)**2)) *(z-1) 
	z[ (z < 0).nonzero() ] = 0
	z = np.array([z])
	z = z.T
	out = X*repmat(z,1,r) / (m0*r) 
	return out

def gradF_t(X,Y,S,M_E,E,m0,rho):
	'''
	Compute F'(t)
	'''
	(n, r) = X.shape
	(m, r) = Y.shape
	
	XS = np.dot(X,S)
	YS = np.dot(Y,np.transpose(S) )
	XSY = np.dot(XS,np.transpose(Y))
	
	Qx = np.dot(np.dot(np.transpose(X),( (M_E - XSY)*E )),YS) /n
	Qy = np.dot(np.transpose(Y), np.dot(np.transpose( (M_E - XSY)*E ),XS)) /m
	
	W = np.dot(( (XSY - M_E)*E ),YS) + np.dot(X,Qx) #+ rho * Gp(X,m0,r)
	Z = np.dot(np.transpose( (XSY - M_E)*E ),XS) + np.dot(Y,Qy) #+ rho * Gp(Y,m0,r)
	return W,Z


def getoptS(X,Y,M_E,E):
	'''
	Function to get Optimal S value given X and Y
	'''
	(n, r) = X.shape
	C = np.dot(np.dot(np.transpose(X),M_E), Y)  
	(t1,t2) = C.shape
	A = np.empty([t1*t2, t1*t2])
	C = C.flatten(1)

	for i in range (r):
		for j in range(r):
			ind = (j)*r+i 
			temp = np.dot(np.dot(np.transpose(X),(  (X[:,i] * np.transpose(Y[:,j]))*E )), Y)
			A[:,ind] = temp.flatten(1) 
	S = np.linalg.solve(A,C)
	out = np.reshape(S,(r,r)).transpose() 
	return out


def getoptT(X,W,Y,Z,S,M_E,E,m0,rho):
	'''
	Function to perform line search
	'''
	norm2WZ = np.power(norm(W,'fro'),2) + np.power(norm(Z,'fro'),2)
	f = []
	f.append( F_t(X, Y,S,M_E,E,m0,rho) )
	
	t = -1e-1 
	for i in range(20):
			f.append( F_t(X+t*W,Y+t*Z,S,M_E,E,m0,rho) )
			if( f[i+1] - f[1] <= 0.5 * t * norm2WZ ):
				out = t 
				break
			t = t/2 
	out = t 
	return out


'''
Main Function
'''
if __name__ == "__main__":
	'''
	Set all the Input Parameters
	'''	
	M_E = readMatrix(sys.argv[1])
	(n,m) = M_E.shape
	E = replaceElements(M_E,1)
	r = int(sys.argv[2])
	m0,rho = 10000,0
	niter  = int(sys.argv[3])
	eps = np.count_nonzero(M_E)/np.sqrt(m*n)
	tol = 1e-3
	'''
	Step 1: rank-r projection
	'''
	frobenius_norm = norm(M_E,'fro')
	rescale_param = np.sqrt((r*np.count_nonzero(E))/(np.power(frobenius_norm,2)))
	M_E = M_E * rescale_param

	'''
	Step 2: Perform Trimming
	'''
	M_Et = trim(M_E,E)
	
	'''
	Step 3: Gradient Descent	
	Singular Value Decomposition
	'''
	(X0,S0,Y0) = svds(M_Et,k = r)
	Y0 = np.transpose(Y0)
	
	'''
	Initial Guess
	'''
	X0 = X0 * np.sqrt(n) 
	Y0 = Y0 * np.sqrt(m) 
	S0 = S0 / eps 
	print('Starting Gradient Descent')
	print('-----------------------------------------------')
	print('Iteration | \t Fit Error \n')
	print('-----------------------------------------------')
	X = X0
	Y=Y0
	S = getoptS(X,Y,M_E,E)
	dist = []
	XSYprime = np.dot(np.dot(X,S),np.transpose(Y))
	dist.append(norm( (M_E - XSYprime)*E ,'fro')/np.sqrt(np.count_nonzero(E)))  
	print('0 |\t',dist[0])

	for i in range(1,niter):
		# Compute the Gradient 
		(W, Z) = gradF_t(X,Y,S,M_E,E,m0,rho)
		
		# Line search for the optimum jump length	
		t = getoptT(X,W,Y,Z,S,M_E,E,m0,rho) 
		X = X + t*W
		Y = Y + t*Z
		S = getoptS(X,Y,M_E,E) 
			
		# Compute the distortion	
		XSYprime = np.dot(np.dot(X,S),np.transpose(Y))
		dist.append(norm( (M_E - XSYprime)*E ,'fro')/np.sqrt(np.count_nonzero(E)) )
		print(i,'|\t',dist[i])
		if( dist[i] < tol ):
			break 
	S = S /rescale_param 
	'''
	Publish results
	'''
	reconstructed_matrix = np.dot(np.dot(X,S),np.transpose(Y))
	print('-----------------------------------------------\n')
	print('Constructing the original matrix ...')
	print('Optimal Objective Function Value F(X,Y,S) = ',dist[-1])
	np.savetxt('reconstructed_matrix.txt', reconstructed_matrix)
	print('Saving the output.....')
	print('The reconstructed matrix is stored in reconstructed_matrix.txt')