# This code is based on the MATLAB implementation available at:https://github.com/airoldilab/SBA/tree/master/OptSpace_matlab

import numpy as np
import sys
from scipy.sparse.linalg import svds
import math
from numpy.linalg import norm
from numpy.matlib import repmat
from scipy.sparse import csr_matrix

import scipy.sparse as sparse

import time
import os

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
	d = E.sum(axis=0)
	d_ = np.mean(d)
	for col in xrange(n):
		if( d[col]>2*d_ ):
			nonzero_indices = np.nonzero(E[:,col])[0]
			p = np.random.permutation(nonzero_indices)
			M_Et[p[math.ceil(2*d_):],col] = 0.0

	d = E.sum(axis=1)
	d_ = np.mean(d)
	for row in xrange(m):
		if( d[row]>2*d_ ):
			nonzero_indices = np.nonzero(E[row,:])[0]
			p = np.random.permutation(nonzero_indices)
			M_Et[row,p[math.ceil(2*d_):]] = 0.0

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
	
	W = np.dot(( (XSY - M_E)*E ),YS) + np.dot(X,Qx)# + rho * Gp(X,m0,r)
	Z = np.dot(np.transpose( (XSY - M_E)*E ),XS) + np.dot(Y,Qy)# + rho * Gp(Y,m0,r)
	return W,Z


def getoptS(X,Y,M_E,E):
	'''
	Function to get Optimal S value given X and Y
	'''
	(n, r) = X.shape
	C = np.dot(np.dot(np.transpose(X),M_E), Y)  
	(t1,t2) = C.shape
	A = np.zeros([t1*t2, t1*t2])
	C = C.flatten(1)

	for i in range (r):
		for j in range(r):
			ind = (j)*r+i 
			X_temp = X[:,i]
			X_temp = X_temp[:,np.newaxis]
			Y_temp = Y[:,j]
			Y_temp = Y_temp[:,np.newaxis]
			temp = np.dot(np.dot(np.transpose(X),(  (X_temp * np.transpose(Y_temp))*E )), Y)
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
			if( f[i+1] - f[0] <= 0.5 * t * norm2WZ ):
				out = t 
				break
			t = t/2 
	out = t 
	return out

def guessRank(M_E):
	(m,n) = M_E.shape
	epsilon = np.count_nonzero(M_E)/np.sqrt(m*n)
	X0,S0,Y0 = svds(M_E,100)
	if(S0[-1]>S0[0]):
		S0 = S0[::-1]

	S1 = S0[0:99] - S0[1:100]

	S1_ = S1/np.mean(S1[89:])

	r1 = 0;
	lam = 0.05;
	cost = np.zeros((S1_.size))
	while(r1<=0):
		for idx in xrange(S1_.size):
			cost[idx] = lam*np.max(S1_[idx:]) + idx

		i1 = np.argmin(cost)
		v1 = np.min(cost)
		r1 = i1-1
		lam = lam + 0.05

'''
Main Function
'''
if __name__ == "__main__":
	'''
	Set all the Input Parameters
	'''	
	# load netflix data
	M0 = np.load('data_matrices/M0.npy')
	M_E = np.load('data_matrices/M_E.npy')

	movies = 2000 #2000
	M0 = M0[0:movies,:]
	M_E = M_E[0:movies,:]

	E0 = 1*(M0!=0)
	user_ratings_count = np.sum(E0,axis=0)
	sorted_users = np.argsort(-user_ratings_count)

	users = 5000 #27062
	M0 = M0[:,sorted_users[0:users]]
	M_E = M_E[:,sorted_users[0:users]]

	tol = 1e-3

	E0 = 1*(M0!=0)
	E = 1*(M_E!=0)
	(m,n) = M_E.shape

	print(np.sum(E0))
	print(np.sum(E))

	average_rating = np.mean(M_E[np.nonzero(M_E)])
	print('Test AVERAGE RATING: ',norm( (M0 - average_rating*(E0-E))*(E0-E) ,'fro')/np.sqrt(np.count_nonzero(E0-E)))

	r = int(sys.argv[1])
	niter = int(sys.argv[2])
	m0,rho = 10000,0

	eps = np.count_nonzero(M_E)/np.sqrt(m*n)

	print("Data Loaded...")
	'''
	Step 1: rank-r projection
	'''
	frobenius_norm = norm(M_E,'fro')
	rescale_param = np.sqrt((r*np.count_nonzero(E))/(np.power(frobenius_norm,2)))
	M_E = M_E * rescale_param

	print("Finished Step 1...")

	# '''
	# Step 2: Perform Trimming
	# '''
	M_Et = trim(np.copy(M_E),np.copy(E))
	print(str(np.sum(M_Et!=0)-np.sum(E)) + " trimmed values")
	#M_Et = np.copy(M_E)

	print("Finished Step 2...")
	
	'''
	Step 3: Gradient Descent	
	Singular Value Decomposition
	'''
	(X0,S0,Y0) = svds(sparse.csr_matrix(M_Et),k = r)
	Y0 = np.transpose(Y0)
	X0 = np.fliplr(X0)
	Y0 = np.fliplr(Y0)

	print("Finished Step 3...")
	t = time.time()

	# (X0,S0,Y0) = np.linalg.svd(M_E)
	# Y0 = np.transpose(Y0)
	# X0 = X0[:,0:r]
	# S0 = np.diag(S0[0:r])
	# Y0 = Y0[0:r,:].T
	
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
	dist_test = []
	XSYprime = np.dot(np.dot(X,S/rescale_param),np.transpose(Y))
	dist.append(norm( (M_E/rescale_param - XSYprime)*E ,'fro')/np.sqrt(np.count_nonzero(E)))  
	dist_test.append(norm( (M0 - np.dot(np.dot(X,S/rescale_param),np.transpose(Y)))*(E0-E) ,'fro')/np.sqrt(np.count_nonzero(E0-E)))
	print('Test: ',dist_test[0])
	print('0 |\t',dist[0])

	for i in range(1,niter):
		time_start = time.time()
		# Compute the Gradient 
		(W, Z) = gradF_t(X,Y,S,M_E,E,m0,rho)
		
		# Line search for the optimum jump length	
		t = getoptT(X,W,Y,Z,S,M_E,E,m0,rho) 
		X = X + t*W
		Y = Y + t*Z
		XSYprime = np.dot(np.dot(X,S),np.transpose(Y))
		#print('After t: ', norm( (M_E - XSYprime)*E ,'fro')/np.sqrt(np.count_nonzero(E)) )
		S = getoptS(X,Y,M_E,E) 
			
		# Compute the distortion	
		XSYprime = np.dot(np.dot(X,S/rescale_param),np.transpose(Y))
		dist.append(norm( (M_E/rescale_param - XSYprime)*E ,'fro')/np.sqrt(np.count_nonzero(E)) )
		dist_test.append(norm( (M0 - np.dot(np.dot(X,S/rescale_param),np.transpose(Y)))*(E0-E) ,'fro')/np.sqrt(np.count_nonzero(E0-E)))
		elapsed_time = time.time() - time_start
		print(i,'|\t',dist[i],dist_test[i],elapsed_time, t)
		print("==========")
		if( dist[i] < tol ):
			break 
	S = S /rescale_param


	dist = np.array(dist)
	dist_test = np.array(dist_test)

	dist = np.stack((dist,dist_test))

	#np.save('save_data/dist_' + str(r) + '.npy',dist)


	# XSYprime = np.dot(np.dot(X,S),np.transpose(Y))
	# print(norm( (M0 - XSYprime),'fro')/np.sqrt(np.count_nonzero(E))) 
	# '''
	# Publish results
	# '''
	# reconstructed_matrix = np.dot(np.dot(X,S),np.transpose(Y))
	# print('-----------------------------------------------\n')
	# print('Constructing the original matrix ...')
	# print('Optimal Objective Function Value F(X,Y,S) = ',dist[-1])
	# np.savetxt('reconstructed_matrix.txt', reconstructed_matrix)
	# print('Saving the output.....')
	# print('The reconstructed matrix is stored in reconstructed_matrix.txt')