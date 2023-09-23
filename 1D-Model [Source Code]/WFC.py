#! /usr/bin/env python3

import numpy as np

#--------------------------------------------#
# NORMALIZATION OF BLOCH STATES COEFFICIENTS #
#--------------------------------------------#

def normCksCHECK(C,nkpts,NH):
	check = True
	thrsh = 1E-5
	for ik in range(nkpts):
		for band in range(NH):
			N = 0.0
			for site in range(NH):
				N += C[ik][band,site] * np.conjugate(C[ik][band,site])
			if (np.abs(N-1.0) >= thrsh):
				check = False
	return check

#--------------------------------------------#
# BLOCH STATES : WAVE FUNCTIONS              #
#--------------------------------------------#

def BLOCHWFC(WFC,nkpts,NH,nXpts,k_vec,Basis,C):
	for ik, k in enumerate(k_vec[:,0]):
		for band in range(NH):
			for site in range(NH):
				WFC[ik,band,:] += C[ik][band,site] * Basis[ik,site,:]
#####################################################################################
				#print('Coeff. [ik=',ik,'n=',band,'j=',site,'] =',C[ik][band,site])
####################################################################################
	return WFC
	
#--------------------------------------------#
# NORMALIZATION                              #
#--------------------------------------------#

def normWFC(WFC,dWFC,k_vec,NH,Ao,dX,Xpts,nXpts):
	Xo = int( Ao/dX )
	Xf = nXpts - int( Ao/dX )
	for ik, k in enumerate(k_vec[:,0]):
		for band in range(NH):
			N = np.trapz((WFC[ik,band,Xo:Xf]*np.conjugate(WFC[ik,band,Xo:Xf])),Xpts[Xo:Xf])
			#N = np.trapz((WFC[ik,band,:]*np.conjugate(WFC[ik,band,:])),Xpts[:])
			#print('NUM. NORM. CONSTANT [ik=',ik,'n=',band,'] | INT  =',N)
			WFC[ik,band,:]  = (1/np.sqrt(N)) *  WFC[ik,band,:]
			dWFC[ik,band,:] = (1/np.sqrt(N)) * dWFC[ik,band,:]
	return WFC, dWFC

def normWFCCHECK(WFC,k_vec,NH,Ao,dX,Xpts,nXpts):
	check = True
	thrsh = 1E-5
	Xo = int( Ao/dX )
	Xf = nXpts - int( Ao/dX )
	for ik, k in enumerate(k_vec[:,0]):
		for band in range(NH):
			N = np.trapz((WFC[ik,band,Xo:Xf]*np.conjugate(WFC[ik,band,Xo:Xf])),Xpts[Xo:Xf])
			if (np.abs(N-1.0) >= thrsh):
				check = False
	return check

