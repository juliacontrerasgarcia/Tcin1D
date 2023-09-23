#! /usr/bin/env python3

from AOs import *
import numpy as np

#------------------------#
# BLOCH SUMS IN N U.C.   #
#------------------------#

def BLOCH(nkpts,NH,Ao,nXpts,Xpts,k_vec,nuc,atoms,alpha,NormCnst):
	basis = np.zeros((nkpts,NH,nXpts)).astype(np.cdouble)
	for ik, k in enumerate(k_vec[:,0]):
		basis[ik,0,:] += np.exp(1j*(2.0*np.pi/Ao)*k*(1.0 + atoms[0][0])*Ao) * AOs(Xpts[:],(nuc + 1 + atoms[0][0])*Ao,alpha)
		basis[ik,0,:] += np.exp(1j*(2.0*np.pi/Ao)*k*(nuc + atoms[0][0])*Ao) * AOs(Xpts[:],(0.0     + atoms[0][0])*Ao,alpha)
		basis[ik,1,:] += np.exp(1j*(2.0*np.pi/Ao)*k*(1.0 + atoms[1][0])*Ao) * AOs(Xpts[:],(nuc + 1 + atoms[1][0])*Ao,alpha)
		basis[ik,1,:] += np.exp(1j*(2.0*np.pi/Ao)*k*(nuc + atoms[1][0])*Ao) * AOs(Xpts[:],(0.0     + atoms[1][0])*Ao,alpha)
		for site in range(NH):
			for m in range(1,nuc+1):
				basis[ik,site,:] += np.exp(1j*(2.0*np.pi/Ao)*k*(m + atoms[site][0])*Ao) * AOs(Xpts[:],(m + atoms[site][0])*Ao,alpha)
########################################################################
				#print('m+j =', m + atoms[site][0])
########################################################################
	basis = basis * NormCnst * (1/(np.sqrt(nkpts)))
	return basis


#------------------------#
# BLOCH SUMS DERIVATIVES #
#------------------------#

def dBLOCH(nkpts,NH,Ao,nXpts,Xpts,k_vec,nuc,atoms,alpha,NormCnst):
	dbasis = np.zeros((nkpts,NH,nXpts)).astype(np.cdouble)
	for ik, k in enumerate(k_vec[:,0]):
		dbasis[ik,0,:] += np.exp(1j*(2.0*np.pi/Ao)*k*(1.0 + atoms[0][0])*Ao) * dAOs(Xpts[:],(nuc + 1 + atoms[0][0])*Ao,alpha)
		dbasis[ik,0,:] += np.exp(1j*(2.0*np.pi/Ao)*k*(nuc + atoms[0][0])*Ao) * dAOs(Xpts[:],(0.0     + atoms[0][0])*Ao,alpha)
		dbasis[ik,1,:] += np.exp(1j*(2.0*np.pi/Ao)*k*(1.0 + atoms[1][0])*Ao) * dAOs(Xpts[:],(nuc + 1 + atoms[1][0])*Ao,alpha)
		dbasis[ik,1,:] += np.exp(1j*(2.0*np.pi/Ao)*k*(nuc + atoms[1][0])*Ao) * dAOs(Xpts[:],(0.0     + atoms[1][0])*Ao,alpha)
		for site in range(NH):
			for m in range(1,nuc+1):
                                dbasis[ik,site,:] += np.exp(1j*(2.0*np.pi/Ao)*k*(m + atoms[site][0])*Ao) * dAOs(Xpts[:],(m + atoms[site][0])*Ao,alpha)
	dbasis = dbasis * NormCnst * (1/(np.sqrt(nkpts)))
	return dbasis

#-------------------------#
# NORMALIZATION           #
#-------------------------#

def normBLOCH(Basis,dBasis,k_vec,NH,Ao,dX,Xpts,nXpts):
	Xo = int( Ao/dX )
	Xf = nXpts - int( Ao/dX )
	for ik, k in enumerate(k_vec[:,0]):
		for site in range(NH):
			N = np.trapz((Basis[ik,site,Xo:Xf]*np.conjugate(Basis[ik,site,Xo:Xf])),Xpts[Xo:Xf])
			#N = np.trapz((Basis[ik,site,:]*np.conjugate(Basis[ik,site,:])),Xpts[:])
################################################################################
			#print('BLOCH SUM [ik=',ik,'j=',site,'] NORM. FACTOR INT=',N)
################################################################################
			Basis[ik,site,:]  = (1/np.sqrt(N)) *  Basis[ik,site,:]
			dBasis[ik,site,:] = (1/np.sqrt(N)) * dBasis[ik,site,:]
	return Basis, dBasis

def normCHECK(Basis,k_vec,NH,Ao,dX,Xpts,nXpts):
	check = True
	thrsh = 1E-5
	Xo = int( Ao/dX )
	Xf = nXpts - int( Ao/dX )
	for ik, k in enumerate(k_vec[:,0]):
		for site in range(NH):
			N = np.trapz((Basis[ik,site,Xo:Xf]*np.conjugate(Basis[ik,site,Xo:Xf])),Xpts[Xo:Xf])
			if (np.abs(N-1.0) >= thrsh):
				check = False
	return check

