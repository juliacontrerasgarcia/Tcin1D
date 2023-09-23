#! /usr/bin/env python3

from AOs   import *
from GRIDS import *
import numpy as np

#-----------------------#
# BASIS SET IN THE U.C. #
#-----------------------#

def BASIS(Ao,xpts,atoms,NH,nxpts,alpha):
	basis = np.zeros((NH,nxpts))
	for site in range(NH):
		basis[site,:] = AOs(xpts[:],Ao*atoms[site][0],alpha)
	return basis


#-----------------------#
# BASIS SET DERIVATIVE  #
#-----------------------#

def dBASIS(Ao,xpts,atoms,NH,nxpts,alpha):
	dbasis = np.zeros((NH,nxpts))
	for site in range(NH):
		dbasis[site,:] = dAOs(xpts[:],Ao*atoms[site][0],alpha)
	return dbasis

#-----------------------#
# BASIS SET NORMALIZAT. #
#-----------------------#

def BASIS_NORM(xpts,nxpts,alpha):
	basis    = np.zeros((nxpts))
	basis[:] = AOs(xpts[:],xpts[int(nxpts/2)],alpha)
	N        = np.trapz((basis[:]*np.conjugate(basis[:])),xpts[:])
	return 1/(np.sqrt(N))

#-----------------------#
# BASIS SET OVERLAP Sv  #
#-----------------------#

def BASIS_OVERLAPv(Ao,atoms,alpha,NormCnst):

	xo         = -5.0 * Ao
	xf         =  5.0 * Ao
	dx         =  0.001

	xpts       = GRID(xo,xf,dx)
	x_lim      = [np.amin(xpts),np.amax(xpts)]
	nxpts      = len(xpts)
	
	basis_A    = np.zeros((nxpts))
	basis_B    = np.zeros((nxpts))

	basis_A[:] = NormCnst * AOs(xpts[:],Ao*atoms[0][0],alpha)
	basis_B[:] = NormCnst * AOs(xpts[:],Ao*atoms[1][0],alpha)

	S_AB       = np.trapz((basis_A[:]*np.conjugate(basis_B[:])),xpts[:])

	return S_AB	

#-----------------------#
# BASIS SET OVERLAP Sv  #
#-----------------------#

def BASIS_OVERLAPw(Ao,atoms,alpha,NormCnst):

	xo         = -5.0 * Ao
	xf         =  5.0 * Ao
	dx         =  0.001

	xpts       = GRID(xo,xf,dx)
	x_lim      = [np.amin(xpts),np.amax(xpts)]
	nxpts      = len(xpts)
	
	basis_A    = np.zeros((nxpts))
	basis_B    = np.zeros((nxpts))

	basis_A[:] = NormCnst * AOs(xpts[:],Ao*atoms[1][0],alpha)
	basis_B[:] = NormCnst * AOs(xpts[:],Ao*(1.0 + atoms[0][0]),alpha)

	S_AB       = np.trapz((basis_A[:]*np.conjugate(basis_B[:])),xpts[:])

	return S_AB	

