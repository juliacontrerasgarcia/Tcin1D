#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from OCC import *

#--------------------------------#
# ELECTRON DENSITY               #
#--------------------------------#

def DENSITIES(WFC,dWFC,evals,k_vec,nkpts,NH,nXpts,Xpts,fermi,temp):

	NS0  = np.zeros((nXpts)).astype(np.double)
	NS   = np.zeros((nXpts)).astype(np.double)
	SC   = np.zeros((nXpts)).astype(np.double)
	
	dNS0 = np.zeros((nXpts)).astype(np.double)
	dNS  = np.zeros((nXpts)).astype(np.double)
	dSC  = np.zeros((nXpts)).astype(np.double)


	for ik, k in enumerate(k_vec[:,0]):
		for band in range (NH):

			nNS0     = OccNumNS0(evals[band,ik])
			nNS      = OccNumNS(evals[band,ik],fermi,temp)
			nSC      = OccNumSC(evals[band,ik],fermi,temp)

			NS0[:]  += nNS0 * np.real( WFC[ik,band,:] * np.conjugate(WFC[ik,band,:]) )
			NS[:]   += nNS  * np.real( WFC[ik,band,:] * np.conjugate(WFC[ik,band,:]) )
			SC[:]   += nSC  * np.real( WFC[ik,band,:] * np.conjugate(WFC[ik,band,:]) )

			dNS0[:] += nNS0 * ( np.real( np.conjugate(dWFC[ik,band,:]) * WFC[ik,band,:] ) + np.real( np.conjugate(WFC[ik,band,:]) * dWFC[ik,band,:] ) )
			dNS[:]  += nNS  * ( np.real( np.conjugate(dWFC[ik,band,:]) * WFC[ik,band,:] ) + np.real( np.conjugate(WFC[ik,band,:]) * dWFC[ik,band,:] ) )
			dSC[:]  += nSC  * ( np.real( np.conjugate(dWFC[ik,band,:]) * WFC[ik,band,:] ) + np.real( np.conjugate(WFC[ik,band,:]) * dWFC[ik,band,:] ) )

	#NS0[:]  = (1.0/nkpts) * NS0[:]
	#NS[:]   = (1.0/nkpts) * NS[:]
	#SC[:]   = (1.0/nkpts) * SC[:]

	#dNS0[:] = (1.0/nkpts) * dNS0[:]
	#dNS[:]  = (1.0/nkpts) * dNS[:]
	#dSC[:]  = (1.0/nkpts) * dSC[:]

			
	return NS0, NS, SC, dNS0, dNS, dSC

#--------------------------------#
# CHECK NORMALIZATION OF DENSITY #
#--------------------------------#

def DENSCHECK(rhoNS0,rhoNS,rhoSC,Xpts,nXpts,Ao,dX,NH,nuc):
	
	check  = True
	thrsh  = 1E-2
	
	Xo     = nuc + 1
	Xf     = Xo + int( (Ao/dX) + 1 )	

	intNS0 = np.trapz(rhoNS0[Xo:Xf], Xpts[Xo:Xf]) 
	intNS  = np.trapz(rhoNS[Xo:Xf] , Xpts[Xo:Xf])
	intSC  = np.trapz(rhoSC[Xo:Xf] , Xpts[Xo:Xf])

	print()
	print('- INTEGRAL OF NS(T=OK) DENSITY IN U.C. :', intNS0)
	print('- INTEGRAL OF NS(T>OK) DENSITY IN U.C. :', intNS )
	print('- INTEGRAL OF SC(T>OK) DENSITY IN U.C. :', intSC )
	print()

	if ( (abs(intNS0 - NH) > thrsh) or (abs(intNS - NH) > thrsh) or (abs(intSC - NH) > thrsh) ):
		check = False

	return check

#--------------------------------#
# KINETIC ENERGY DENSITIES       #
#--------------------------------#

def KINETIC(dWFC,evals,k_vec,nkpts,NH,nXpts,Xpts,fermi,temp):

	NS0  = np.zeros((nXpts)).astype(np.double)
	NS   = np.zeros((nXpts)).astype(np.double)
	SC   = np.zeros((nXpts)).astype(np.double)
	
	dNS0 = np.zeros((nXpts)).astype(np.double)
	dNS  = np.zeros((nXpts)).astype(np.double)
	dSC  = np.zeros((nXpts)).astype(np.double)


	for ik, k in enumerate(k_vec[:,0]):
		for band in range (NH):

			nNS0     = OccNumNS0(evals[band,ik])
			nNS      = OccNumNS(evals[band,ik],fermi,temp)
			nSC      = OccNumSC(evals[band,ik],fermi,temp)

			NS0[:]  += 0.5 * nNS0 * np.real( dWFC[ik,band,:] * np.conjugate(dWFC[ik,band,:]) )
			NS[:]   += 0.5 * nNS  * np.real( dWFC[ik,band,:] * np.conjugate(dWFC[ik,band,:]) )
			SC[:]   += 0.5 * nSC  * np.real( dWFC[ik,band,:] * np.conjugate(dWFC[ik,band,:]) )

	#NS0[:]  = (1.0/nkpts) * NS0[:]
	#NS[:]   = (1.0/nkpts) * NS[:]
	#SC[:]   = (1.0/nkpts) * SC[:]

	return NS0, NS, SC

#--------------------------------#
# KINETIC ENERGY DENSITIES VW|TF #
#--------------------------------#


def KINETICvw(DENS,dDENS,Xpts,nXpts):

	Tvw    = np.zeros((nXpts)).astype(np.double)

	Tvw[:] = (1.0)/(8.0) * (np.real( dDENS[:] * np.conjugate(dDENS[:]) ))/(DENS[:])

	return Tvw

def KINETICtf(DENS,Xpts,nXpts):

	Ttf    = np.zeros((nXpts)).astype(np.double)

	#Ttf[:] = (3.0)/(10.0) * np.power(3.0*np.pi**2, 2./3) * np.power(DENS[:], 5./3)
	Ttf[:] = (np.pi**2)/(24.0) * np.power(DENS[:], 3.0)
	
	return Ttf
