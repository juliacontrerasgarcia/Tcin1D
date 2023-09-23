#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from OCC import *

#--------------------------------#
# ELECTRON LOCALIZATION FUNCTION #
#--------------------------------#

def ELF(T, Tvw, Ttf, nXpts):

	ELF       = np.zeros((nXpts)).astype(np.double)
	KERNEL    = np.zeros((nXpts)).astype(np.double)

	KERNEL[:] = (T[:] - Tvw[:]) / (Ttf[:])

	ELF[:]    = 1.0 / (1.0 + KERNEL[:]**2 )

	return ELF 

#--------------------------------#
# NETWORKING VALUE : PHI         #
#--------------------------------#

def NWV(ELF,Ao,dX,nuc):
	
	Xo  = 2  * ( nuc + 1 )
	Xf  = 10 * ( Xo + 2 * int( (Ao/dX) + 1 ) )
	
	phi = min(ELF[Xo:Xf])

	return phi 
