#! /usr/bin/env python3

import numpy as np

#--------------------------------#
# ATOMIC ORBITALS : Gaussian     #
#--------------------------------#

def AOs(r,Ro,alpha):
	return np.exp(-alpha*(r-Ro)**2)


#--------------------------------#
# ATOMIC ORBITALS DERIVATIVE     #
#--------------------------------#

def dAOs(r,Ro,alpha):
        return -2.0 * alpha * (r - Ro) * AOs(r,Ro,alpha)
