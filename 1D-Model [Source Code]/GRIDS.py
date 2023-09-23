#! /usr/bin/env python3

import numpy as np

#--------------------------------#
# GRIDS                          #
#--------------------------------#

def GRID(init,final,step):
	return np.arange(init,final,step)

#--------------------------------#
# KPOINTS MESH IN THE BZ         #
#--------------------------------#

def KPOINTS(k,Nk):
	k = np.zeros((Nk))
	s = int((Nk+1)/2)	
	for i in range(1,s):
		k[i-1]   = (i+s-1)/Nk
		k[i+s-1] = i/Nk
	k[s-1] = Nk/Nk
	return k
