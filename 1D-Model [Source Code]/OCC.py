#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

#--------------------------------#
# NS OCCUPATION NUMBER (T = 0K)  #
#--------------------------------#

def XI(energy,fermi):
	return energy - fermi


#---------------------------------------------------------#
# NS OCCUPATION NUMBER (T = 0K): FERMI-DIRAC HEAVISIDE    #
#---------------------------------------------------------#

def OccNumNS0(energy):
	if (energy < 0.0):
		return  2
	elif (energy > 0.0):
		return  0
	else:
		return 1

#---------------------------------------------------------#
# NS OCCUPATION NUMBER (T > 0K): FERMI-DIRAC DISTRIBUTION #
#---------------------------------------------------------#

def OccNumNS(energy,fermi,temp):
	kB   = 8.617333262e-5;                      #[eV/K]
	beta = 1/(kB * temp)                        #[1/eV]
	ekn  = XI(energy,fermi)
	return 2/(1+ np.exp(beta*ekn))

#---------------------------------------------------------#
# GAP FUNCTION: LORENTZIAN OF WIDTH w AROUND FERMI LEVEL  #
#---------------------------------------------------------#

def GAP(energy,fermi):
        w    = 0.2                                    #[eV]
        ekn  = XI(energy,fermi)
        return ( (w)/(2*np.pi) ) / ((ekn)**2+(w/2.0)**2)

#---------------------------------------------------------#
# CHI ENERGY FUNCTION                                     #
#---------------------------------------------------------#

def CHI(energy,fermi,temp):
	kB   = 8.617333262e-5;                      #[eV/K]
	beta = 1/(kB * temp)                        #[1/eV]
	ekn  = XI(energy,fermi)
	Dkn  = GAP(energy,fermi)
	Ekn  = np.sqrt(ekn**2 + Dkn**2)
	return ( Dkn/(2.0 * Ekn) ) * np.tanh((beta)/(2.0) * Ekn)
	
#---------------------------------------------------------#
# SC OCCUPATION NUMBER: MODIFIED FERMI-DIRAC DISTRIB.     #
#---------------------------------------------------------#

def OccNumSC(energy,fermi,temp):
	kB   = 8.617333262e-5;                      #[eV/K]
	beta = 1/(kB * temp)                        #[1/eV]
	ekn  = XI(energy,fermi)
	Dkn  = GAP(energy,fermi)
	Ekn  = np.sqrt(ekn**2 + Dkn**2)
	return 1.0 - (ekn/Ekn) * np.tanh((beta)/(2.0) * Ekn)
