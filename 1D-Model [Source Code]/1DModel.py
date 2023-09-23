#! /usr/bin/env python3

#-----------------------------------#
# SET UP THE ENVIRONMENT            #
#-----------------------------------#

from pythtb import *
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from PROCEDURES import *
from GRIDS      import *
from AOs        import *
from BASIS      import *
from BLOCH      import *
from WFC        import *
from ERR        import *
from OCC        import *
from PROPERTIES import *
from ELF        import *
from MPDs       import *

#-----------------------------------#
# PLOTTING DIRECTIVES               #
#-----------------------------------#

plotAOs    = True
plotBB     = True
plotWFC    = True
plotBND    = True
plotOCC    = True
plotDENS   = True
plotKIN    = True
plotELOCF  = True

#-----------------------------------#
# ERROR CALCULATION DIRECTIVES      #
#-----------------------------------#

errBB  = True
errWFC = True

#-----------------------------------#
# N. OF K POITNS | ALPHA | t | TEMP #
#-----------------------------------#

nkpts = int(sys.argv[1])
Ao    = float(sys.argv[2])
dHH   = float(sys.argv[3])
alpha = float(sys.argv[4])
#t     = float(sys.argv[4])
fermi = float(sys.argv[5])
temp  = int(sys.argv[6])
dX    = float(sys.argv[7])

#-----------------------------------#
# OUTPUT DIRECTORIES                #
#-----------------------------------#

fig_dir          = './Figures-kpt{}'.format(nkpts)
data_dir         = './Data-kpt{}'.format(nkpts)

fig_AOsBASIS     = '{}/AOsBASIS'.format(fig_dir)
fig_BlochBASIS   = '{}/BlochBASIS'.format(fig_dir)
fig_dBlochBASIS  = '{}/dBlochBASIS'.format(fig_dir)
fig_BWFC         = '{}/BlochWFC'.format(fig_dir)
fig_dBWFC        = '{}/dBlochWFC'.format(fig_dir)
fig_BND          = '{}/BANDS'.format(fig_dir)
fig_OCC          = '{}/OCC'.format(fig_dir)
fig_DENSITY      = '{}/DENS'.format(fig_dir)
fig_KINETIC      = '{}/KIN'.format(fig_dir)
fig_ELF          = '{}/ELF'.format(fig_dir)

os.system('mkdir {}'.format(fig_dir))
os.system('mkdir {}'.format(data_dir))

os.system('mkdir {}'.format(fig_AOsBASIS))
os.system('mkdir {}'.format(fig_BlochBASIS)) 
os.system('mkdir {}'.format(fig_dBlochBASIS))
os.system('mkdir {}'.format(fig_BWFC))
os.system('mkdir {}'.format(fig_dBWFC))
os.system('mkdir {}'.format(fig_BND))
os.system('mkdir {}'.format(fig_OCC))
os.system('mkdir {}'.format(fig_DENSITY))
os.system('mkdir {}'.format(fig_KINETIC))
os.system('mkdir {}'.format(fig_ELF))

#-----------------------------------#
# PYTHTB 1D MODEL                   #
#-----------------------------------#

dim_r   = 1
dim_k   = 1
lattice = [[Ao]]
#Ao      = 2.0
NH      = 2
atoms   = [[0.0],[dHH]]

MODEL   = tb_model(dim_r,dim_k,lattice,atoms)

#-----------------------------------#
# THE WFC INSIDE THE U.C.: x-GRID   #
#-----------------------------------#

xo       = -2.0 * Ao
xf       =  2.0 * Ao
dx       =  0.001

#alpha    = 10

xpts     = GRID(xo,xf,dx)
x_lim    = [np.amin(xpts),np.amax(xpts)]
nxpts    = len(xpts)

AOBasis  =  BASIS(Ao,xpts[:],atoms[:],NH,nxpts,alpha)
dAOBasis = dBASIS(Ao,xpts[:],atoms[:],NH,nxpts,alpha)

NormCnst = BASIS_NORM(xpts[:],nxpts,alpha)

SBasisV   = BASIS_OVERLAPv(Ao,atoms[:],alpha,NormCnst)
SBasisW   = BASIS_OVERLAPw(Ao,atoms[:],alpha,NormCnst)

print()
print('- OVERLAP OF GAUSSIAN BASIS FUNCTIONS AOs Sv = <m,A|m,B>   =',SBasisV)
print('- OVERLAP OF GAUSSIAN BASIS FUNCTIONS AOs Sw = <m,B|m+1,A> =',SBasisW)
print()

if plotAOs:
	plotBASIS(AOBasis[:],dAOBasis[:],NH,xpts[:],x_lim,fig_AOsBASIS)


#-----------------------------------#
# HOPPING PARAMETERS                #
#-----------------------------------#

to = -1.5
#to = - 3.0
So = 0.28650479686019015

v  = to * (SBasisV/So)
w  = to * (SBasisW/So)

#-----------------------------------#
# PYTHTB 1D MODEL PRINTING          #
#-----------------------------------#

print()
print('CALCULATION FOR',nkpts,'NUMBER OF K-POINTS')
print()
print('Dimension of Real Space                ',dim_r)
print('Dimnesion of Reciprocal Space          ',dim_k)
print('Lattice Vector                         ',lattice)
print('Total number of Atoms in the Unit Cell:',NH)
print('Positions                             :',atoms[0],atoms[1])
print('Lattice Parameter Ao                  :',Ao)
print('Alpha Parameter of Gaussian AOs       :',alpha)
print('Intracell Hopping Parameter v [eV]    :',v)
print('Intercell Hopping Parameter w [eV]    :',w)
print('Fermi Energy                          :',fermi)
print('Temperature [K]                       :',temp)
print('Real Space Grid Density dX [a.u.]     :',dX)

#-----------------------------------#
# PYTHTB MODEL PARAMETERS           #
#-----------------------------------#

#v = t
#w = t 
e =  0.0

MODEL.set_onsite(e,0,mode='reset')
MODEL.set_onsite(e,1,mode='reset')
MODEL.set_hop(v,0,1,[0])
MODEL.set_hop(w,1,0,[1])

MODEL.display()

#-----------------------------------#
# MESH IN THE BRILLOUIN ZONE [BZ]   #
#-----------------------------------#

(k_vec,k_dist,k_node) = MODEL.k_path('full',nkpts,report=False)

###################################
#print(k_vec)
####################################

#-----------------------------------#
# SECULAR PROBLEM: COEFF. OF B.SUMS #
#-----------------------------------#

print('--------------------------------------------------------------')
print()
print('DIAGONALIZATION')
print()

Ckn          = wf_array(MODEL, [nkpts])

(evals,evec) = MODEL.solve_all(k_vec[:], eig_vectors=True)

for ik in range(nkpts):
	Ckn[ik] = evec[:,ik,:]

Ckn.impose_pbc(0,0)

check       = normCksCHECK(Ckn,nkpts,NH)
if check:
	print('- CHECK: NORMALIZATION OF BLOCH STATES COEFFICIENTS CORRECT')
	print()
else:
	print('- ERROR: BLOCH STATES COEFFICIENTS NOT NORMALIZED')

print('DIAGONALIZATION FINISHED')
print()
print('--------------------------------------------------------------')
print()

#-----------------------------------#
# BAND STRUCTURE                    #
#-----------------------------------#

if plotBND:
	plotBANDS(k_vec[:],evals[:,:],fig_BND)

#-----------------------------------#
# CHANGED MESH BRILLOUIN ZONE [BZ]  #
#-----------------------------------#

for i in range(1,nkpts+1):
        k_vec[i-1] = i/nkpts

#s = int((nkpts+1)/2)
#for i in range(1,s):
#	k_vec[i-1]   = (i+s-1)/nkpts
#	k_vec[i+s-1] = i/nkpts
#k_vec[s-1] = nkpts/nkpts

###################################
#print(k_vec)
####################################

#-----------------------------------#
# DEFINITION OF REAL SPACE X-GRID   #
#-----------------------------------#

nuc         = nkpts

Xo          = 0.0
Xf          = (nuc+2.0)*Ao
#dX          = 0.0001

Xpts        = GRID(Xo,Xf,dX)
X_lim       = [np.amin(Xpts),np.amax(Xpts)]
nXpts       = len(Xpts)

Xin         = int( Ao/dX )
Xfin        = nXpts - int( Ao/dX )

#-----------------------------------#
# BLOCH SUM BASIS ON N U.C.: X-GRID #
#-----------------------------------#

print('--------------------------------------------------------------')
print()
print('CALCULATIONS OF BLOCH SUMS')
print()

BLOCHBasis  =  BLOCH(nkpts,NH,Ao,nXpts,Xpts[:],k_vec[:],nuc,atoms[:],alpha,NormCnst)
dBLOCHBasis = dBLOCH(nkpts,NH,Ao,nXpts,Xpts[:],k_vec[:],nuc,atoms[:],alpha,NormCnst)

(BLOCHBasis, dBLOCHBasis) = normBLOCH(BLOCHBasis[:,:,:],dBLOCHBasis[:,:,:],k_vec[:],NH,Ao,dX,Xpts[:],nXpts)  

check       = normCHECK(BLOCHBasis[:,:,:],k_vec[:],NH,Ao,dX,Xpts[:],nXpts)
if check:
	print('- CHECK: NORMALIZATION OF BLOCH SUMS CORRECT')
	print()
else:
	print('- ERROR: BLOCH SUMS NOT NORMALIZED')


if errBB:
	print('- CALCULATION OF BLOCH SUMS ORTHONORMALITY ERROR')
	(errBLOCHBasis,errBBik,errBBjk,errBBsite_i,errBBsite_j) = errBASIS(BLOCHBasis[:,:,:],nkpts,NH,Ao,dX,Xpts[:],nXpts)
	print()
	print('  ERROR IN BLOCH SUMS ORTHONORMALITY:',errBLOCHBasis)
	print('  STATE [K=',errBBik,', SITE=',errBBsite_i,'] WITH STATE [K=',errBBjk,', SITE=',errBBsite_j,']')
	print()
	print('- CALCULATION OF BLOCH SUMS ORTH. ERROR FINISHED')
	print()

if plotBB:
	plotBLOCH(BLOCHBasis[:,:,Xin:Xfin],dBLOCHBasis[:,:,Xin:Xfin],k_vec[:],nkpts,NH,Xpts[Xin:Xfin],X_lim,fig_BlochBASIS,fig_dBlochBASIS)

print('CALCULATION OF BLOCH SUMS FINISHED')
print()
print('--------------------------------------------------------------')
print()

#-----------------------------------#
# BLOCH WAVE FUNCTIONS CONSTRUCTION #
#-----------------------------------#

print('--------------------------------------------------------------')
print()
print('CALCULATION OF BLOCH EIGENSTATES')
print()

WFCkx = np.zeros((nkpts,NH,nXpts)).astype(np.cdouble)
WFCkx = BLOCHWFC(WFCkx[:,:,:],nkpts,NH,nXpts,k_vec[:],BLOCHBasis[:,:,:],Ckn)

dWFCkx = np.zeros((nkpts,NH,nXpts)).astype(np.cdouble)
dWFCkx = BLOCHWFC(dWFCkx[:,:,:],nkpts,NH,nXpts,k_vec[:],dBLOCHBasis[:,:,:],Ckn)

(WFCkx, dWFCkx) = normWFC(WFCkx[:,:,:],dWFCkx[:,:,:],k_vec[:],NH,Ao,dX,Xpts[:],nXpts)

check       = normWFCCHECK(WFCkx[:,:,:],k_vec[:],NH,Ao,dX,Xpts[:],nXpts)
if check:
	print('- CHECK: NORMALIZATION OF BLOCH WAVE FUNCTIONS CORRECT')
	print()
else:
	print('- ERROR: BLOCH WAVE FUNCTIONS NOT NORMALIZED')

if errWFC:
	print('- CALCULATION OF BLOCH WFCs ORTHONORMALITY ERROR')
	(errBLOCHWFC,errBWFCik,errBWFCjk,errBWFCband_i,errBWFCband_j) = errBWFC(WFCkx[:,:,:],nkpts,NH,Ao,dX,Xpts[:],nXpts)
	print()
	print('  ERROR IN BLOCH WFCs ORTHONORMALITY:',errBLOCHBasis)
	print()
	print('  STATE [K=',errBWFCik,', BAND=',errBWFCband_i,'] WITH STATE [K=',errBWFCjk,', BAND=',errBWFCband_j,']')
	print()
	print('- CALCULATION OF BLOCH WFCs ORTH. ERROR FINISHED')
	print()

#############################################################################################################################

#ind_inf = int( (Ao/2.0)/dX )
#ind_sup = nXpts - int( (Ao/2.0)/dX )

#lim_inf = Xpts[ind_inf] 
#lim_sup = Xpts[ind_sup]

#NORM_COEFF = np.ones((nkpts,NH))

#print()
#print('NORMALIZATION CONSTANTS : ANALYTICAL INTEGRATION')
#print()

#for ik, k in enumerate(k_vec[:,0]):
#	for band in range(NH):
#		N                  = OV_INTEGRAL(NH,nuc,k,k,Ckn[ik][band,:],Ckn[ik][band,:],atoms[:],alpha,lim_inf,lim_sup,Ao)
#		print('AN.  NORM. CONSTANT [ik=',ik,'n=',band,'] N =',N)
#		NORM_COEFF[ik,band] = 1/(np.sqrt(N)) 

#INT = OV_INTEGRAL(NH,nuc,k_vec[0],k_vec[50],Ckn[0][0,:],Ckn[50][0,:],atoms[:],alpha,lim_inf,lim_sup,Ao)
#INT = INT * NORM_COEFF[0,0] * NORM_COEFF[50,0]
#print('Xo = ',lim_inf,'Xf = ',lim_sup,'TRIAL INTEGRAL =', INT)

#for ik, k in enumerate(k_vec[:,0]):
#	for jk, kp in enumerate(k_vec[:,0]):
#		for bandi in range(NH):
#			for bandj in range(NH):
#				INT = OV_INTEGRAL(NH,nuc,k,kp,Ckn[ik][bandi,:],Ckn[jk][bandj,:],atoms[:],alpha,lim_inf,lim_sup,Ao)
#				INT = INT * NORM_COEFF[ik,bandi] * NORM_COEFF[jk,bandj]


#####################################################################################################################

if plotWFC:
	plotWFCkx(WFCkx[:,:,Xin:Xfin],dWFCkx[:,:,Xin:Xfin],evals[:,:],k_vec[:],nkpts,NH,Xpts[Xin:Xfin],X_lim,fig_BWFC,fig_dBWFC)

print('CALCULATION OF BLOCH EIGENSTATES FINISHED')
print()
print('--------------------------------------------------------------')
print()

#####################################################################################################################

#print()
#print('Ckn COEFFICIENTS')
#for ik in range(nkpts):
#	print('ik= ',ik,' C[ik]= ',Ckn[ik], 'E[ik]= ',evals[:,ik])


#print()
#print('K VECTORS')
#for ik, k in enumerate(k_vec[:,0]):
#	print('ik= ',ik,'K= ',k)

#print()
#print('CHECK FOR ORTHOGONALITY')
#for ik, k in enumerate(k_vec[:,0]):
#	for bandi in range(NH):
#		for jk, kp in enumerate(k_vec[:,0]):
#			for bandj in range(NH):
#				INT = np.trapz((WFCkx[ik,bandi,:]*np.conjugate(WFCkx[jk,bandj,:])),Xpts[:])
#				if ( (np.real(INT) > 0.001 ) or ( np.imag(INT) > 0.001) ):
#					print('ik= ',ik,' jk= ',jk,' bandi= ',bandi,' bandj= ',bandj,' NON ORTHOGONAL ', INT)
#print()


#INT1 = np.trapz((WFCkx[10,0,:]*np.conjugate(WFCkx[10,1,:])),Xpts[:])
#INT2 = np.trapz((WFCkx[10,1,:]*np.conjugate(WFCkx[10,0,:])),Xpts[:])

#print('INT[1] =  ', INT1)
#print('INT[1]* = ', INT2)

######################################################################################################################

#-----------------------------------#
# OCCUPATION NUMBERS 'NS' AND 'SC'  #
#-----------------------------------#

Eo    = -2.0
Ef    =  2.0
dE    =  0.01

#temp  = 300
#fermi = 0.0

en    = np.arange(Eo,Ef,dE)

if plotOCC:
	plotGAP(en,fig_OCC)
	plotCHI(en,fermi,temp,fig_OCC)
	plotOccNum(en,fermi,temp,fig_OCC)

#-----------------------------------#
# ELECTRONIC DENSITY 'NS' AND 'SC'  #
#-----------------------------------#

print('--------------------------------------------------------------')
print()
print('CALCULATION OF ELECTRONIC DENSITY')
print()

rhoNS0  = np.zeros((nXpts)).astype(np.double)
rhoNS   = np.zeros((nXpts)).astype(np.double)
rhoSC   = np.zeros((nXpts)).astype(np.double)

drhoNS0 = np.zeros((nXpts)).astype(np.double)
drhoNS  = np.zeros((nXpts)).astype(np.double)
drhoSC  = np.zeros((nXpts)).astype(np.double)


(rhoNS0, rhoNS, rhoSC, drhoNS0, drhoNS, drhoSC) = DENSITIES(WFCkx[:,:,:],dWFCkx[:,:,:],evals[:,:],k_vec[:],nkpts,NH,nXpts,Xpts[:],fermi,temp)

check   = DENSCHECK(rhoNS0[:],rhoNS[:],rhoSC[:],Xpts[:],nXpts,Ao,dX,NH,nuc)
if check:
        print('- CHECK: NORMALIZATION OF DENSITIES CORRECT')
        print()
else:
	print('- ERROR: DENSITIES NOT NORMALIZED')

if plotDENS:
	plotDENSITY(rhoNS0[:],rhoNS[:],rhoSC[:],drhoNS0[:],drhoNS[:],drhoSC[:],Xpts[:],nXpts,Ao,dX,nuc,temp,fig_DENSITY)

print('CALCULATION OF ELECTRONIC DENSITY FINISHED')
print()
print('--------------------------------------------------------------')
print()

#-----------------------------------#
# KINETIC EN. DENSITY 'NS' AND 'SC' #
#-----------------------------------#

print('--------------------------------------------------------------')
print()
print('CALCULATION OF KINETIC ENERGY DENSITIES')
print()

T_NS0   = np.zeros((nXpts)).astype(np.double)
T_NS    = np.zeros((nXpts)).astype(np.double)
T_SC    = np.zeros((nXpts)).astype(np.double)

TNS0_vw = np.zeros((nXpts)).astype(np.double)
TNS_vw  = np.zeros((nXpts)).astype(np.double)
TSC_vw  = np.zeros((nXpts)).astype(np.double)

TNS0_tf = np.zeros((nXpts)).astype(np.double)
TNS_tf  = np.zeros((nXpts)).astype(np.double)
TSC_tf  = np.zeros((nXpts)).astype(np.double)

(T_NS0, T_NS, T_SC) = KINETIC(dWFCkx[:,:,:],evals[:,:],k_vec[:],nkpts,NH,nXpts,Xpts[:],fermi,temp)

TNS0_vw = KINETICvw(rhoNS0[:], drhoNS0[:], Xpts[:], nXpts)
TNS_vw  = KINETICvw(rhoNS[:] , drhoNS[:] , Xpts[:], nXpts)
TSC_vw  = KINETICvw(rhoSC[:] , drhoSC[:] , Xpts[:], nXpts)

TNS0_tf = KINETICtf(rhoNS0[:], Xpts[:], nXpts)
TNS_tf  = KINETICtf(rhoNS[:] , Xpts[:], nXpts)
TSC_tf  = KINETICtf(rhoSC[:] , Xpts[:], nXpts)

if plotKIN:
	plotKINETIC(T_NS0[:],T_NS[:],T_SC[:],TNS0_vw[:],TNS_vw[:],TSC_vw[:],TNS0_tf[:],TNS_tf[:],TSC_tf[:],Xpts[:],nXpts,Ao,dX,nuc,temp,fig_KINETIC)

print('CALCULATION OF KINETIC ENERGY DENSITIES FINISHED')
print()
print('--------------------------------------------------------------')
print()

#-----------------------------------#
# ELECTRON LOCALIZATION FUNCTION    #
#-----------------------------------#

print('--------------------------------------------------------------')
print()
print('CALCULATION OF ELF')
print()

ELF_NS0 = np.zeros((nXpts)).astype(np.double)
ELF_NS  = np.zeros((nXpts)).astype(np.double)
ELF_SC  = np.zeros((nXpts)).astype(np.double)
LOC_CP  = np.zeros((nXpts)).astype(np.double)

ELF_NS0 = ELF(T_NS0[:], TNS0_vw[:], TNS0_tf[:], nXpts)
ELF_NS  = ELF(T_NS[:] , TNS_vw[:] , TNS_tf[:] , nXpts)
ELF_SC  = ELF(T_SC[:] , TSC_vw[:] , TSC_tf[:] , nXpts)
LOC_CP  = ELF(T_SC[:] , T_NS[:]   , TNS_tf[:] , nXpts)

if plotELOCF:
	plotELF(ELF_NS0[:],ELF_NS[:],ELF_SC[:],LOC_CP[:],Xpts[:],nXpts,Ao,dX,nuc,temp,fig_ELF)

print('CALCULATION OF ELF FINISHED')
print()
print('--------------------------------------------------------------')
print()

#-----------------------------------#
# NETWORKING VALUE NWV              #
#-----------------------------------#

print('--------------------------------------------------------------')
print()
print('CALCULATION OF THE NWV')
print()

NWV_NS0 = NWV(ELF_NS0[:],Ao,dX,nuc)
NWV_NS  = NWV(ELF_NS[:] ,Ao,dX,nuc)
NWV_SC  = NWV(ELF_SC[:] ,Ao,dX,nuc)

print('- NETWORKING VALUE NS(T=  0  K):'   , NWV_NS0)
print('- NETWORKING VALUE NS(T=',temp,'K):', NWV_NS)
print('- NETWORKING VALUE SC(T=',temp,'K):', NWV_SC)
print()

print('CALCULATION OF THE NWV FINISHED')
print()
print('--------------------------------------------------------------')
print()

#-----------------------------------#
# PROBABILITIES INDEPENDENT ELECTR. #
#-----------------------------------#

print('--------------------------------------------------------------')
print()
print('CALCULATION OF PROBABILITIES FOR INDEPENDENT SYSTEM')
print()

NEL              = NH * nkpts
NELvec           = np.arange(0, NEL + 1 ,1)

indPROB_NS0      = np.zeros((NEL + 1)).astype(np.cdouble)
indPROB_NS       = np.zeros((NEL + 1)).astype(np.cdouble)
indPROB_SC       = np.zeros((NEL + 1)).astype(np.cdouble)

indPROB_TOT      = np.zeros((NEL + 1,3)).astype(np.cdouble)

indPROB_NS0      = indPROB(rhoNS0[:],ELF_NS0[:],Xpts[:],nXpts,dX,NH,nkpts,Ao,dHH)
indPROB_NS       = indPROB(rhoNS[:] ,ELF_NS[:] ,Xpts[:],nXpts,dX,NH,nkpts,Ao,dHH)
indPROB_SC       = indPROB(rhoSC[:] ,ELF_SC[:] ,Xpts[:],nXpts,dX,NH,nkpts,Ao,dHH)

indPROB_TOT[:,0] = indPROB_NS0[:]
indPROB_TOT[:,1] = indPROB_NS[:]
indPROB_TOT[:,2] = indPROB_SC[:]

print()
print('- PROBABILITIES FOR INDEPENDENT ELECTRONS NS(T=0K)')
print()
print('  INDEP PROB NS0 0 =', np.real(indPROB_NS0[0]) )
print('  INDEP PROB NS0 1 =', np.real(indPROB_NS0[1]) )
print('  INDEP PROB NS0 2 =', np.real(indPROB_NS0[2]) )
print('  INDEP PROB NS0 3 =', np.real(indPROB_NS0[3]) )
print('  INDEP PROB NS0 4 =', np.real(indPROB_NS0[4]) )
print('  INDEP PROB NS0 5 =', np.real(indPROB_NS0[5]) )
print('  INDEP PROB NS0 6 =', np.real(indPROB_NS0[6]) )
print('  INDEP PROB NS0 7 =', np.real(indPROB_NS0[7]) )
print()
print('- PROBABILITIES FOR INDEPENDENT ELECTRONS NS(T=300K)')
print()
print('  INDEP PROB NS  0 =', np.real(indPROB_NS[0]) )
print('  INDEP PROB NS  1 =', np.real(indPROB_NS[1]) )
print('  INDEP PROB NS  2 =', np.real(indPROB_NS[2]) )
print('  INDEP PROB NS  3 =', np.real(indPROB_NS[3]) )
print('  INDEP PROB NS  4 =', np.real(indPROB_NS[4]) )
print('  INDEP PROB NS  5 =', np.real(indPROB_NS[5]) )
print('  INDEP PROB NS  6 =', np.real(indPROB_NS[6]) )
print('  INDEP PROB NS  7 =', np.real(indPROB_NS[7]) )
print()
print('- PROBABILITIES FOR INDEPENDENT ELECTRONS SC(T=300K)')
print()
print('  INDEP PROB SC  0 =', np.real(indPROB_SC[0]) )
print('  INDEP PROB SC  1 =', np.real(indPROB_SC[1]) )
print('  INDEP PROB SC  2 =', np.real(indPROB_SC[2]) )
print('  INDEP PROB SC  3 =', np.real(indPROB_SC[3]) )
print('  INDEP PROB SC  4 =', np.real(indPROB_SC[4]) )
print('  INDEP PROB SC  5 =', np.real(indPROB_SC[5]) )
print('  INDEP PROB SC  6 =', np.real(indPROB_SC[6]) )
print('  INDEP PROB SC  7 =', np.real(indPROB_SC[7]) )
print()

print('CALCULATION OF PROB. FOR INDEP. SYSTEM FINISHED')
print()
print('--------------------------------------------------------------')
print()
	
#-----------------------------------#
# MPD's NS(T=OK) RECURSIVE FORMULA  #
#-----------------------------------#

print('--------------------------------------------------------------')
print()
print('CALCULATION OF PROBABILITIES [CANCÉS FORMULA]')
print()

#####
#print()
#print('K VECTORS AND OCCUPATION NUMBERS')
#for ik, k in enumerate(k_vec[:,0]):
#	for band in range(NH):
#		occ =  OccNumNS0(evals[band,ik])
#		print('k= ',k_vec[ik],' e= ',evals[band,ik],' nk= ',occ)
#print()
#####

PROB   = np.zeros((NEL + 1)).astype(np.cdouble)

PROB   = recPROB(WFCkx[:,:,:],k_vec[:],evals[:,:],ELF_NS0[:],Xpts[:],nXpts,dX,NH,nkpts,Ao,dHH,errBLOCHWFC)

print()
print('- PROBABILITIES [ELF BASIN]')
print()
print('  PROB 0 =', np.real(PROB[0]) )
print('  PROB 1 =', np.real(PROB[1]) )
print('  PROB 2 =', np.real(PROB[2]) )
print('  PROB 3 =', np.real(PROB[3]) )
print('  PROB 4 =', np.real(PROB[4]) )
print('  PROB 5 =', np.real(PROB[5]) )
print()
	
print('CALCULATION OF PROB. [CANCÉS FORMULA] FINISHED')
print()
print('--------------------------------------------------------------')
print()

#-----------------------------------#
# LOC. AND DELOC. INDICES           #
#-----------------------------------#

print('--------------------------------------------------------------')
print()
print('CALCULATION OF LOCALIZATION|DELOCALIZATION INDICES')
print()

NN         = 12
NNvec      =  np.arange(2,NN+2,1)

lambda_NS0 = 0.0
lambda_NS  = 0.0
lambda_SC  = 0.0

delta_NS0  = np.zeros((NN)).astype(np.cdouble)
delta_NS   = np.zeros((NN)).astype(np.cdouble)
delta_SC   = np.zeros((NN)).astype(np.cdouble)

(lambda_NS0, delta_NS0) = INDICES(WFCkx[:,:,:],k_vec[:],evals[:,:],ELF_NS0[:],Xpts[:],nXpts,dX,NH,nkpts,Ao,dHH,NN,fermi,temp,errBLOCHWFC,'NS0')
(lambda_NS , delta_NS)  = INDICES(WFCkx[:,:,:],k_vec[:],evals[:,:],ELF_NS[:] ,Xpts[:],nXpts,dX,NH,nkpts,Ao,dHH,NN,fermi,temp,errBLOCHWFC,'NS' )
(lambda_SC , delta_SC)  = INDICES(WFCkx[:,:,:],k_vec[:],evals[:,:],ELF_SC[:] ,Xpts[:],nXpts,dX,NH,nkpts,Ao,dHH,NN,fermi,temp,errBLOCHWFC,'SC' )

print()
print('- LOCALIZATION INDICES')
print()
print('  LAMBDA NS0   = '  , np.real(lambda_NS0))
print('  LAMBDA NS    = '  , np.real(lambda_NS) )
print('  LAMBDA SC    = '  , np.real(lambda_SC) )
print()
print('- DELOCALIZATION INDICES')
print()
print('  DELTA(2)  NS0 = ', np.real(delta_NS0[0] ))
print('  DELTA(3)  NS0 = ', np.real(delta_NS0[1] ))
print('  DELTA(4)  NS0 = ', np.real(delta_NS0[2] ))
print('  DELTA(5)  NS0 = ', np.real(delta_NS0[3] ))
print('  DELTA(6)  NS0 = ', np.real(delta_NS0[4] ))
print('  DELTA(7)  NS0 = ', np.real(delta_NS0[5] ))
print('  DELTA(8)  NS0 = ', np.real(delta_NS0[6] ))
print('  DELTA(9)  NS0 = ', np.real(delta_NS0[7] ))
print('  DELTA(10) NS0 = ', np.real(delta_NS0[8] ))
print('  DELTA(11) NS0 = ', np.real(delta_NS0[9] ))
print('  DELTA(12) NS0 = ', np.real(delta_NS0[10]))
print('  DELTA(13) NS0 = ', np.real(delta_NS0[11]))
print()
print('  DELTA(2)  NS  = ', np.real(delta_NS[0] ))
print('  DELTA(3)  NS  = ', np.real(delta_NS[1] ))
print('  DELTA(4)  NS  = ', np.real(delta_NS[2] ))
print('  DELTA(5)  NS  = ', np.real(delta_NS[3] ))
print('  DELTA(6)  NS  = ', np.real(delta_NS[4] ))
print('  DELTA(7)  NS  = ', np.real(delta_NS[5] ))
print('  DELTA(8)  NS  = ', np.real(delta_NS[6] ))
print('  DELTA(9)  NS  = ', np.real(delta_NS[7] ))
print('  DELTA(10) NS  = ', np.real(delta_NS[8] ))
print('  DELTA(11) NS  = ', np.real(delta_NS[9] ))
print('  DELTA(12) NS  = ', np.real(delta_NS[10]))
print('  DELTA(13) NS  = ', np.real(delta_NS[11]))
print()
print('  DELTA(2)  SC  = ', np.real(delta_SC[0] ))
print('  DELTA(3)  SC  = ', np.real(delta_SC[1] ))
print('  DELTA(4)  SC  = ', np.real(delta_SC[2] ))
print('  DELTA(5)  SC  = ', np.real(delta_SC[3] ))
print('  DELTA(6)  SC  = ', np.real(delta_SC[4] ))
print('  DELTA(7)  SC  = ', np.real(delta_SC[5] ))
print('  DELTA(8)  SC  = ', np.real(delta_SC[6] ))
print('  DELTA(9)  SC  = ', np.real(delta_SC[7] ))
print('  DELTA(10) SC  = ', np.real(delta_SC[8] ))
print('  DELTA(11) SC  = ', np.real(delta_SC[9] ))
print('  DELTA(12) SC  = ', np.real(delta_SC[10]))
print('  DELTA(13) SC  = ', np.real(delta_SC[11]))
print()

print('CALCULATION OF LOC.|DELOC. INDICES FINISHED')
print()
print('--------------------------------------------------------------')
print()

#-----------------------------------#
# WRITING OUTPUTS                   #
#-----------------------------------#

WRITEWFC(NH,nkpts,k_vec[:],evals[:,:],Ckn,temp,fermi,data_dir)

WRITE(rhoNS0[:]  , Xpts[:]   ,nXpts,'density_NS0'     ,data_dir)
WRITE(rhoNS[:]   , Xpts[:]   ,nXpts,'density_NS'      ,data_dir)
WRITE(rhoSC[:]   , Xpts[:]   ,nXpts,'density_SC'      ,data_dir)

WRITE(drhoNS0[:] , Xpts[:]   ,nXpts,'ddensity_NS0'    ,data_dir)
WRITE(drhoNS[:]  , Xpts[:]   ,nXpts,'ddensity_NS'     ,data_dir)
WRITE(drhoSC[:]  , Xpts[:]   ,nXpts,'ddensity_SC'     ,data_dir)

WRITE(T_NS0[:]   , Xpts[:]   ,nXpts,'kinetic_NS0'     ,data_dir)
WRITE(TNS0_vw[:] , Xpts[:]   ,nXpts,'kinetic_NS0_vw'  ,data_dir)
WRITE(TNS0_tf[:] , Xpts[:]   ,nXpts,'kinetic_NS0_tf'  ,data_dir)

WRITE(T_NS[:]    , Xpts[:]   ,nXpts,'kinetic_NS'      ,data_dir)
WRITE(TNS_vw[:]  , Xpts[:]   ,nXpts,'kinetic_NS_vw'   ,data_dir)
WRITE(TNS_tf[:]  , Xpts[:]   ,nXpts,'kinetic_NS_tf'   ,data_dir)

WRITE(T_SC[:]    , Xpts[:]   ,nXpts,'kinetic_SC'      ,data_dir)
WRITE(TSC_vw[:]  , Xpts[:]   ,nXpts,'kinetic_SC_vw'   ,data_dir)
WRITE(TSC_tf[:]  , Xpts[:]   ,nXpts,'kinetic_SC_tf'   ,data_dir)

WRITE(ELF_NS0[:] , Xpts[:]   ,nXpts,'elf_NS0'         ,data_dir)
WRITE(ELF_NS[:]  , Xpts[:]   ,nXpts,'elf_NS'          ,data_dir)
WRITE(ELF_SC[:]  , Xpts[:]   ,nXpts,'elf_SC'          ,data_dir)
WRITE(LOC_CP[:]  , Xpts[:]   ,nXpts,'loc_CP'          ,data_dir)

WRITEc(indPROB_TOT[:], NELvec[:] ,NEL+1,'indep_probabilities',data_dir)
WRITE(PROB[:]        , NELvec[:] ,NEL+1,'probabilities'      ,data_dir)

WRITE_LIs(lambda_NS0   , lambda_NS   , lambda_SC              , 3 , 'lambda',data_dir)
WRITE_DIs(delta_NS0[:] , delta_NS[:] , delta_SC[:] , NNvec[:], NN, 'delta' ,data_dir) 
