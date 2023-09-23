#! /usr/bin/env python3

import numpy as np
import scipy.special

from OCC import *

#----------------------------------#
# PROBABILITIES FOR INDEP. ELECTR. #
#----------------------------------#

def indPROB(DENS,ELF,Xpts,nXpts,dX,NH,nkpts,Ao,dHH):

	NEL   = NH * nkpts
	Basin = np.zeros((2)).astype(np.intc)
	
	PROB  = np.zeros((NEL + 1)).astype(np.cdouble)
	
	Basin = ELFBasin(ELF[:],Xpts[:],dX,nXpts,Ao,dHH)

	B     = (1/NEL) * np.trapz( DENS[Basin[0]:Basin[1]] ,Xpts[Basin[0]:Basin[1]]) 

	#print()
	#print('B(OMEGA) =', B)
	#print()

	for nu in range(NEL + 1):
		PROB[nu] = scipy.special.comb(NEL,nu) * (B**nu) * (1.0 - B)**(NEL-nu)	

	return PROB

#----------------------------------#
# RECURSIVE FORMULA FOR S.D. WFC   #
#----------------------------------#

def recPROB(WFCkx,k_vec,evals,ELF,Xpts,nXpts,dX,NH,nkpts,Ao,dHH,errBLOCHBasis):

	NEL   = NH * nkpts
	Np    = int(NEL/2)
	
	Basin = np.zeros((2)).astype(np.intc)
	
	ORBs  = np.zeros((Np,nXpts)).astype(np.cdouble)
	ORBen = np.zeros((Np)).astype(np.double)

	S     = np.zeros((Np,Np)).astype(np.cdouble)
	wS    = np.zeros((Np)).astype(np.cdouble)

	PROB  = np.zeros((NEL + 1)).astype(np.cdouble)

	Basin         = ELFBasin(ELF[:],Xpts[:],dX,nXpts,Ao,dHH)

	(ORBs, ORBen) = getORB(WFCkx[:,:,:],k_vec[:],evals[:,:],Np,NH,nkpts,nXpts)

	S             = OVERLAP(ORBs[:,:],Basin[:],Ao,dX,Xpts[:],nXpts,Np,errBLOCHBasis)
	wS            = DIAG(S[:,:],Np)

	PROB          = REC_PROB_SD(wS[:],NEL,Np)	
	
	return PROB


#----------------------------------#
# ELF BASIN | INTEGRATION INTERVAL #
#----------------------------------#

def ELFBasin(ELF,Xpts,dX,nXpts,Ao,dHH):

	#thrs     = 1.0e-10
	Basin    = np.zeros((2)).astype(np.intc)

	N        = int(Ao/dX)
	Np       = int((dHH/2.0)/dX)

	#Basin[0] = np.argmin(ELF[int(nXpts/2):nXpts])
	#Basin[0] = np.argmin(ELF[1000:int(nXpts/2)])
	#Basin[0] = np.argmin(ELF[:])

	Basin[0] = N + Np
	Basin[1] = int((3.0/2.0)*N+Np)	

	#ELFmin   = ELF[Basin[0]]
	#Basin[1] = Basin[0] + 1
	#while ( abs(ELF[Basin[1]] - ELFmin) > thrs ):
	#	Basin[1] += 1

	#print()
	#print('Index[1]',Basin[0],'Index[2]',Basin[1])
	#print('ELF[1]=',ELF[Basin[0]],'ELF[2]=',ELF[Basin[1]])
	#print('Xpts[1]=',Xpts[Basin[0]],'Xpts[2]=',Xpts[Basin[1]])
	#print()

	return Basin

#----------------------------------#
# GET OCC. ORBITALS NS (T=0K)      #
#----------------------------------#

def getORB(WFCkx,k_vec,evals,Np,NH,nkpts,nXpts):

	ORBs  = np.zeros((Np,nXpts)).astype(np.cdouble)
	ORBen = np.zeros((Np)).astype(np.double)

	i = 0
	for ik, k in enumerate(k_vec[:,0]):
		for band in range(NH):
			occ = OccNumNS0(evals[band,ik])
			if (occ != 0.0 ):
				ORBs[i,:]  = WFCkx[ik,band,:]
				ORBen[i]   = evals[band,ik]
				#print('ORBs[',i,'] : K[',ik,'] =',k,'e[',band,',',ik,'] =',evals[band,ik],'nk =',occ)
				i += 1
	if ( i != Np ):
		print('ERROR: AN ERROR HAS OCCURED WITH getORB SUBROUTINE')
	
	return ORBs, ORBen

#----------------------------------#
# 'S' OVERLAP MATRIX IN OMEGA VOL. #
#----------------------------------#

def OVERLAP(ORBs,Basin,Ao,dX,Xpts,nXpts,Np,errBLOCHWFC):

	thrs = errBLOCHWFC

	Xo   = int( Ao/dX )
	Xf   = nXpts - int( Ao/dX )
	
	S    = np.zeros((Np,Np)).astype(np.cdouble)

	for i in range(Np):
		for j in range(i+1):
			S[i,j] = np.trapz(( np.conjugate(ORBs[i,Basin[0]:Basin[1]]) * ORBs[j,Basin[0]:Basin[1]] ),Xpts[Basin[0]:Basin[1]])
			#S[i,j] = np.trapz(( np.conjugate(ORBs[i,Xo:Xf]) * ORBs[j,Xo:Xf] ),Xpts[Xo:Xf])
	
			#if ( abs( S[i,j] ) < thrs ):
			 #if ( abs( np.real(S[i,j]) ) < thrs ):
			#	S[i,j] = 0.0
			 #if ( abs( np.imag(S[i,j]) ) < thrs ):
			 #	 np.imag(S[i,j]) = 0.0
	
			S[j,i] = np.conjugate(S[i,j])
	
			#print( 'S[',i,',',j,'] =', S[i,j])

	#print()
	#print('OVERLAP MATRIX')
	#print(S)
	#print()

	return S

#----------------------------------#
# DIAGONALIZATION OF HERMITIAN MAT #
#----------------------------------#

def DIAG(MAT,n):

	eigenvals = np.zeros((n)).astype(np.cdouble)

	eigenvals = np.linalg.eigvalsh(MAT)

	#print()
	#print('S MATRIX EIGENVALUES')
	#print(eigenvals)
	#print()

	return eigenvals

#----------------------------------#
# RECURSIVE FORMULA FOR S.D. WFC   #
#----------------------------------#

def REC_PROB_SD(wS,NEL,Np):

	a     = np.zeros((NEL+1,NEL+1)).astype(np.cdouble)
	beta  = np.zeros((NEL)).astype(np.cdouble)
	alpha = np.zeros((NEL)).astype(np.cdouble)
	PROB  = np.zeros((NEL+1)).astype(np.cdouble)

	for k in range(Np):
	
		#print('ITERATION: ',k)
	
		beta[2*k]      = wS[k]
		beta[2*k + 1]  = wS[k]

		alpha[2*k]     = 1.0 - wS[k]
		alpha[2*k + 1] = 1.0 - wS[k]
	

	#print()
	#for k in range(NEL):
	#	print('BETA [',k,'] = ',beta[k])
	#	print('ALPHA[',k,'] = ',alpha[k])
	#	print()

	
	a[0,0] = 1.0
	for k in range(NEL):                                #CHECK!
		a[k+1,0]   = alpha[k] * a[k,0]
		a[k+1,k+1] = beta[k]  * a[k,k]
		for j in range(k):
			a[k+1,j+1] = beta[k] * a[k,j] + alpha[k] * a[k,j+1]
			#print('A[',k+1,',',j+1,']= ','BETA[',k,'] * A[',k,',',j,'] + ALPHA[',k,'] * A[',k,',',j+1,']')
	
	PROB[:] = a[NEL,:] #CHECK IF IT ACTUALLY GIVES THE LAST ROW A[:,:]

	#print()
	#for k in range(NEL+1):
	#	for j in range(NEL+1):
	#		print('A[',k,',',j,'] = ',a[k,j])
 
	#print()
	#print()	
	#print('A MATRIX')
	#print(a)
	#print()
	#print('PROBABILITIES')
	#print()
	#print('PROB 0 =', np.real(PROB[0]) )
	#print('PROB 1 =', np.real(PROB[1]) )
	#print('PROB 2 =', np.real(PROB[2]) )
	#print('PROB 3 =', np.real(PROB[3]) )
	#print('PROB 4 =', np.real(PROB[4]) )
	#print('PROB 5 =', np.real(PROB[5]) )
	
	return PROB

#----------------------------------#
# RECURSIVE FORMULA FOR S.D. WFC   #
#----------------------------------#

def INDICES(WFC,k_vec,evals,ELF,Xpts,nXpts,dX,NH,nkpts,Ao,dHH,NN,fermi,temp,errBLOCHWFC,SYSTEM):

	NEL        = NH * nkpts

	BasinA     = np.zeros((2)).astype(np.intc)
	BasinB     = np.zeros((2)).astype(np.intc)
	BasinC     = np.zeros((2)).astype(np.intc)
	BasinD     = np.zeros((2)).astype(np.intc)
	BasinE     = np.zeros((2)).astype(np.intc)
	BasinF     = np.zeros((2)).astype(np.intc)
	BasinG     = np.zeros((2)).astype(np.intc)
	BasinH     = np.zeros((2)).astype(np.intc)
	BasinI     = np.zeros((2)).astype(np.intc)
	BasinJ     = np.zeros((2)).astype(np.intc)
	BasinK     = np.zeros((2)).astype(np.intc)
	BasinL     = np.zeros((2)).astype(np.intc)
	BasinM     = np.zeros((2)).astype(np.intc)


	LAMBDA     = 0.0
	DELTA      = np.zeros((NN)).astype(np.cdouble)

	BasinA     = ELFBasin(ELF[:],Xpts[:],dX,nXpts,Ao,dHH)
	BasinB[0]  = BasinA[1]
	BasinB[1]  = 2 * BasinA[1] - BasinA[0]
	BasinC[0]  = BasinB[1]
	BasinC[1]  = 2 * BasinB[1] - BasinB[0]
	BasinD[0]  = BasinC[1]
	BasinD[1]  = 2 * BasinC[1] - BasinC[0]
	BasinE[0]  = BasinD[1]
	BasinE[1]  = 2 * BasinD[1] - BasinD[0]
	BasinF[0]  = BasinE[1]
	BasinF[1]  = 2 * BasinE[1] - BasinE[0]
	BasinG[0]  = BasinF[1]
	BasinG[1]  = 2 * BasinF[1] - BasinF[0]
	BasinH[0]  = BasinG[1]
	BasinH[1]  = 2 * BasinG[1] - BasinG[0]
	BasinI[0]  = BasinH[1]
	BasinI[1]  = 2 * BasinH[1] - BasinH[0]
	BasinJ[0]  = BasinI[1]
	BasinJ[1]  = 2 * BasinI[1] - BasinI[0]
	BasinK[0]  = BasinJ[1]
	BasinK[1]  = 2 * BasinJ[1] - BasinJ[0]
	BasinL[0]  = BasinK[1]
	BasinL[1]  = 2 * BasinK[1] - BasinK[0]
	BasinM[0]  = BasinL[1]
	BasinM[1]  = 2 * BasinL[1] - BasinL[0]
	

	#print()
	#print('CALCULATION FOR:',SYSTEM)
	#print()
	#print('BASIN [A]')
	#print('Index[1]',BasinA[0],'Index[2]',BasinA[1])
	#print('ELF[1]=',ELF[BasinA[0]],'ELF[2]=',ELF[BasinA[1]])
	#print('Xpts[1]=',Xpts[BasinA[0]],'Xpts[2]=',Xpts[BasinA[1]])
	#print()
	#print('BASIN [B]')
	#print('Index[1]',BasinB[0],'Index[2]',BasinB[1])
	#print('ELF[1]=',ELF[BasinB[0]],'ELF[2]=',ELF[BasinB[1]])
	#print('Xpts[1]=',Xpts[BasinB[0]],'Xpts[2]=',Xpts[BasinB[1]])
	#print()
	#print('BASIN [C]')
	#print('Index[1]',BasinC[0],'Index[2]',BasinC[1])
	#print('ELF[1]=',ELF[BasinC[0]],'ELF[2]=',ELF[BasinC[1]])
	#print('Xpts[1]=',Xpts[BasinC[0]],'Xpts[2]=',Xpts[BasinC[1]])
	#print()
	#print('BASIN [D]')
	#print('Index[1]',BasinD[0],'Index[2]',BasinD[1])
	#print('ELF[1]=',ELF[BasinD[0]],'ELF[2]=',ELF[BasinD[1]])
	#print('Xpts[1]=',Xpts[BasinD[0]],'Xpts[2]=',Xpts[BasinD[1]])
	#print()
	#print('BASIN [E]')
	#print('Index[1]',BasinE[0],'Index[2]',BasinE[1])
	#print('ELF[1]=',ELF[BasinE[0]],'ELF[2]=',ELF[BasinE[1]])
	#print('Xpts[1]=',Xpts[BasinE[0]],'Xpts[2]=',Xpts[BasinE[1]])
	#print()
	#print('BASIN [F]')
	#print('Index[1]',BasinF[0],'Index[2]',BasinF[1])
	#print('ELF[1]=',ELF[BasinF[0]],'ELF[2]=',ELF[BasinF[1]])
	#print('Xpts[1]=',Xpts[BasinF[0]],'Xpts[2]=',Xpts[BasinF[1]])
	#print()
	#print('BASIN [G]')
	#print('Index[1]',BasinG[0],'Index[2]',BasinG[1])
	#print('ELF[1]=',ELF[BasinG[0]],'ELF[2]=',ELF[BasinG[1]])
	#print('Xpts[1]=',Xpts[BasinG[0]],'Xpts[2]=',Xpts[BasinG[1]])
	#print()
	#print('BASIN [H]')
	#print('Index[1]',BasinH[0],'Index[2]',BasinH[1])
	#print('ELF[1]=',ELF[BasinH[0]],'ELF[2]=',ELF[BasinH[1]])
	#print('Xpts[1]=',Xpts[BasinH[0]],'Xpts[2]=',Xpts[BasinH[1]])
	#print()
	#print('BASIN [I]')
	#print('Index[1]',BasinI[0],'Index[2]',BasinI[1])
	#print('ELF[1]=',ELF[BasinI[0]],'ELF[2]=',ELF[BasinI[1]])
	#print('Xpts[1]=',Xpts[BasinI[0]],'Xpts[2]=',Xpts[BasinI[1]])
	#print()
	#print('BASIN [J]')
	#print('Index[1]',BasinJ[0],'Index[2]',BasinJ[1])
	#print('ELF[1]=',ELF[BasinJ[0]],'ELF[2]=',ELF[BasinJ[1]])
	#print('Xpts[1]=',Xpts[BasinJ[0]],'Xpts[2]=',Xpts[BasinJ[1]])
	#print()
	#print('BASIN [K]')
	#print('Index[1]',BasinK[0],'Index[2]',BasinK[1])
	#print('ELF[1]=',ELF[BasinK[0]],'ELF[2]=',ELF[BasinK[1]])
	#print('Xpts[1]=',Xpts[BasinK[0]],'Xpts[2]=',Xpts[BasinK[1]])
	#print()
	#print('BASIN [L]')
	#print('Index[1]',BasinL[0],'Index[2]',BasinL[1])
	#print('ELF[1]=',ELF[BasinL[0]],'ELF[2]=',ELF[BasinL[1]])
	#print('Xpts[1]=',Xpts[BasinL[0]],'Xpts[2]=',Xpts[BasinL[1]])
	#print()
	#print('BASIN [M]')
	#print('Index[1]',BasinM[0],'Index[2]',BasinM[1])
	#print('ELF[1]=',ELF[BasinM[0]],'ELF[2]=',ELF[BasinM[1]])
	#print('Xpts[1]=',Xpts[BasinM[0]],'Xpts[2]=',Xpts[BasinM[1]])
	#print()

	for ik, k in enumerate(k_vec[:,0]):
		for jk, kp in enumerate(k_vec[:,0]):
			for bandi in range(NH):
				for bandj in range(NH):
				
					if (SYSTEM=='NS0'):
						n_occ  = OccNumNS0(evals[bandi,ik])
						np_occ = OccNumNS0(evals[bandj,jk])

					if (SYSTEM=='NS'):
						n_occ  = OccNumNS(evals[bandi,ik],fermi,temp)
						np_occ = OccNumNS(evals[bandj,jk],fermi,temp)

					if (SYSTEM=='SC'):
						n_occ  = OccNumSC(evals[bandi,ik],fermi,temp)
						np_occ = OccNumSC(evals[bandj,jk],fermi,temp)
					
					Sij_A  = np.trapz((np.conjugate(WFC[ik,bandi,BasinA[0]:BasinA[1]])*WFC[jk,bandj,BasinA[0]:BasinA[1]]),Xpts[BasinA[0]:BasinA[1]])					
					Sij_B  = np.trapz((np.conjugate(WFC[ik,bandi,BasinB[0]:BasinB[1]])*WFC[jk,bandj,BasinB[0]:BasinB[1]]),Xpts[BasinB[0]:BasinB[1]])
					Sij_C  = np.trapz((np.conjugate(WFC[ik,bandi,BasinC[0]:BasinC[1]])*WFC[jk,bandj,BasinC[0]:BasinC[1]]),Xpts[BasinC[0]:BasinC[1]])
					Sij_D  = np.trapz((np.conjugate(WFC[ik,bandi,BasinD[0]:BasinD[1]])*WFC[jk,bandj,BasinD[0]:BasinD[1]]),Xpts[BasinD[0]:BasinD[1]])
					Sij_E  = np.trapz((np.conjugate(WFC[ik,bandi,BasinE[0]:BasinE[1]])*WFC[jk,bandj,BasinE[0]:BasinE[1]]),Xpts[BasinE[0]:BasinE[1]])
					Sij_F  = np.trapz((np.conjugate(WFC[ik,bandi,BasinF[0]:BasinF[1]])*WFC[jk,bandj,BasinF[0]:BasinF[1]]),Xpts[BasinF[0]:BasinF[1]])			 			
					Sij_G  = np.trapz((np.conjugate(WFC[ik,bandi,BasinG[0]:BasinG[1]])*WFC[jk,bandj,BasinG[0]:BasinG[1]]),Xpts[BasinG[0]:BasinG[1]])
					Sij_H  = np.trapz((np.conjugate(WFC[ik,bandi,BasinH[0]:BasinH[1]])*WFC[jk,bandj,BasinH[0]:BasinH[1]]),Xpts[BasinH[0]:BasinH[1]])
					Sij_I  = np.trapz((np.conjugate(WFC[ik,bandi,BasinI[0]:BasinI[1]])*WFC[jk,bandj,BasinI[0]:BasinI[1]]),Xpts[BasinI[0]:BasinI[1]])
					Sij_J  = np.trapz((np.conjugate(WFC[ik,bandi,BasinJ[0]:BasinJ[1]])*WFC[jk,bandj,BasinJ[0]:BasinJ[1]]),Xpts[BasinJ[0]:BasinJ[1]])
					Sij_K  = np.trapz((np.conjugate(WFC[ik,bandi,BasinK[0]:BasinK[1]])*WFC[jk,bandj,BasinK[0]:BasinK[1]]),Xpts[BasinK[0]:BasinK[1]])
					Sij_L  = np.trapz((np.conjugate(WFC[ik,bandi,BasinL[0]:BasinL[1]])*WFC[jk,bandj,BasinL[0]:BasinL[1]]),Xpts[BasinL[0]:BasinL[1]])
					Sij_M  = np.trapz((np.conjugate(WFC[ik,bandi,BasinM[0]:BasinM[1]])*WFC[jk,bandj,BasinM[0]:BasinM[1]]),Xpts[BasinM[0]:BasinM[1]])






					LAMBDA    += Sij_A * np.conjugate(Sij_A) * n_occ * np_occ
					DELTA[0]  += Sij_A * np.conjugate(Sij_B) * n_occ * np_occ
					DELTA[1]  += Sij_A * np.conjugate(Sij_C) * n_occ * np_occ
					DELTA[2]  += Sij_A * np.conjugate(Sij_D) * n_occ * np_occ
					DELTA[3]  += Sij_A * np.conjugate(Sij_E) * n_occ * np_occ
					DELTA[4]  += Sij_A * np.conjugate(Sij_F) * n_occ * np_occ
					DELTA[5]  += Sij_A * np.conjugate(Sij_G) * n_occ * np_occ
					DELTA[6]  += Sij_A * np.conjugate(Sij_H) * n_occ * np_occ
					DELTA[7]  += Sij_A * np.conjugate(Sij_I) * n_occ * np_occ
					DELTA[8]  += Sij_A * np.conjugate(Sij_J) * n_occ * np_occ
					DELTA[9]  += Sij_A * np.conjugate(Sij_K) * n_occ * np_occ
					DELTA[10] += Sij_A * np.conjugate(Sij_L) * n_occ * np_occ
					DELTA[11] += Sij_A * np.conjugate(Sij_M) * n_occ * np_occ


	LAMBDA = LAMBDA / 2.0 #(nkpts**2)
	#DELTA  = DELTA * (2.0)/(nkpts**2)

	return LAMBDA, DELTA 
