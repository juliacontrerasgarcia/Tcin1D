#! /usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from OCC import *

#--------------------------------#
# PLOT AO's BASIS                #
#--------------------------------#

def plotBASIS(AOBasis,dAOBasis,NH,xpts,x_lim,fig_AOsBASIS):
	
	print('- PLOTTING AOs BASIS')
	print()

	fig_Basis , ax_Basis  = plt.subplots()
	fig_dBasis, ax_dBasis = plt.subplots()
        
	site_label = ['A','B']
	
	for site in range(NH):
               	ax_Basis.plot(xpts,   AOBasis[site,:], label = r'SITE $j={}$'.format(site_label[site]))
               	ax_dBasis.plot(xpts, dAOBasis[site,:], label = r'SITE $j={}$'.format(site_label[site]))
        
	ax_Basis.set_title('AOs BASIS SET')
	ax_Basis.set_xlabel(r'$x \ [u]$')
	ax_Basis.set_xlim(x_lim[0], x_lim[1])
	ax_Basis.set_ylabel(r'$\phi_{j}(x)$')
	ax_Basis.grid()
	ax_Basis.legend(loc=2)
	fig_Basis.tight_layout()
	fig_Basis.savefig('{}/BASIS.png'.format(fig_AOsBASIS))
	plt.close()
	
	ax_dBasis.set_title('AOsDERIVATIVE OF BASIS SET')
	ax_dBasis.set_xlabel(r'$x \ [u]$')
	ax_dBasis.set_xlim(x_lim[0], x_lim[1])
	ax_dBasis.set_ylabel(r'$\frac{d\phi_{j}(x)}{dx}$')
	ax_dBasis.grid()
	ax_dBasis.legend(loc=2)
	fig_dBasis.tight_layout()
	fig_dBasis.savefig('{}/dBASIS.png'.format(fig_AOsBASIS))
	plt.close()

	print('- PLOTTING AOs BASIS FINISHED')
	print()

#--------------------------------#
# PLOT BLOCH BASIS               #
#--------------------------------#

def plotBLOCH(BLOCHBasis,dBLOCHBasis,k_vec,nkpts,NH,Xpts,X_lim,fig_BlochBASIS,fig_dBlochBASIS):

	print('- PLOTTING BLOCH SUMS BASIS')
	print()

	for ik, k in enumerate(k_vec[:,0]):
		fig_BBasis  , ax_BBasis  = plt.subplots()
		fig_dBBasis , ax_dBBasis  = plt.subplots()
		for site in range(NH):
			ax_BBasis.plot(Xpts , np.real(BLOCHBasis[ik, site,:]), label=r'Re $\chi_{}^k$'.format(site))
			ax_dBBasis.plot(Xpts, np.real(dBLOCHBasis[ik, site,:]), label=r'Re $d\chi_{}^k/dx$'.format(site))

		ax_BBasis.set_title('NORMALIZED BLOCH SUMS BASIS SET, k={:.2f}'.format(k))
		ax_BBasis.set_xlabel(r'$x \ [u]$')
		ax_BBasis.set_xlim(X_lim[0], X_lim[1])
		ax_BBasis.set_ylabel(r'$\chi^{k}_{j}(x)$')
		ax_BBasis.legend()
		fig_BBasis.tight_layout()
		fig_BBasis.savefig('{}/BlochBASIS-k{:.2f}.png'.format(fig_BlochBASIS,k))
		plt.close()

		ax_dBBasis.set_title('NORMALIZED BLOCH SUMS BASIS SET DERIVATIVE, k={:.2f}'.format(k))
		ax_dBBasis.set_xlabel(r'$x \ [u]$')
		ax_dBBasis.set_xlim(X_lim[0], X_lim[1])
		ax_dBBasis.set_ylabel(r'$\frac{d\chi^{k}_{j}(x)}{dx}$')
		ax_dBBasis.legend()
		fig_dBBasis.tight_layout()
		fig_dBBasis.savefig('{}/dBlochBASIS-k{:.2f}.png'.format(fig_dBlochBASIS,k))
		plt.close()
	
	print('- PLOTTING BLOCH SUMS BASIS FINISHED')
	print()

#--------------------------------#
# PLOT NORMALIZED BLOCH BASIS    #
#--------------------------------#

def plotNormBLOCH(BLOCHBasis,k_vec,nkpts,NH,Xpts,X_lim,fig_normBlochBASIS):

	print('- PLOTTING NORMALIZED BLOCH SUM BASIS')
	print()

	for ik, k in enumerate(k_vec[:,0]):
		fig_BBasis , ax_BBasis  = plt.subplots()
		for site in range(NH):
			ax_BBasis.plot(Xpts, np.real(BLOCHBasis[ik, site,:]), label=r'Re $\chi_{}^k$'.format(site))

		ax_BBasis.set_title('NORMALIZED BLOCH SUMS BASIS SET, k={:.2f}'.format(k))
		ax_BBasis.set_xlabel(r'$x \ [u]$')
		ax_BBasis.set_xlim(X_lim[0], X_lim[1])
		ax_BBasis.set_ylabel(r'$\chi^{k}_{j}(x)$')
		ax_BBasis.legend()
		fig_BBasis.tight_layout()
		fig_BBasis.savefig('{}/normBlochBASIS-k{:.2f}.png'.format(fig_normBlochBASIS,k))
		plt.close()

	print('- PLOTTING NORMALIZED BLOCH SUM BASIS FINISHED')
	print()

#--------------------------------#
# PLOT BLOCH EIGENSTATES WFC     #
#--------------------------------#

def plotWFCkx(WFC,dWFC,evals,k_vec,nkpts,NH,Xpts,X_lim,fig_BWFC,fig_dBWFC):

	print('- PLOTTING BLOCH WAVE FUNCTIONS')
	print()
	
	for ik, k in enumerate(k_vec[:,0]):
		fig_WFC , ax_WFC  = plt.subplots()
		fig_dWFC, ax_dWFC  = plt.subplots()
		for band in range(NH):
			ax_WFC.plot(Xpts[:] , np.real(WFC[ik, band,:]),label=r'Re $\psi_{}^k$, $\epsilon_{}^k = {:.2f}$'.format(band,band,evals[band,ik]))
			ax_dWFC.plot(Xpts[:], np.real(dWFC[ik, band,:]),label=r'Re $d\psi_{}^k/dx$, $\epsilon_{}^k = {:.2f}$'.format(band,band,evals[band,ik]))
		
		ax_WFC.set_title('BLOCH EIGENFUNCTION, k={:.2f}'.format(k))
		ax_WFC.set_xlabel(r'$x \ [u]$')
		#ax_WFC.set_xlim(X_lim[0], X_lim[1])
		ax_WFC.set_ylabel(r'$\psi_{nk}(x)$')
		ax_WFC.legend()
		fig_WFC.tight_layout()
		fig_WFC.savefig('{}/BlochWFC-k{:.2f}.png'.format(fig_BWFC,k))
		plt.close()
	

		ax_dWFC.set_title('BLOCH EIGENFUNCTION DERIVATIVE, k={:.2f}'.format(k))
		ax_dWFC.set_xlabel(r'$x \ [u]$')
		#ax_dWFC.set_xlim(X_lim[0], X_lim[1])
		ax_dWFC.set_ylabel(r'$\frac{d\psi_{nk}(x)}{dx}$')
		ax_dWFC.legend()
		fig_dWFC.tight_layout()
		fig_dWFC.savefig('{}/dBlochWFC-k{:.2f}.png'.format(fig_dBWFC,k))
		plt.close()

	print('- PLOTTING BLOCH WAVE FUNCTIONS FINISHED')
	print()

#--------------------------------#
# PLOT BAND STRUCTURE            #
#--------------------------------#

def plotBANDS(k_vec,evals,fig_BND):

	print('- PLOTTING BANDS')
	print()

	k_label = [r'$-\frac{\pi}{a_{o}}$',r'$0$',r'$\frac{\pi}{a_{o}}$']
	nodes   = [-0.5,0.0,0.5]

	fig_BAND, ax_BAND = plt.subplots()

	for band in range(evals.shape[0]):
		ax_BAND.plot(k_vec[:,0],evals[band,:],'k-')

	ax_BAND.set_title('BAND STRUCTURE')
	ax_BAND.set_xlabel('PATH IN K-SPACE')
	ax_BAND.set_ylabel('BAND ENERGIES')
	ax_BAND.set_xticks(nodes)
	ax_BAND.set_xticklabels(k_label)
	for n in range(len(nodes)):
		ax_BAND.axvline(x=nodes[n], linewidth=0.5, color='k')

	fig_BAND.tight_layout()
	fig_BAND.savefig('{}/BANDS.png'.format(fig_BND))
	plt.close()

	print('- PLOTTING BANDS FINISHED')
	print()

#--------------------------------#
# PLOT GAP: LORENTZIAN AROUND EF #
#--------------------------------#

def plotGAP(en,fig_OCC):

	print('- PLOTTING GAP FUNCTION')
	print()

	fermi = 0.0

	fig_GAP, ax_GAP = plt.subplots()

	ax_GAP.plot(en[:],GAP(en,fermi), color='black')
	ax_GAP.set_title('GAP')
	ax_GAP.set_xlabel(r'$\xi_{kn} = \epsilon_{kn} - \mu$ [eV]')
	ax_GAP.set_ylabel(r'$\Delta_{kn}$ [eV]')

	fig_GAP.tight_layout()
	fig_GAP.savefig('{}/GAP.png'.format(fig_OCC))
	plt.close()

	print('- PLOTTING GAP FUNCTION FINSIHED')
	print()

#--------------------------------#
# PLOT CHI ENERGY                #
#--------------------------------#

def plotCHI(en,fermi,temp,fig_OCC):

	print('- PLOTTING CHI ENERGY')
	print()

	fig_CHI, ax_CHI = plt.subplots()

	ax_CHI.plot(en[:],CHI(en,fermi,temp), color='black')
	ax_CHI.set_title('CHI ENERGY')
	ax_CHI.set_xlabel(r'$\xi_{kn} = \epsilon_{kn} - \mu$ [eV]')
	ax_CHI.set_ylabel(r'$\chi_{kn}$ [eV]')

	fig_CHI.tight_layout()
	fig_CHI.savefig('{}/CHI.png'.format(fig_OCC))
	plt.close()

	print('- PLOTTING CHI ENERGY FINISHED')
	print()

#--------------------------------#
# PLOT OCCUPATION NUMBERS        #
#--------------------------------#

def plotOccNum(en,fermi,temp,fig_OCC):

	print('- PLOTTING OCCUPATION NUMBERS')
	print()
	
	NS0   = np.array([OccNumNS0(ene)           for ene in en])
	NS    = np.array([OccNumNS(ene,fermi,temp) for ene in en])
	SC    = np.array([OccNumSC(ene,fermi,temp) for ene in en])

	fig_ON, ax_ON = plt.subplots()
	ax_ON.plot(en,NS0, label = r'NS at $T =0K$'              , color='black')
	ax_ON.plot(en,NS , label = r'NS at $T ={}K$'.format(temp), color='blue'  )
	ax_ON.plot(en,SC , label = r'SC at $T ={}K$'.format(temp), color='red' )

	ax_ON.set_title('OCCUPATION NUMBERS')
	ax_ON.set_xlabel(r'$\xi_{kn} = \epsilon_{kn} - \mu$ [eV]')
	ax_ON.set_ylabel(r'$\langle n_{kn} \rangle$')
	ax_ON.legend()
	
	fig_ON.tight_layout()
	fig_ON.savefig('{}/OCC.png'.format(fig_OCC))
	plt.close()

	print('- PLOTTING OCCUPATION NUMBERS FINISHED')
	print()

#--------------------------------#
# PLOT DENSITIES                 #
#--------------------------------#

def plotDENSITY(rhoNS0,rhoNS,rhoSC,drhoNS0,drhoNS,drhoSC,Xpts,nXpts,Ao,dX,nuc,temp,fig_DENSITY):

	print('- PLOTTING DENSITIES')
	print()

	Xo  = nuc + 1
	Xf  = Xo + 2 * int( (Ao/dX) + 1 )

	fig_DENS , ax_DENS  = plt.subplots()
	fig_dDENS, ax_dDENS = plt.subplots()

	ax_DENS.plot(Xpts[Xo:Xf], rhoNS0[Xo:Xf], label = r'NS at $T =0K$'              , color='black')
	ax_DENS.plot(Xpts[Xo:Xf], rhoNS[Xo:Xf] , label = r'NS at $T ={}K$'.format(temp), color='blue' )
	ax_DENS.plot(Xpts[Xo:Xf], rhoSC[Xo:Xf] , label = r'SC at $T ={}K$'.format(temp), color='red'  )

	ax_dDENS.plot(Xpts[Xo:Xf], drhoNS0[Xo:Xf], label = r'NS at $T =0K$'              , color='black')
	ax_dDENS.plot(Xpts[Xo:Xf], drhoNS[Xo:Xf] , label = r'NS at $T ={}K$'.format(temp), color='blue' )
	ax_dDENS.plot(Xpts[Xo:Xf], drhoSC[Xo:Xf] , label = r'SC at $T ={}K$'.format(temp), color='red'  )

	ax_DENS.set_title('ELECTRONIC DENSITIES')
	ax_DENS.set_xlabel(r'$x \ [u]$')
	ax_DENS.set_ylabel(r'$\rho (x)$')
	ax_DENS.legend()

	ax_dDENS.set_title('ELECTRONIC DENSITIES DERIVATIVES')
	ax_dDENS.set_xlabel(r'$x \ [u]$')
	ax_dDENS.set_ylabel(r'$\frac{d\rho (x)}{dx}$')
	ax_dDENS.legend()
	
	fig_DENS.tight_layout()
	fig_DENS.savefig('{}/DENS.png'.format(fig_DENSITY))

	fig_dDENS.tight_layout()
	fig_dDENS.savefig('{}/dDENS.png'.format(fig_DENSITY))

	plt.close()

	print('- PLOTTING DENSITIES FINISHED')
	print()

#--------------------------------#
# PLOT KINETIC ENERGY DENSITIES  #
#--------------------------------#

def plotKINETIC(T_NS0,T_NS,T_SC,TNS0_vw,TNS_vw,TSC_vw,TNS0_tf,TNS_tf,TSC_tf,Xpts,nXpts,Ao,dX,nuc,temp,fig_KINETIC):

	print('- PLOTTING KINETIC ENERGY DENSITIES')
	print()

	Xo  = nuc + 1
	Xf  = Xo + 2 * int( (Ao/dX) + 1 )

	fig_T  , ax_T   = plt.subplots()
	fig_Tvw, ax_Tvw = plt.subplots()
	fig_Ttf, ax_Ttf = plt.subplots()

	ax_T.plot(Xpts[Xo:Xf], T_NS0[Xo:Xf], label = r'NS at $T =0K$'              , color='black')
	ax_T.plot(Xpts[Xo:Xf], T_NS[Xo:Xf] , label = r'NS at $T ={}K$'.format(temp), color='blue' )
	ax_T.plot(Xpts[Xo:Xf], T_SC[Xo:Xf] , label = r'SC at $T ={}K$'.format(temp), color='red'  )

	ax_Tvw.plot(Xpts[Xo:Xf], TNS0_vw[Xo:Xf], label = r'NS at $T =0K$'              , color='black')
	ax_Tvw.plot(Xpts[Xo:Xf], TNS_vw[Xo:Xf] , label = r'NS at $T ={}K$'.format(temp), color='blue' )
	ax_Tvw.plot(Xpts[Xo:Xf], TSC_vw[Xo:Xf] , label = r'SC at $T ={}K$'.format(temp), color='red'  )

	ax_Ttf.plot(Xpts[Xo:Xf], TNS0_tf[Xo:Xf], label = r'NS at $T =0K$'              , color='black')
	ax_Ttf.plot(Xpts[Xo:Xf], TNS_tf[Xo:Xf] , label = r'NS at $T ={}K$'.format(temp), color='blue' )
	ax_Ttf.plot(Xpts[Xo:Xf], TSC_tf[Xo:Xf] , label = r'SC at $T ={}K$'.format(temp), color='red'  )

	ax_T.set_title('KINETIC ENERGY DENSITIES DENSITIES')
	ax_T.set_xlabel(r'$x \ [u]$')
	ax_T.set_ylabel(r'$\tau (x)$')
	ax_T.legend()

	ax_Tvw.set_title('KINETIC ENERGY DENSITIES | VON WEIZSACKER')
	ax_Tvw.set_xlabel(r'$x \ [u]$')
	ax_Tvw.set_ylabel(r'$\tau_{vW}(x) = \frac{1}{8}\frac{\vert\nabla\rho(x)\vert^{2}}{\rho(x)}$')
	ax_Tvw.legend()

	ax_Ttf.set_title('KINETIC ENERGY DENSITIES | THOMAS FERMI')
	ax_Ttf.set_xlabel(r'$x \ [u]$')
	ax_Ttf.set_ylabel(r'$\tau_{TF}(x) = \frac{\pi^{2}}{24}\rho^{3}(x)$')
	ax_Ttf.legend()
	
	fig_T.tight_layout()
	fig_T.savefig('{}/KINETIC.png'.format(fig_KINETIC))

	fig_Tvw.tight_layout()
	fig_Tvw.savefig('{}/KINETICvw.png'.format(fig_KINETIC))

	fig_Ttf.tight_layout()
	fig_Ttf.savefig('{}/KINETICtf.png'.format(fig_KINETIC))

	plt.close()

	print('- PLOTTING KINETIC ENERGY DENSITIES FINISHED')
	print()

#--------------------------------#
# PLOT ELECTRON LOC. FUNCTION    #
#--------------------------------#

def plotELF(ELF_NS0,ELF_NS,ELF_SC,LOC_CP,Xpts,nXpts,Ao,dX,nuc,temp,fig_ELF):

	print('- PLOTTING ELECTRON LOCALIZATION FUNCTIONS')
	print()

	Xo  = nuc + 1
	Xf  = Xo + 2 * int( (Ao/dX) + 1 )

	fig_ELOC  , ax_ELOC   = plt.subplots()

	ax_ELOC.plot(Xpts[Xo:Xf], ELF_NS0[Xo:Xf], label = r'NS at $T =0K$'              , color='black' )
	ax_ELOC.plot(Xpts[Xo:Xf], ELF_NS[Xo:Xf] , label = r'NS at $T ={}K$'.format(temp), color='blue'  )
	ax_ELOC.plot(Xpts[Xo:Xf], ELF_SC[Xo:Xf] , label = r'SC at $T ={}K$'.format(temp), color='red'   )
	#ax_ELOC.plot(Xpts[Xo:Xf], LOC_CP[Xo:Xf] , label = r'CP at $T ={}K$'.format(temp), color='orange')
	
	ax_ELOC.set_title('ELECTRON LOCALIZATION FUNCTIONS')
	ax_ELOC.set_xlabel(r'$x \ [u]$')
	ax_ELOC.set_ylabel(r'$\eta (x)$')
	ax_ELOC.legend()
	
	fig_ELOC.tight_layout()
	fig_ELOC.savefig('{}/ELF.png'.format(fig_ELF))

	plt.close()

	print('- PLOTTING ELECTRON LOCALIZATION FUNCTIONS FINISHED')
	print()


#--------------------------------#
# WRITE BLOCH WFC PROPERTIES     #
#--------------------------------#

def WRITEWFC(NH,nkpts,k_vec,evals,Ckn,temp,fermi,data_dir):

	BEout = np.zeros((NH*nkpts,9))
	BEout = BEarrange(NH,nkpts,k_vec[:],evals[:,:],Ckn,temp,fermi)
	
	FMT  = '%d', '%d', '%1.4f', '%1.2f', '%1.2f', '%1.4f', '%d', '%1.4f', '%1.4f'
	HEAD = 'BAND    Nk    K    C[A]     C[B]      E       n[NSo]  n[NS]     n[SC]'
	np.savetxt('{}/states.dat'.format(data_dir), BEout, fmt = FMT, delimiter = '    ', header = HEAD)

#--------------------------------#
# WRITE OBSERVABLES ON A GRID    #
#--------------------------------#

def WRITEc(data,Xpts,nXpts,flag,data_dir):

	aux      = np.zeros((nXpts,4))
	aux[:,0] = Xpts[:]
	aux[:,1] = np.real(data[:,0])
	aux[:,2] = np.real(data[:,1])
	aux[:,3] = np.real(data[:,2])
	
	np.savetxt('{}/{}.dat'.format(data_dir,flag), aux)

#--------------------------------#
# WRITE OBSERVABLES ON A GRID    #
#--------------------------------#

def WRITE(data,Xpts,nXpts,flag,data_dir):
	
	aux      = np.zeros((nXpts,2))
	aux[:,0] = Xpts[:]
	aux[:,1] = np.real(data[:])

	np.savetxt('{}/{}.dat'.format(data_dir,flag), aux)

#--------------------------------#
# WRITE OBSERVABLES ON A GRID    #
#--------------------------------#

def WRITE(data,Xpts,nXpts,flag,data_dir):
	
	aux      = np.zeros((nXpts,2))
	aux[:,0] = Xpts[:]
	aux[:,1] = np.real(data[:])

	np.savetxt('{}/{}.dat'.format(data_dir,flag), aux)

#--------------------------------#
# WRITE LOCALIZATION INDICES     #
#--------------------------------#

def WRITE_LIs(LNS0,LNS,LSC,n,flag,data_dir):
	
	aux    = np.zeros((n))
	aux[0] = np.real(LNS0)
	aux[1] = np.real(LNS )
	aux[2] = np.real(LSC )

	np.savetxt('{}/{}.dat'.format(data_dir,flag), aux)

#--------------------------------#
# WRITE DELOCALIZATION INDICES   #
#--------------------------------#

def WRITE_DIs(DNS0,DNS,DSC,NNvec,NN,flag,data_dir):
	
	aux      = np.zeros((NN,4))
	aux[:,0] = NNvec[:]
	aux[:,1] = np.real(DNS0[:])
	aux[:,2] = np.real(DNS[:] )
	aux[:,3] = np.real(DSC[:] )

	np.savetxt('{}/{}.dat'.format(data_dir,flag), aux)

#--------------------------------#
# BLOCH EIG. PROP. COLLECTION    #
#--------------------------------#

def BEarrange(NH,nkpts,k_vec,evals,Ckn,temp,fermi):
	BEout = np.zeros((NH*nkpts,9))
	i = 0
	for band in range(NH):
		for ik, k in enumerate(k_vec[:,0]):
			BEout[i,0] = band 
			BEout[i,1] = ik
			BEout[i,2] = k
			BEout[i,3] = np.real(Ckn[ik][band,0])
			BEout[i,4] = np.real(Ckn[ik][band,1])
			BEout[i,5] = evals[band,ik]
			BEout[i,6] = OccNumNS0(evals[band,ik]) 
			BEout[i,7] = OccNumNS(evals[band,ik],fermi,temp)
			BEout[i,8] = OccNumSC(evals[band,ik],fermi,temp)
			i += 1
	
	return BEout
		
