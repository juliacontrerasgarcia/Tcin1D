#! /usr/bin/env python3

import numpy as np

#-------------------------------------------------#
# ERROR EVALUATION: ORTHOGONALITY OF BLOCH SUMS   #
#-------------------------------------------------#

def errBASIS(Basis,nkpts,NH,Ao,dX,Xpts,nXpts):

	Xo  = int(Ao/dX)
	Xf  = nXpts - int(Ao/dX)	
	err = 0.0
########################################
	#print('ORTHOGONALITY CHECK BLOCH SUMS')
	#print('Xo =',Xo,'Xf =',Xf)
	#print('Xpts[Xo] =',Xpts[Xo],'Xpts[Xf] =',Xpts[Xf])
########################################
	for ik in range(nkpts): 
		for sitei in range(NH):
			for jk in range(ik):
				for sitej in range(NH):
					INT = np.trapz((Basis[ik,sitei,Xo:Xf]*np.conjugate(Basis[jk,sitej,Xo:Xf])),Xpts[Xo:Xf])
					#INT = np.trapz((Basis[ik,sitei,:]*np.conjugate(Basis[jk,sitej,:])),Xpts[:])
#################################################################
					#print(' NUMERICAL <k=',jk,'j=',sitej,'| k=',ik,'j=',sitei,' > = ', INT)
#################################################################
					if ( err < abs( np.real(INT) ) < 0.99999 ):
						err        = abs( np.real(INT) )
						err_ik     = ik
						err_jk     = jk
						err_sitei = sitei 
						err_sitej = sitej
					if ( err < abs( np.imag(INT) ) < 0.99999 ):
						err        = abs( np.imag(INT) )
						err_ik     = ik
						err_jk     = jk
						err_sitei = sitei
						err_sitej = sitej

	return err, err_ik, err_jk, err_sitei, err_sitej
					

#-------------------------------------------------#
# ERROR EVALUATION: ORTHOGONALITY OF BLOCH WFCs   #
#-------------------------------------------------#


def errBWFC(WFC,nkpts,NH,Ao,dX,Xpts,nXpts):

	Xo  = int( Ao/dX )
	Xf  = nXpts - int( Ao/dX )	
	err = 0.0
#################################
	#print('ORTHOGONALITY CHECK WFCs')
################################
	for ik in range(nkpts): 
		for bandi in range(NH):
			for jk in range(ik+1):
				for bandj in range(NH):
					INT = np.trapz((WFC[ik,bandi,Xo:Xf]*np.conjugate(WFC[jk,bandj,Xo:Xf])),Xpts[Xo:Xf])
					#INT = np.trapz((WFC[ik,bandi,:]*np.conjugate(WFC[jk,bandj,:])),Xpts[:])
###############################################################################################################
					#print(' NUMERICAL <k=',jk,'n=',bandj,'| k=',ik,'n=',bandi,' > = ', INT)
###############################################################################################################
					if ( err < abs( np.real(INT) ) < 0.99999 ):
						err        = abs( np.real(INT) )
						err_ik     = ik
						err_jk     = jk
						err_bandi = bandi 
						err_bandj = bandj
					if ( err < abs( np.imag(INT) ) < 0.99999 ):
						err        = abs( np.imag(INT) )
						err_ik     = ik
						err_jk     = jk
						err_bandi = bandi
						err_bandj = bandj

	return err, err_ik, err_jk, err_bandi, err_bandj
