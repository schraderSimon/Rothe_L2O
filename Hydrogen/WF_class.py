import os
nthread="6"
os.environ["OMP_NUM_THREADS"] = nthread# export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = nthread # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] =nthread # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = nthread # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = nthread# export NUMEXPR_NUM_THREADS=6

import numpy as np
import numpy
import matplotlib.pyplot as plt
from numpy import einsum
from numpy import exp, pi, sqrt
#from einsumt import einsumt as einsum
from scipy import linalg
from scipy.linalg import det, inv,eig
from scipy.special import erf, erfi, dawsn
from scipy.optimize import basinhopping
import sys
import time
import warnings
import h5py
import os
from numpy import nan_to_num as ntn
from numba import jit
from utils import *
precision="normal"
if precision=="single":
    dtype_float=numpy.float32
    dtype_complex=numpy.complex64
elif precision=="extended":
    dtype_float=np.float128
    dtype_complex=np.complex256
else:
    dtype_float=np.float64
    dtype_complex=np.complex128
warnings.filterwarnings("ignore")
sqrt_pi_inv=1/sqrt(pi)
def getExpIntermediates_numba(clk,ylk,glk,gaussLinear,gaussExponent):
    len_clk=len(clk)
    invR=np.zeros((len_clk,len_clk),dtype=dtype_complex)
    dinvRdy=np.zeros((len_clk,len_clk),dtype=dtype_complex)
    dinvRdc=np.zeros((len_clk,len_clk),dtype=dtype_complex)
    d2invRdc2=np.zeros((len_clk,len_clk),dtype=dtype_complex)
    d2invRdy2=np.zeros((len_clk,len_clk),dtype=dtype_complex)
    d2invRdcdy=np.zeros((len_clk,len_clk),dtype=dtype_complex)
    invR_int3 = ylk**2
    for i in range(len(gaussLinear)):
        ge_i=gaussExponent[i]
        gl_i=gaussLinear[i]
        invR_int2 = 1/(clk + ge_i)
        invR_int5 = pi**1.5*exp(glk + invR_int2*0.25*invR_int3)/sqrt(clk + ge_i)**3
        invR=invR+gl_i*(invR_int5)
        dinvRdc+=+gl_i*invR_int5*(-3*invR_int2/2 - 0.25*invR_int3*invR_int2**2)
        dinvRdy+=+gl_i*(0.5*invR_int2*invR_int5*ylk)
        d2invRdc2+=+gl_i*invR_int5*(1.25*invR_int2**3*invR_int3 + 0.0625*invR_int2**4*invR_int3**2 + 15*invR_int2**2/4)
        d2invRdcdy+=+gl_i*invR_int5*(-1.25*invR_int2**2*ylk - 0.125*invR_int2**3*invR_int3*ylk)
        d2invRdy2+=+gl_i*invR_int5*(0.25*invR_int3*invR_int2**2 + 0.5*invR_int2)
    return invR,dinvRdc,dinvRdy,d2invRdc2,d2invRdcdy,d2invRdy2
def getExpsquaredIntermediates_numba(clk,ylk,glk,gaussLinear,gaussExponent):
    len_clk=len(clk)
    invRsq=np.zeros((len_clk,len_clk),dtype=dtype_complex)
    dinvRsqdc=np.zeros((len_clk,len_clk),dtype=dtype_complex)
    dinvRsqdy=np.zeros((len_clk,len_clk),dtype=dtype_complex)
    for i in range(len(gaussLinear)):
        ge_i=gaussExponent[i]
        gl_i=gaussLinear[i]
        invRsq_int2 = (clk + ge_i)**(-1)
        invRsq_int5 = pi**1.5*exp(glk + invRsq_int2*0.25*ylk**2)/sqrt(clk + ge_i)**3
        invRsq+=gl_i*(invRsq_int5)
        dinvRsqdc+=gl_i*(-1.5*invRsq_int2*invRsq_int5 -  0.25*ylk**2*invRsq_int5*(invRsq_int2**2))
        dinvRsqdy+=+gl_i*( 0.5*invRsq_int2*invRsq_int5*ylk)
    return invRsq,dinvRsqdc,dinvRsqdy

class WF():
    def __init__(self,params,basis_coefficients,potential_params=None,fieldFunc=None,t=0,beta_ovlp=0,gauge="length"):
        """
        WF for a system where the potential is written as a linear combination of Gaussians.
        """
        self.params=params

        self.t=0
        if fieldFunc is not None:
            self.fieldFunc=fieldFunc
        else:
            def fieldFunc(t=0):
                return 0
            self.fieldFunc=fieldFunc
        self.potential_params=potential_params
        num_basis_funcs=self.num_basis_funcs=len(basis_coefficients)
        self.coefficients=np.array(basis_coefficients)#,dtype=np.complex128)
        self.a=np.array(params[::4],dtype=dtype_float)
        self.b=np.array(params[1::4],dtype=dtype_float)
        self.mu=np.array(params[2::4],dtype=dtype_float)
        self.q=np.array(params[3::4],dtype=dtype_float)
        self.c=self.a**2+1j*self.b
        self.M=M=(2*self.c*self.mu+1j*self.q)

        self.setUpIntermediates()
        #self.normalize()
        self.setUp_param_derivs()
        self.beta_ovlp=beta_ovlp
        self.gauge=gauge
    def normalize(self):
        norm=np.conj(self.coefficients).T@self.overlap_normalized@self.coefficients
        self.coefficients=self.coefficients/sqrt(norm)
    def setUpIntermediates(self):
        pass #This is the only difference between the different type of WFs
    def setUp_param_derivs(self):
        """
        Input: None
        Returns: A 4*4*num_gauss tensor T, containing the derivatives of intermediate_(l)k[i] with respect to input_param n for Gaussian k
        Elg. T[i,j,k] contains the derivative of the i'th element in [clk,ylk,g_lk,M_k] w.r.t. the j'th element in [a_k,b_k,mu_k,q_k]
        """
        T=np.zeros((4,4,self.num_basis_funcs),dtype=dtype_complex)
        T[0,0,:]=2*self.a #dcda
        T[0,1,:]=1j #dcdb
        T[0,2,:]=0 #dcdmu
        T[0,3,:]=0 #dcdq
        T[1,0,:]= 4*self.a*self.mu #dyda
        T[1,1,:]= 2*1j*self.mu#dydb
        T[1,2,:]= 2*self.c#dydmu
        T[1,3,:]=1j #dydq
        T[2,0,:]=-2*self.a*self.mu**2 #dgda
        T[2,1,:]= -1j*self.mu**2#dgdb
        T[2,2,:]= -1j*self.q-2*self.c*self.mu#dgdmu
        T[2,3,:]=-1j*self.mu #dgdq
        T[3,0,:]= 4*self.a*self.mu#dMda
        T[3,1,:]= 2*1j*self.mu#dMdb
        T[3,2,:]=2*self.c #dMdmu
        T[3,3,:]=1j #dMdq
        self.T=T
    def calculate_overlap_normalized(self,start_index_WFnew):
        overlap=self.overlap[start_index_WFnew:,start_index_WFnew:].copy()
        for i in range(len(overlap)):
            for j in range(len(overlap)):
                if i==j:
                    continue
                overlap[i,j]=overlap[i,j]/sqrt(overlap[i,i]*overlap[j,j])
        for i in range(len(overlap)):
            overlap[i,i]=1
        return overlap
    def calculate_diag_derivs(self):
        deriv=np.zeros(self.num_basis_funcs*4,dtype=np.complex128)
        deriv[0::4]=-3/self.a
        return deriv
    def calculate_all_overlap_deriv(self,start_index_WFnew):
        """
        This calculates the overlap of the derivative matrix.
        Here is what it processes:
        Returns the matrix deriv.shape = [num_gaussians_tot,num_gaussians_newWF,num_params_per_gaussians]
        The element deriv[:,j,k] is the derivative of of the column of the overlap matrix, e.g. dS[:,j]/dparam_(4*j+k)
        (for diagonal elements, it is technically not THAT, it is the inner product of the <:|d/dparm g_j>)
        """
        si=start_index_WFnew*4
        si_div4=si//4
        diag_derivs=self.calculate_diag_derivs()
        diag_derivs=np.reshape(diag_derivs[si:],[(len(diag_derivs)-si)//4,4]).T
        all_param_derivs=self.T[:,:,si_div4:]
        deriv_firstPart=einsum("abc,cb->abc",self.dOdc[:, si_div4:, np.newaxis],all_param_derivs[0,:,:])
        deriv_firstPart+=einsum("abc,cb->abc",self.dOdy[:, si_div4:, np.newaxis],all_param_derivs[1,:,:])
        deriv_firstPart+=einsum("abc,cb->abc",self.dOdg[:, si_div4:, np.newaxis],all_param_derivs[2,:,:])
        deriv_firstPart=deriv_firstPart/self.sqrt_overlap_mat[:, si_div4:, np.newaxis]
        deriv_secondPart=-0.5*einsum("abc,cb->abc",self.overlap_normalized[:, si_div4:, np.newaxis],diag_derivs)
        return deriv_firstPart+deriv_secondPart
    def calculate_overlap_deriv(self,param_index):
        """Returns the "slice" d/dparam <phi_l|phi_k> with fixed k (depending on param_index) and flexible l
        """
        diag_derivs=self.calculate_diag_derivs()
        type=param_index%4 #a,b,mu or q
        gauss_index=param_index//4
        param_derivs=self.T[:,type,gauss_index]
        deriv_firstPart=self.dOdc[:,gauss_index]*param_derivs[0]+self.dOdy[:,gauss_index]*param_derivs[1]+self.dOdg[:,gauss_index]*param_derivs[2]
        diag_term=deriv_firstPart[gauss_index]
        deriv_firstPart=deriv_firstPart/self.sqrt_overlap_mat[:,gauss_index]
        deriv_secondPart=-0.5*self.overlap_normalized[:,gauss_index]*diag_derivs[param_index]
        return deriv_firstPart+deriv_secondPart
    def calculate_all_H_deriv(self,start_index_WFnew,t=0):
        si=start_index_WFnew*4
        si_div4=si//4
        diag_derivs=self.calculate_diag_derivs()
        diag_derivs=np.reshape(diag_derivs[si:],[(len(diag_derivs)-si)//4,4]).T
        all_param_derivs=np.swapaxes(self.T[:,:,si_div4:],1,2)
        #all_param_derivs=self.T[:,:,si_div4:]
        oneOverR_deriv=self.dinvRdc[:, si_div4:, np.newaxis]*all_param_derivs[0,:,:]
        oneOverR_deriv+=self.dinvRdy[:, si_div4:, np.newaxis]*all_param_derivs[1,:,:]
        oneOverR_deriv+=self.invR[:, si_div4:, np.newaxis]*all_param_derivs[2,:,:]

        c=self.c
        M=self.M
        cs=c**2
        cM=c*M
        M6c=M**2-6*c
        dck_dparam=self.T[0,:,si_div4:]
        dM_dparam=self.T[3,:,si_div4:]

        dLaplacedc=-4*(cs*self.d2Odc2+cM*self.d2Odcdy)+M6c*self.dOdc
        dLaplacedy=-4*(cs*self.d2Odcdy+cM*self.d2Ody2)+M6c*self.dOdy
        dLaplacedg=-4*(cs*self.dOdc+cM*self.dOdy)+M6c*self.overlap
        laplace_derivs=dLaplacedc[:, si_div4:, np.newaxis]*all_param_derivs[0,:,:]
        laplace_derivs+=dLaplacedy[:, si_div4:, np.newaxis]*all_param_derivs[1,:,:]
        laplace_derivs+=dLaplacedg[:, si_div4:, np.newaxis]*all_param_derivs[2,:,:]
        laplace_derivs+=-8*self.dOdc[:, si_div4:, np.newaxis]*(c[si_div4:]*dck_dparam).T
        laplace_derivs+=-4*self.dOdy[:, si_div4:, np.newaxis]*(c[si_div4:]*dM_dparam+M[si_div4:]*dck_dparam).T
        laplace_derivs+=self.overlap[:, si_div4:, np.newaxis]*(-6*dck_dparam+2*M[si_div4:]*dM_dparam).T
        if self.gauge=="length":
            z_deriv =self.d2Odcdy[:, si_div4:, np.newaxis]*all_param_derivs[0,:,:]
            z_deriv+=self.d2Ody2[:, si_div4:, np.newaxis]*all_param_derivs[1,:,:]
            z_deriv+=self.dOdy[:, si_div4:, np.newaxis]*all_param_derivs[2,:,:]
        elif self.gauge=="velocity":
            dpdc=M*self.dOdc-2*c*self.d2Odcdy
            dpdy=M*self.dOdy-2*c*self.d2Ody2
            dpdg=M*self.overlap-2*c*self.dOdy
            ddz_deriv=dpdc[:, si_div4:, np.newaxis]*all_param_derivs[0,:,:]
            ddz_deriv+=dpdy[:, si_div4:, np.newaxis]*all_param_derivs[1,:,:]
            ddz_deriv+=dpdg[:, si_div4:, np.newaxis]*all_param_derivs[2,:,:]
            ddz_deriv+=-2*self.dOdy[:, si_div4:, np.newaxis]*dck_dparam.T
            ddz_deriv+=self.overlap[:, si_div4:, np.newaxis]*dM_dparam.T

        if self.gauge=="length":
            deriv_firstPart=(-0.5*laplace_derivs-oneOverR_deriv+self.fieldFunc(t)*z_deriv)/np.expand_dims(self.sqrt_overlap_mat[:, si_div4:], axis=-1)

        elif self.gauge=="velocity":
            deriv_firstPart=(-0.5*laplace_derivs-oneOverR_deriv-1j*self.fieldFunc(t)*ddz_deriv)/np.expand_dims(self.sqrt_overlap_mat[:, si_div4:], axis=-1)
        deriv_secondPart=-0.5*self.calculate_H_field(t)[:, si_div4:, np.newaxis]*diag_derivs.T
        return deriv_firstPart+deriv_secondPart

    def calculate_all_Hsquared_deriv(self,start_index_WFnew,t=0):
        si=start_index_WFnew*4
        si_div4=si//4
        diag_derivs=self.calculate_diag_derivs()
        diag_derivs=np.reshape(diag_derivs[si:],[(len(diag_derivs)-si)//4,4]).T
        all_param_derivs=np.swapaxes(self.T[:,:,si_div4:],1,2)

        oneOverSquared_derivs=einsum("abc,bc->abc",self.dinvRsqdc[:, si_div4:, np.newaxis],all_param_derivs[0,:,:])
        oneOverSquared_derivs+=einsum("abc,bc->abc",self.dinvRsqdy[:, si_div4:, np.newaxis],all_param_derivs[1,:,:])
        oneOverSquared_derivs+=einsum("abc,bc->abc",self.invRsq[:, si_div4:, np.newaxis],all_param_derivs[2,:,:])

        c=self.c
        q=self.q
        mu=self.mu
        cs=c**2

        M=self.M
        cM=c*self.M
        M6c=M**2-6*c
        cg=c
        Mg=c
        cl_c=np.conj(c)
        cl_c2=cl_c**2
        Ml_c=np.conj(M)
        Mg_cg=M6c
        Mlc_clc=(Ml_c**2-6*cl_c)
        cl_c2cg2=cl_c2*cg**2
        cl_ccg2Ml_c=cl_c*cg**2*Ml_c
        cl_c2cgMg=cl_c2*cg*Mg
        cg2Mlcclc=cg**2*Mlc_clc
        cl_ccgMgMl_c=cl_c*cg*Mg*Ml_c
        cl_c2Mg_cg=cl_c2*Mg_cg
        cgMl_c_clc_Mg=cg*Mlc_clc*Mg
        cl_cMg_cgMl_c=cl_c*Mg_cg*Ml_c
        cl_c2c2=np.outer(cl_c2,cs)
        cl_c_Ml_cc2=np.outer(cl_c*Ml_c,cs)
        cl_c2_cM=np.outer(cl_c2,cM)
        clcmlccm=np.outer(cl_c*Ml_c,cM)
        cl_c2Mgcg=np.outer(cl_c2,Mg_cg)
        Mlcclcc2=np.outer(Mlc_clc,cs)
        Mlc_clccM=np.outer(Mlc_clc,cM)
        clc_Mlc_Mc=np.outer(cl_c*Ml_c,Mg_cg)
        Mlc_clc_M_c=np.outer(Mlc_clc,Mg_cg)

        dn4dc=16*cl_c2c2*self.d3Odc3
        dn4dc+=16*cl_c_Ml_cc2*self.d3Odc2dy
        dn4dc+=16*cl_c2_cM*self.d3Odc2dy
        dn4dc+=16*clcmlccm*self.d3Odcdy2
        dn4dc+=-4*Mlcclcc2*self.d2Odc2
        dn4dc+=-4*cl_c2Mgcg*self.d2Odc2
        dn4dc+=-4*Mlc_clccM*self.d2Odcdy
        dn4dc+=-4*clc_Mlc_Mc*self.d2Odcdy
        dn4dc+=Mlc_clc_M_c*self.dOdc
        dn4dy=16*cl_c2c2*self.d3Odc2dy
        dn4dy+=16*cl_c_Ml_cc2*self.d3Odcdy2
        dn4dy+=16*cl_c2_cM*self.d3Odcdy2
        dn4dy+=16*clcmlccm*self.d3Ody3
        dn4dy+=-4*Mlcclcc2*self.d2Odcdy
        dn4dy+=-4*cl_c2Mgcg*self.d2Odcdy
        dn4dy+=-4*Mlc_clccM*self.d2Ody2
        dn4dy+=-4*clc_Mlc_Mc*self.d2Ody2
        dn4dy+=Mlc_clc_M_c*self.dOdy
        nabla_4=16*cl_c2c2*self.d2Odc2
        nabla_4+=16*cl_c_Ml_cc2*self.d2Odcdy
        nabla_4+=16*cl_c2_cM*self.d2Odcdy
        nabla_4+=16*clcmlccm*self.d2Ody2
        nabla_4+=-4*Mlcclcc2*self.dOdc
        nabla_4+=-4*cl_c2Mgcg*self.dOdc
        nabla_4+=-4*Mlc_clccM*self.dOdy
        nabla_4+=-4*clc_Mlc_Mc*self.dOdy
        nabla_4+=Mlc_clc_M_c*self.overlap
        dck_dparam=self.T[0,:,si_div4:]
        dM_dparam=self.T[3,:,si_div4:]
        dOdc_rep=self.dOdc[:, si_div4:, np.newaxis]
        dOdy_rep=self.dOdy[:, si_div4:, np.newaxis]
        o_rep=self.overlap[:, si_div4:, np.newaxis]
        d2Ody2_rep=self.d2Ody2[:, si_div4:, np.newaxis]
        d2Odcdy_rep=self.d2Odcdy[:, si_div4:, np.newaxis]

        laplaceSquared_deriv=dn4dc[:, si_div4:, np.newaxis]*all_param_derivs[0,:,:]
        laplaceSquared_deriv+=dn4dy[:, si_div4:, np.newaxis]*all_param_derivs[1,:,:]
        laplaceSquared_deriv+=nabla_4[:, si_div4:, np.newaxis]*all_param_derivs[2,:,:]
        laplaceSquared_deriv += 32 * np.einsum("igT,i,g,Tg->igT", self.d2Odc2[:, si_div4:, np.newaxis], cl_c2, c[si_div4:], dck_dparam)
        laplaceSquared_deriv+=32*einsum("igT,i,g,Tg->igT",d2Odcdy_rep,cl_c*Ml_c,c[si_div4:],dck_dparam)
        laplaceSquared_deriv+=16*einsum("igT,i,g,Tg->igT",d2Odcdy_rep,cl_c2,M[si_div4:],dck_dparam)
        laplaceSquared_deriv+=16*einsum("igT,i,g,Tg->igT",d2Odcdy_rep,cl_c2,c[si_div4:],dM_dparam)
        laplaceSquared_deriv+=16*einsum("igT,i,g,Tg->igT",d2Ody2_rep,cl_c*Ml_c,c[si_div4:],dM_dparam)
        laplaceSquared_deriv+=16*einsum("igT,i,g,Tg->igT",d2Ody2_rep,cl_c*Ml_c,M[si_div4:],dck_dparam)
        laplaceSquared_deriv+=-8*einsum("igT,i,g,Tg->igT",dOdc_rep,Mlc_clc,c[si_div4:],dck_dparam)
        laplaceSquared_deriv+=-4*einsum("igT,i,Tg->igT",dOdc_rep,cl_c2,-6*dck_dparam)
        laplaceSquared_deriv+=-4*einsum("igT,i,g,Tg->igT",dOdc_rep,cl_c2,M[si_div4:],2*dM_dparam)
        laplaceSquared_deriv+=-4*einsum("igT,i,Tg->igT",dOdy_rep,cl_c*Ml_c,-6*dck_dparam)
        laplaceSquared_deriv+=-4*einsum("igT,i,g,Tg->igT",dOdy_rep,cl_c*Ml_c,M[si_div4:],2*dM_dparam)
        laplaceSquared_deriv+=-4*einsum("igT,i,g,Tg->igT",dOdy_rep,Mlc_clc,c[si_div4:],dM_dparam)
        laplaceSquared_deriv+=-4*einsum("igT,i,g,Tg->igT",dOdy_rep,Mlc_clc,M[si_div4:],dck_dparam)
        laplaceSquared_deriv+=einsum("igT,i,Tg->igT",o_rep,Mlc_clc,-6*dck_dparam)
        laplaceSquared_deriv+=einsum("igT,i,g,Tg->igT",o_rep,Mlc_clc,M[si_div4:],2*dM_dparam)
        drLaplacedc=-4*cs*self.d2invRdc2-4*cM*self.d2invRdcdy+(M6c)*self.dinvRdc
        drLaplacedy=-4*cs*self.d2invRdcdy-4*cM*self.d2invRdy2+(M6c)*self.dinvRdy
        drLaplacedg=-4*cs*self.dinvRdc-4*cM*self.dinvRdy+(M6c)*self.invR

        rlaplace_derivs=einsum("abc,bc->abc",drLaplacedc[:, si_div4:, np.newaxis],all_param_derivs[0,:,:])
        rlaplace_derivs+=einsum("abc,bc->abc",drLaplacedy[:, si_div4:, np.newaxis],all_param_derivs[1,:,:])
        rlaplace_derivs+=einsum("abc,bc->abc",drLaplacedg[:, si_div4:, np.newaxis],all_param_derivs[2,:,:])
        rlaplace_derivs+=einsum("bac,bc->abc",np.conj(drLaplacedc[ si_div4:,:, np.newaxis]),all_param_derivs[0,:,:])
        rlaplace_derivs+=einsum("bac,bc->abc",np.conj(drLaplacedy[ si_div4:, :, np.newaxis]),all_param_derivs[1,:,:])
        rlaplace_derivs+=einsum("bac,bc->abc",np.conj(drLaplacedg[ si_div4:, :, np.newaxis]),all_param_derivs[2,:,:])
        rlaplace_derivs+=einsum("abc,cb->abc",self.dinvRdc[:, si_div4:, np.newaxis],-8*c[si_div4:]*dck_dparam)
        rlaplace_derivs+=-4*einsum("abc,cb->abc",self.dinvRdy[:, si_div4:, np.newaxis],(c[si_div4:]*dM_dparam+M[si_div4:]*dck_dparam))
        rlaplace_derivs+=einsum("abc,cb->abc",self.invR[:, si_div4:, np.newaxis],-6*dck_dparam+2*M[si_div4:]*dM_dparam)
        if self.gauge=="length":
            pass
            dzLaplacedc=-4*cs*self.d3Odc2dy-4*cM*self.d3Odcdy2+(M6c)*self.d2Odcdy
            dzLaplacedy=-4*cs*self.d3Odcdy2-4*cM*self.d3Ody3+(M6c)*self.d2Ody2
            dzLaplacedg=-4*cs*self.d2Odcdy-4*cM*self.d2Ody2+(M6c)*self.dOdy
            zlaplace_derivs=einsum("abc,bc->abc",dzLaplacedc[:, si_div4:, np.newaxis],all_param_derivs[0,:,:])
            zlaplace_derivs+=einsum("abc,bc->abc",dzLaplacedy[:, si_div4:, np.newaxis],all_param_derivs[1,:,:])
            zlaplace_derivs+=einsum("abc,bc->abc",dzLaplacedg[:, si_div4:, np.newaxis],all_param_derivs[2,:,:])
            zlaplace_derivs+=einsum("bac,bc->abc",np.conj(dzLaplacedc[ si_div4:,:, np.newaxis]),all_param_derivs[0,:,:])
            zlaplace_derivs+=einsum("bac,bc->abc",np.conj(dzLaplacedy[ si_div4:, :, np.newaxis]),all_param_derivs[1,:,:])
            zlaplace_derivs+=einsum("bac,bc->abc",np.conj(dzLaplacedg[ si_div4:, :, np.newaxis]),all_param_derivs[2,:,:])
            zlaplace_derivs+=einsum("abc,cb->abc",d2Odcdy_rep,-8*c[si_div4:]*dck_dparam)
            zlaplace_derivs+=-4*einsum("abc,cb->abc",d2Ody2_rep,(c[si_div4:]*dM_dparam+M[si_div4:]*dck_dparam))
            zlaplace_derivs+=einsum("abc,cb->abc",dOdy_rep,-6*dck_dparam+2*M[si_div4:]*dM_dparam)

            zSquared_derivs=einsum("abc,bc->abc",self.d3Odcdy2[:, si_div4:, np.newaxis],all_param_derivs[0,:,:])
            zSquared_derivs+=einsum("abc,bc->abc",self.d3Ody3[:, si_div4:, np.newaxis],all_param_derivs[1,:,:])
            zSquared_derivs+=einsum("abc,bc->abc",self.d2Ody2[:, si_div4:, np.newaxis],all_param_derivs[2,:,:])

            zOverR_derivs=einsum("abc,bc->abc",self.d2invRdcdy[:, si_div4:, np.newaxis],all_param_derivs[0,:,:])
            zOverR_derivs+=einsum("abc,bc->abc",self.d2invRdy2[:, si_div4:, np.newaxis],all_param_derivs[1,:,:])
            zOverR_derivs+=einsum("abc,bc->abc",self.dinvRdy[:, si_div4:, np.newaxis],all_param_derivs[2,:,:])

            deriv_firstPart=(+oneOverSquared_derivs+0.25*laplaceSquared_deriv+0.5*rlaplace_derivs+self.fieldFunc(t)**2*zSquared_derivs+self.fieldFunc(t)*(-0.5*zlaplace_derivs-2*zOverR_derivs))/self.sqrt_overlap_mat[:, si_div4:, np.newaxis]
        elif self.gauge=="velocity":
            z2_terms=4*cs #Terms that have a <z^2> contribution
            ovlp_terms=-2*self.c+M**2
            z_terms=-4*self.c*M
            dp2dc   =ovlp_terms*self.dOdc+z2_terms*self.d3Odcdy2+z_terms*self.d2Odcdy
            dp2dy   =ovlp_terms*self.dOdy+z2_terms*self.d3Ody3+z_terms*self.d2Ody2
            dp2dg=p2=ovlp_terms*self.overlap+z2_terms*self.d2Ody2+z_terms*self.dOdy
            p2_derivs=dp2dc[:, si_div4:, np.newaxis]*all_param_derivs[0,:,:]
            p2_derivs+=dp2dy[:, si_div4:, np.newaxis]*all_param_derivs[1,:,:]
            p2_derivs+=dp2dg[:, si_div4:, np.newaxis]*all_param_derivs[2,:,:]
            p2_derivs+=8*self.d2Ody2[:, si_div4:, np.newaxis]*(c[si_div4:]*dck_dparam).T #explicit c derivative
            p2_derivs+=-4*self.dOdy[:, si_div4:, np.newaxis]*(c[si_div4:]*dM_dparam+M[si_div4:]*dck_dparam).T
            p2_derivs+=self.overlap[:, si_div4:, np.newaxis]*(-2*dck_dparam+2*M[si_div4:]*dM_dparam).T
            #Derive laplace times p
            ddz_invR=0
            dp_invRdg=self.M*self.invR-2*self.c*self.dinvRdy
            dp_invRdc=self.M*self.dinvRdc-2*self.c*self.d2invRdcdy
            dp_invRdy=self.M*self.dinvRdy-2*self.c*self.d2invRdy2
            dp_invr_derivs=einsum("abc,bc->abc",dp_invRdc[:, si_div4:, np.newaxis],all_param_derivs[0,:,:])
            dp_invr_derivs+=einsum("abc,bc->abc",dp_invRdy[:, si_div4:, np.newaxis],all_param_derivs[1,:,:])
            dp_invr_derivs+=einsum("abc,bc->abc",dp_invRdg[:, si_div4:, np.newaxis],all_param_derivs[2,:,:])
            dp_invr_derivs-=einsum("bac,bc->abc",np.conj(dp_invRdc[ si_div4:,:, np.newaxis]),all_param_derivs[0,:,:])
            dp_invr_derivs-=einsum("bac,bc->abc",np.conj(dp_invRdy[ si_div4:,:, np.newaxis]),all_param_derivs[1,:,:])
            dp_invr_derivs-=einsum("bac,bc->abc",np.conj(dp_invRdg[ si_div4:,:, np.newaxis]),all_param_derivs[2,:,:])
            dp_invr_derivs+=einsum("abc,cb->abc",self.dinvRdy[:, si_div4:, np.newaxis],-2*dck_dparam)
            dp_invr_derivs+=einsum("abc,cb->abc",self.invR[:, si_div4:, np.newaxis],dM_dparam)

            #Last one: laplacian times p... At least they commute lol.
            Mc_c=np.conj(M)**2-6*np.conj(c)
            cs_c=np.conj(c)**2
            dnabla_ddzdc=-8*np.einsum("k,l,lk->lk",c,cs_c,-self.d3Odc2dy)+8*np.einsum("k,l,lk->lk",c,np.conj(c*M),self.d3Odcdy2)-2*np.einsum("k,l,lk->lk",c,Mc_c,self.d2Odcdy)
            dnabla_ddzdc+=-4*np.einsum("k,l,lk->lk",M,cs_c,self.d2Odc2)-4*np.einsum("k,l,lk->lk",M,np.conj(c*M),self.d2Odcdy)+np.einsum("k,l,lk->lk",M,Mc_c,self.dOdc)
            dnabla_ddzdy=-8*np.einsum("k,l,lk->lk",c,cs_c,-self.d3Odcdy2)+8*np.einsum("k,l,lk->lk",c,np.conj(c*M),self.d3Ody3)-2*np.einsum("k,l,lk->lk",c,Mc_c,self.d2Ody2)
            dnabla_ddzdy+=-4*np.einsum("k,l,lk->lk",M,cs_c,self.d2Odcdy)-4*np.einsum("k,l,lk->lk",M,np.conj(c*M),self.d2Ody2)+np.einsum("k,l,lk->lk",M,Mc_c,self.dOdy)


            nabla_ddz=-8*np.einsum("k,l,lk->lk",c,cs_c,-self.d2Odcdy)+8*np.einsum("k,l,lk->lk",c,np.conj(c*M),self.d2Ody2)-2*np.einsum("k,l,lk->lk",c,Mc_c,self.dOdy)
            nabla_ddz+=-4*np.einsum("k,l,lk->lk",M,cs_c,self.dOdc)-4*np.einsum("k,l,lk->lk",M,np.conj(c*M),self.dOdy)+np.einsum("k,l,lk->lk",M,Mc_c,self.overlap)
            laplace_ddz_deriv=dnabla_ddzdc[:, si_div4:, np.newaxis]*all_param_derivs[0,:,:]
            laplace_ddz_deriv+=dnabla_ddzdy[:, si_div4:, np.newaxis]*all_param_derivs[1,:,:]
            laplace_ddz_deriv+=nabla_ddz[:, si_div4:, np.newaxis]*all_param_derivs[2,:,:]

            laplace_ddz_deriv+=-8*einsum("igT,i,Tg->igT",-d2Odcdy_rep,cs_c,dck_dparam) #C deriv of first
            laplace_ddz_deriv+=8*einsum("igT,i,Tg->igT",d2Ody2_rep,np.conj(c*M),dck_dparam) #C deriv of first
            laplace_ddz_deriv+=-2*einsum("igT,i,Tg->igT",dOdy_rep,Mc_c,dck_dparam)
            laplace_ddz_deriv+=-4*einsum("igT,i,Tg->igT",dOdc_rep,cs_c,dM_dparam)
            laplace_ddz_deriv+=-4*einsum("igT,i,Tg->igT",dOdy_rep,np.conj(c*M),dM_dparam)
            laplace_ddz_deriv+=einsum("igT,i,Tg->igT",o_rep,Mc_c,dM_dparam)



            deriv_firstPart=(+oneOverSquared_derivs+0.25*laplaceSquared_deriv+0.5*rlaplace_derivs-self.fieldFunc(t)**2*p2_derivs-1j*self.fieldFunc(t)*(-laplace_ddz_deriv-dp_invr_derivs))/self.sqrt_overlap_mat[:, si_div4:, np.newaxis]

        deriv_secondPart=-0.5*einsum("abc,cb->abc",self.calculate_Hsquared_field(t)[:, si_div4:, np.newaxis],diag_derivs)
        return deriv_firstPart+deriv_secondPart


    def evaluate_WF(self,R):
        returnval=0
        for i in range(len(self.coefficients)):
            ci=self.coefficients[i]/sqrt(self.overlap[i,i])
            params_i=self.params[i*4:i*4+4]
            mu=np.array([0,0,params_i[2]])
            q=np.array([0,0,params_i[3]])
            R=np.array(R)
            c=-params_i[0]**2+1j*params_i[1]
            returnval+=ci*np.exp(c*(R-mu).T@(R-mu)+1j*q.T@(R-mu))
        return returnval

    def c_alpha_deriv(self,param_index,start_index_WFnew,t,h=1e-3):
        self.rothe_optimal_c(start_index_WFnew,t,h)
        S_deriv=self.calculate_S_mat_deriv(param_index,start_index_WFnew,t,h)
        rho=(self.rho)
        S=self.Smat
        rho_deriv=(self.calculate_rho_vec_deriv(param_index,start_index_WFnew,t,h)) #This one is wrong somehow?
        c=self.opt_c
        c_deriv=solve(S,-S_deriv@c+rho_deriv)
        return c_deriv

    def calculate_z(self):
        return self.dOdy
    def calculate_ddz(self):
        return self.M*self.overlap-2*self.c*self.dOdy
    def calculate_d2dz2(self):
        cs=self.c**2
        M=self.M
        z2_terms=4*cs #Terms that have a <z^2> contribution
        ovlp_terms=-2*self.c+M**2
        z_terms=-4*self.c*M
        p2_exp_mat=ovlp_terms*self.overlap+z2_terms*self.d2Ody2+z_terms*self.dOdy
        return p2_exp_mat
    def calculate_ddz_laplacian(self):
        #This should be inspired by zLaplacian
        #This calculates laplacian*d/dz. Due to commuting operators, it is NOT necessary to calculate d/dz*laplacian explicitely.
        clk,ylk=self.clk,self.ylk
        c=self.c
        M=self.M
        q=self.q
        mu=self.mu
        overlap=self.overlap
        d2Odcdy=self.d2Odcdy
        d2Ody2=self.d2Ody2
        dOdy=self.dOdy
        dOdc=self.dOdc
        Mc_c=np.conj(M)**2-6*np.conj(c)
        cs_c=np.conj(c)**2
        returnval=-8*np.einsum("k,l,lk->lk",c,cs_c,-d2Odcdy)+8*np.einsum("k,l,lk->lk",c,np.conj(c*M),d2Ody2)-2*np.einsum("k,l,lk->lk",c,Mc_c,dOdy)
        returnval+=-4*np.einsum("k,l,lk->lk",M,cs_c,dOdc)-4*np.einsum("k,l,lk->lk",M,np.conj(c*M),dOdy)+np.einsum("k,l,lk->lk",M,Mc_c,overlap)
        return returnval
    def calculate_ddz_overR(self):
        vals=self.M*self.invR-2*self.c*self.dinvRdy
        return vals-np.conj(vals.T)
    #Similarly, I need to update the derivatives for H and Hsquared... This will not be so easy. But it CAN be done. I think this can have an if else for the gauge...
    def calculate_laplacian_over_r(self):
        c=self.c
        M=self.M
        invR=self.invR
        dinvRdc=self.dinvRdc
        dinvRdy=self.dinvRdy
        first= -4*c**2*dinvRdc-4*c*M*dinvRdy+(M**2-6*c)*invR
        second= (np.conj(-4*c**2)*dinvRdc.T-np.conj(4*c*M)*dinvRdy.T+np.conj(M**2-6*c)*invR.T).T
        return first+second

    def calculate_zSquared(self):
        return self.d2Ody2
    def calculate_r(self):
        return self.invR
    def calculate_rsquared(self):
        return self.invRsq
    def calculate_zOverr(self):
        """
        Calculate the expectation value z/r
        """
        return self.dinvRdy
    def calculate_laplacian(self):
        clk,ylk=self.clk, self.ylk
        M=self.M
        c=self.c
        overlap=self.overlap
        dOdc=self.dOdc
        dOdy=self.dOdy
        laplace=-4*c**2*dOdc-4*c*M*dOdy+(M**2-6*c)*overlap
        return laplace
    def calculate_z_Laplacian(self):
        clk,ylk=self.clk,self.ylk
        c=self.c
        M=self.M
        overlap=self.overlap
        d2Odcdy=self.d2Odcdy
        d2Ody2=self.d2Ody2
        dOdy=self.dOdy
        first= -4*c**2*d2Odcdy-4*c*M*d2Ody2+(M**2-6*c)*dOdy
        c=self.c
        M=self.M
        second= np.conj(-4*c**2)*d2Odcdy.T-np.conj(4*c*M)*d2Ody2.T+np.conj(M**2-6*c)*dOdy.T
        return first+second.T
    def calculate_laplacian_squared(self):
        clk,ylk=self.clk,self.ylk
        c=self.c
        M=self.M
        M6c=M**2-6*c
        c_c=np.conj(self.c)
        M_c=np.conj(self.M)
        M6c_c=M_c**2-6*c_c
        overlap=self.overlap
        dOdc=self.dOdc
        d2Odc2=self.d2Odc2
        d2Odcdy=self.d2Odcdy
        dOdy=self.dOdy
        d2Ody2=self.d2Ody2
        nabla_4=16*np.outer(c_c**2,c**2)*d2Odc2
        nabla_4+=16*(np.outer(c_c*M_c,c**2)+np.outer(c_c**2,c*M))*d2Odcdy
        nabla_4+=16*np.outer(c_c*M_c,c*M)*d2Ody2
        nabla_4+=-4*(np.outer(M6c_c,c**2)+np.outer(c_c**2,M6c))*dOdc
        nabla_4+=-4*(np.outer(M6c_c,c*M)+np.outer(c_c*M_c,M6c))*dOdy
        nabla_4+=np.outer(M6c_c,M6c)*overlap
        return nabla_4

    def calculate_H_noField(self):
        H_nofield=(-0.5*self.calculate_laplacian()-self.calculate_r())/self.sqrt_overlap_mat
        return H_nofield
    def calculate_H_field(self,t=0):
        Hnofield=self.calculate_H_noField()
        if self.gauge=="length":
            return Hnofield+self.fieldFunc(t)*self.calculate_z()/self.sqrt_overlap_mat
        elif self.gauge=="velocity":
            return Hnofield-1j*self.fieldFunc(t)*self.calculate_ddz()/self.sqrt_overlap_mat
    def calculate_Hsquared_field(self,t):
        Hnofield=self.calculate_Hsquared_noField()
        if self.gauge=="length":
            Hfield=Hnofield+(self.fieldFunc(t)**2*self.calculate_zSquared()+self.fieldFunc(t)*(-2*self.calculate_zOverr()+-0.5*self.calculate_z_Laplacian()))/self.sqrt_overlap_mat
        elif self.gauge=="velocity":
            Hfield=Hnofield+(-self.fieldFunc(t)**2*self.calculate_d2dz2()+1j*self.fieldFunc(t)*(self.calculate_ddz_overR()+self.calculate_ddz_laplacian()))/self.sqrt_overlap_mat
        return Hfield
    def calculate_Hsquared_noField(self):
        H_nofieldsquared=self.calculate_rsquared()+0.25*self.calculate_laplacian_squared()+0.5*self.calculate_laplacian_over_r()
        return H_nofieldsquared/self.sqrt_overlap_mat

    def calculate_Hexp_noField(self):
        H_nofield=self.calculate_H_noField()
        return np.conj(self.coefficients).T@H_nofield@self.coefficients
    def calculate_Hexp_field(self,t=0):
        H_field=self.calculate_H_field(t)
        return np.conj(self.coefficients).T@H_field@self.coefficients
    def calculate_Hsquaredexp_noField(self):
        Hsquared_nofield=self.calculate_Hsquared_noField()
        return np.conj(self.coefficients).T@Hsquared_nofield@self.coefficients
    def calculate_Hsquaredexp_field(self,t):
        Hsquared_field=self.calculate_Hsquared_field(t)
        return np.conj(self.coefficients).T@Hsquared_field@self.coefficients
    def calculate_dipole_moment(self):
        return np.conj(self.coefficients).T@(self.calculate_z()/self.sqrt_overlap_mat)@self.coefficients
    def calculate_S_mat(self,start_index_WFnew,t,h=1e-3):
        if h<1e-16:
            self.S_mat=(self.overlap_normalized )[start_index_WFnew:,start_index_WFnew:]
        else:
            self.S_mat=(self.overlap_normalized+h**2/4*self.calculate_Hsquared_field(t) )[start_index_WFnew:,start_index_WFnew:]
        return self.S_mat
    def calculate_rho_vec(self,start_index_WFnew,t,h=1e-3):
        if h<1e-16:
            rho_mat=self.overlap_normalized[:start_index_WFnew,start_index_WFnew:]
        else:
            rho_mat=(self.overlap_normalized-h**2/4*self.calculate_Hsquared_field(t) +1j*h*self.calculate_H_field(t))[:start_index_WFnew,start_index_WFnew:] #Upper right corner
        self.rho_mat=rho_mat
        self.rho_vec=einsum("i,ij->j",self.coefficients[:start_index_WFnew],np.conj(rho_mat))
        return self.rho_vec
    def calculate_all_S_deriv_vecs(self,start_index_WFnew,t,h=1e-3):
        S_deriv_vecs=self.calculate_all_overlap_deriv(start_index_WFnew)+self.calculate_all_Hsquared_deriv(start_index_WFnew,t)*h**2/4
        return S_deriv_vecs[start_index_WFnew:]
    def calculate_all_rho_derivs(self,start_index_WFnew,t,h=1e-3):
        rho_deriv_vecs=self.calculate_all_overlap_deriv(start_index_WFnew)-h**2/4*self.calculate_all_Hsquared_deriv(start_index_WFnew,t)+1j*self.calculate_all_H_deriv(start_index_WFnew,t)*h
        return np.einsum("abc,a->bc",rho_deriv_vecs[:start_index_WFnew],np.conj(self.coefficients[:start_index_WFnew]))
    def calculate_overlap_tildePhim(self,start_index_WFnew,t,h=1e-3): #Calculate <\tilde \Phi_m|\tilde \Phi_m>
        S=self.overlap_normalized[:start_index_WFnew,:start_index_WFnew].copy() # NEEDS TO BE A FUCKING COPY OTHERWISE THE WHOLE SHIT CRASHES LIKE THE FUCKING MALAYSIA AIRLINES
        if h>1e-16:
            contribution=self.calculate_Hsquared_field(t)[:start_index_WFnew,:start_index_WFnew]
            S+=0.25*h**2*contribution
        return np.conj(self.coefficients[:start_index_WFnew]).T@S@self.coefficients[:start_index_WFnew]
    def rothe_optimal_c(self,start_index_WFnew,t,h=1e-3):
        rho=self.calculate_rho_vec(start_index_WFnew,t,h)
        S=self.calculate_S_mat(start_index_WFnew,t,h)
        self.Smat=S
        self.opt_c=solve(S,rho)
        self.rho=rho

    def rothe_error(self,start_index_WFnew,t,h=1e-3):

        overlap_term=self.calculate_overlap_tildePhim(start_index_WFnew,t,h)#1. Calculate <\tilde \Phi_m|\tilde \Phi_m>
        self.rothe_optimal_c(start_index_WFnew,t,h)
        projection_term=np.conj(self.rho).T@self.opt_c
        difference=overlap_term-projection_term
        if np.isnan(difference):
            print("Rothe error is nan...")
            return 0.1
        else:
            return np.real(difference) #numerical noise makes this not exactly real
    def rothe_jacobian(self,start_index_WFnew,t,h=1e-3):
        if "opt_c" not in dir():
            self.rothe_optimal_c(start_index_WFnew,t,h)
        S_deriv_vecs=self.calculate_all_S_deriv_vecs(start_index_WFnew,t,h)
        rho_indices=self.calculate_all_rho_derivs(start_index_WFnew,t,h)
        opt_c=self.opt_c
        opt_c_rep=np.repeat(opt_c[:, np.newaxis], 4, axis=1)
        rho_optc_prods=np.real((rho_indices)*opt_c_rep+np.conj((rho_indices)*opt_c_rep))
        optc_smat_prod=np.einsum("a,aij,i->ij",np.conj(opt_c),S_deriv_vecs,opt_c)+np.einsum("i,aij,a->ij",np.conj(opt_c),np.conj(S_deriv_vecs),opt_c)
        jacobian=np.real(optc_smat_prod-rho_optc_prods)
        jacobian=jacobian.flatten()
        jac=jacobian
        return jacobian
    def rothe_error_overlap_control(self,start_index_WFnew,t,h=1e-3,err_t=0.99):
        beta=self.beta_ovlp
        #Implements the "overlap punishment" from https://pubs.acs.org/doi/10.1021/cr200419d
        overlap_normalized_abs=self.overlap_normalized*self.overlap_normalized.T-np.eye(len(self.overlap_normalized))
        overlap_normalized_abs=overlap_normalized_abs[start_index_WFnew:,start_index_WFnew:]
        rothe_error=self.rothe_error(start_index_WFnew,t,h)
        overlap_new=beta*(overlap_normalized_abs-err_t**2)/(1-err_t**2)
        overlap_new=overlap_new*(overlap_new>0)
        return np.real(np.sum(overlap_new.flatten()))+rothe_error
    def rothe_jacobian_overlap_control(self,start_index_WFnew,t,h=1e-3,err_t=0.99):
        beta=self.beta_ovlp
        jac=self.rothe_jacobian(start_index_WFnew,t,h)
        jac_old=jac.copy()
        overlap_normalized_abs=self.overlap_normalized*self.overlap_normalized.T-np.eye(len(self.overlap_normalized))
        overlap_normalized_abs=overlap_normalized_abs[start_index_WFnew:,start_index_WFnew:]
        len_jac=4*(self.num_basis_funcs-start_index_WFnew)
        for i in range(len_jac):
            overlap_deriv_i=self.calculate_overlap_deriv(i+4*start_index_WFnew)[start_index_WFnew:]
            overlap_deriv_i_of_interest=overlap_deriv_i*(overlap_normalized_abs[i//4,:]>err_t)
            jac[i]+=4*beta*np.real(overlap_deriv_i_of_interest.T@self.overlap_normalized[start_index_WFnew:,start_index_WFnew:][i//4,:])/(1-err_t**2)
        return jac
class lincombGauss_WF(WF):


    def setUpIntermediates(self):
        c=self.c
        q=self.q
        M=self.M
        self.clk=clk=np.add.outer(np.conj(c),c)
        self.ylk=ylk=np.add.outer((2*self.mu*np.conj(c)),(2*self.mu*c))+1j*np.add.outer(-q,q)
        self.glk=glk=1j*np.add.outer((self.mu*q),-(self.mu*q))-np.add.outer((self.mu**2*np.conj(c)),(self.mu**2*c))
        self.exp_glk=exp_glk=exp(glk)
        self.gaussLinear=self.potential_params[0]
        self.gaussExponent=self.potential_params[1]
        try: #Either these are given, or there are not given. If they're not given, they're simply the potential squared!
            if self.t<100:
                raise Exception
            self.gaussExponentsquared=self.potential_params[3]
            self.gaussLinearsquared=self.potential_params[2]
        except:


            gaussLinearsquared=[]
            gaussExponentsquared=[]

            for i in range(len(self.gaussLinear)):
                for j in range(i,len(self.gaussLinear)):
                    if i==j:
                        gaussLinearsquared.append(self.gaussLinear[i]*self.gaussLinear[j])
                        gaussExponentsquared.append(self.gaussExponent[i]+self.gaussExponent[j])
                    else:
                        gaussLinearsquared.append(2*self.gaussLinear[i]*self.gaussLinear[j])
                        gaussExponentsquared.append(self.gaussExponent[i]+self.gaussExponent[j])

            self.gaussExponentsquared=np.array(gaussExponentsquared).ravel()
            self.gaussLinearsquared=np.array(gaussLinearsquared).ravel()

        """Powers of ylk and clk"""

        ylkp2=ylk**2
        ylkp3=ylkp2*ylk
        ylkp4=ylkp3*ylk
        ylkp5=ylkp4*ylk
        """Overlap and its derivatives, with intermediates generated using sympy"""
        clk_inv=1/clk
        clk_inv_squared=clk_inv**2
        ovlp_int1 = clk_inv
        ovlp_int2 = ylkp2
        ovlp_int4= pi**1.5*exp(glk + ovlp_int1*0.25*ovlp_int2)/sqrt(clk)**3
        ovlp_int5 = ovlp_int1*ovlp_int4
        ovlp_int6 = ovlp_int4*clk_inv_squared
        ovlp_int7 = 0.25*ovlp_int2*ovlp_int6
        ovlp_int8 = 0.5*ovlp_int5
        ovlp_int10 = ovlp_int4*clk_inv**4
        ovlp_int11 = 0.0625*ovlp_int10*ylkp4
        ovlp_int12 = ovlp_int4/clk**3
        ovlp_int13 = ovlp_int12*ovlp_int2
        ovlp_int14 = 1.25*ovlp_int6
        ovlp_int16 = 0.125*ovlp_int12*ylkp3
        ovlp_int20 = ovlp_int4*clk_inv_squared**2*clk_inv
        overlap=ovlp_int4
        overlap[np.isnan(overlap)]=0
        self.overlap=overlap
        self.dOdc=-3*ovlp_int5/2 - ovlp_int7
        self.dOdy=ovlp_int8*ylk
        self.d2Odc2=ovlp_int11 + 1.25*ovlp_int13 + 15*ovlp_int6/4
        self.d2Odcdy=-ovlp_int14*ylk - ovlp_int16
        self.d2Ody2=ovlp_int7 + ovlp_int8
        self.d3Odc3=-105*ovlp_int12/8 - 0.015625*clk_inv**6*ovlp_int4*ylkp5*ylk - 0.65625*ovlp_int20*ylkp4 - 6.5625*ovlp_int10*ovlp_int2
        self.d3Odc2dy=4.375*ovlp_int12*ylk + 0.03125*ovlp_int20*ylkp5 + 0.875*ovlp_int10*ylkp3
        self.d3Odcdy2=-ovlp_int11 - ovlp_int13 - ovlp_int14
        self.d3Ody3=ovlp_int16 + 0.75*ovlp_int6*ylk
        self.dOdg=self.overlap #This one is simple

        self.invRsq,self.dinvRsqdc,self.dinvRsqdy=getExpsquaredIntermediates_numba(self.clk,self.ylk,self.glk,self.gaussLinearsquared,self.gaussExponentsquared)
        self.invR,self.dinvRdc,self.dinvRdy,self.d2invRdc2,self.d2invRdcdy,self.d2invRdy2=getExpIntermediates_numba(self.clk,self.ylk,self.glk,self.gaussLinear,self.gaussExponent)
        sqrt_overlaps=sqrt(np.diag(self.overlap))
        self.sqrt_overlap_mat=np.outer(sqrt_overlaps,sqrt_overlaps)
        self.overlap_normalized=self.overlap/self.sqrt_overlap_mat
class erf_WF(WF):
    def __init__(self,params,basis_coefficients,fieldFunc=None,potential_params=None,t=0,beta_ovlp=0,gauge=None):
        self.invR_mu=potential_params[0]
        """The 'fix' parameters used for mu=10"""
        a_10=[102.4542879057064, 4677.895053160093, 113.42187800074939, 158.8773264758503, 184.8976861413018, 393949941.23615706, 133.00479758269714, 201690.70809892353, 4692833.187090565, 256.94641883515504]
        c_10=[5.183106241629503, -1.751958578211088e-10, 8.538828301454487, 4.5641177135985345, 1.1852011849441624, 0.0022121805545205486, 7.852543353517831, 2.8167690402369772e-11, 1.1868417360005878e-10, 0.00015767850391057436]

        """The 'fix' parameters used for mu=100"""
        a_100=[13300.44229943767, 15884.837695608596, 18485.05069229749, 1770595228.681136, 280926207340.81494, 4019260292973.267, 25141.373797892582, 10245.585264615638, 17285292.824737024, 11342.743871555964]
        c_100=[784.6992876031436, 456.226640985813, 119.01023695082404, 8.692310871083464e-08, 0.11065301306939546, 0.11061165500686343, 0.021845305454917252, 518.5788334310637, -4.251489826856414e-09, 853.8586030676961]
        if self.invR_mu==100:
            self.oneOverRsquared_Gaussian_exponents=a_100
            self.oneOverRsquared_Gaussian_lincoeff=c_100
        elif self.invR_mu==10:
            self.oneOverRsquared_Gaussian_exponents=a_10
            self.oneOverRsquared_Gaussian_lincoeff=c_10
        else:
            raise NotImplementedError("This mu is not implemented in erf")
        super().__init__(params,basis_coefficients,potential_params,fieldFunc,t,beta_ovlp,gauge)

    def setUpIntermediates(self):
        invR_mu=self.invR_mu
        c=self.c
        q=self.q
        M=self.M
        self.clk=clk=np.add.outer(np.conj(c),c)
        self.ylk=ylk=np.add.outer((2*self.mu*np.conj(c)),(2*self.mu*c))+1j*np.add.outer(-q,q)
        self.glk=glk=1j*np.add.outer((self.mu*q),-(self.mu*q))-np.add.outer((self.mu**2*np.conj(c)),(self.mu**2*c))
        self.exp_glk=exp_glk=exp(glk)

        """Powers of ylk and clk"""

        ylkp2=ylk**2
        ylkp3=ylkp2*ylk
        ylkp4=ylkp3*ylk
        ylkp5=ylkp4*ylk
        ylkp6=ylkp5*ylk
        clkp2=clk**2
        clkp3=clkp2*clk

        """Overlap and its derivatives, with intermediates generated using sympy"""
        clk_inv=1/clk
        clk_inv_squared=clk_inv**2
        ovlp_int0 = clkp3
        ovlp_int1 = clk_inv
        ovlp_int2 = ylkp2
        ovlp_int3 = 0.25*ovlp_int2
        ovlp_int4= pi**1.5*exp(glk + ovlp_int1*ovlp_int3)/sqrt(clk)**3
        ovlp_int5 = ovlp_int1*ovlp_int4
        ovlp_int6 = ovlp_int4*clk_inv_squared
        ovlp_int7 = ovlp_int3*ovlp_int6
        ovlp_int8 = 0.5*ovlp_int5
        ovlp_int10 = ovlp_int4*clk_inv**4
        ovlp_int11 = 0.0625*ovlp_int10*ylkp4
        ovlp_int12 = ovlp_int4/ovlp_int0
        ovlp_int13 = ovlp_int12*ovlp_int2
        ovlp_int14 = 1.25*ovlp_int6
        ovlp_int16 = 0.125*ovlp_int12*ylkp3
        ovlp_int20 = ovlp_int4*clk_inv_squared**2*clk_inv
        overlap=ovlp_int4
        overlap[np.isnan(overlap)]=0
        self.overlap=overlap
        dOdc=self.dOdc=-3*ovlp_int5/2 - ovlp_int7
        dOdy=self.dOdy=ovlp_int8*ylk
        d2Odc2=self.d2Odc2=ovlp_int11 + 1.25*ovlp_int13 + 15*ovlp_int6/4
        d2Odcdy=self.d2Odcdy=-ovlp_int14*ylk - ovlp_int16
        d2Ody2=self.d2Ody2=ovlp_int7 + ovlp_int8
        d3Odc3=self.d3Odc3=-105*ovlp_int12/8 - 0.015625*clk_inv**6*ovlp_int4*ylkp6 - 0.65625*ovlp_int20*ylkp4 - 6.5625*ovlp_int10*ovlp_int2
        d3Odc2dy=self.d3Odc2dy=4.375*ovlp_int12*ylk + 0.03125*ovlp_int20*ylkp5 + 0.875*ovlp_int10*ylkp3
        d3Odcdy2=self.d3Odcdy2=-ovlp_int11 - 1.0*ovlp_int13 - ovlp_int14
        d3Ody3=self.d3Ody3=ovlp_int16 + 0.75*ovlp_int6*ylk
        self.dOdg=dOdg=self.overlap #This one is simple
        sqrt_overlaps=sqrt(np.diag(self.overlap))
        self.sqrt_overlap_mat=np.outer(sqrt_overlaps,sqrt_overlaps)
        self.overlap_normalized=self.overlap/self.sqrt_overlap_mat

        threshold_discard=1e-280 #
        threshold_ylk=1e-3#This seems to give "good" results!!
        large_ylk=(abs(ylk)>threshold_ylk)*1
        small_ylk=(abs(ylk)<=threshold_ylk)*1


        """<1/r> and its derivatives, with intermediates generated using sympy"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x0=ntn(1/ylk,posinf=0,neginf=0,nan=0)#*large_ylk
            x11=ylkp2
            x18=ntn(1/x11,posinf=0,neginf=0,nan=0)#*large_ylk
            x30= ylkp3
            x31= ntn(1/x30,posinf=0,neginf=0,nan=0)#*large_ylk
            x40= ylkp4
            x40_inv=ntn(1/x40,posinf=0,neginf=0,nan=0)#*large_ylk
            x1=invR_mu**2
            x2= clkp2 + clk*x1
            x3=1/sqrt(x2)
            x4=invR_mu*x3
            x5=0.5*ylk
            x6=erf(np.array(x4*x5,dtype=np.complex128))
            x7=x0*x6
            x8=-clk - x1*0.5
            x9=x2**(-3/2)
            x10=sqrt_pi_inv
            x12=0.25*x11
            x13=exp(-x1*x12/x2)
            x14=x10*x13
            x15=x14*x9
            x16=invR_mu*x15
            x17=clk

            x19=x18*x6
            x20=x14*x4
            x21=-3*clk - 3*x1/2
            x22=x2**(-5/2)
            x23=x2**(-7/2)
            x24= invR_mu**3
            x25= -2*clk - x1
            x26= clk*x12*x14*x24*x25*x8
            x27= x14*x22
            x28= x27*x8
            x29= clk*x24*x5

            x32= 0.5*x24
            x33= x15*x32
            x34= 2.0*x18*x20
            x39= invR_mu**5
            k=clk*x7
            dkdc= x16*x17*x8 + x7
            dkdy=-clk*x19 + x0*x17*x20
            d2kdc2=(clk*invR_mu*x10*x21*x22*x8 + 2.0*invR_mu*x10*x8*x9)*x13 - x16*x17 - x23*x26
            d2kdcdy=invR_mu*x0*x10*x13*x3 - x19 - x28*x29
            d2kdy2=2*clk*x31*x6 - clk*x33 - clk*x34

            #There are two different sets of expressions: large_ylk for ylk>threshold, small_ylk for ylk<=threshold
            invR=large_ylk*2*overlap*k
            dinvRdc=2*large_ylk*(k*dOdc+overlap*dkdc)
            dinvRdy=2*large_ylk*(k*dOdy+overlap*dkdy)
            d2invRdc2 =large_ylk*2*(d2Odc2*k+2*dkdc*dOdc+overlap*d2kdc2)
            d2invRdy2 =large_ylk*2*(d2Ody2*k+2*dkdy*dOdy+overlap*d2kdy2)
            d2invRdcdy=large_ylk*2*(dOdc*dkdy+dOdy*dkdc+d2Odcdy*k+d2kdcdy*overlap)

            indices=np.argwhere(abs(self.overlap_normalized)<threshold_discard)



            """<1/r²> and its derivatives, with intermediates generated using sympy"""

            invRsq=np.zeros_like(large_ylk,dtype=np.complex128)
            dinvRsqdc=np.zeros_like(large_ylk,dtype=np.complex128)
            dinvRsqdy=np.zeros_like(large_ylk,dtype=np.complex128)
            rsquared_int0 = x0
            rsquared_int6 = clk**0.5
            rsquared_int1 = rsquared_int6**3
            rsquared_int2 = ylkp2
            rsquared_int3 = 2*dawsn(np.array(0.5*ylk/rsquared_int6,dtype=np.complex128))
            rsquared_int4 = rsquared_int1*rsquared_int3
            rsquared_int5 = rsquared_int4/2
            rsquared_int9 = 0.125*rsquared_int3*ylk
            rsquared_int11 = clk
            rsquared_int13 = 0.25*rsquared_int3*rsquared_int6
            rsquared_int14 = x18
            invRsquared_noOvlp=rsquared_int0*rsquared_int5
            dinvRsquared_noOvlp_dc=0.75*rsquared_int0*rsquared_int3*rsquared_int6 + rsquared_int9/rsquared_int6 - 0.25
            dinvRsquared_noOvlp_dy=0.5*rsquared_int0*rsquared_int11 - rsquared_int13 - x18*rsquared_int5


            invRsq+=4*large_ylk*overlap*invRsquared_noOvlp
            dinvRsqdc+=4*large_ylk*(invRsquared_noOvlp*dOdc+overlap*dinvRsquared_noOvlp_dc)
            dinvRsqdy+=4*large_ylk*(invRsquared_noOvlp*dOdy+overlap*dinvRsquared_noOvlp_dy)
            clkPmu=clk+invR_mu**2 # Extra contribution to c_lk for calculating friends
            """extra term in 1/r^2 - first, need to calculate the "alternative overlap" """
            inv_sqrt_clkPmu=1/sqrt(clkPmu)
            clk_Pmu_inv=1/clkPmu
            ovlp_pMu_int0 = clk + invR_mu**2
            ovlp_pMu_int1 = ovlp_pMu_int0**3
            ovlp_pMu_int2 = 1/ovlp_pMu_int0
            ovlp_pMu_int3 = ylkp2
            ovlp_pMu_int4 = 0.25*ovlp_pMu_int3
            ovlp_pMu_int5 = pi**1.5*exp(glk + ovlp_pMu_int2*ovlp_pMu_int4)/sqrt(ovlp_pMu_int0)**3
            ovlp_pMu_int6 = ovlp_pMu_int2*ovlp_pMu_int5
            ovlp_pMu_int7 = ovlp_pMu_int5/ovlp_pMu_int0**2
            ovlp_pMu_int8 = ovlp_pMu_int4*ovlp_pMu_int7
            ovlp_pMu_int9 = 0.5*ovlp_pMu_int6
            ovlp_pMu_int10 = ovlp_pMu_int5/ovlp_pMu_int1
            overlapPmu = ovlp_pMu_int5
            dOdcPmu = -1.5*ovlp_pMu_int6 - ovlp_pMu_int8
            dOdyPmu = ovlp_pMu_int9*ylk
            d2Odc2Pmu = 1.25*ovlp_pMu_int10*ovlp_pMu_int3 + 15/4*ovlp_pMu_int7 + 0.0625*ovlp_pMu_int5*ylkp4/ovlp_pMu_int0**4
            d2OdcdyPmu = -0.125*ovlp_pMu_int10*ylkp3 - 1.25*ovlp_pMu_int7*ylk
            d2Ody2Pmu = ovlp_pMu_int8 + ovlp_pMu_int9
            """extra term in 1/r^2 - now, calculate the alternative 1/r^2"""
            rsquared_int1 = clkPmu**1.5
            rsquared_int3 = dawsn(np.array(0.5*ylk*inv_sqrt_clkPmu,dtype=np.complex128))*2
            rsquared_int5 = rsquared_int1*rsquared_int3/2
            rsquared_int6 = clkPmu**0.5
            rsquared_int10 = 0.125*rsquared_int3*ylk/rsquared_int6 - 0.25
            rsquared_int11 = clkPmu
            rsquared_int13 = 0.25*rsquared_int3*rsquared_int6
            invRsquared_noOvlp=rsquared_int0*rsquared_int5
            dinvRsquared_noOvlp_dc=0.75*rsquared_int0*rsquared_int3*rsquared_int6 + rsquared_int10
            dinvRsquared_noOvlp_dy=0.5*rsquared_int0*rsquared_int11 - rsquared_int13 - rsquared_int14*rsquared_int5

            invRsq-=4*large_ylk*overlapPmu*invRsquared_noOvlp
            dinvRsqdc-=4*large_ylk*(invRsquared_noOvlp*dOdcPmu+overlapPmu*dinvRsquared_noOvlp_dc)
            dinvRsqdy-=4*large_ylk*(invRsquared_noOvlp*dOdyPmu+overlapPmu*dinvRsquared_noOvlp_dy)

            """Taylor expansion for 1/r"""
            kT_int0 = clk_inv
            kT_int1 = invR_mu**2
            kT_int2 = clk + kT_int1
            kT_int3 = clk*kT_int2
            kT_int4 = ylkp2
            kT_int5 = kT_int1*kT_int4
            kT_int6 = sqrt_pi_inv
            kT_int7 = sqrt(kT_int3)
            kT_int8 = kT_int6*kT_int7
            kT_int9 = clkp2
            kT_int10 = clkp3
            kT_int11 = kT_int10
            kT_int12 = kT_int1**3
            kT_int13 = kT_int1**2
            kT_int14 = clk*kT_int13
            kT_int15 = kT_int1*kT_int9
            kT_int16 = 0.5*kT_int9
            kT_int17 = clk*kT_int1
            kT_int18 = invR_mu**3
            kT_int19 = kT_int18*kT_int8
            kT_int20 = 0.166666666666667*kT_int17 + 0.166666666666667*kT_int9
            kT_int21 = kT_int18*kT_int6
            kT_int22 = kT_int0*kT_int21/(kT_int7*(kT_int13 + 2.0*kT_int17 + kT_int9))
            kT_int23 = kT_int13 + 2*kT_int17 + kT_int9
            k = invR_mu*kT_int0*kT_int8*(kT_int3 - 0.0833333333333333*kT_int5)/kT_int2**2
            dkdc = kT_int19*(0.166666666666667*clk*kT_int4 + kT_int16 + 0.5*kT_int17 + 0.0416666666666667*kT_int5)/(kT_int9*(kT_int11 + kT_int12 + 3.0*kT_int14 + 3.0*kT_int15))
            dkdy = kT_int22*ylk*(0.025*kT_int1*kT_int4 - kT_int20)
            d2kdc2 = kT_int19*(-0.25*clk*kT_int5 - kT_int11 - 0.0625*kT_int13*kT_int4 - 0.25*kT_int14 - 1.25*kT_int15 - kT_int16*kT_int4)/(kT_int10*(clk**4 + 4.0*clk*kT_int12 + invR_mu**8 + 4.0*kT_int1*kT_int10 + 6.0*kT_int13*kT_int9))
            d2kdcdy = clk*kT_int21*ylk*(kT_int23*kT_int3*(0.333333333333333*clk + 0.0833333333333333*kT_int1) - kT_int5*(-0.03125*kT_int2**3 + 0.00625*kT_int2*kT_int23 + kT_int23*(0.125*clk + 0.0625*kT_int1)))/(kT_int23*kT_int3**(7/2))
            d2kdy2 = kT_int22*(0.075*kT_int1*kT_int4 - kT_int20)

            invR=invR+small_ylk*(2*overlap*k)
            dinvRdc=dinvRdc+2*small_ylk*(k*dOdc+overlap*dkdc)
            dinvRdy=dinvRdy+2*small_ylk*(k*dOdy+overlap*dkdy)
            d2invRdc2 =d2invRdc2+small_ylk*2*(d2Odc2*k+2*dkdc*dOdc+overlap*d2kdc2)
            d2invRdy2 =d2invRdy2+small_ylk*2*(d2Ody2*k+2*dkdy*dOdy+overlap*d2kdy2)
            d2invRdcdy=d2invRdcdy+small_ylk*2*(dOdc*dkdy+dOdy*dkdc+d2Odcdy*k+d2kdcdy*overlap)

            """Taylor expansion for 1/r^2"""
            kR2T_int0 = clk
            kR2T_int1 = ylkp2
            kR2T_int2 = ylkp4
            kR2T_int3 = clk_inv
            kR2T_int4 = clk_inv
            kR2T_int5 = clk_inv_squared
            kR2T_int6 = clk_inv_squared
            kR2 = 0.5*kR2T_int0 - 0.0833333333333333*kR2T_int1 + kR2T_int2*(0.01875*kR2T_int3 - 0.0104166666666667*kR2T_int4)
            dkR2dc = kR2T_int1*(0.1875*kR2T_int3 - 0.1875*kR2T_int4) + kR2T_int2*(-0.0317708333333333*kR2T_int5 + 0.0234375*kR2T_int6) + 0.5
            dkR2dy = ylkp5*(0.00554315476190476*kR2T_int5 - 0.00911458333333333*kR2T_int6) + ylkp3*(-0.0395833333333333*kR2T_int3 + 0.0729166666666667*kR2T_int4) - 0.166666666666667*ylk
            invRsq+=small_ylk*4*overlap*kR2
            dinvRsqdc+=small_ylk*4*(dOdc*kR2+overlap*dkR2dc)
            dinvRsqdy+=small_ylk*4*(dOdy*kR2+overlap*dkR2dy)
            kR2T_int0 = clkPmu
            kR2T_int1 = ylkp2
            kR2T_int2 = ylkp4
            kR2T_int3 = 1/clkPmu
            kR2T_int4 = kR2T_int3
            kR2T_int5 = kR2T_int3**2
            kR2T_int6 = kR2T_int5
            kR2 = 0.5*kR2T_int0 - 0.0833333333333333*kR2T_int1 + kR2T_int2*(0.01875*kR2T_int3 - 0.0104166666666667*kR2T_int4)
            dkR2dc = kR2T_int1*(0.1875*kR2T_int3 - 0.1875*kR2T_int4) + kR2T_int2*(-0.0317708333333333*kR2T_int5 + 0.0234375*kR2T_int6) + 0.5
            dkR2dy = ylkp5*(0.00554315476190476*kR2T_int5 - 0.00911458333333333*kR2T_int6) + ylkp3*(-0.0395833333333333*kR2T_int3 + 0.0729166666666667*kR2T_int4) - 0.166666666666667*ylk
            invRsq-=4*small_ylk*overlapPmu*kR2
            dinvRsqdc-=4*small_ylk*(kR2*dOdcPmu+overlapPmu*dkR2dc)
            dinvRsqdy-=4*small_ylk*(kR2*dOdyPmu+overlapPmu*dkR2dy)


        """Finally, the "extra Gaussians" needed to make 1/r^2 = (1/r)^2"""
        for m in range(len(self.oneOverRsquared_Gaussian_exponents)):
            alpha=self.oneOverRsquared_Gaussian_exponents[m]
            linearCoeff=self.oneOverRsquared_Gaussian_lincoeff[m]
            invR_Gaussian_contrib0 = alpha + clk
            invR_Gaussian_contrib2 = 1/invR_Gaussian_contrib0
            invR_Gaussian_contrib3 = ylkp2
            invR_Gaussian_contrib4 = 0.25*invR_Gaussian_contrib3
            invR_Gaussian_contrib5 = pi**1.5*exp(glk + invR_Gaussian_contrib2*invR_Gaussian_contrib4)/sqrt(invR_Gaussian_contrib0)**3
            invR_Gaussian_contrib6 = invR_Gaussian_contrib2*invR_Gaussian_contrib5
            invRsq += linearCoeff*invR_Gaussian_contrib5
            dinvRsqdc += linearCoeff*(-3*invR_Gaussian_contrib6/2 - invR_Gaussian_contrib4*invR_Gaussian_contrib5/invR_Gaussian_contrib0**2)
            dinvRsqdy += linearCoeff*0.5*invR_Gaussian_contrib6*ylk

        self.invR=np.asarray(invR,dtype=np.complex128)
        self.dinvRdc=np.asarray(dinvRdc,dtype=np.complex128)
        self.dinvRdy=np.asarray(dinvRdy,dtype=np.complex128)
        self.d2invRdc2=np.asarray(d2invRdc2,dtype=np.complex128)
        self.d2invRdcdy=np.asarray(d2invRdcdy,dtype=np.complex128)
        self.d2invRdy2=np.asarray(d2invRdy2,dtype=np.complex128)
        self.invRsq=np.asarray(invRsq,dtype=np.complex128)
        self.dinvRsqdc=np.asarray(dinvRsqdc,dtype=np.complex128)
        self.dinvRsqdy=np.asarray(dinvRsqdy,dtype=np.complex128)

        self.invR[abs(overlap)<threshold_discard]=0
        self.dinvRdc[abs(overlap)<threshold_discard]=0
        self.overlap[abs(overlap)<threshold_discard]=0
        self.overlap_normalized[abs(overlap)<threshold_discard]=0
        self.dinvRdy[abs(overlap)<threshold_discard]=0
        self.d2invRdy2[abs(overlap)<threshold_discard]=0
        self.d2invRdcdy[abs(overlap)<threshold_discard]=0
        self.d2invRdc2[abs(overlap)<threshold_discard]=0
        self.invRsq[abs(overlap)<threshold_discard]=0
        self.dinvRsqdc[abs(overlap)<threshold_discard]=0
        self.dinvRsqdy[abs(overlap)<threshold_discard]=0
        self.dOdc[abs(overlap)<threshold_discard]=0
        self.dOdy[abs(overlap)<threshold_discard]=0
        self.d2Odc2[abs(overlap)<threshold_discard]=0
        self.d2Ody2[abs(overlap)<threshold_discard]=0
        self.d2Odcdy[abs(overlap)<threshold_discard]=0
        self.d3Odc3[abs(overlap)<threshold_discard]=0
        self.d3Odcdy2[abs(overlap)<threshold_discard]=0
        self.d3Ody3[abs(overlap)<threshold_discard]=0
        self.d3Odc2dy[abs(overlap)<threshold_discard]=0
class erfGau_WF(WF):
    def __init__(self,params,basis_coefficients,fieldFunc=None,potential_params=None,t=0,beta_ovlp=0,gauge="length"):
        self.invR_mu=potential_params[0]
        correctingGaussians_c_mu1=np.array([-0.004424028014000214, 0.06436669830327446, 0.09421572509745602, 0.0775185569139012, 0.03583929908131722, 0.005723319078413169, -2.5730300556858765e-08])
        correctingGaussians_alpha_mu1=np.array([1.0087931229154938, 1.013934257361394, 1.076385530800151, 1.177111009094002, 1.2906260195061192, 1.379380907712712, 1.9993743225685336])**2

        if self.invR_mu==1:
            self.oneOverRsquared_Gaussian_exponents=correctingGaussians_alpha_mu1
            self.oneOverRsquared_Gaussian_lincoeff=correctingGaussians_c_mu1
        super().__init__(params,basis_coefficients,potential_params,fieldFunc,t,beta_ovlp,gauge)
    def setUpIntermediates(self):
        invR_mu=self.invR_mu
        c=self.c
        q=self.q
        M=self.M
        self.clk=clk=np.add.outer(np.conj(c),c)
        self.ylk=ylk=np.add.outer((2*self.mu*np.conj(c)),(2*self.mu*c))+1j*np.add.outer(-q,q)
        self.glk=glk=1j*np.add.outer((self.mu*q),-(self.mu*q))-np.add.outer((self.mu**2*np.conj(c)),(self.mu**2*c))
        self.exp_glk=exp_glk=exp(glk)

        """Powers of ylk and clk"""

        ylkp2=ylk**2
        ylkp3=ylkp2*ylk
        ylkp4=ylkp3*ylk
        ylkp5=ylkp4*ylk
        ylkp6=ylkp5*ylk
        clkp2=clk**2
        clkp3=clkp2*clk

        """Overlap and its derivatives, with intermediates generated using sympy"""
        clk_inv=1/clk
        clk_inv_squared=clk_inv**2
        ovlp_int0 = clkp3
        ovlp_int1 = clk_inv
        ovlp_int2 = ylkp2
        ovlp_int3 = 0.25*ovlp_int2
        ovlp_int4= pi**1.5*exp(glk + ovlp_int1*ovlp_int3)/sqrt(clk)**3
        ovlp_int5 = ovlp_int1*ovlp_int4
        ovlp_int6 = ovlp_int4*clk_inv_squared
        ovlp_int7 = ovlp_int3*ovlp_int6
        ovlp_int8 = 0.5*ovlp_int5
        ovlp_int10 = ovlp_int4*clk_inv**4
        ovlp_int11 = 0.0625*ovlp_int10*ylkp4
        ovlp_int12 = ovlp_int4/ovlp_int0
        ovlp_int13 = ovlp_int12*ovlp_int2
        ovlp_int14 = 1.25*ovlp_int6
        ovlp_int16 = 0.125*ovlp_int12*ylkp3
        ovlp_int20 = ovlp_int4*clk_inv_squared**2*clk_inv
        overlap=ovlp_int4
        overlap[np.isnan(overlap)]=0
        self.overlap=overlap
        dOdc=self.dOdc=-3*ovlp_int5/2 - ovlp_int7
        dOdy=self.dOdy=ovlp_int8*ylk
        d2Odc2=self.d2Odc2=ovlp_int11 + 1.25*ovlp_int13 + 15*ovlp_int6/4
        d2Odcdy=self.d2Odcdy=-ovlp_int14*ylk - ovlp_int16
        d2Ody2=self.d2Ody2=ovlp_int7 + ovlp_int8
        d3Odc3=self.d3Odc3=-105*ovlp_int12/8 - 0.015625*clk_inv**6*ovlp_int4*ylkp6 - 0.65625*ovlp_int20*ylkp4 - 6.5625*ovlp_int10*ovlp_int2
        d3Odc2dy=self.d3Odc2dy=4.375*ovlp_int12*ylk + 0.03125*ovlp_int20*ylkp5 + 0.875*ovlp_int10*ylkp3
        d3Odcdy2=self.d3Odcdy2=-ovlp_int11 - ovlp_int13 - ovlp_int14
        d3Ody3=self.d3Ody3=ovlp_int16 + 0.75*ovlp_int6*ylk
        self.dOdg=dOdg=self.overlap #This one is simple
        sqrt_overlaps=sqrt(np.diag(self.overlap))
        self.sqrt_overlap_mat=np.outer(sqrt_overlaps,sqrt_overlaps)
        self.overlap_normalized=self.overlap/self.sqrt_overlap_mat

        threshold_discard=1e-280 #If the overlap is less than this, then the whole thing isn't calculated.
        threshold_ylk=1e-3#This seems to give "good" results!!
        large_ylk=(abs(ylk)>threshold_ylk)*1
        small_ylk=(abs(ylk)<=threshold_ylk)*1

        c_term=0.923+1.568*invR_mu
        alpha_term=0.2411+1.405*invR_mu
        """<1/r> and its derivatives, with intermediates generated using sympy"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x0 =ntn(1/ylk,posinf=0,neginf=0,nan=0)
            x1 = invR_mu**2
            x2 = clk**2 + clk*x1
            x3 = 1/sqrt(x2)
            x4 = invR_mu*x3
            x5 = 0.5*ylk
            x6 = erf(x4*x5)
            x7 = x0*x6
            x8 = -clk - x1/2
            x9 = x2**(-3/2)
            x10 = 1/sqrt(pi)
            x11 = ylkp2
            x12 = 0.25*x11
            x13 = exp(-x1*x12/x2)
            x14 = x10*x13
            x15 = x14*x9
            x16 = invR_mu*x15
            x17 = clk
            x18 = ntn(1/ylkp2,posinf=0,neginf=0,nan=0)#*large_ylk
            x19 = x18*x6
            x20 = x14*x4
            x21 = x2**(-5/2)
            x22 = clk*invR_mu**3
            x23 = x14*x8
            k = clk*x7
            dkdc = x16*x17*x8 + x7
            dkdy = -clk*x19 + x0*x17*x20
            d2kdc2 = clk*invR_mu*x10*x13*x21*x8*(-3*clk - 3*x1/2) + 2.0*invR_mu*x10*x13*x8*x9 - x12*x22*x23*(-2*clk - x1)/x2**(7/2) - x16*x17
            d2kdcdy = invR_mu*x0*x10*x13*x3 - x19 - x21*x22*x23*x5
            d2kdy2 = -2.0*clk*x18*x20 + 2*clk*x6*x18*x0 - 0.5*x15*x22


            #There are two different sets of expressions: large_ylk for ylk>threshold, small_ylk for ylk<=threshold
            invR=large_ylk*2*overlap*k
            dinvRdc=2*large_ylk*(k*dOdc+overlap*dkdc)
            dinvRdy=2*large_ylk*(k*dOdy+overlap*dkdy)
            d2invRdc2 =large_ylk*2*(d2Odc2*k+2*dkdc*dOdc+overlap*d2kdc2)
            d2invRdy2 =large_ylk*2*(d2Ody2*k+2*dkdy*dOdy+overlap*d2kdy2)
            d2invRdcdy=large_ylk*2*(dOdc*dkdy+dOdy*dkdc+d2Odcdy*k+d2kdcdy*overlap)

            """Taylor expansion for erf part of 1/r"""
            kT_int0 = clk_inv
            kT_int1 = x1
            kT_int2 = clk + kT_int1
            kT_int3 = clk*kT_int2
            kT_int4 = ylkp2
            kT_int5 = kT_int1*kT_int4
            kT_int6 = sqrt_pi_inv
            kT_int7 = sqrt(kT_int3)
            kT_int8 = kT_int6*kT_int7
            kT_int9 = clkp2
            kT_int10 = clkp3
            kT_int11 = kT_int10
            kT_int13 = kT_int1**2
            kT_int12 = kT_int13*kT_int1
            kT_int14 = clk*kT_int13
            kT_int15 = kT_int1*kT_int9
            kT_int16 = 0.5*kT_int9
            kT_int17 = clk*kT_int1
            kT_int18 = invR_mu**3
            kT_int19 = kT_int18*kT_int8
            kT_int20 = 0.166666666666667*kT_int17 + 0.166666666666667*kT_int9
            kT_int21 = kT_int18*kT_int6
            kT_int22 = kT_int0*kT_int21/(kT_int7*(kT_int13 + 2.0*kT_int17 + kT_int9))
            kT_int23 = kT_int13 + 2*kT_int17 + kT_int9
            k = invR_mu*kT_int0*kT_int8*(kT_int3 - 0.0833333333333333*kT_int5)/kT_int2**2
            dkdc = kT_int19*(0.166666666666667*clk*kT_int4 + kT_int16 + 0.5*kT_int17 + 0.0416666666666667*kT_int5)/(kT_int9*(kT_int11 + kT_int12 + 3.0*kT_int14 + 3.0*kT_int15))
            dkdy = kT_int22*ylk*(0.025*kT_int1*kT_int4 - kT_int20)
            d2kdc2 = kT_int19*(-0.25*clk*kT_int5 - kT_int11 - 0.0625*kT_int13*kT_int4 - 0.25*kT_int14 - 1.25*kT_int15 - kT_int16*kT_int4)/(kT_int10*(clk**4 + 4.0*clk*kT_int12 + invR_mu**8 + 4.0*kT_int1*kT_int10 + 6.0*kT_int13*kT_int9))
            d2kdcdy = clk*kT_int21*ylk*(kT_int23*kT_int3*(0.333333333333333*clk + 0.0833333333333333*kT_int1) - kT_int5*(-0.03125*kT_int2**3 + 0.00625*kT_int2*kT_int23 + kT_int23*(0.125*clk + 0.0625*kT_int1)))/(kT_int23*kT_int3**(7/2))
            d2kdy2 = kT_int22*(0.075*kT_int1*kT_int4 - kT_int20)

            invR      = invR       +small_ylk*(2*overlap*k)
            dinvRdc   = dinvRdc    +2*small_ylk*(k*dOdc+overlap*dkdc)
            dinvRdy   = dinvRdy    +2*small_ylk*(k*dOdy+overlap*dkdy)
            d2invRdc2 = d2invRdc2  +small_ylk*2*(d2Odc2*k+2*dkdc*dOdc+overlap*d2kdc2)
            d2invRdy2 = d2invRdy2  +small_ylk*2*(d2Ody2*k+2*dkdy*dOdy+overlap*d2kdy2)
            d2invRdcdy= d2invRdcdy +small_ylk*2*(dOdc*dkdy+dOdy*dkdc+d2Odcdy*k+d2kdcdy*overlap)

            """Gaussian part of 1/r"""
            gauss_int0 = alpha_term**2 + clk
            gauss_int1 = gauss_int0**3
            gauss_int2 = 1/gauss_int0
            gauss_int3 = ylkp2
            gauss_int4 = 0.25*gauss_int3
            gauss_int5 = pi**1.5*c_term*exp(gauss_int2*gauss_int4 + glk)/sqrt(gauss_int1)
            gauss_int6 = gauss_int2*gauss_int5
            gauss_int7 = gauss_int5/gauss_int0**2
            gauss_int8 = gauss_int4*gauss_int7
            gauss_int9 = 0.5*gauss_int6
            gauss_int10 = gauss_int5/gauss_int1
            invR =invR+ gauss_int5
            dinvRdc =dinvRdc -3*gauss_int6/2 - gauss_int8
            dinvRdy =dinvRdy+ gauss_int9*ylk
            d2invRdc2 =d2invRdc2+ 1.25*gauss_int10*gauss_int3 + 15*gauss_int7/4 + 0.0625*gauss_int5*ylk**4/gauss_int0**4
            d2invRdcdy =d2invRdcdy -0.125*gauss_int10*ylk**3 - 1.25*gauss_int7*ylk
            d2invRdy2 =d2invRdy2 + gauss_int8 + gauss_int9

            """<1/r^2> consists of 5 terms: 1/r^2, -e^(-mu^2r^2)/r^2, 2c_term e^(-alpha_term^2 r^2), c_term^2e^(-2alpha_term^2 r^2) and correcting Gaussians"""

            """<1/r²> and its derivatives, with intermediates generated using sympy"""

            """1/r²"""
            invRsq=np.zeros_like(large_ylk,dtype=np.complex128)
            dinvRsqdc=np.zeros_like(large_ylk,dtype=np.complex128)
            dinvRsqdy=np.zeros_like(large_ylk,dtype=np.complex128)
            rsquared_int0 = x0
            rsquared_int6 = clk**0.5
            rsquared_int1 = rsquared_int6**3
            rsquared_int2 = ylkp2
            rsquared_int3 = 2*dawsn(np.array(0.5*ylk/rsquared_int6,dtype=np.complex128))
            rsquared_int4 = rsquared_int1*rsquared_int3
            rsquared_int5 = rsquared_int4/2
            rsquared_int9 = 0.125*rsquared_int3*ylk
            rsquared_int11 = clk
            rsquared_int13 = 0.25*rsquared_int3*rsquared_int6
            rsquared_int14 = x18
            invRsquared_noOvlp=rsquared_int0*rsquared_int5
            dinvRsquared_noOvlp_dc=0.75*rsquared_int0*rsquared_int3*rsquared_int6 + rsquared_int9/rsquared_int6 - 0.25
            dinvRsquared_noOvlp_dy=0.5*rsquared_int0*rsquared_int11 - rsquared_int13 - x18*rsquared_int5
            invRsq+=4*large_ylk*overlap*invRsquared_noOvlp
            dinvRsqdc+=4*large_ylk*(invRsquared_noOvlp*dOdc+overlap*dinvRsquared_noOvlp_dc)
            dinvRsqdy+=4*large_ylk*(invRsquared_noOvlp*dOdy+overlap*dinvRsquared_noOvlp_dy)



            """extra term in 1/r^2 - first, need to calculate the "alternative overlap" """
            """Alternative overlap term that shows up in the evaluation of expectation value - it is now no longer the overlap"""
            clkPmu=clk+invR_mu**2 # Extra contribution to c_lk for calculating friends
            inv_sqrt_clkPmu=1/sqrt(clkPmu)
            clk_Pmu_inv=1/clkPmu
            ovlp_pMu_int0 = clk + invR_mu**2
            ovlp_pMu_int1 = ovlp_pMu_int0**3
            ovlp_pMu_int2 = 1/ovlp_pMu_int0
            ovlp_pMu_int3 = ylkp2
            ovlp_pMu_int4 = 0.25*ovlp_pMu_int3
            ovlp_pMu_int5 = pi**1.5*exp(glk + ovlp_pMu_int2*ovlp_pMu_int4)/sqrt(ovlp_pMu_int0)**3
            ovlp_pMu_int6 = ovlp_pMu_int2*ovlp_pMu_int5
            ovlp_pMu_int7 = ovlp_pMu_int5/ovlp_pMu_int0**2
            ovlp_pMu_int8 = ovlp_pMu_int4*ovlp_pMu_int7
            ovlp_pMu_int9 = 0.5*ovlp_pMu_int6
            ovlp_pMu_int10 = ovlp_pMu_int5/ovlp_pMu_int1
            overlapPmu = ovlp_pMu_int5
            dOdcPmu = -1.5*ovlp_pMu_int6 - ovlp_pMu_int8
            dOdyPmu = ovlp_pMu_int9*ylk
            d2Odc2Pmu = 1.25*ovlp_pMu_int10*ovlp_pMu_int3 + 15/4*ovlp_pMu_int7 + 0.0625*ovlp_pMu_int5*ylkp4/ovlp_pMu_int0**4
            d2OdcdyPmu = -0.125*ovlp_pMu_int10*ylkp3 - 1.25*ovlp_pMu_int7*ylk
            d2Ody2Pmu = ovlp_pMu_int8 + ovlp_pMu_int9
            """extra term in 1/r^2 - now, calculate the alternative 1/r^2"""
            rsquared_int1 = clkPmu**1.5
            rsquared_int3 = dawsn(np.array(0.5*ylk*inv_sqrt_clkPmu,dtype=np.complex128))*2
            rsquared_int5 = rsquared_int1*rsquared_int3/2
            rsquared_int6 = clkPmu**0.5
            rsquared_int10 = 0.125*rsquared_int3*ylk/rsquared_int6 - 0.25
            rsquared_int11 = clkPmu
            rsquared_int13 = 0.25*rsquared_int3*rsquared_int6
            invRsquared_noOvlp=rsquared_int0*rsquared_int5
            dinvRsquared_noOvlp_dc=0.75*rsquared_int0*rsquared_int3*rsquared_int6 + rsquared_int10
            dinvRsquared_noOvlp_dy=0.5*rsquared_int0*rsquared_int11 - rsquared_int13 - rsquared_int14*rsquared_int5

            invRsq-=4*large_ylk*overlapPmu*invRsquared_noOvlp
            dinvRsqdc-=4*large_ylk*(invRsquared_noOvlp*dOdcPmu+overlapPmu*dinvRsquared_noOvlp_dc)
            dinvRsqdy-=4*large_ylk*(invRsquared_noOvlp*dOdyPmu+overlapPmu*dinvRsquared_noOvlp_dy)


            """Taylor expansion for 1/r^2"""
            """1/r² term"""
            kR2T_int0 = clk
            kR2T_int1 = ylkp2
            kR2T_int2 = ylkp4
            kR2T_int3 = clk_inv
            kR2T_int4 = clk_inv
            kR2T_int5 = clk_inv_squared
            kR2T_int6 = clk_inv_squared
            kR2 = 0.5*kR2T_int0 - 0.0833333333333333*kR2T_int1 + kR2T_int2*(0.01875*kR2T_int3 - 0.0104166666666667*kR2T_int4)
            dkR2dc = kR2T_int1*(0.1875*kR2T_int3 - 0.1875*kR2T_int4) + kR2T_int2*(-0.0317708333333333*kR2T_int5 + 0.0234375*kR2T_int6) + 0.5
            dkR2dy = ylkp5*(0.00554315476190476*kR2T_int5 - 0.00911458333333333*kR2T_int6) + ylkp3*(-0.0395833333333333*kR2T_int3 + 0.0729166666666667*kR2T_int4) - 0.166666666666667*ylk
            invRsq+=small_ylk*4*overlap*kR2
            dinvRsqdc+=small_ylk*4*(dOdc*kR2+overlap*dkR2dc)
            dinvRsqdy+=small_ylk*4*(dOdy*kR2+overlap*dkR2dy)
            """e^(-mu^2r²)/r² term"""
            kR2T_int0 = clkPmu
            kR2T_int1 = ylkp2
            kR2T_int2 = ylkp4
            kR2T_int3 = 1/clkPmu
            kR2T_int4 = kR2T_int3
            kR2T_int5 = kR2T_int3**2
            kR2T_int6 = kR2T_int5
            kR2 = 0.5*kR2T_int0 - 0.0833333333333333*kR2T_int1 + kR2T_int2*(0.01875*kR2T_int3 - 0.0104166666666667*kR2T_int4)
            dkR2dc = kR2T_int1*(0.1875*kR2T_int3 - 0.1875*kR2T_int4) + kR2T_int2*(-0.0317708333333333*kR2T_int5 + 0.0234375*kR2T_int6) + 0.5
            dkR2dy = ylkp5*(0.00554315476190476*kR2T_int5 - 0.00911458333333333*kR2T_int6) + ylkp3*(-0.0395833333333333*kR2T_int3 + 0.0729166666666667*kR2T_int4) - 0.166666666666667*ylk
            invRsq-=4*small_ylk*overlapPmu*kR2
            dinvRsqdc-=4*small_ylk*(kR2*dOdcPmu+overlapPmu*dkR2dc)
            dinvRsqdy-=4*small_ylk*(kR2*dOdyPmu+overlapPmu*dkR2dy)


        """The "extra Gaussians" needed to make 1/r^2 = (1/r)^2. Fixing terms as well as c^2 e^(-2alpha^2)"""
        linear_coeff_extra=[c_term**2]+list(self.oneOverRsquared_Gaussian_lincoeff)#+
        oneOverRsquared_Gaussian_exponents_extra=[2*alpha_term**2]+list(self.oneOverRsquared_Gaussian_exponents)#+
        for m in range(len(linear_coeff_extra)):
            alpha=oneOverRsquared_Gaussian_exponents_extra[m]
            linearCoeff=linear_coeff_extra[m]
            invR_Gaussian_contrib0 = alpha + clk
            invR_Gaussian_contrib2 = 1/invR_Gaussian_contrib0
            invR_Gaussian_contrib3 = ylkp2
            invR_Gaussian_contrib4 = 0.25*invR_Gaussian_contrib3
            invR_Gaussian_contrib5 = pi**1.5*exp(glk + invR_Gaussian_contrib2*invR_Gaussian_contrib4)/sqrt(invR_Gaussian_contrib0)**3
            invR_Gaussian_contrib6 = invR_Gaussian_contrib2*invR_Gaussian_contrib5
            invRsq += linearCoeff*invR_Gaussian_contrib5
            dinvRsqdc += linearCoeff*(-3*invR_Gaussian_contrib6/2 - invR_Gaussian_contrib4*invR_Gaussian_contrib5/invR_Gaussian_contrib0**2)
            dinvRsqdy += linearCoeff*0.5*invR_Gaussian_contrib6*ylk

        """Finally, we need to implement the 2ce^(...)erf(mur)/r term"""
        Gauss_correction0 = alpha_term**2 + clk
        Gauss_correction1 = 1/sqrt(Gauss_correction0**3)
        Gauss_correction2 = 1/Gauss_correction0
        Gauss_correction3 = ylk**2
        Gauss_correction4 = 0.25*Gauss_correction3
        Gauss_correction5 = exp(Gauss_correction2*Gauss_correction4 + glk)
        Gauss_correction6 = Gauss_correction1*Gauss_correction5*c_term
        Gauss_correction7 = Gauss_correction6*x0
        Gauss_correction8 = 0.5*ylk
        Gauss_correction9 = invR_mu**2
        Gauss_correction10 = Gauss_correction0**2 + Gauss_correction0*Gauss_correction9
        Gauss_correction11 = invR_mu/sqrt(Gauss_correction10)
        Gauss_correction12 = pi**1.5*erf(Gauss_correction11*Gauss_correction8)
        Gauss_correction13 = Gauss_correction12*Gauss_correction7
        Gauss_correction14 = 2*Gauss_correction0
        Gauss_correction15 = Gauss_correction12*Gauss_correction6
        Gauss_correction16 = pi
        Gauss_correction17 = exp(-Gauss_correction4*Gauss_correction9/Gauss_correction10)
        erf_gauss_correction = Gauss_correction13*Gauss_correction14
        derf_gauss_correctiondc = Gauss_correction0*Gauss_correction1*Gauss_correction16*Gauss_correction17*Gauss_correction5*c_term*invR_mu*(-Gauss_correction0 - Gauss_correction9/2)/Gauss_correction10**(3/2) - Gauss_correction13 - Gauss_correction15*Gauss_correction2*Gauss_correction8
        derf_gauss_correctiondy = Gauss_correction0*Gauss_correction11*Gauss_correction16*Gauss_correction17*Gauss_correction7 - Gauss_correction14*Gauss_correction15*x18 + Gauss_correction15
        gaussC0 = invR_mu**2
        gaussC1 = alpha_term**2
        gaussC2 = clk + gaussC1
        gaussC3 = alpha_term**4
        gaussC4 = clk**2
        gaussC5 = clk*gaussC0
        gaussC6 = clk*gaussC1
        gaussC7 = 2*gaussC6
        gaussC8 = gaussC0*gaussC1
        gaussC9 = gaussC3 + gaussC4 + gaussC5 + gaussC7 + gaussC8
        gaussC10 = ylkp2
        gaussC11 = 0.5*gaussC10
        gaussC12 = 2.0*gaussC3 + 2.0*gaussC4 + 4.0*gaussC6
        gaussC13 = alpha_term**6
        gaussC14 = clkp3
        gaussC15 = 3*clk*gaussC3 + 3*gaussC1*gaussC4 + gaussC13 + gaussC14
        gaussC16 = pi**1.0*c_term*invR_mu*exp(glk)/sqrt(gaussC15)
        gaussC17 = gaussC0*gaussC3 + gaussC0*gaussC4 + 2*gaussC1*gaussC5 + gaussC15
        gaussC18 = invR_mu**4
        gaussC19 = 6*gaussC3
        gaussC20 = 2*gaussC0
        gaussC21 = alpha_term**8 + clk**4 + 4*clk*gaussC13 + 4*gaussC1*gaussC14 + gaussC13*gaussC20 + gaussC14*gaussC20 + gaussC18*gaussC3 + gaussC18*gaussC4 + gaussC18*gaussC7 + gaussC19*gaussC4 + gaussC19*gaussC5 + 6*gaussC4*gaussC8
        gaussC22 = gaussC21*gaussC9
        gaussC23 = gaussC17*gaussC22
        gaussC24 = 1.0*gaussC5 + 1.0*gaussC8
        gaussC25 = gaussC17*gaussC21
        gaussC26 = 0.5*gaussC3 + 0.5*gaussC4 + 0.25*gaussC5 + 1.0*gaussC6 + 0.25*gaussC8
        gaussC27 = gaussC0*gaussC2
        gaussC28 = gaussC17*gaussC9
        gaussC29 = gaussC16/(gaussC17*gaussC2*gaussC21*gaussC9**(3/2))
        gaussC30 = gaussC2**2

        erfgaus_Taylor = gaussC16*(clk*gaussC11 + 0.333333333333333*gaussC0*gaussC10 + gaussC1*gaussC11 + gaussC12 + 2.0*gaussC5 + 2.0*gaussC8)/(sqrt(gaussC9)*(gaussC0 + gaussC2))
        derfgaus_Taylordc = gaussC29*(gaussC10*(-gaussC2*gaussC22*gaussC26 - 0.75*gaussC23 + 0.0833333333333333*gaussC25*gaussC27 + gaussC26*gaussC27*gaussC28) - 1.0*gaussC2*gaussC23 - gaussC2*gaussC25*(gaussC12 + gaussC24))
        derfgaus_Taylordy = gaussC29*ylk*(-gaussC10*(0.0833333333333333*gaussC0*gaussC17*gaussC2*gaussC21 + 0.0833333333333333*gaussC0*gaussC21*gaussC30*gaussC9 - 0.05*gaussC18*gaussC28*gaussC30 - 0.25*gaussC23) + gaussC17*gaussC2*gaussC21*(gaussC24 - 0.333333333333333*gaussC27 + 1.0*gaussC3 + 1.0*gaussC4 + 2.0*gaussC6))

        invRsq+=2*large_ylk*erf_gauss_correction #OK
        dinvRsqdc+=2*large_ylk*derf_gauss_correctiondc #OK
        dinvRsqdy+=2*large_ylk*derf_gauss_correctiondy #OK
        invRsq+=2*small_ylk*erfgaus_Taylor #OK
        dinvRsqdc+=2*small_ylk*derfgaus_Taylordc #OK
        dinvRsqdy+=2*small_ylk*derfgaus_Taylordy #OK
        self.invR=np.asarray(invR,dtype=np.complex128)
        self.dinvRdc=np.asarray(dinvRdc,dtype=np.complex128)
        self.dinvRdy=np.asarray(dinvRdy,dtype=np.complex128)
        self.d2invRdc2=np.asarray(d2invRdc2,dtype=np.complex128)
        self.d2invRdcdy=np.asarray(d2invRdcdy,dtype=np.complex128)
        self.d2invRdy2=np.asarray(d2invRdy2,dtype=np.complex128)
        self.invRsq=np.asarray(invRsq,dtype=np.complex128)
        self.dinvRsqdc=np.asarray(dinvRsqdc,dtype=np.complex128)
        self.dinvRsqdy=np.asarray(dinvRsqdy,dtype=np.complex128)

        self.invR[abs(overlap)<threshold_discard]=0
        self.dinvRdc[abs(overlap)<threshold_discard]=0
        self.overlap[abs(overlap)<threshold_discard]=0
        self.overlap_normalized[abs(overlap)<threshold_discard]=0
        self.dinvRdy[abs(overlap)<threshold_discard]=0
        self.d2invRdy2[abs(overlap)<threshold_discard]=0
        self.d2invRdcdy[abs(overlap)<threshold_discard]=0
        self.d2invRdc2[abs(overlap)<threshold_discard]=0
        self.invRsq[abs(overlap)<threshold_discard]=0
        self.dinvRsqdc[abs(overlap)<threshold_discard]=0
        self.dinvRsqdy[abs(overlap)<threshold_discard]=0
        self.dOdc[abs(overlap)<threshold_discard]=0
        self.dOdy[abs(overlap)<threshold_discard]=0
        self.d2Odc2[abs(overlap)<threshold_discard]=0
        self.d2Ody2[abs(overlap)<threshold_discard]=0
        self.d2Odcdy[abs(overlap)<threshold_discard]=0
        self.d3Odc3[abs(overlap)<threshold_discard]=0
        self.d3Odcdy2[abs(overlap)<threshold_discard]=0
        self.d3Ody3[abs(overlap)<threshold_discard]=0
        self.d3Odc2dy[abs(overlap)<threshold_discard]=0
