import numpy as np
from numba import jit
from numpy import cosh, tanh, arctanh, sin, cos, tan, arcsin, arccos, exp, array, sqrt, pi

import scipy
from scipy.optimize import minimize
import time
from scipy.optimize import OptimizeResult



def Coulomb(x, Z, x_c=0.0, alpha=1.0):
    """
    Coulomb potential in 1D.

    Args:
        x (np.ndarray): The grid points.
        Z (float): The nuclear charge.
        x_c (float): The nuclear position.
        a (float): The regularization parameter.
    """
    return -Z / np.sqrt((x - x_c) ** 2 + alpha)

class Molecule1D:
    def __init__(self, R=[0.0], Z=[1], alpha=1.0):
        """
        Molecular potential in 1D.

        Args:
            R (list): The nuclear positions.
            Z (list): The nuclear charges.
            alpha (float): The regularization parameter.
        """
        self.R_list = R
        self.Z_list = Z
        if alpha <= 0:
            raise ValueError("The regularization parameter must be positive.")
        self.alpha = alpha

    def __call__(self, x):
        if isinstance(x, float):
            potential = 0
        else:
            potential = np.zeros(len(x),dtype=np.complex128)
        for R, Z in zip(self.R_list, self.Z_list):
            potential += Coulomb(x, Z=Z, x_c=R, alpha=self.alpha)
        return potential
def laserfield(E0, omega, td):
    """
    Sine-squared laser pulse.

    Args:
        t (float): Time.
        E0 (float): Maximum field strength.
        omega (float): Laser frequency.
        td (float): Duration of the laser pulse.
    """
    def field(t):
        return -E0 * np.sin(omega * t) * np.sin(t*np.pi / td) ** 2
    return field


class ConvergedError(Exception):
    def __init__(self):
        pass
def save_wave_function(filename, wave_function, time_step,xval,time,rothe_error,norms,ngauss,norbs):
    try:
        # Try to load the existing data
        existing_data = np.load(filename)
        params=existing_data['params']
        params=list(params)
        xvals=list(existing_data['xvals'])
        times=list(existing_data['times'])
        rothe_errors=list(existing_data['rothe_errors'])
        norms_list=list(existing_data['norms'])
        number_of_basis_functions=list(existing_data['nbasis'])
    except FileNotFoundError:
        params=[]
        xvals=[]
        times=[]
        rothe_errors=[]
        norms_list=[]
        number_of_basis_functions=[]
    if len(params)==0 or len(params[-1])==len(wave_function):
        pass
    elif len(params[-1])<len(wave_function):
        print("Wave function has wrong length")
        print(ngauss)
        for i in range(len(params)):
            params_i=np.zeros_like(wave_function)
            lincoeff_real=params[i][:(ngauss-1)*norbs]#.reshape((ngauss,norbs))
            lincoeff_complex=params[i][(ngauss-1)*norbs:(ngauss-1)*norbs*2]#.reshape((ngauss,norbs))
            gaussian_nonlincoeffs=params[i][(ngauss-1)*norbs*2:]
            params_i[:(ngauss-1)*norbs]=lincoeff_real
            params_i[ngauss*norbs:(2*ngauss-1)*norbs]=lincoeff_complex
            params_i[ngauss*norbs*2:-4]=gaussian_nonlincoeffs
            params_i[-4:]=[1,0,0,0]
            params[i]=params_i
    elif len(params[-1])>len(wave_function):
        print("Wave function has wrong length")
        for i in range(len(params)):
            ngauss_wrong=len(params[i])//(4+norbs*2)
            params_i=np.zeros_like(wave_function)
            lincoeff_real=params[i][:ngauss*norbs]
            lincoeff_complex=params[i][ngauss_wrong*norbs:(ngauss+ngauss_wrong)*norbs]
            gaussian_nonlincoeffs=params[i][ngauss_wrong*norbs*2:-4*(ngauss_wrong-ngauss)]
            params_i[:len(lincoeff_real)]=lincoeff_real
            params_i[len(lincoeff_real):len(lincoeff_real)+len(lincoeff_complex)]=lincoeff_complex
            params_i[len(lincoeff_real)+len(lincoeff_complex):]=gaussian_nonlincoeffs
            params[i]=params_i
    params.append(wave_function)
    xvals.append(xval)
    times.append(time)
    rothe_errors.append(rothe_error)
    norms_list.append(norms)
    number_of_basis_functions.append(ngauss)
    # Append new wave function at the current time step
    np.savez(filename, params=params, time_step=time_step,times=times,xvals=xvals,rothe_errors=rothe_errors,norms=norms_list,nbasis=number_of_basis_functions)
    print(" \nTime: %.2f, Cumul R.E.: %.2e"%(np.real(times[-1]),np.sum((rothe_errors))))
def minimize_hessian(error_function,start_params,num_gauss):
    params=start_params
    param_list=[]
    error_list=[]
    i_min=5
    i_max=15
    for i in range(i_max):
        error,grad,hess=error_function(params,num_gauss,hessian=True)
        diag_hess=np.diag(np.diag(hess))
        hess_inv=np.linalg.inv(hess+1000*diag_hess)
        paramsNew=params-np.dot(hess_inv,grad)
        #print("Iteration %d: Error: %.3e"%(i,error))
        params=paramsNew
        param_list.append(params)
        error_list.append(error)
        if i>=i_min:
            if error_list[i]/error_list[i-i_min]>0.995:
                break
    print("Niter: %d"%len(error_list))
    best_error=np.argmin(error_list)
    return param_list[best_error],error_list[best_error]
def calculate_potential(Z_list,R_list,alpha,points):
    V=Molecule1D(R_list,Z_list,alpha)(points)
    return V
def e_e_interaction(x):
    """
    Calculate the electron-electron interaction term
    """
    Vee = np.zeros((len(x), len(x)),dtype=np.complex128)
    for i in range(len(x)):
        for j in range(len(x)):
            Vee[i, j] = Coulomb(x[i] - x[j],x_c=0, Z=-1, alpha=1)
    return Vee
def gauss(x,a,b,p,q):
    bredde = a**2 + 1j*b
    qminx = q - x
    jp=1j*p
    gaussval =sqrt(abs(a)/sqrt(pi/2))* exp(-qminx * (jp + bredde*qminx))
    return gaussval
def minus_half_laplacian(x,a,b,p,q):
    bredde=(a**2 + 1j*b)
    qminx=q-x
    return (bredde - 2.0*(0.5j*p + bredde*qminx)**2)*gauss(x,a,b,p,q)
@jit(nopython=True, fastmath=True,cache=True)
def gauss_and_minushalflaplacian(x, a, b, p, q):
    bredde = a**2 + 1j*b
    qminx = q - x
    jp=1j*p
    gaussval =sqrt(abs(a)/sqrt(pi/2))* exp(-qminx * (jp + bredde*qminx))
    minus_half_laplace = (bredde - 0.5 * (jp + 2*bredde*qminx)**2) * gaussval
    return gaussval, minus_half_laplace

@jit(nopython=True, fastmath=True,cache=True)
def gauss_and_minushalflaplacian_and_derivs(x,a,b,p,q):
    asq=a**2
    bredde = asq + 1j*b
    qminx = q - x
    jp=1j*p
    br_qminx=bredde*qminx
    gaussval =sqrt(abs(a)/sqrt(pi/2))* exp(-qminx * (jp + br_qminx))
    minus_half_laplace = (bredde - 0.5 * (jp + 2*br_qminx)**2) * gaussval
    
    aderiv=(-2*a*q**2 + 4*a*q*x - 2*a*x**2 + 1/(2*a))*gaussval
    bderiv=1j*(-q**2 + 2.0*q*x - x**2)*gaussval
    pderiv=1j*(-q + x)*gaussval
    qderiv=(-2.0*asq*q + 2.0*asq*x - 2j*b*q + 2j*b*x - jp)*gaussval
    tempval=0.5*jp + br_qminx
    tempvalsq=tempval**2
    brmttvsq=bredde - 2*tempvalsq
    b_inv=1/brmttvsq
    aderiv_kin=minus_half_laplace*(-asq*(qminx**2*(4*asq + 4*1j*b - 8*tempvalsq) + 16*qminx*tempval - 4) + brmttvsq)/(2*a)*b_inv
    bderiv_kin=minus_half_laplace*1j*(-qminx**2*brmttvsq - 4*qminx*tempval + 1)*b_inv
    pderiv_kin=minus_half_laplace*1j*(-jp - 2.0*br_qminx - qminx*brmttvsq)*b_inv
    qderiv_kin=minus_half_laplace*(-2.0*(2*asq + 2j*b)*tempval - (jp + 2*br_qminx)*brmttvsq)*b_inv
    return (gaussval, minus_half_laplace, aderiv, bderiv, pderiv, qderiv, 
            aderiv_kin, bderiv_kin, pderiv_kin, qderiv_kin)

def make_minimizer_function(avals_min,avals_max,bvals_min,bvals_max,pvals_min,pvals_max,muvals_min,muvals_max):
    def minimize_transformed_bonds(error_function,start_params,multi_bonds=0.1,gtol=1e-9,maxiter=20
                                   ,gradient=None,both=False,lambda_grad0=1e-14,hess_inv=None,intervene=True,timepoint=0,write_file=False):
        """
        Minimizes with min_max bonds as described in https://lmfit.github.io/lmfit-py/bounds.html
        """
        def transform_params(untransformed_params):
            newparams=np.zeros_like(untransformed_params)
            """
            for i,param in enumerate(2*(untransformed_params-mins)/(maxs-mins)-1):
                if param>0.9999:
                    newparams[i]=4
                elif param<-0.9999:
                    newparams[i]=-4
                else:
                    newparams[i]=arctanh(param)
            """
            for i,param in enumerate(2*(untransformed_params-mins)/(maxs-mins)-1):
                if param>0.9999:
                    newparams[i]=1.5566
                elif param<-0.9999:
                    newparams[i]=-1.5566
                else:
                    newparams[i]=arcsin(param)
            
            return newparams
        def untransform_params(transformed_params):
            #return mins+(maxs-mins)/2*(1+tanh(transformed_params))
            return mins+(maxs-mins)/2*(1+sin(transformed_params))
        def chainrule_params(transformed_params):
            #coshvals=cosh(transformed_params)
            #print("Biggest coshval: ",np.max(np.abs(coshvals)))
            #sys.exit(0)
            
            #return returnval= 0.5*(maxs-mins)/(coshvals**2)
            return 0.5*(maxs-mins)*cos(transformed_params)

        def transformed_error(transformed_params):
            error=error_function(untransform_params(transformed_params))
            return error
        def transformed_gradient(transformed_params):
            orig_grad=gradient(untransform_params(transformed_params))
            chainrule_grad=chainrule_params(transformed_params)
            grad=orig_grad*chainrule_grad
            if(np.isnan(sum(grad))):

                print("gradient has nan")
                print(grad)
                print(transformed_error(transformed_params))
                if np.isnan(np.sum(orig_grad)):
                    print("Original gradient has nan...")
                return np.nan_to_num(grad)
            return grad
        def transform_error_and_gradient(transformed_params):
            untransformed_params=untransform_params(transformed_params)
            error,orig_grad=error_function(untransformed_params)
            chainrule_grad=chainrule_params(transformed_params)
            grad=orig_grad*chainrule_grad
            if(np.isnan(sum(grad))):

                print("gradient has nan")
                print(grad)
                print(transformed_error(transformed_params))
                if np.isnan(np.sum(orig_grad)):
                    print("Original gradient has nan...")
                return np.nan_to_num(grad)
            return error,grad

        num_gauss=len(start_params)//4
        range_nonlin=[0.05,0.1,0.1,0.1]*num_gauss
        rangex=np.array(range_nonlin)
        mins=start_params-rangex
        
        maxs=(start_params+rangex)
        for i in range(num_gauss):
            mins[i*4]-=multi_bonds*abs(start_params[i*4])
            maxs[i*4]+=multi_bonds*abs(start_params[i*4])
            mins[i*4+1]-=multi_bonds*abs(start_params[i*4+1])
            maxs[i*4+1]+=multi_bonds*abs(start_params[i*4+1])
            mins[i*4+2]-=multi_bonds*abs(start_params[i*4+2])
            maxs[i*4+2]+=multi_bonds*abs(start_params[i*4+2])
            mins[i*4+3]-=multi_bonds*abs(start_params[i*4+3])
            maxs[i*4+3]+=multi_bonds*abs(start_params[i*4+3])
        mmins=np.zeros_like(mins)
        mmaxs=np.zeros_like(maxs)
        mmins[0::4]=avals_min
        mmaxs[0::4]=avals_max
        mmins[1::4]=bvals_min
        mmaxs[1::4]=bvals_max
        mmins[2::4]=pvals_min
        mmaxs[2::4]=pvals_max
        mmins[3::4]=muvals_min
        mmaxs[3::4]=muvals_max

        for i in range(len(mins)):
            if mins[i]<mmins[i]:
                mins[i]=mmins[i]
            if maxs[i]>mmaxs[i]:
                maxs[i]=mmaxs[i]
        for i,sp in enumerate(start_params):
            if abs(sp-mins[i])<1e-1:
                pass
                #start_params[i]=mins[i]+1e-2
            if abs(sp-maxs[i])<1e-2:
                pass
                #start_params[i]=maxs[i]-1e-2

        transformed_params=transform_params(start_params)
        transformed_params=np.real(np.nan_to_num(transformed_params))

        start=time.time()
        if both:
            err0,grad0=transform_error_and_gradient(transformed_params)
        else:
            grad0=transformed_gradient(transformed_params)
        if hess_inv is None:
            #hess_inv0=np.eye(len(grad0))/np.linalg.norm(grad0)
            hess_inv0=np.diag(1/(abs(grad0)+lambda_grad0))

        else:
            hess_inv0=hess_inv
        numiter=0
        scale=np.ones(len(grad0))
        hess_inv0=hess_inv0/abs(scale)
        x_scaled=transformed_params/scale
        def f_scaled(x_scaled):
            err,grad=transform_error_and_gradient(x_scaled*scale)
            #print(err)
            newgrad=grad/scale
            #print(np.sort(abs(newgrad)))
            return err,newgrad
        err,grad0=f_scaled(x_scaled)
        f_storage=[]
        def callback_func(intermediate_result: scipy.optimize.OptimizeResult):
            nonlocal numiter
            nonlocal f_storage
            nonlocal transformed_sol
            nonlocal minval
            transformed_sol=intermediate_result.x
            fun=intermediate_result.fun
            minval=fun
            re=sqrt(fun)
            f_storage.append(re)
            miniter=20
            compareto=miniter-1
            if  numiter>=miniter and intervene: 
                if f_storage[-1]/f_storage[-compareto]>0.999 and f_storage[-1]/f_storage[-compareto]<1:
                    raise ConvergedError
            numiter+=1

        converged=False
        try:
            
            if both is False:
                sol=minimize(transformed_error,transformed_params,jac=transformed_gradient,
                            method='BFGS',options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':gtol},
                            callback=callback_func)
            else:
                sol=minimize(f_scaled,x_scaled,jac=True,
                            method='BFGS',options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':gtol,"c1":1e-4,"c2":0.9},
                            callback=callback_func)
            transformed_sol = sol.x
            minval=sol.fun #Not really fun, I am lying here
            numiter=sol.nit
            """
            transformed_sol,numiter,minval=diis(transform_error_and_gradient,transformed_params)
            """
        except ConvergedError:
            converged=True
            """
            This means that the callback function tells me that the function is not decreasing anymore.
            The important values are updated in the callback function (self.transformed_sol and self.minval)
            so that this "error" can safely be ignored.
            """
            pass

        end=time.time()
        if len(f_storage)!=0 and write_file:
            filename="error_trajectory/trajectory_%.2f.npz"%timepoint
            np.savez(filename,f_storage=f_storage)
        return untransform_params(transformed_sol*scale), minval, end-start,numiter
    return minimize_transformed_bonds