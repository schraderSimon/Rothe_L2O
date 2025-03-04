import numpy as np
def diis(func, x0, grad=True, learning_rate=0.01, tol=1e-4, max_iter=100, m=10):
    x = x0
    residuals = []
    x_list = []
    func_vals=[]
    for i in range(max_iter):
        if grad is not True:

            grad_val = grad(x)
            func_val=func(x)
        else:
            func_val,grad_val=func(x)    
        # Perform a gradient descent step
        func_vals.append(func_val)
        x_new = x - learning_rate * grad_val
        #hess_inv0=np.diag(1/(1e-15+abs(grad_val)))
        #result = minimize(func, x, method='BFGS', jac=grad, tol=1e-14, options={'maxiter': 1,"hess_inv0":hess_inv0})
        #x_new = result.x
        # Calculate the error (difference between new and old x)
        error = x_new - x
        
        residuals.append(error)
        x_list.append(x_new)
        if len(residuals) > m:
            residuals.pop(0)
            x_list.pop(0)
        try:
            change=abs((func_vals[-1]-func_vals[-5])/func_vals[-1])
            if change < tol:
                break
        except:
            pass
        # Construct DIIS step
        b_mat = np.array([[np.dot(r1, r2) for r2 in residuals] for r1 in residuals])
        b_mat = np.pad(b_mat, ((0, 1), (0, 1)), 'constant', constant_values=-1)
        b_mat[-1, -1] = 0
        
        rhs = np.zeros(len(b_mat))
        rhs[-1] = -1
        #print(b_mat)
        b_mat=b_mat+1e-20*np.eye(len(b_mat))
        #coeffs = np.linalg.lstsq(b_mat, rhs)[0][:-1]
        coeffs = np.linalg.solve(b_mat, rhs)[:-1]
        # Update x using DIIS extrapolation
        x_diis = sum(c * xk for c, xk in zip(coeffs, x_list))
        
        # Set x for next iteration
        x = x_diis
    return x, i, func_vals[-1]
def gradient_descent(func, grad, x0, learning_rate=0.001, tol=1e-6, max_iter=10000):
    x = x0
    for i in range(max_iter):
        grad_val = grad(x)
        x = x - learning_rate * grad_val
        if np.linalg.norm(grad_val) < tol:
            break
    return x, i

from scipy.optimize import minimize

def optimize_bfgs(func,grad, x0,hess=None,max_iter=100):
    result = minimize(func, x0, method='BFGS', jac=grad,options={"hess_inv0":hess,"maxiter":max_iter,"gtol":1e-11})
    return result.x, result.nfev

def minimize_transformed_bonds(error_function,start_params,multi_bonds=0.1,gtol=1e-9,maxiter=20,gradient=None,both=False,lambda_grad0=1e-14,hess_inv=None,scale="log",intervene=True):
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
        #returnval= 0.5*(maxs-mins)/(coshvals**2)
        #return returnval
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

    dp=multi_bonds*np.ones(len(start_params)) #Percentage (times 100) how much the parameters are alowed to change compared to previous time step
    range_nonlin=[0.05,0.1,1,1]*(len(start_params)//4)
    rangex=range_nonlin
    rangex=np.array(rangex)
    mins=start_params-rangex-dp*abs(start_params)
    maxs=start_params+rangex+dp*abs(start_params)
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
        hess_inv0=np.eye(len(grad0))/np.linalg.norm(grad0)*1e2
        hess_inv0=np.diag(1/abs(grad0+lambda_grad0*np.array(len(grad0))))

    else:
        hess_inv0=hess_inv
    
    numiter=0
    f_storage=[]
    def callback_func(intermediate_result: scipy.optimize.OptimizeResult):
        if intervene:
            nonlocal numiter
            nonlocal f_storage
            nonlocal transformed_sol
            nonlocal minval
            transformed_sol=intermediate_result.x
            fun=intermediate_result.fun
            minval=fun
            if scale=="log":
                re=sqrt(np.exp(fun))
            else:
                re=sqrt(fun)
            f_storage.append(re)
            miniter=10
            compareto_opt=10
            compareto=compareto_opt if compareto_opt<miniter else miniter-1
            if  numiter>=miniter: 
                if f_storage[-1]/f_storage[-compareto]>0.995 and f_storage[-1]/f_storage[-compareto]<1:
                    raise ConvergedError
            numiter+=1

    converged=False
    try:
        
        if both is False:
            sol=minimize(transformed_error,transformed_params,jac=transformed_gradient,
                        method='BFGS',options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':gtol},
                        callback=callback_func)
        else:
            sol=minimize(transform_error_and_gradient,transformed_params,jac=True,
                        method='BFGS',options={"hess_inv0":hess_inv0,'maxiter':maxiter,'gtol':gtol},
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
    return untransform_params(transformed_sol), minval, end-start,numiter
