from time_evolution_and_optimization_frozen import *
def sine_field_func(t):
    omega=0.057
    t_cycle = 2 * np.pi / omega
    td = 3 * t_cycle
    dt = t
    pulse = (
        (np.sin(np.pi * dt / td) ** 2)
        * np.heaviside(dt, 1.0)
        * np.heaviside(td - dt, 1.0)
        * np.sin(omega * dt)
        * E0
    )
    return pulse
def fieldfunc(t):
    return sine_field_func(t)
def make_error_and_gradient_functions(E0,quality,t0):
    h=0.2
    invR_mu=100
    omega=0.057
    epsilon_values = {
        (0.03, 1): 3e-1,
        (0.03, 2): 3e-2,
        (0.03, 3): 5e-3,
        (0.06, 1): 1,
        (0.06, 2): 2.5e-1,
        (0.06, 3): 2.5e-2,
        (0.12, 1): 12,
        (0.12, 2): 6,
        (0.12, 3): 3,
    }
    epsilon = epsilon_values.get((E0, quality), None)  # Returns None if (E0, quality) isn't found

    filename="outputs/NEWNEW/data_E0%.3f_omega%.3f_invRmu%d_dt%.2f_epsilon%.2e_frozenCore_compressed.h5"%(E0,omega,invR_mu,h,epsilon)
    try:
        with h5py.File(filename, "r+") as data_file:
            times=np.array(data_file["times"])
            t0_index=np.argmin(abs(times-t0))
            params=np.array(data_file["parameters_t=%.2f"%t0])
            cvals=np.array(data_file["coefficients_t=%.2f"%t0])
            err_t0=np.array(data_file["rothe_error"])[t0_index]
    except FileNotFoundError:
        raise Exception("File not found")
    tfinal=500
    wave_function=erf_WF(params=params,basis_coefficients=cvals,fieldFunc=fieldfunc,potential_params=[invR_mu])
    timeEvolver=TimeEvolution(wave_function,fieldFunc=sine_field_func,potential_params=[invR_mu],potential_type="erf",h=h,
                            T=tfinal,epsilon=epsilon,t0=t0,error_t0=err_t0,filename=filename,write_to_file=False)
    wfoptimizer=WFOptimizer(wavefunction=timeEvolver.WF,potential_params=timeEvolver.potential_params,t=timeEvolver.t,
                            h=timeEvolver.h,error_threshold=timeEvolver.error_threshold,fieldFunc=timeEvolver.fieldFunc,
                            potential_type=timeEvolver.potential_type,file=timeEvolver.filename,
                            allow_adding=False,hlm=False,gauge="length")
    def error_function_optimization(parameters,frozen_params):
        full_parameters=np.concatenate((frozen_params,parameters))
        error=wfoptimizer.error_for_optimization(full_parameters)
        return error
    def gradient_optimization(parameters,frozen_params):
        full_parameters=np.concatenate((frozen_params,parameters))
        grad=wfoptimizer.jac_for_optimization(full_parameters)[len(frozen_params):]
        return grad
    def create_wrapper(function, parameters_frozen):
        def wrapped(parameters):
            return function(parameters, parameters_frozen)
        return wrapped
    parameters=params[100:]
    parameters_frozen=params[:100]

    optimizee=create_wrapper(error_function_optimization, parameters_frozen)
    optimizee_grad=create_wrapper(gradient_optimization, parameters_frozen)
    return optimizee,optimizee_grad,parameters


E0=float(sys.argv[1])
quality=int(sys.argv[2]) #1, 2 or 3
t0=float(sys.argv[3])
optimizee,optimizee_grad,parameters=make_error_and_gradient_functions(E0,quality,t0)
from scipy.optimize import minimize
error_initial=optimizee(parameters)
grad_initial=optimizee_grad(parameters)
print("Initial error: ", error_initial)
print("Initial gradient: ", grad_initial)
hess_inv0=np.diag(1/np.abs(grad_initial))
result=minimize(optimizee,parameters,method="BFGS",jac=optimizee_grad,options={"hess_inv0":hess_inv0,"gtol":1e-10})
print(result)