from scipy.optimize import minimize
from numba import jit
import sys
from quadratures import gaussian_quadrature, trapezoidal_quadrature
from mean_field_grid_rothe import *
np.set_printoptions(linewidth=300, precision=16, suppress=True, formatter={'float': '{:0.7e}'.format})
from exchange_correlation_functionals import v_xc,epsilon_xc
# Function to save or append data
def v_ee_coulomb(grids):
    vp = 1 / np.sqrt(grids**2 + 1)
    return vp
def hartree_potential(grid,rho,weights,vee=v_ee_coulomb):
    v_h = np.zeros_like(rho)
    for i, xi in enumerate(grid):
        v_h[i] = np.sum(rho*vee(xi-grid)*weights)
    return v_h



@jit(nopython=True,cache=True, fastmath=False,parallel=True)
def setupfunctions(gaussian_nonlincoeffs,points):
    if gaussian_nonlincoeffs.ndim==1:
        num_gauss=1
    else:
        num_gauss = len(gaussian_nonlincoeffs)
    functions = np.zeros((num_gauss, len(points)), dtype=np.complex128)
    minus_half_laplacians = np.zeros((num_gauss, len(points)), dtype=np.complex128)
    if gaussian_nonlincoeffs.ndim==1:
        avals=[gaussian_nonlincoeffs[0]]
        bvals=[gaussian_nonlincoeffs[1]]
        pvals=[gaussian_nonlincoeffs[2]]
        qvals=[gaussian_nonlincoeffs[3]]
    else:
        avals=gaussian_nonlincoeffs[:,0]
        bvals=gaussian_nonlincoeffs[:,1]
        pvals=gaussian_nonlincoeffs[:,2]
        qvals=gaussian_nonlincoeffs[:,3]
    for i in range(num_gauss):
        indices_of_interest=np.where((np.abs(points-qvals[i])*avals[i])<6)
        funcvals, minus_half_laplacian_vals = gauss_and_minushalflaplacian(points[indices_of_interest], avals[i], bvals[i], pvals[i], qvals[i])

        functions[i][indices_of_interest] = funcvals
        minus_half_laplacians[i][indices_of_interest] = minus_half_laplacian_vals
    
    return functions, minus_half_laplacians
@jit(nopython=True,cache=True, fastmath=False,parallel=True)
def setupfunctionsandDerivs(gaussian_nonlincoeffs, points):
    """
    Evaluate complex Gaussian functions and their derivatives on a grid,
    but only at grid points where |x - q| * a < 6.

    Parameters
    ----------
    gaussian_nonlincoeffs : (N,4) array or 1D array of length 4
        The nonlinear parameters for the Gaussians, where each row is
        [a, b, p, q]. (If a 1D array is given, there is one Gaussian.)
    points : 1D array
        The grid points on which the functions and derivatives are evaluated.

    Returns
    -------
    functions : (N, len(points)) complex array
        The values of the Gaussians.
    minus_half_laplacians : (N, len(points)) complex array
        The values of the minus-half-laplacian operator on the Gaussians.
    aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs : (N, len(points)) complex arrays
        The derivatives of the Gaussians with respect to a, b, p, and q.
    aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs : (N, len(points)) complex arrays
        The derivatives of the kinetic (minus-half-laplacian) term with respect to a, b, p, and q.
    """
    # Determine number of Gaussians and extract parameters.
    if gaussian_nonlincoeffs.ndim == 1:
        num_gauss = 1
        # Convert to 1D numpy arrays of length 1 for each parameter.
        avals = np.array([gaussian_nonlincoeffs[0]])
        bvals = np.array([gaussian_nonlincoeffs[1]])
        pvals = np.array([gaussian_nonlincoeffs[2]])
        qvals = np.array([gaussian_nonlincoeffs[3]])
    else:
        num_gauss = gaussian_nonlincoeffs.shape[0]
        avals = gaussian_nonlincoeffs[:, 0]
        bvals = gaussian_nonlincoeffs[:, 1]
        pvals = gaussian_nonlincoeffs[:, 2]
        qvals = gaussian_nonlincoeffs[:, 3]
    
    nPoints = points.shape[0]
    
    # Allocate output arrays.
    functions         = np.zeros((num_gauss, nPoints), dtype=np.complex128)
    minus_half_laplacians = np.zeros((num_gauss, nPoints), dtype=np.complex128)
    aderiv_funcs      = np.zeros((num_gauss, nPoints), dtype=np.complex128)
    bderiv_funcs      = np.zeros((num_gauss, nPoints), dtype=np.complex128)
    pderiv_funcs      = np.zeros((num_gauss, nPoints), dtype=np.complex128)
    qderiv_funcs      = np.zeros((num_gauss, nPoints), dtype=np.complex128)
    aderiv_kin_funcs  = np.zeros((num_gauss, nPoints), dtype=np.complex128)
    bderiv_kin_funcs  = np.zeros((num_gauss, nPoints), dtype=np.complex128)
    pderiv_kin_funcs  = np.zeros((num_gauss, nPoints), dtype=np.complex128)
    qderiv_kin_funcs  = np.zeros((num_gauss, nPoints), dtype=np.complex128)
    
    # Loop over Gaussians in parallel.
    for i in range(num_gauss):
        # Select grid indices where |x - q|*a < 6.
        # np.nonzero returns a tuple; [0] extracts the 1D indices.
        indices_of_interest = np.nonzero((np.abs(points - qvals[i]) * avals[i]) < 6)[0]
        
        # Evaluate the Gaussian and its derivatives on these indices.
        funcvals, minus_half_laplacian_vals, da, db, dp, dq, dTa, dTb, dTp, dTq = \
            gauss_and_minushalflaplacian_and_derivs(points[indices_of_interest],
                                                    avals[i], bvals[i], pvals[i], qvals[i])
        
        # Store the computed values.
        functions[i, indices_of_interest]         = funcvals
        minus_half_laplacians[i, indices_of_interest] = minus_half_laplacian_vals
        aderiv_funcs[i, indices_of_interest]        = da
        bderiv_funcs[i, indices_of_interest]        = db
        pderiv_funcs[i, indices_of_interest]        = dp
        qderiv_funcs[i, indices_of_interest]        = dq
        aderiv_kin_funcs[i, indices_of_interest]    = dTa
        bderiv_kin_funcs[i, indices_of_interest]    = dTb
        pderiv_kin_funcs[i, indices_of_interest]    = dTp
        qderiv_kin_funcs[i, indices_of_interest]    = dTq

    return (functions, minus_half_laplacians,
            aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs,
            aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs)
def calculate_overlapmatrix(functions,wT):
    num_gauss=len(functions)
    overlap_matrix=np.zeros((num_gauss,num_gauss),dtype=np.complex128)
    for i in range(num_gauss):
        i_conj=np.conj(functions[i])
        for j in range(i,num_gauss):
            integrand_overlap=i_conj*functions[j]
            overlap_integraded=wT@integrand_overlap
            overlap_matrix[i,j]=overlap_integraded
            overlap_matrix[j,i]=np.conj(overlap_integraded)
    return overlap_matrix

def make_orbitals(C,gaussian_nonlincoeffs):
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs.reshape((C.shape[0],4)),points)
    return make_orbitals_numba(C,gaussian_nonlincoeffs,functions)

@jit(nopython=True,cache=True,fastmath=False)
def make_orbitals_numba(C,gaussian_nonlincoeffs,functions):
    nbasis=C.shape[0]
    norbs=C.shape[1]
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((nbasis,4))
    orbitals=np.zeros((norbs,len(points)),dtype=np.complex128)
    for i in range(norbs):
        orbital=np.zeros_like(points,dtype=np.complex128)
        for j in range(nbasis):
            orbital+=C[j,i]*functions[j]
        orbitals[i]=orbital
    return orbitals

def calculate_Fgauss(fockOrbitals,gaussian_nonlincoeffs,num_gauss,time_dependent_potential=None):
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs,points)
    return calculate_Fgauss_fast(np.array(fockOrbitals),num_gauss,time_dependent_potential,np.array(functions),np.array(minus_half_laplacians))
def calculate_v_gauss(fockOrbitals,gaussian_nonlincoeffs,num_gauss,time_dependent_potential=None):
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((num_gauss,4))
    functions,minus_half_laplacians=setupfunctions(gaussian_nonlincoeffs,points)
    return calculate_v_gauss_fast(np.array(fockOrbitals),num_gauss,time_dependent_potential,np.array(functions),np.array(minus_half_laplacians))

@jit(nopython=True,fastmath=True)
def calculate_Fgauss_fast(fockOrbitals,num_gauss,time_dependent_potential,functions,minus_half_laplacians):
    nFock=len(fockOrbitals)
    Fgauss=minus_half_laplacians
    potential_term = potential_grid + time_dependent_potential
    electron_density=np.zeros(fockOrbitals.shape[1],dtype=np.complex128)
    for j in range(nFock):
        electron_density+=2*np.abs(fockOrbitals[j])**2
    coulomb_term=np.dot(electron_density,weighted_e_e_grid)
    Fgauss+=(potential_term+coulomb_term)*functions
    fock_orbitals_conj=np.conj(fockOrbitals)
    for i in range(num_gauss):
        for j in range(nFock):
            exchange_term =(fock_orbitals_conj[j] * functions[i]).T@weighted_e_e_grid
            Fgauss[i] += -exchange_term * fockOrbitals[j]
    return Fgauss
def calculate_v_gauss_fast(previous_time_orbitals,num_gauss,time_dependent_potential,functions,minus_half_laplacians):
    nOrbs=len(previous_time_orbitals)
    Agauss=minus_half_laplacians
    potential_term = potential_grid + time_dependent_potential
    electron_density=np.zeros(previous_time_orbitals.shape[1],dtype=np.complex128)
    for j in range(nOrbs):
        electron_density+=2*np.abs(previous_time_orbitals[j])**2
    coulomb_term=np.dot(electron_density,weighted_e_e_grid)
    potential_term+=coulomb_term
    potential_term+=v_xc(electron_density)
    Agauss+=potential_term*functions
    return Agauss
@jit(nopython=True,cache=True,fastmath=False)
def calculate_Ftimesorbitals(orbitals,FocktimesGauss):
    nbasis=orbitals.shape[0]
    norbs=orbitals.shape[1]
    FockOrbitals=np.empty((norbs,len(points)),dtype=np.complex128)
    for i in range(norbs):
        FockOrbital=np.zeros_like(points,dtype=np.complex128)
        for j in range(nbasis):
            FockOrbital+=orbitals[j,i]*FocktimesGauss[j]
        FockOrbitals[i]=FockOrbital
    return FockOrbitals



class Rothe_evaluator:
    def __init__(self,old_params,old_lincoeff,time_dependent_potential,timestep,number_of_frozen_orbitals=0,method="HF"):
        """
        old_params: The parameters for the Gaussians from the previous iteration
        old_lincoeff: The linear coefficients for the Gaussians in the basis of the old ones, from the previous iteration
        time_dependent_potential: The time-dependent potential evaluated at the relevant time
        timestep: The timestep used in the propagation
        """
        self.nbasis=old_lincoeff.shape[0]
        self.norbs=old_lincoeff.shape[1]
        self.method=method
        if method=="HF":
            self.orbital_operator=calculate_Fgauss_fast
            self.orbital_operator_slow=calculate_Fgauss
        elif method=="DFT":
            self.orbital_operator=calculate_v_gauss_fast
            self.orbital_operator_slow=calculate_v_gauss
        self.old_params=old_params
        self.old_lincoeff=old_lincoeff
        self.pot=time_dependent_potential
        self.dt=timestep
        self.nfrozen=number_of_frozen_orbitals
        self.params_frozen=old_params[:4*self.nfrozen]
        self.orbitals_that_represent_Fock=make_orbitals(self.old_lincoeff,self.old_params) #Orbitals that define the Fock operator; which are the old orbitals

        self.old_action=self.calculate_Adagger_oldOrbitals() #Essentially, the thing we want to approximate with the new orbitals
        self.f_frozen,self.fock_act_on_frozen_gauss=self.calculate_frozen_orbital_stuff()
        
    def calculate_Adagger_oldOrbitals(self):
        fock_act_on_old_gauss=self.orbital_operator_slow(self.orbitals_that_represent_Fock,self.old_params,num_gauss=self.nbasis,time_dependent_potential=self.pot) #Act with the OLD Fock operator on the OLD Gaussians
        Fock_times_Orbitals=calculate_Ftimesorbitals(self.old_lincoeff,fock_act_on_old_gauss)
        rhs=self.orbitals_that_represent_Fock-1j*self.dt/2*Fock_times_Orbitals
        return rhs
    def calculate_frozen_orbital_stuff(self):
        functions,minus_half_laplacians,aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs, aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs=setupfunctionsandDerivs(self.params_frozen.reshape((-1,4)),points)
        
        fock_act_on_frozen_gauss=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(functions),time_dependent_potential=self.pot,
                                                    functions=np.array(functions),minus_half_laplacians=np.array(minus_half_laplacians))
        return functions,fock_act_on_frozen_gauss
    
    def rothe_plus_gradient(self,nonlin_params_unfrozen):
        old_action=self.old_action *sqrt_weights
        gradient=np.zeros_like(nonlin_params_unfrozen)

        nonlin_params=np.concatenate((self.params_frozen,nonlin_params_unfrozen))
        functions_u,minus_half_laplacians_u,aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs, aderiv_kin_funcs, bderiv_kin_funcs, pderiv_kin_funcs, qderiv_kin_funcs=setupfunctionsandDerivs(nonlin_params_unfrozen.reshape((-1,4)),points)
        fock_act_on_new_gauss=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(functions_u),time_dependent_potential=self.pot,
                                                    functions=np.array(functions_u),minus_half_laplacians=np.array(minus_half_laplacians_u))
        functions=np.concatenate((self.f_frozen,functions_u))
        fock_act_on_functions=np.concatenate((self.fock_act_on_frozen_gauss,fock_act_on_new_gauss))
        function_derivs=[]
        kin_derivs=[]
        for i in range(len(aderiv_funcs)):
            function_derivs+=[aderiv_funcs[i],bderiv_funcs[i],pderiv_funcs[i],qderiv_funcs[i]]
            kin_derivs+=[aderiv_kin_funcs[i],bderiv_kin_funcs[i],pderiv_kin_funcs[i],qderiv_kin_funcs[i]]
        function_derivs=np.array(function_derivs)
        kin_derivs=np.array(kin_derivs)
        indices_random=np.random.choice(len(old_action[0]), len(old_action[0])//2, replace=False);multiplier=2
        indices_random=np.array(np.arange(len(old_action[0]))); multiplier=1
        X=functions+1j*self.dt/2*fock_act_on_functions
        n_gridpoints=X.shape[1]
        n_params=len(nonlin_params_unfrozen)
        new_lincoeff=np.empty((self.nbasis,self.norbs),dtype=np.complex128)
        old_action=old_action[:,indices_random]
        X=X.T
        Xderc=np.zeros_like(X)[indices_random,:]
        X = X * sqrt_weights.reshape(-1, 1)
        X=X[indices_random,:]
        X_dag=X.conj().T
        XTX =X_dag @ X
        I=np.eye(XTX.shape[0])
        rothe_error=0
        zs=np.zeros_like(old_action)
        invmats=[]
        #
        for orbital_index in range(old_action.shape[0]):
            Y=old_action[orbital_index]
            XTy = X_dag @ Y
            invmats.append(np.linalg.inv(XTX+ lambd * I))
            new_lincoeff[:,orbital_index]=invmats[-1]@XTy
            zs[orbital_index]=Y-X@new_lincoeff[:,orbital_index]
            rothe_error+=np.linalg.norm(zs[orbital_index])**2*multiplier
        
        self.optimal_lincoeff=new_lincoeff
        Fock_act_on_derivs=self.orbital_operator(np.array(self.orbitals_that_represent_Fock),
                                                    num_gauss=len(function_derivs),time_dependent_potential=self.pot,
                                                    functions=np.array(function_derivs),minus_half_laplacians=np.array(kin_derivs))
        #Fock_act_on_derivs=np.concatenate((self.Fock_act_on_frozen_derivs,Fock_act_on_derivs))
        #function_derivs=np.concatenate((self.f_frozen_derivs,function_derivs))
        Xders=function_derivs+1j*self.dt/2*Fock_act_on_derivs
        
        Xders=Xders.T
        Xders = Xders * sqrt_weights.reshape(-1, 1)
        Xders=Xders[indices_random,:]
        gradvecs=np.zeros((n_params,n_gridpoints),dtype=np.complex128)
        for i in range(len(nonlin_params_unfrozen)):
            Xder=Xderc.copy()
            Xder[:,self.nfrozen+i//4]=Xders[:,i]
            Xder_dag=Xder.conj().T
            for orbital_index in range(old_action.shape[0]):
                Y=old_action[orbital_index]
                invmat=invmats[orbital_index]
                XTYder=Xder_dag @ Y
                XTY=X_dag @ Y
                matrix_der=-invmat@(X_dag@Xder+Xder_dag@X)@invmat
                cder=matrix_der@XTY+invmat@XTYder
                gradvec=(-Xder@new_lincoeff[:,orbital_index]-X@cder)
                gradient[i]+=2*np.real(zs[orbital_index].conj().T@gradvec)*multiplier
                gradvecs[i]+=gradvec
        
        gradvecs=np.array(gradvecs)

        return rothe_error,gradient

class Rothe_propagation:
    def __init__(self,params_initial,lincoeffs_initial,pulse,timestep,points,nfrozen=0,t=0,norms=None,params_previous=None,method="HF"):
        self.nbasis=lincoeffs_initial.shape[0]
        self.norbs=lincoeffs_initial.shape[1]
        self.method=method
        if norms is not None:
            self.norms=norms
        else:
            self.norms=np.ones(self.norbs)
        self.pulse=pulse
        self.dt=timestep
        params_initial=params_initial.flatten()
        self.lincoeffs=lincoeffs_initial
        self.params=params_initial
        self.functions=None
        self.nfrozen=nfrozen
        if params_previous is not None:
            try:
                self.adjustment=params_initial[4*self.nfrozen:]-params_previous[4*self.nfrozen:]
            except:
                self.adjustment=None
        else:
            self.adjustment=None
        self.full_params=np.concatenate((lincoeffs_initial.flatten().real,lincoeffs_initial.flatten().imag,params_initial))
        self.t=t
        self.lambda_grad0=1e-14
        self.last_added_t=0
def make_error_and_gradient_functions(method,molecule,quality,t0):
    omega = 0.06075  # Laser frequency
    t_c = 2 * np.pi / omega  # Optical cycle
    E0=0.0534
    n_cycles = 3
    dt=0.05#*(-1j)
    td = n_cycles * t_c  # Duration of the laser pulse
    fieldfunc=laserfield(E0, omega, td)
    filenames = {
            ("HF", "LiH", 1):   "WF_HF_LiH_0.0534_20_24_300_1.000e+04.npz",
            ("HF", "LiH", 2):   "WF_HF_LiH_0.0534_20_24_300_5.000e-01.npz",
            ("HF", "LiH", 3):   "WF_HF_LiH_0.0534_20_24_300_3.000e-02.npz",
            ("HF", "LiH2", 1):  "WF_HF_LiH2_0.0534_34_38_300_1.000e+04.npz",
            ("HF", "LiH2", 2):  "WF_HF_LiH2_0.0534_34_38_300_1.500e+00.npz",
            ("HF", "LiH2", 3):  "WF_HF_LiH2_0.0534_34_38_300_6.000e-01.npz",
            ("DFT", "LiH", 1):  "WF_DFT_LiH_0.0534_20_24_300_1.000e+04.npz",
            ("DFT", "LiH", 2):  "WF_DFT_LiH_0.0534_20_24_300_1.000e+00.npz",
            ("DFT", "LiH", 3):  "WF_DFT_LiH_0.0534_20_24_300_3.000e-01.npz",
            ("DFT", "LiH2", 1): "WF_DFT_LiH2_0.0534_34_38_300_5.000e+01.npz",
            ("DFT", "LiH2", 2): "WF_DFT_LiH2_0.0534_34_38_300_1.000e+01.npz",
            ("DFT", "LiH2", 3): "WF_DFT_LiH2_0.0534_34_38_300_4.000e+00.npz"
        }
    if molecule=="LiH2":
        norbs=4
        nfrozen=34
    else:
        norbs=2
        nfrozen=20
    
    
    infilename=filenames[(method,molecule,quality)]
    data=np.load(infilename)
    times=data["times"]
    params=data["params"]
    nbasis=data["nbasis"]
    norms=data["norms"]
    start_time=t0
    closest_index = np.abs(times - start_time).argmin()
    ngauss=nbasis[closest_index]
    ngauss_wrong=len(params[closest_index])//(4+norbs*2)
    norms_initial=norms[closest_index]
    lincoeff_initial_real=params[closest_index][:ngauss*norbs]#.reshape((ngauss,norbs))
    lincoeff_initial_complex=params[closest_index][ngauss_wrong*norbs:(ngauss+ngauss_wrong)*norbs]#.reshape((ngauss,norbs))
    lincoeff_initial=lincoeff_initial_real+1j*lincoeff_initial_complex
    lincoeff_initial=lincoeff_initial.reshape((ngauss,norbs))
    if ngauss_wrong-ngauss>0:
        gaussian_nonlincoeffs=params[closest_index][ngauss_wrong*norbs*2:-4*(ngauss_wrong-ngauss)]
    else:
        gaussian_nonlincoeffs=params[closest_index][ngauss*norbs*2:]
    gaussian_nonlincoeffs_unreshaped=gaussian_nonlincoeffs.copy()
    gaussian_nonlincoeffs=gaussian_nonlincoeffs.reshape((ngauss,4))
    inner_grid=10
    points_inner,weights_inner=gaussian_quadrature(-inner_grid,inner_grid,24*inner_grid+1)
    grid_spacing=0.4
    grid_a=-200
    grid_b=200
    num_points=int((grid_b-inner_grid)/grid_spacing)
    points_outer1,weights_outer1=trapezoidal_quadrature(grid_a, -inner_grid, num_points)
    points_outer2,weights_outer2=trapezoidal_quadrature(inner_grid, grid_b, num_points)
    points=np.concatenate((points_outer1,points_inner,points_outer2))
    weights=np.concatenate((weights_outer1,weights_inner,weights_outer2))
    tmax=500
    rothepropagator=Rothe_propagation(gaussian_nonlincoeffs,lincoeff_initial,pulse=fieldfunc,
                                timestep=dt,points=points,nfrozen=nfrozen,t=tmax,norms=norms_initial,params_previous=None,method=method)
    rothepropagator.time_dependent_potential=rothepropagator.pulse(t0+dt/2)*points
    rothe_evaluator=Rothe_evaluator(gaussian_nonlincoeffs_unreshaped,lincoeff_initial,rothepropagator.time_dependent_potential,dt,rothepropagator.nfrozen,rothepropagator.method)
    initial_full_new_params=gaussian_nonlincoeffs_unreshaped[4*nfrozen:]
    def error_function_optimization(parameters):
        error=rothe_evaluator.rothe_plus_gradient(parameters)[0]
        print(error)
        return error
    def gradient_optimization(parameters):
        return rothe_evaluator.rothe_plus_gradient(parameters)[1]
    return error_function_optimization,gradient_optimization,initial_full_new_params
method=sys.argv[1] # HF or DFT
molecule=sys.argv[2]
quality=int(sys.argv[3]) #1, 2 or 3
t0=float(sys.argv[4])



if molecule=="LiH":
    R_list=[-1.15, 1.15]
    Z_list=[3,1]
elif molecule=="LiH2":
    R_list=[-4.05, -1.75, 1.75, 4.05]
    Z_list=[3, 1,3,1]
norbs=sum(Z_list)//2
alpha=0.5


inner_grid=10
points_inner,weights_inner=gaussian_quadrature(-inner_grid,inner_grid,24*inner_grid+1)
grid_spacing=0.4
grid_a=-200
grid_b=200
num_points=int((grid_b-inner_grid)/grid_spacing)
points_outer1,weights_outer1=trapezoidal_quadrature(grid_a, -inner_grid, num_points)
points_outer2,weights_outer2=trapezoidal_quadrature(inner_grid, grid_b, num_points)
points=np.concatenate((points_outer1,points_inner,points_outer2))
weights=np.concatenate((weights_outer1,weights_inner,weights_outer2))
potential_grid=calculate_potential(Z_list,R_list,alpha,points)
lambd=5e-10
sqrt_weights=np.sqrt(weights)
V=external_potential=calculate_potential(Z_list,R_list,alpha,points)
wT=weights.T
e_e_grid=e_e_interaction(points)
weighted_e_e_grid = e_e_grid * weights[:, np.newaxis]
optimizee,optimizee_grad,parameters=make_error_and_gradient_functions(method,molecule,quality,t0)
from scipy.optimize import minimize
error_initial=optimizee(parameters)
grad_initial=optimizee_grad(parameters)
print("Initial error: ", error_initial)
print("Initial gradient: ", grad_initial)
hess_inv0=np.diag(1/np.abs(grad_initial))

res=minimize(optimizee,parameters,method="BFGS",jac=optimizee_grad,options={"maxiter":50,"gtol":1e-9,"hess_inv0":hess_inv0})

print("Best function value found:", res.fun)
print("Best x found:", res.x)
