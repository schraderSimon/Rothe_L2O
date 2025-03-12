from WF_class import *
from scipy.optimize import minimize
from scipy.optimize import line_search
from numpy import sin, arcsin,cos, tanh, sinh, cosh,arctanh
import time
np.set_printoptions(precision=6,linewidth=300)


def WFmaker(params,c,fieldFunc,potential_params,potential_type,beta_ovlp=2e-12,gauge="length"):
    """
    Returns a WF object with a given set of
    - params (the nonlinear parameters)
    - c (the linear parameters)
    - fieldfunc (The laser pulse function.)
    - potential_params (parameters necessary for the 1/r approximation)
    - potential_type: The type of 1/r approximation
    - beta_ovlp: The amount of "punishment" used for avoiding too large values for S_ij
    - Gauge: The gauge ("length" or "velocity")
    """
    if potential_type=="lincombGauss":
        return lincombGauss_WF(params,c,fieldFunc=fieldFunc,potential_params=potential_params,beta_ovlp=beta_ovlp,gauge=gauge)
    elif potential_type=="erf":
        return erf_WF(params,c,fieldFunc,potential_params,beta_ovlp=beta_ovlp,gauge=gauge)
    elif potential_type=="erfGau":
        return erfGau_WF(params,c,fieldFunc,potential_params,beta_ovlp=beta_ovlp,gauge=gauge)

    elif potential_type=="projErf":
        return projected_H2_WF(params,c,fieldFunc,potential_params,beta_ovlp=beta_ovlp,gauge=gauge)
    else:
        print("Error, potential type does not exist")
        raise ValueError

class TimeEvolution():
    def __init__(self,WF_t0,potential_params,potential_type="lincombGauss",fieldFunc=None,h=1,T=10,epsilon=1e-2,t0=0,error_t0=0,filename="dumps/dumpfile",hlm=False,gauge="length",write_to_file=True):
        """
        input:
        WF_t0: The wave function object for the wave function at time t=0
        potential_params (parameters necessary for the 1/r approximation)'
        potential_type: The type of 1/r approximation
        fieldFunc: The laser pulse function.
        h: The time step for time evolution
        T: The final time
        epsilon: The overall maximal time evolution error
        t0: Starting time
        error_t0: the error at t_0
        filename: The file name in which to dump data
        hlm: Outdated parameter, should always be "False" (False by default)
        gauge: The gauge to use (length of velocity)
        """
        self.dipole_moment=[]
        self.WF=WF_t0
        self.h=h
        if fieldFunc is not None:
            self.fieldFunc=fieldFunc
        else:
            def fieldFunc(t=0):
                return 0
            self.fieldFunc=fieldFunc
        self.t0=t0
        self.t=t0 #Starting time
        self.T=T
        self.epsilon=epsilon
        self.error_threshold=epsilon/T*h
        self.hlm=hlm
        self.gauge=gauge
        self.dipole_moment.append(self.WF.calculate_dipole_moment())
        self.accumulated_error=error_t0
        self.n_steps=self.T/self.h
        self.potential_params=potential_params
        self.potential_type=potential_type
        self.filename=filename
        if write_to_file: 
            if not os.path.isfile(filename):
                with h5py.File(filename, "w") as data_file:
                    data_file.create_dataset("parameters_t=%.2f"%self.t,data=self.WF.params)
                    data_file.create_dataset("coefficients_t=%.2f"%self.t,data=self.WF.coefficients)
                    new_times=[t0]
                    dpm=[self.WF.calculate_dipole_moment()]
                    accumulated_error=[0]
                    data_file.create_dataset("times", data=new_times)
                    data_file.create_dataset("dpm", data=dpm)
                    data_file.create_dataset("rothe_error",data=accumulated_error)
                    data_file.create_dataset("hlm",data=[hlm])
            else:
                with h5py.File(filename, "r+") as data_file:
                    #print(data_file.keys())
                    times_read=np.array(data_file["times"])
                    try:
                        data_file.create_dataset("hlm",data=[hlm])
                    except:
                        pass
                    index_to_delete_after=np.argmin(abs(times_read-t0))+1
                    if abs(times_read[index_to_delete_after-1]-t0)>1e-5:
                        print("You cannot start your run from this time, as no reference data exists")
                        raise IndexError
                    else:
                        times_new=times_read[:index_to_delete_after]
                        mu_new=np.array(data_file["dpm"])[:index_to_delete_after]
                        re_new=np.array(data_file["rothe_error"])[:index_to_delete_after]
                        del data_file["dpm"]; del data_file["rothe_error"]; del data_file["times"]; del data_file["hlm"]
                        data_file.create_dataset("times", data=times_new)
                        data_file.create_dataset("dpm", data=mu_new)
                        data_file.create_dataset("rothe_error",data=re_new)
                        data_file.create_dataset("hlm",data=[hlm])
    def time_evolve_step(self):
        """
        create wave function at the next time step
        """
        allow_adding=True #Outdated parameter
        wfoptimizer=WFOptimizer(wavefunction=self.WF,potential_params=self.potential_params,t=self.t,h=self.h,error_threshold=self.error_threshold,fieldFunc=self.fieldFunc,potential_type=self.potential_type,file=self.filename,allow_adding=allow_adding,hlm=self.hlm,gauge=self.gauge)
        new_WF,added=wfoptimizer.find_next_Timestep() #Run the actual optimization
        self.accumulated_error+=wfoptimizer.error
        del wfoptimizer
        del self.WF
        self.t=self.t+self.h
        self.WF=new_WF
        self.dipole_moment.append(self.WF.calculate_dipole_moment())
        print("Accumulated error: %f"%self.accumulated_error)
        print("Presumed error: %f"%(self.accumulated_error/(self.t)*(self.T)))

    def time_evolve(self):
        """
        Run time evolution up to time T
        """
        while self.t<self.T:
            self.time_evolve_step() #Find WF at next time step
            print("%.3f/%.3f"%(self.t,self.T))
            with h5py.File(self.filename, "r+") as data_file:
                try:
                    data_file.create_dataset("parameters_t=%.2f"%self.t,data=self.WF.params)
                    data_file.create_dataset("coefficients_t=%.2f"%self.t,data=self.WF.coefficients)
                except ValueError:
                    del data_file["parameters_t=%.2f"%self.t]
                    del data_file["coefficients_t=%.2f"%self.t]
                    data_file.create_dataset("parameters_t=%.2f"%self.t,data=self.WF.params)
                    data_file.create_dataset("coefficients_t=%.2f"%self.t,data=self.WF.coefficients)
                old_times=data_file["times"]
                new_times=list(old_times)
                new_times.append(self.t)
                del data_file["times"]
                data_file.create_dataset("times", data=new_times)

                old_dipole=data_file["dpm"]
                new_dipole=list(old_dipole)
                new_dipole.append(self.WF.calculate_dipole_moment())
                del data_file["dpm"]
                data_file.create_dataset("dpm", data=new_dipole)

                old_re=data_file["rothe_error"]
                new_re=list(old_re)
                new_re.append(self.accumulated_error)
                del data_file["rothe_error"]
                data_file.create_dataset("rothe_error", data=new_re)

            if (self.t-0.001)%(5*self.h)>(5*self.h)-self.h:
                #At every 5th time step, print dipole moment, coefficients and parameters
                print(list(self.WF.coefficients))
                print(list(self.WF.params))
                print(list(-np.real(self.dipole_moment)))
class WFOptimizer():
    """
    Class to carry out optimization.
    """
    def __init__(self,wavefunction,potential_params,t,h,error_threshold,fieldFunc,potential_type,file,allow_adding=True,hlm=False,gauge="length"):
        self.WF_init=wavefunction
        self.gauge=gauge
        self.potential_type=potential_type
        self.potential_params=potential_params
        self.t=t
        self.h=h
        self.hlm=hlm
        self.error_threshold=error_threshold
        self.fieldFunc=fieldFunc
        self.error=0 #Simply set to 0, could also be None - just make it exist in all cases
        self.allow_adding=allow_adding #
        self.file=file
        self.setUpOptimizer()
    def setUpOptimizer(self):
        """
        Take parameters from the WF object and set up initial guess for paremters
        """
        self.params_old=self.WF_init.params
        self.coefficients_old=self.WF_init.coefficients #... and the linear coefficients.
        self.len_old_WF=len(self.coefficients_old) #Keep the number of basis elements of the old WF, as we might be adding new WFs.
        self.params_new=self.params_old.copy()
        self.coefficients_new=self.coefficients_old.copy()
        self.len_new_WF=self.len_old_WF #In the beginning, start with the same number of Gaussians
    def get_start_parameters(self,alphas=np.linspace(0,1.2,13)):
        """
        Find the best alpha, i.e. how much of the change between last and second-to-last timestep to add to initial guess
        """
        with h5py.File(self.file, "r") as data_file:
            times=np.array(data_file["times"])
            if len(times)<2:
                return self.params_old
            x=-1
            params_old=np.array(data_file["parameters_t=%.2f"%self.t])
            params_oldold=np.array(data_file["parameters_t=%.2f"%(self.t-self.h)])
        if len(params_old) > len(params_oldold):
            diff=len(params_old)-len(params_oldold)
            direction=(params_old-np.concatenate([params_oldold,[0]*diff]))
        elif len(params_old)==len(params_oldold):
            direction=params_old-params_oldold
        elif len(params_old)==len(params_oldold)-4:
            direction=params_old-params_oldold[:-4]
        else:
            return params_old


        error_set=[]
        for alpha in alphas:
            val=self.calculate_error_from_params(params_old+alpha*direction)
            if val>0:
                error_set.append(sqrt(val))
            else:
                error_set.append(1e100)
        error_set=np.nan_to_num(error_set,nan=1e15,posinf=1e15,neginf=1e15)
        i=np.argmin(error_set)
        print("alpha: %.2f"%alphas[i])
        return params_old+direction*alphas[i]
    def calculate_error_from_params(self,params):
        """
        Calculates the Rothe error for a set of parameters (t and old lin/nonlin parameters are stored)
        """
        c_old=self.coefficients_old
        params_old=self.params_old
        c_new=np.zeros(self.len_new_WF)
        cmerged=np.concatenate((c_old,c_new))
        params_merged=np.concatenate((params_old,params))
        self.merged_WF=WFmaker(params_merged,cmerged,fieldFunc=self.fieldFunc,potential_params=self.potential_params,potential_type=self.potential_type,gauge=self.gauge)
        error=self.merged_WF.rothe_error(len(c_old),self.t+self.h/2,self.h)
        return error
    def calculate_jac_from_params(self,params):
        """
        Calculates the Rothe gradient for a set of parameters (t and old parameters are stored)
        """
        c_old=self.coefficients_old
        params_old=self.params_old
        c_new=np.zeros(self.len_new_WF)
        cmerged=np.concatenate((c_old,c_new))
        params_merged=np.concatenate((params_old,params))
        return self.merged_WF.rothe_jacobian(len(c_old),self.t+self.h/2,self.h)
    def error_for_optimization(self,params):
        """
        Calculates the actual optimization function (i.e. with penaty on S_ij) for a set of parameters (t and old lin/nonlin parameters are stored)
        """

        c_old=self.coefficients_old
        params_old=self.params_old
        c_new=np.zeros(self.len_new_WF)
        cmerged=np.concatenate((c_old,c_new))
        params_merged=np.concatenate((params_old,params))
        self.merged_WF=WFmaker(params_merged,cmerged,self.fieldFunc,potential_params=self.potential_params,potential_type=self.potential_type,gauge=self.gauge)
        error=self.merged_WF.rothe_error_overlap_control(len(c_old),self.t+self.h/2,self.h)

        return error
    def jac_for_optimization(self,params):
        """
        Calculates the actual optimization gradient (i.e. with penaty on S_ij) for a set of parameters (t and old lin/nonlin parameters are stored)
        """

        c_old=self.coefficients_old
        params_old=self.params_old
        c_new=np.zeros(self.len_new_WF)
        cmerged=np.concatenate((c_old,c_new))
        params_merged=np.concatenate((params_old,params))
        return self.merged_WF.rothe_jacobian_overlap_control(len(c_old),self.t+self.h/2,self.h)

    def optimize(self): #The actual optimization procedure
        old_error=1e100
        gtol=1e-16
        num_max_WF=120

        def findLeastImportantGaussian(parameters):
            #Finds the Gaussian which, if removed, increases the Rothe error the least.
            #The first 20 Gaussians are ignored. Ideally, this should however be 25.
            error_assumed=1e100 # some big ass number
            worst_i=0
            self.len_new_WF-=1
            for i in range(20,len(parameters)//4):
                to_delete_parameters=np.array([0,1,2,3])+4*i
                params_reduced=np.delete(parameters,to_delete_parameters)
                if i<20:
                    error=1e100
                else:
                    error=error_function(params_reduced)
                if error<error_assumed:
                    worst_i=i
                    error_assumed=error
            self.len_new_WF+=1
            return worst_i, error_assumed
        def error_function(parameters):
            error=self.calculate_error_from_params(parameters)
            #print(sqrt(error))
            return error
        def error_function_optimization(parameters,frozen_params):
            full_parameters=np.concatenate((frozen_params,parameters))

            error=self.error_for_optimization(full_parameters)
            return error
        def gradient_optimization(parameters,frozen_params):
            full_parameters=np.concatenate((frozen_params,parameters))
            grad=self.jac_for_optimization(full_parameters)[len(frozen_params):]
            return grad
        def gradient(parameters):
            grad=self.calculate_jac_from_params(parameters)
            return grad
        def get_Guess_distribution(vals,n):
            x=np.linspace(np.min(vals),np.max(vals),10000)
            returnvals=np.zeros(len(x))
            for a in vals:
                returnvals+=np.exp(-(x-a)**2/(2*(a+1e-10)**2))*1/np.sqrt(2*np.pi*(a+1e-10)**2)
            p=returnvals/np.sum(returnvals)
            return np.random.choice(x,p=p,size=n)
        def gaussianAddingStrategy_ludwik(parameters):
            """
            The strategy to add Gaussians, e.g. producing guess distributgions and then sampling from them.
            The Gaussian suggested that reduces the Rothe error the most, is returned.
            """
            a_params=parameters[::4]#[20:]
            b_params=parameters[1::4]#[20:]
            mu_params=parameters[2::4]#[20:]
            q_params=parameters[3::4]#[20:]
            n=500
            avals_sample=get_Guess_distribution(a_params,n)
            bvals_sample=get_Guess_distribution(b_params,n)
            muvals_sample=get_Guess_distribution(mu_params,n)
            qvals_sample=get_Guess_distribution(q_params,n)
            errors=np.zeros(n)
            for i in range(n):
                a=avals_sample[i]
                b=bvals_sample[i]
                mu=muvals_sample[i]
                q=qvals_sample[i]
                params_to_add=[a,b,mu,q]
                c_old=self.coefficients_old
                params_old=self.params_old
                c_new=np.zeros(self.len_new_WF+len(params_to_add)//4)
                cmerged=np.concatenate((c_old,c_new))
                params=np.concatenate((parameters,params_to_add))
                params_merged=np.concatenate((params_old,params))
                merged_WF=WFmaker(params_merged,cmerged,self.fieldFunc,potential_params=self.potential_params,potential_type=self.potential_type,gauge=self.gauge)
                error=merged_WF.rothe_error_overlap_control(len(c_old),self.t+self.h/2,self.h,err_t=0.990)
                errors[i]=error
            errors=np.nan_to_num(errors, nan=1e100, posinf=1e100, neginf=1e100)
            i=np.argmin(errors)
            a=avals_sample[i]
            b=bvals_sample[i]
            mu=muvals_sample[i]
            q=qvals_sample[i]
            #return [a,b,mu,q,a,-b,mu,q,a,b,mu,-q,a,-b,mu,-q],4
            return [a,b,mu,q],1
        def minimize_transformed_bonds(error_function,gradient,maxiter,start_params,args,gtol,multi_bonds=5e-1,repeat=0):
            """
            Minimizes with min_max bonds as described in https://lmfit.github.io/lmfit-py/bounds.html
            """
            start_params[::4]=abs(start_params[::4])
            def transform_params(untransformed_params):
                return arctanh(2*(untransformed_params-mins)/(maxs-mins)-1)
                return arcsin(2*(untransformed_params-mins)/(maxs-mins)-1)
            def untransform_params(transformed_params):
                return mins+(maxs-mins)/2*(1+tanh(transformed_params))
                return mins+(maxs-mins)/2*(1+sin(transformed_params))
            def chainrule_params(transformed_params):
                return 0.5*(maxs-mins)/(cosh(transformed_params)**2)
                return 0.5*(maxs-mins)*cos(transformed_params)

            def transformed_error(transformed_params):
                error=error_function(untransform_params(transformed_params),[])
                return error
            def transformed_gradient(transformed_params):
                orig_grad=gradient(untransform_params(transformed_params),[])
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
            freeze_GS=True
            n_frozen_cutoff=0
            if freeze_GS:
                n_frozen_cutoff=100
            dp=multi_bonds*np.ones(len(start_params)) #Percentage (times 100) how much the parameters are alowed to change compared to previous time step
            dp[:n_frozen_cutoff]=1e-14 #Essentially forbid the initial parameters to change

            range_notmu=0.1
            range_mu=0.1
            rangex=np.array([range_notmu,range_notmu,range_notmu,range_mu]*(len(start_params)//4))
            mins=start_params-dp*abs(start_params)-rangex
            maxs=start_params+dp*abs(start_params)+rangex
            mins[:n_frozen_cutoff]=start_params[:n_frozen_cutoff]-dp[:n_frozen_cutoff]*abs(start_params)[:n_frozen_cutoff]-1e-14
            maxs[:n_frozen_cutoff]=start_params[:n_frozen_cutoff]+dp[:n_frozen_cutoff]*abs(start_params)[:n_frozen_cutoff]+1e-14
            transformed_params=transform_params(start_params)
            transformed_params=np.real(np.nan_to_num(transformed_params))
            startErr=transformed_error(transformed_params)
            start=time.time()
            print("Starting to transform")
            import scipy
            sol=minimize(transformed_error,transformed_params,jac=transformed_gradient,method="BFGS",options={"maxiter":5000,"gtol":1e-16,"hess_inv0":np.eye(len(transformed_params))*1e8})
            #sol=minimize(transformed_error,transformed_params,method="BFGS",options={"maxiter":30,"gtol":1e-16,"eps":1e-6})
            transformed_sol=sol.x
            end=time.time()

            print(f"Number of iterations: {sol.nit}")
            print("Time used: %.2f seconds"%(end-start))

            solErr=transformed_error(transformed_sol)
            print(startErr,solErr,startErr-solErr)
            return untransform_params(transformed_sol), sol.nit

        err_threshold=self.error_threshold
        start_params=self.get_start_parameters()

        new_params=start_params
        new_params_backup=new_params.copy()
        old_error=error_function(new_params)
        optc_oldParams=self.merged_WF.opt_c
        print(old_error)
        print("Old Error: %.10f; num WF: %d"%(sqrt(old_error),len(optc_oldParams)))
        new_error=True

        tol=1e-17
        new_error_middle=old_error
        print("Minimizing transformed bonds")

        new_params,nit=minimize_transformed_bonds(error_function_optimization,gradient_optimization,maxiter=1500,start_params=new_params,args=None,gtol=gtol,multi_bonds=5e-1)
        new_error=error_function(new_params)
        if new_error>old_error:
            new_params=start_params
            new_error=old_error
        print("IBT error: %.10f/%.10f (%.3f proc.)"%(sqrt(new_error),err_threshold,sqrt(new_error)/err_threshold))
        bigger=False


        if (sqrt(new_error)>err_threshold*1.004 and (nit<50)) or np.isnan(sqrt(new_error)) :
             """IF the error is too large and the number of iterations is less than 50, if the the new error is (numerically nan),
             the input parameter is slightly perturbed and the optimization is carried out a second time.
              """
             start_params=self.get_start_parameters()
             new_start_params=new_params_backup+1e-4*(1-2*np.random.rand(len(new_params_backup)))*new_params_backup
             new_start_params[:100]=new_params_backup[:100]
             new_paramss,nitnew=minimize_transformed_bonds(error_function_optimization,gradient_optimization,maxiter=1500,start_params=new_start_params,args=None,gtol=gtol,multi_bonds=5e-1)
             new_errorr=error_function(new_paramss)
             print("New IBT error: %.10f/%.10f (%.3f proc.)"%(sqrt(new_errorr),err_threshold,sqrt(new_errorr)/err_threshold))
             if (new_errorr<new_error or np.isnan(sqrt(new_error))) and new_errorr>0:
                 new_params=new_paramss
                 new_error=new_errorr

        new_params_backup=new_params.copy()
        count=0
        counterino=0
        iteration=0
        divisible_by_10_timestep= int(np.round(self.t/self.h))%10==0
        if divisible_by_10_timestep:
            least_important_gaussian,error_least_important=findLeastImportantGaussian(new_params)
        removed_WF_num=0
        while (self.len_new_WF>50) and (divisible_by_10_timestep) and removed_WF_num<10:
            """If there's at least 51 Gaussians and it's a 10'th time step, attempt to remove a Gaussian"""
            # break #This was used for E0=0.12
            new_params_removed=np.delete(new_params,[least_important_gaussian*4,least_important_gaussian*4+1,least_important_gaussian*4+2,least_important_gaussian*4+3])
            new_params_removed_copy=new_params_removed.copy()
            old_params_copy=new_params.copy()
            self.len_new_WF-=1
            old_error=new_error
            new_error_removed=1e100
            delete_counter=0
            while sqrt(new_error_removed)>sqrt(error_least_important) and delete_counter<1:
                new_params_removed=new_params_removed_copy#+self.h*1e-3*(1-2*np.random.rand(len(new_params_removed)))*new_params_removed_copy #Heavily perturb for improved optimization
                new_params_removed[:100]=new_params_removed_copy[:100]

                print("Attempting to Delete one unimportant Gaussian")
                new_params_removed,nit=minimize_transformed_bonds(error_function_optimization,gradient_optimization,maxiter=400,start_params=new_params_removed,args=None,gtol=gtol,multi_bonds=5e-1)
                new_error_removed=error_function(new_params_removed)
                delete_counter+=1

            if sqrt(new_error_removed)>err_threshold and sqrt(new_error_removed)<1.01*sqrt(old_error): #If the Gaussian is really unimportant
                new_params=new_params_removed
                new_error=new_error_removed
                print("New error (removed): %.10f"%sqrt(new_error_removed))
                print("A Gaussian was removed")
                least_important_gaussian,error_least_important=findLeastImportantGaussian(new_params)
            elif sqrt(new_error_removed)<err_threshold and sqrt(new_error_removed)<1.01*sqrt(old_error): #If removing it is good and doesn't give a huge error
                new_params=new_params_removed
                new_error=new_error_removed
                print("New error (removed): %.10f"%sqrt(new_error_removed))
                print("A Gaussian was removed")
                least_important_gaussian,error_least_important=findLeastImportantGaussian(new_params)
            else:
                new_params=old_params_copy
                new_error=old_error
                self.len_new_WF+=1
                print("New error (removed): %.10f"%sqrt(new_error_removed))
                print("No Gaussian was removed")
                break

            removed_WF_num+=1

        S=self.merged_WF.calculate_overlap_normalized(len(self.coefficients_old))
        S=S-np.eye(len(S))
        print("Largest overlap element (abs): %.5f"%np.max(abs(S)))
        new_params_backup_FirstOptimization=new_params.copy()
        new_error_backup=new_error
        error_after_optimization=sqrt(new_error)

        while sqrt(new_error)>err_threshold*1.004 and iteration<3:
            if self.len_new_WF>num_max_WF:
                #No Gaussians are added if number of WFS has reached its maximum
                break
            iteration+=1
            #least_important_gaussian,error_least_important=findLeastImportantGaussian(new_params)
            new_basis1,num_new_elements=gaussianAddingStrategy_ludwik(new_params)
            params_added=np.array(list(new_params)+list(new_basis1))#+list(new_basis1))
            old_err=sqrt(error_function(new_params))
            original_params=new_params.copy()
            self.len_new_WF=len(params_added)//4

            print("Number of WFS: %d %d"%(self.len_new_WF,len(params_added)//4))

            new_err=sqrt(error_function(params_added))


            c_old=self.coefficients_old
            params_old=self.params_old
            c_new=np.zeros(self.len_new_WF)
            cmerged=np.concatenate((c_old,c_new))
            params_merged=np.concatenate((params_old,params_added))
            merged_WF=WFmaker(params_merged,cmerged,self.fieldFunc,potential_params=self.potential_params,potential_type=self.potential_type,gauge=self.gauge)
            merged_WF.rothe_optimal_c(len(c_old),self.t,self.h)
            merged_WF.opt_c
            error=merged_WF.rothe_error(len(c_old),self.t+self.h/2,self.h)
            print("Error before optimization: %.10f"%sqrt(error))
            new_params_backup=params_added.copy()
            first_err=sqrt(error_function(params_added))
            new_error_final=1e100
            add_counter=0
            while sqrt(new_error_final)>first_err and add_counter<1:
                new_params=new_params_backup+(1e-4)*(1-2*np.random.rand(len(new_params_backup)))*new_params_backup
                new_params[:100]=new_params_backup[:100]
                new_params,nit=minimize_transformed_bonds(error_function_optimization,gradient_optimization,maxiter=700,start_params=new_params,args=None,gtol=gtol,multi_bonds=5e-1)
                new_error_final=error_function(new_params)
                add_counter+=1
            if sqrt(new_error_final)>first_err:
                #In case the optimization produces a worse estimate than the initial one
                new_params=new_params_backup
                new_error_final=first_err**2
            new_error=new_error_final
            print("Error after BFGS optimization: %.10f"%sqrt(new_error))
            iteration+=1
            print("Added basis. New error: %.10f"%sqrt(new_error))
            #least_important_gaussian,error_least_important=findLeastImportantGaussian(new_params)
        if (sqrt(new_error)>self.error_threshold and sqrt(new_error)/(error_after_optimization)>0.99 and len(new_params) != len(new_params_backup_FirstOptimization)) or new_error<0:
            print("The error has not been decreased below the threshold (or sufficiently). No Gaussian is added nevertheless")
            new_params=new_params_backup_FirstOptimization
            new_error=new_error_backup
            self.len_new_WF-=1
        c_old=self.coefficients_old
        params_old=self.params_old
        c_new=np.zeros(self.len_new_WF)
        cmerged=np.concatenate((c_old,c_new))
        params_merged=np.concatenate((params_old,new_params))
        merged_WF=WFmaker(params_merged,cmerged,self.fieldFunc,potential_params=self.potential_params,potential_type=self.potential_type,gauge=self.gauge)
        merged_WF.rothe_optimal_c(len(c_old),self.t,self.h)
        opt_c_return=merged_WF.opt_c
        self.error=sqrt(new_error)
        print("Done with the time step")

        return new_params,opt_c_return,len(c_new)>len(c_old) #the new parameters and wether the number of wave functions has changed
    def find_next_Timestep(self):
        new_params,opt_c,added=self.optimize() #Added means wether a WF has been added or not
        return WFmaker(new_params,opt_c,self.fieldFunc,potential_params=self.potential_params,potential_type=self.potential_type,gauge=self.gauge),added
