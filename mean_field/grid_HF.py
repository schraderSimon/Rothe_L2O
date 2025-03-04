import numpy as np
from numpy import pi
np.set_printoptions(precision=5, linewidth=300)
import matplotlib.pyplot as plt
import sys
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

class matrixElement():
    def __init__(self, gridsize,num_gridpoints):
        self.gridsize = gridsize
        self.num_gridpoints = num_gridpoints
        self.grid=np.linspace(-gridsize,gridsize,num_gridpoints)
        self.dx=2*gridsize/(num_gridpoints-1)
    def calculate_kinetic(self):
        dx=self.dx
        T=np.zeros((self.num_gridpoints,self.num_gridpoints),dtype=np.complex128)
        for i in range(self.num_gridpoints):
            for j in range(self.num_gridpoints):
                if i==j:
                    T[i,j]=pi**2/(6*dx**2)
                else:
                    T[i,j]=(-1)**(i-j)/(dx**2*(i-j)**2)
        self.kinetic=T
        return T
    def calculate_potential(self,Z_list,R_list,alpha):
        self.V=np.diag(Molecule1D(R_list,Z_list,alpha)(self.grid))
        return self.V
    def calculate_dipole_operator(self):
        self.dipole_operator=np.diag(self.grid)
        return self.dipole_operator
    def calculate_w(self):
        n_dvr = len(self.grid)
        w = np.zeros((n_dvr, n_dvr))
        for i in range(n_dvr):
            w[i, :] = Coulomb(-self.grid[i]+ self.grid, Z=-1, alpha=1)
        self.w=w#No dx needed
        return self.w

class solver():
    def __init__(self,gridsize,num_gridpoints,Z_list,R_list,alpha,nelec):
        self.nelec=4
        self.norb=self.nelec//2
        self.num_gridpoints=num_gridpoints
        self.grid=np.linspace(-gridsize,gridsize,num_gridpoints)
        self.M=matrixElement(gridsize,num_gridpoints)
        self.T_mat=self.M.calculate_kinetic()
        self.V_mat=self.M.calculate_potential(Z_list,R_list,alpha)
        self.Z=Z_list
        self.R=R_list
        self.w_repr=self.M.calculate_w()
        self.C_init=np.linalg.eigh(self.T_mat+self.V_mat)[1]
    def calculate_density(self,C):
        self.D = np.einsum("mj,vj->mv", C[:,:self.norb], C.conj()[:,:self.norb])
        return self.D
    def calculate_Fock_matrix(self,D):
        self.Fock=self.T_mat+self.V_mat
        
        dir=np.diag(np.einsum("kk,ik->i",D,self.w_repr))
        exchange=D*self.w_repr
        self.Fock+=2*dir-1*exchange
        return self.Fock
    def get_energy(self,C=None):
        """Returns the ground-state energy <H>"""
        if C is None:
            C=self.C
        D=self.calculate_density(C)
        tot_vec=self.T_mat+self.V_mat+self.calculate_Fock_matrix(D)
        energy=np.trace(np.dot(D, tot_vec))
        repulsion_contribution=0
        for i in range(len(self.Z)):
            for j in range(i+1,len(self.Z)):
                repulsion_contribution+=self.Z[i]*self.Z[j]/np.abs(self.R[i]-self.R[j])
        
        return energy+repulsion_contribution
    def get_dipole_moment(self,C=None):
        """Returns the dipole moment <mu>"""
        if C is None:
            C=self.C
        dipole_moment=0
        for i in range(self.norb):
            dipole_moment+=sum(abs(C[:,i])**2*self.grid)*2
        return dipole_moment
    def SCF(self):
        self.C=self.C_init
        for i in range(50):
            D=self.calculate_density(self.C)
            self.Fock=self.calculate_Fock_matrix(D)

            eigval, eigvec = np.linalg.eigh(self.Fock)
            self.C = eigvec
            err=np.linalg.norm(D@self.Fock-self.Fock@D)
            print(i,self.get_energy())
            if err<1e-11:
                break
        return self.C
    def propagate(self,field,dt,Tmax,E0):
        C=self.C
        C_trajectory=[]
        t=0
        self.dipole=[self.get_dipole_moment(C)]
        self.energy=[self.get_energy(C)]
        self.t=[0]
        C_trajectory.append(C[:,:2])
        dipole_trajectory=[]
        dipole_trajectory.append(self.get_dipole_moment(C))
        while t<Tmax:
            print(t/Tmax)
            C=self.propagate_onestep_first_order(C,field,t,dt)
            t+=dt
            self.t.append(t)
            self.dipole.append(self.get_dipole_moment(C))
            self.energy.append(self.get_energy(C))
            #plt.plot(self.grid,abs(C[:,0])**2);plt.plot(self.grid,abs(C[:,1])**2);plt.show()
            C_trajectory.append(C)
            dipole_trajectory.append(self.get_dipole_moment(C))
        filename="grid_solution_%.3f.npz"%E0
        np.savez(filename,gridpoints=self.grid,Cvals=C_trajectory,dipole=dipole_trajectory)
        return self.t,self.dipole,self.energy
    def propagate_onestep_second_order(self,C,field,t,dt):
        C_old=C.copy()
        #1. Calculate "temporary" \tilde C(t+dt/2) using F(C(t),t+dt/4)
        D=self.calculate_density(C)
        Fock=self.calculate_Fock_matrix(D)
        Fock+=field(t+dt/4)*np.diag(self.grid) #Add the field to the Fock matrix
        I=np.eye(self.num_gridpoints,dtype=np.complex128)
        A_op=I+1j*dt/4*Fock
        C_tilde=np.linalg.solve(A_op,np.conj(A_op).T@C[:,:self.norb])
        D_tilde=self.calculate_density(C_tilde)        
        #2. Calculate C(t+dt/2) using F(\tilde C(t+dt/2),t+dt/4)
        Fock=self.calculate_Fock_matrix(D_tilde)
        Fockfield=Fock+field(t+dt/4)*np.diag(self.grid)
        A_op=I+1j*dt/4*Fockfield
        C_dt2=np.linalg.solve(A_op,np.conj(A_op).T@C_old[:,:self.norb])
        #3. Calculate C(t+dt) using F(C(t+dt/2),t+3dt/4)
        #Fock=self.calculate_Fock_matrix(D_dt2)
        #Fock+=field(t+3*dt/4)*np.diag(self.grid)
        Fockfield=Fock+field(t+3*dt/4)*np.diag(self.grid)
        A_op=I+1j*dt/4*Fockfield
        C_dt=np.linalg.solve(A_op,np.conj(A_op).T@C_dt2[:,:self.norb])

        self.C=C_dt
        return C_dt
    def propagate_onestep_first_order(self,C,field,t,dt):
        C_old=C.copy()
        #1. Calculate "temporary" \tilde C(t+dt/2) using F(C(t),t+dt/4)
        D=self.calculate_density(C)
        Fock=self.calculate_Fock_matrix(D)
        Fock+=field(t+dt/2)*np.diag(self.grid) #Add the field to the Fock matrix
        
        print(np.linalg.eigh(Fock)[0][:5])
        I=np.eye(self.num_gridpoints,dtype=np.complex128)
        A_op=I+1j*dt/2*Fock
        C_nexstep=np.linalg.solve(A_op,np.conj(A_op).T@C[:,:self.norb])
        self.C=C_nexstep
        return C_nexstep
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
if __name__ == "__main__":
    R_list=[-1.15, 1.15]
    Z_list=[3,1]
    alpha=0.5
    gridsize = 15
    num_gridpoints = 400
    
    S=solver(gridsize,num_gridpoints,Z_list,R_list,alpha,nelec=sum(Z_list))
    orbs=S.C_init[:,:S.norb]

    #plt.plot(S.grid,(orbs[:,0]),label="Orbital 1")
    #plt.plot(S.grid,(orbs[:,1]),label="Orbital 2")
    S.SCF()
    orbs=S.C[:,:S.norb]
    print(orbs.shape)
    plt.plot(S.grid,abs(orbs[:,0]),label="Orbital 1 converged")
    plt.plot(S.grid,(orbs[:,1]),label="Orbital 2 converged")
    filename="orbitals_converged"
    np.savez(filename,grid=S.grid,orbitals=orbs)

    plt.legend()
    plt.show()
    E0 = float(sys.argv[1])  # Maximum field strength
    omega = 0.06075  # Laser frequency
    t_c = 2 * np.pi / omega  # Optical cycle
    n_cycles = 3

    td = n_cycles * t_c  # Duration of the laser pulse
    tfinal = td  # Total time of the simulation
    t=np.linspace(0,tfinal,300)
    fieldfunc=laserfield(E0, omega, td)
    plt.plot(t, fieldfunc(t))

    plt.show()
    t, dipole, energy = S.propagate(fieldfunc, 0.05, tfinal,E0)
    plt.plot(t, np.array(dipole))
    plt.ylim(-2.4, -0.6)
    plt.show()