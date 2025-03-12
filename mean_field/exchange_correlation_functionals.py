import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import splrep, splev, interp1d
from scipy.interpolate import CubicSpline
# Define sin_integral function
def sin_integral(rho):
    def integrand(y, rho):
        ledd1=np.sin(y)**2/y**2
        ledd2=-0.5/np.sqrt(np.pi**2/4 + (y/rho)**2)
        return ledd1*ledd2
    if rho<1e-12:
        pass;return -0.291*2*rho+np.log(2*rho)*(0.5*rho)
    if rho>10000:
        pass;return -1/2+0.203/(2*rho)
    result, _ = quad(integrand, 0, np.inf, args=(rho,), epsabs=1e-12, epsrel=1e-12,limit=5000)
    return result

# Define epsilon_x as a vectorized function
def epsilon_x(rho):
    return np.vectorize(sin_integral)(rho)
def epsilon_x_alt(rho, f_spline):
    # Apply the first condition: rho < 1e-10
    condition1 = rho < 1e-12
    result1 = -0.291*2*rho+np.log(2*rho)*(0.5*rho)
    
    # Apply the second condition: rho > 100
    condition2 = rho > 1000
    result2 =-1/2+0.203/(2*rho)
    
    # Default case: use f_spline
    default_result = f_spline(rho)
    
    # Combine conditions using np.where
    result = np.where(condition1, result1, np.where(condition2, result2, default_result))
    
    return result# Step 1: Sample points
def epsilon_c(rho):
    rho=abs(rho)+1e-17
    A=18.4
    B=0
    C=7.501
    D=0.10185
    E=0.012827
    alpha=1.511
    beta=0.258
    m=4.424
    rs=1/(2*rho)
    term1=-0.5*np.log1p(alpha*rs+beta*rs**m)
    term2=rs+E*rs**2
    term3=A+B*rs+C*rs**2+D*rs**3
    negligible_rho=np.abs(rho)<1e-15
    result=term1*term2/term3
    result[negligible_rho] = 0
    return result
def epsilon_c_deriv(rho):
    """
    Computes the derivative of epsilon_C with respect to rho, i.e. dεC/dρ.
    
    Parameters
    ----------
    rho : float or array-like
        The density value(s) at which to compute the derivative.
        
    Returns
    -------
    dεC/dρ : float or np.ndarray
        The derivative of εC with respect to ρ for the input rho.
    """
    # Ensure rho is a numpy array for vectorized operations
    rho = np.atleast_1d(rho)
    rho=abs(rho)+1e-17
    # Define parameters
    A = 18.4
    B = 0.0
    C = 7.501
    D = 0.10185
    E = 0.012827
    alpha = 1.511
    beta = 0.258
    m = 4.424

    # rs = 1/(2*rho)
    rs = 1.0/(2.0*rho)

    # U(rs) = ln(1 + alpha*rs + beta*rs^m)
    inner =alpha*rs + beta*(rs**m)
    U = np.log1p(inner)
    
    # V(rs) = rs + E*rs^2
    V = rs + E*(rs**2)
    
    # H(rs) = A + B*rs + C*rs^2 + D*rs^3
    H = A + B*rs + C*(rs**2) + D*(rs**3)
    
    # Derivatives:
    # U'(rs) = (alpha + beta*m*rs^(m-1)) / (1 + alpha*rs + beta*rs^m)
    U_prime = (alpha + beta*m*(rs**(m-1))) / (1+inner)
    
    # V'(rs) = 1 + 2E*rs
    V_prime = 1.0 + 2.0*E*rs
    
    # H'(rs) = B + 2C*rs + 3D*rs^2
    H_prime = B + 2.0*C*rs + 3.0*D*(rs**2)
    
    # G(rs) = U(rs)*V(rs)
    G = U*V
    
    # G'(rs) = U'(rs)*V(rs) + U(rs)*V'(rs)
    G_prime = U_prime*V + U*V_prime
    
    # f(rs) = εC(rs) = -1/2 * (G/H)
    # f'(rs) = -1/2 * (G'H - GH') / H^2
    numerator = G_prime*H - G*H_prime
    f_prime = -0.5 * numerator / (H**2)
    
    # dr_s/dρ = -1/(2ρ²)
    drs_drho = -1.0/(2.0*(rho**2))
    
    # dεC/dρ = f'(rs)*drs_drho
    d_eC_drho = f_prime * drs_drho

    # If the input was a single float, return a float
    if d_eC_drho.size == 1:
        return d_eC_drho.item()
    return np.nan_to_num(d_eC_drho)    
def first_order_finite_difference(f, rho, h=5e-8):
    """
    Calculate the first-order finite difference derivative of a function.

    Parameters:
    f (function): The function to differentiate.
    rho (float): The point at which to compute the first derivative.
    h (float): The step size for finite difference (default: 1e-5).

    Returns:
    float: The first-order finite difference derivative.
    """
    return (f(rho + h) - f(rho - h)) / (2 * h)
def make_exchange_correlation_and_potential(x_sample,y_sample):
    epsilon_x_spline = CubicSpline(x_sample, y_sample)
    epsilon_x_spline_deriv = epsilon_x_spline.derivative()
    def epsilon_x_deriv(rho):
        rho=abs(rho)+1e-17
        condition1 = rho < 1e-12
        result1 = 0.5*np.log(rho) + 0.5*np.log(2) - 0.082
        
        condition2 = rho > 1000
        result2 =-0.1015/((rho)**2)
        
        default_result = epsilon_x_spline_deriv(rho)
        
        result = np.where(condition1, result1, np.where(condition2, result2, default_result))
        
        return result# Step 1: Sample points
    def epsilon_x(rho):
        rho=abs(rho)+1e-17
        condition1 = rho < 1e-12
        result1 = -0.291*2*rho+np.log(2*(rho))*(0.5*rho)
        
        condition2 = rho > 1000
        result2 =-1/2+0.203/(2*rho)
        
        default_result = epsilon_x_spline(rho)
        
        result = np.where(condition1, result1, np.where(condition2, result2, default_result))
        return result
    def epsilon_xc(rho):
        return epsilon_c(rho) + epsilon_x(rho)
    def epsilon_xc_deriv(rho):
        return epsilon_c_deriv(rho) + epsilon_x_deriv(rho)
    def v_xc(rho):
        returnval=rho*epsilon_xc_deriv(rho)+epsilon_xc(rho)
        return np.nan_to_num(returnval)
    return epsilon_xc,v_xc
import os 
module_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the exchange_values.npz file
file_path = os.path.join(module_dir, "exchange_values.npz")
try:
    data = data = np.load(file_path)
    x_sample = data['x']
    y_sample = data['y']
    print("Loaded exchange values from file.")
except:
    print("Computing exchange values")
    x_sample=np.sort(np.concatenate((np.logspace(-14,4,500),np.linspace(0.001,10,200))))
    y_sample = epsilon_x(x_sample)       
    np.savez(file_path, x=x_sample, y=y_sample)
epsilon_xc,v_xc=make_exchange_correlation_and_potential(x_sample,y_sample)
