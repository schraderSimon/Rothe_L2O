import numpy as np
from numpy import exp,pi,cos,sqrt
def get_Guess_distribution(vals,n):
    x=np.linspace(np.min(vals),np.max(vals),10000)
    returnvals=np.zeros(len(x))
    for a in vals:
        returnvals+=exp(-(x-a)**2/(2*(a+1e-10)**2))*1/sqrt(2*np.pi*(a+1e-10)**2)
    p=returnvals/np.sum(returnvals)
    return np.random.choice(x,p=p,size=n)
def cosine4_mask(x, a, b):
    a0=0.85*a
    b0=b*0.85
    returnval=np.zeros_like(x)
    left_absorber=cos(pi/2*(a0-x)/(a-a0))**(1/4)
    right_absorber=cos(pi/2*(b0-x)/(b0-b))**(1/4)
    returnval[(x<a0) & (x>a)]=left_absorber[(x<a0) & (x>a)]
    returnval[(x>b0) & (x<b)]=right_absorber[(x>b0) & (x<b)]
    returnval[(x<a) | (x>b)]=0
    returnval[(x>a0) & (x<b0)]=1
    return returnval
