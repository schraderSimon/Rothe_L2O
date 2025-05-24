import numpy as np
import numpy
import matplotlib.pyplot as plt
from numpy import exp, pi, sinh, cosh, sqrt
from numpy import linalg
from scipy.linalg import det, inv,eig
from scipy.special import erf, erfi, dawsn
import scipy
import sys
import time
import warnings
import mpmath
np.set_printoptions(precision=6,linewidth=300)

def solve(A,b):
    try:
        return np.linalg.solve(A+np.eye(len(A))*1e-14,b)
    except:
        return np.linalg.solve(A+np.eye(len(A))*1e-12,b)

