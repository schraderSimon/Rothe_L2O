import numpy as np

# from numba import jit
from scipy.integrate import simpson
import scipy.fftpack
import scipy.signal


def T_sine_dvr(k1, k2, dr):
    # Reference: doi.org/10.1063/1.462100
    # Eq.(A8)
    if k1 == k2:
        return (
            1
            / (2 * dr**2)
            * (-1) ** (k1 - k2)
            * (np.pi**2 / 3 - 1 / (2 * k1**2))
        )
    else:
        return (
            1
            / (2 * dr**2)
            * (-1) ** (k1 - k2)
            * (2 / (k1 - k2) ** 2 - 2 / (k1 + k2) ** 2)
        )


def get_T_dvr(n_r, dr):
    T = np.zeros((n_r, n_r))
    for k1 in range(n_r):
        for k2 in range(n_r):
            T[k1, k2] = T_sine_dvr(k1 + 1, k2 + 1, dr)
    return T


def regularized_Coulomb(r, eps):
    if np.abs(r) <= eps:
        return (1 + 2 / np.pi * np.cos(np.pi * r / (2 * eps))) / eps
    else:
        return 1 / np.abs(r)


class Gaussian_nuclear_potential:
    def __init__(self, b=5, n_gauss=10):
        a = -b
        N = n_gauss * 2 - 1
        h = (b - a) / (N)
        self.weights = np.zeros(N + 1)
        self.exponents = np.zeros(N + 1)

        self.weights[0] = np.cosh(a) * 1 / np.sqrt(np.pi)
        self.weights[-1] = np.cosh(b) * 1 / np.sqrt(np.pi)
        for i in range(1, N):
            self.weights[i] = 2 * np.cosh(a + i * h) * 1 / np.sqrt(np.pi)
        for i in range(0, N + 1):
            self.exponents[i] = np.sinh(a + i * h) ** 2

        self.weights = self.weights * h * 0.5
        self.exponents = self.exponents[: len(self.exponents) // 2]
        self.weights = self.weights[: len(self.weights) // 2] * 2

    def __call__(self, r):
        result = 0
        counter = 0
        for i in range(len(self.weights)):
            result += self.weights[i] * np.exp(-self.exponents[i] * r**2)
        return result


delta = lambda x, y: x == y


def compute_numerical_states(l_max, n_max, r):
    # Compute states
    eigenenergies = {}
    eigenstates = {}

    # Assuming equispace grid
    dr = r[1] - r[0]
    n_grid = len(r)

    for l in range(l_max):
        h_diag = 1.0 / (dr**2) + l * (l + 1) / (2 * r**2) - 1 / r
        h_off_diag = -1.0 / (2 * dr**2) * np.ones(n_grid - 1)
        H = (
            np.diag(h_diag)
            + np.diag(h_off_diag, k=-1)
            + np.diag(h_off_diag, k=1)
        )
        eps_l, u_l = np.linalg.eigh(H)
        eigenstates[l] = u_l[:, :n_max]
        eigenenergies[l] = eps_l[:n_max]

    # normalize states
    for l in eigenstates:
        states = eigenstates[l]
        normalized_states = np.zeros_like(states)
        for i, state in enumerate(states.T):
            normalized_states[:, i] = state / np.sqrt(
                simps(np.abs(state) ** 2, r)
            )
        eigenstates[l] = normalized_states

    return eigenenergies, eigenstates


def mask_function(r, r_max, r0):
    if r < r0:
        return 1
    else:
        return np.cos(np.pi * (r - r0) / (2 * (r_max - r0))) ** (1 / 8)


# @jit(nopython=True)
# def tridiag_prod(a, b, c, v):

#     v_new = np.zeros(v.shape, dtype=np.complex128)
#     L = len(v)

#     v_new[0] = a[0] * v[0] + b[0] * v[1]
#     v_new[L - 1] = c[L - 2] * v[L - 2] + a[L - 1] * v[L - 1]

#     for i in range(1, L - 1):
#         v_new[i] = c[i - 1] * v[i - 1] + a[i] * v[i] + b[i] * v[i + 1]

#     return v_new


# ## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
# @jit(nopython=True)
# def TDMAsolver(a, b, c, d, size):
#     """
#     TDMA solver, a b c d can be NumPy array type or Python list type.
#     refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
#     and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
#     """
#     nf = size  # number of equations
#     ac, bc, cc, dc = (
#         a.copy(),
#         b.copy(),
#         c.copy(),
#         d.copy(),
#     )  # map(np.array, (a, b, c, d)) # copy arrays

#     for it in range(1, nf):
#         mc = ac[it - 1] / bc[it - 1]
#         bc[it] = bc[it] - mc * cc[it - 1]
#         dc[it] = dc[it] - mc * dc[it - 1]

#     xc = np.zeros(size, np.complex128)
#     xc += bc

#     xc[-1] = dc[-1] / bc[-1]

#     for il in range(nf - 2, -1, -1):
#         xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

#     return xc


def compute_dipole_moment(r, psi):

    dipole_moment = 0j

    for l in range(psi.shape[0] - 1):

        dipole_moment += (
            2
            * (l + 1)
            / np.sqrt((2 * l + 1) * (2 * l + 3))
            * simps(r * psi[l].conj() * psi[l + 1], r)
        )

    return dipole_moment.real


def compute_overlap(r, psi_t, psi_nl):

    overlap = simps(psi_t.conj() * psi_nl, r)
    return overlap


def coeff(l):
    return (l + 1) / np.sqrt((2 * l + 1) * (2 * l + 3))


# Setup Hamiltonian: -1/2*nabla^2 + l*(l+1)/(2*r) - 1/r + z*cos(theta)*E(t)
def setup_Hamiltonian(r_max, dr, l_max):

    n_grid = int(r_max / dr)
    r = np.arange(1, n_grid + 1) * dr

    M = l_max * n_grid
    H0 = np.zeros((M, M))

    # H0 is tridiagnoal in space (index j)
    for l in range(l_max):

        h_diag = 1.0 / (dr**2) + l * (l + 1) / (2 * r**2) - 1 / r
        h_off_diag = -1.0 / (2 * dr**2) * np.ones(n_grid - 1)

        i0 = l * n_grid
        i1 = (l + 1) * n_grid

        H0[i0:i1, i0:i1] = (
            np.diag(h_diag)
            + np.diag(h_off_diag, k=-1)
            + np.diag(h_off_diag, k=1)
        )

    Hint = np.zeros((M, M))

    # Hint is tridiagonal in l.
    for j in range(n_grid):

        upper = coeff(np.arange(l_max - 1)) * r[j]
        lower = coeff(np.arange(l_max - 1)) * r[j]

        i0 = j * l_max
        i1 = (j + 1) * l_max

        Hint[i0:i1, i0:i1] = np.diag(lower, k=-1) + np.diag(upper, k=1)

    return H0, Hint


def compute_hhg_spectrum(time_points, dipole_moment, hann_window=False):

    dip = scipy.signal.detrend(dipole_moment, type="constant")
    if hann_window:
        Px = (
            np.abs(
                scipy.fftpack.fftshift(
                    scipy.fftpack.fft(
                        dip * np.sin(np.pi * time_points / time_points[-1]) ** 2
                    )
                )
            )
            ** 2
        )
    else:
        Px = np.abs(scipy.fftpack.fftshift(scipy.fftpack.fft(dip))) ** 2

    dt = time_points[1] - time_points[0]
    print(dt)
    omega = (
        scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(time_points)))
        * 2
        * np.pi
        / dt
    )

    return omega, Px
