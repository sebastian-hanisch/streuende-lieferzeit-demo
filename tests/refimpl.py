"""Unabhängige Referenzrechnungen für die Tests (anderer Rechenweg als `slz_dynamics`): Handschritt in Brüchen, feste Lieferzeit mit ausdrücklicher Vorgeschichte,
Grenze über Eigenwerte (Raster und Bisektion) statt über die Wurzelortskurve, Moment-Operator durch Probevektoren statt über Kronecker-Produkte."""

import math
from fractions import Fraction as F

import numpy as np


def step_fractions(i, x, alpha, beta, p, e=F(0)):
    """Ein Schritt der Kette in Brüchen (Handrechnung): Rückgabe i', x', u, Ankunft. x_1 bekommt u, aus jeder Stufe geht der Anteil p weiter."""
    k = len(x)
    u = -alpha * (i + beta * sum(x))
    xp = list(x)
    xp[0] = xp[0] + u
    m = [p * v for v in xp]
    xn = [xp[0] - m[0]]
    for j in range(1, k):
        xn.append(xp[j] - m[j] + m[j - 1])
    return i + m[k - 1] - e, xn, u, m[k - 1]


def matrix_from_step(alpha, beta, p, k):
    """Zustandsmatrix der Kette aus dem Bruch-Schritt (Probevektoren), als float-Matrix."""
    cols = []
    for j in range(k + 1):
        v = [F(0)] * (k + 1)
        v[j] = F(1)
        i2, x2, _, _ = step_fractions(v[0], v[1:], alpha, beta, p)
        cols.append([i2] + x2)
    return np.array([[float(cols[j][r]) for j in range(k + 1)] for r in range(k + 1)])


def fixed_lag_path(L, alpha, beta, e):
    """Feste Lieferzeit L mit ausdrücklicher Bestellhistorie: i_{t+1} = i_t + u_{t+1-L} - e_t, u_t = -alpha (i_t + beta sum_{j=1}^{L-1} u_{t-j}); i_0 = 0.
    Rückgabe i_0..i_{T-1} und u_0..u_{T-1} (gleiche Indizierung wie `fluid_path`)."""
    T = len(e)
    i = [0.0]
    u = []
    for t in range(T):
        pipeline = sum(u[t - j] for j in range(1, L) if t - j >= 0)
        u.append(-alpha * (i[t] + beta * pipeline))
        arriving = u[t + 1 - L] if t + 1 - L >= 0 else 0.0
        if L == 1:
            arriving = u[t]
        i.append(i[t] + arriving - e[t])
    return np.array(i[:T]), np.array(u)


def rho(A):
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def crit_by_eigenvalues(matrix_of_alpha, amax=2.6, step=0.01):
    """Kleinstes α mit Spektralradius > 1: Raster mit Schritt `step`, danach Bisektion; inf, wenn bis amax keines."""
    lo = 0.0
    a = step
    while a <= amax + 1e-12:
        if rho(matrix_of_alpha(a)) > 1 + 1e-9:
            hi = a
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if rho(matrix_of_alpha(mid)) > 1 + 1e-12:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi)
        lo = a
        a += step
    return math.inf


def ms_operator_by_probing(k, p, alpha, beta, chain_matrix, phi_matrix):
    """Moment-Operator spaltenweise durch die Anwendung der Vorschrift auf Einheitsvektoren (statt Kronecker-Produkte)."""
    q = 1.0 - p
    n = k + 1
    A, c = chain_matrix(k, p, alpha, beta)
    Phi = phi_matrix(k)
    N = n * n + k

    def apply(y):
        S = y[:n * n].reshape(n, n)
        Q = y[n * n:]
        Qp = Q.copy()
        Qp[0] += c @ S @ c
        Sn = A @ S @ A.T + Phi @ np.diag(p * q * Qp) @ Phi.T
        Qn = q * Qp
        Qn[1:] += p * Qp[:-1]
        return np.concatenate([Sn.ravel(), Qn])

    M = np.zeros((N, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = 1.0
        M[:, j] = apply(e)
    return M
