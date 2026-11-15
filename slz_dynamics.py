"""Kern: Lager mit Bestellregel und streuender Lieferzeit (linear, Erwartungswert-Modell und Einzelbestellung).

Modell (Perioden): Bestand i (minus Soll), Bestellung u (minus Mittelbedarf D), Bedarfsabweichung e.
  i_{t+1} = i_t + Ankunft_{t+1} - e_t,   u_t = -alpha (i_t + beta * Leitung_t)
Lieferzeit-Verteilung f_l = P(Lieferzeit = l). Hauptfamilie ("diskretes Erlang", eigene Wahl): Summe aus k geometrischen Zeiten mit
Fortschrittswahrscheinlichkeit p je Periode, q = 1 - p; mindestens k Perioden, Mittel tau = k / p. p = 1 ist die feste Lieferzeit L = k.
  anteilig (Erwartungswert-Modell, linear): von jeder Bestellung kommt der Anteil f_l nach l Perioden an.
  Einzelbestellung: jede Bestellung kommt als Ganzes nach einer gezogenen Lieferzeit (Bestellungen dürfen sich überholen).
Zustand der Kette: (i, x_1, ..., x_k), x_j Inhalt der Stufe j; Leitung = Summe der Stufeninhalte.

Eigene Herleitungen in diesem Modell (nicht aus einer Quelle; Literatur zu Lieferzeit-Verteilungen im Lager wurde nicht gelesen):
  beta = 0: charakteristisches Polynom (z - 1)(z - q)^k + alpha p^k z;  beta = 1: (z - (1 - alpha))(z - q)^k.
  Grenze alpha_krit als kleinster positiver reeller Wert von alpha(z) = (1 - z) / G(z) auf dem Einheitskreis, G(z) = sum_l f_l z^(1-l)
  (Wurzelortskurve; für beliebige Verteilungen und unabhängig von den Eigenwerten der Zustandsmatrix).
  zweites Moment der Einzelbestellung: lineares System für E[z z^T] und die Summe der Quadrate der Bestellklumpen je Stufe.

Herkunft des Codes: neu geschrieben für dieses Repo (Messlogik der Vormessung `rk_v18*` nicht kopiert); die Zufallsfolgen kommen aus `slz_rng.py`.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

import slz_rng as R

# ---------------------------------------------------------------------------------------------------------------- Kette


def stages(tau: int, k: int) -> tuple[int, float]:
    """Stufenzahl k (auf 1 bis tau begrenzt) und Fortschrittswahrscheinlichkeit p = k / tau; k = tau ist die feste Lieferzeit."""
    k_eff = max(1, min(int(k), int(tau)))
    return k_eff, k_eff / float(tau)


def chain_matrix(k: int, p: float, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Zustandsmatrix A (ohne Bedarfsabweichung) der Kette mit k Stufen und die Zeile c mit u = c·z; Zustand (i, x_1, ..., x_k)."""
    q = 1.0 - p
    n = k + 1
    c = np.zeros(n)
    c[0] = -alpha
    c[1:] = -alpha * beta
    e1 = np.zeros(n)
    e1[1] = 1.0
    x1p = e1 + c                       # Stufe 1 nach dem Eintrag der neuen Bestellung
    A = np.zeros((n, n))
    A[0, 0] = 1.0
    A[0] += p * (x1p if k == 1 else _unit(n, k))
    A[1] = q * x1p
    for j in range(2, k + 1):
        A[j] = q * _unit(n, j) + p * (x1p if j == 2 else _unit(n, j - 1))
    return A, c


def _unit(n: int, j: int) -> np.ndarray:
    r = np.zeros(n)
    r[j] = 1.0
    return r


def dist_matrix(f, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Zustandsmatrix für eine beliebige Verteilung f (f[l-1] = P(Lieferzeit = l)); Zustand (i_t, u_{t-1}, ..., u_{t-L+1}), L = len(f).

    Nur für kurze Verteilungen und Zeitreihen gedacht: Eigenwerte langer Matrizen mit vielen Wurzeln nahe 0 sind schlecht konditioniert."""
    f = np.asarray(f, float)
    L = len(f)
    c = np.zeros(L)
    c[0] = -alpha
    for j in range(1, L):
        c[j] = -alpha * beta * f[j:].sum()
    A = np.zeros((L, L))
    A[0] = np.eye(L)[0] + f[0] * c
    for l in range(2, L + 1):
        A[0, l - 1] += f[l - 1]
    if L >= 2:
        A[1] = c
        for j in range(2, L):
            A[j, j - 1] = 1.0
    return A, c


def spectral_radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def charpoly_chain(k: int, p: float, alpha: float, beta: float) -> np.ndarray:
    """Koeffizienten (höchster Grad zuerst) des charakteristischen Polynoms der Kette; nur beta = 0 und beta = 1 sind hergeleitet."""
    q = 1.0 - p
    base = np.poly([q] * k)
    if beta == 0:
        poly = np.polymul([1.0, -1.0], base)
        poly[-2] += alpha * p ** k
        return poly
    if beta == 1:
        return np.polymul([1.0, -(1.0 - alpha)], base)
    raise ValueError("Polynom nur für beta = 0 und beta = 1 hergeleitet")


# ------------------------------------------------------------------------------------------------- Verteilungen


def erlang_pmf(k: int, p: float, lmax: int) -> np.ndarray:
    """f_l für l = 1..lmax der Summe aus k geometrischen Zeiten: C(l-1, k-1) p^k q^(l-k); Index l-1."""
    f = np.zeros(lmax)
    q = 1.0 - p
    for l in range(k, lmax + 1):
        f[l - 1] = math.comb(l - 1, k - 1) * p ** k * (q ** (l - k) if l > k else 1.0)
    return f


def mean_var(f) -> tuple[float, float]:
    f = np.asarray(f, float)
    ls = np.arange(1, len(f) + 1)
    m = float(np.sum(ls * f))
    return m, float(np.sum((ls - m) ** 2 * f))


def fixed_pmf(tau: int) -> np.ndarray:
    f = np.zeros(int(tau))
    f[-1] = 1.0
    return f


def two_point_pmf(tau: int, d: int) -> np.ndarray:
    """Lieferzeit tau - d oder tau + d, je mit Wahrscheinlichkeit 1/2 (Mittel tau); d auf 0 bis tau - 1 begrenzt."""
    d = max(0, min(int(d), int(tau) - 1))
    f = np.zeros(int(tau) + d)
    f[tau - d - 1] += 0.5
    f[tau + d - 1] += 0.5
    return f


def uniform_pmf(tau: int, w: int) -> np.ndarray:
    """Gleichverteilt auf tau - w, ..., tau + w (Mittel tau); w auf 0 bis tau - 1 begrenzt."""
    w = max(0, min(int(w), int(tau) - 1))
    f = np.zeros(int(tau) + w)
    f[tau - w - 1:tau + w] = 1.0 / (2 * w + 1)
    return f


RANDOM_FORMS = {"glatt": 1.0 / 3.0, "gemischt": 1.0, "spitz": 3.0}   # Exponent der Exponentialgewichte: kleiner = gleichmäßiger


def random_pmf(seed: int, tau: int, form: str = "gemischt") -> np.ndarray:
    """Zufällige Verteilung auf 1..lmax mit genau dem Mittel tau (eigene Konstruktion): lmax = tau + 1 bis tau + 4,
    Gewichte w_l = E_l^gamma mit E_l = -ln u_l (gamma je Form), danach exponentielle Neigung exp(theta l), theta so, dass das Mittel tau ist."""
    gamma = RANDOM_FORMS[form]
    u = R.uniform_array(seed, 5 + 4 + 64)
    lmax = int(tau) + 1 + int(u[0] * 4)
    e = -np.log(np.maximum(u[1:1 + lmax], 1e-12))
    logw = gamma * np.log(np.maximum(e, 1e-300))
    ls = np.arange(1, lmax + 1)

    def tilted(theta):
        lg = logw + theta * ls
        g = np.exp(lg - lg.max())
        return g / g.sum()

    lo, hi = -60.0, 60.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if float(np.sum(ls * tilted(mid))) < tau:
            lo = mid
        else:
            hi = mid
    return tilted(0.5 * (lo + hi))


# ---------------------------------------------------------------------------------------- Grenze (Wurzelortskurve)


def _theta_grid(n: int = 6000) -> np.ndarray:
    return np.unique(np.concatenate([np.geomspace(1e-6, math.pi, n), np.linspace(1e-3, math.pi, n)]))


_GRID = _theta_grid()


def _crossings(alpha_of_theta, grid=None) -> list[float]:
    """Alle positiven reellen Werte von alpha(e^{i theta}), theta in (0, pi]: Anzahlen von Wurzeln auf dem Einheitskreis können nur dort wechseln.
    Vorzeichenwechsel des Imaginärteils auf dem Raster, danach Bisektion; theta = pi (z = -1) gesondert."""
    th = _GRID if grid is None else grid
    inner = th[:-1]                                    # th[-1] = pi: der Imaginärteil dort ist Rundungsrauschen
    im = alpha_of_theta(inner).imag
    s = np.sign(im)
    idx = np.nonzero(s[:-1] * s[1:] < 0)[0]
    lo, hi = inner[idx], inner[idx + 1]
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        m_im = alpha_of_theta(mid).imag
        left = np.sign(m_im) == np.sign(alpha_of_theta(lo).imag)
        lo = np.where(left, mid, lo)
        hi = np.where(left, hi, mid)
    values = list(alpha_of_theta(0.5 * (lo + hi)).real) if len(idx) else []
    values.append(float(alpha_of_theta(np.array([math.pi])).real[0]))
    return sorted(v for v in values if v > 1e-12 and math.isfinite(v))


def _alpha_chain(k: int, p: float):
    q = 1.0 - p

    def f(theta):
        z = np.exp(1j * np.asarray(theta, float))
        return (1.0 - z) * (z - q) ** k / (z * p ** k)
    return f


def _alpha_dist(pmf):
    fl = np.asarray(pmf, float)
    ls = np.arange(1, len(fl) + 1)

    def f(theta):
        th = np.atleast_1d(np.asarray(theta, float))
        z = np.exp(1j * th)
        g = np.sum(fl[None, :] * np.exp(-1j * th[:, None] * (ls[None, :] - 1)), axis=1)
        return (1.0 - z) / g
    return f


def crossings_chain(k: int, p: float) -> list[float]:
    """Alle α, bei denen die Kette (beta = 0) eine Wurzel auf dem Einheitskreis hat (aufsteigend)."""
    return _crossings(_alpha_chain(k, p))


def critical_alpha_chain(k: int, p: float, beta: float = 0.0) -> float:
    """Erste Instabilität der Kette beim Hochdrehen von α: beta = 0 kleinster positiver Wert der Wurzelortskurve (inf, wenn es keinen gibt),
    beta = 1 gilt α = 2 für jede Verteilung (Bestandsposition IP' = (1 - α) IP - e; für p = 1 und beta = 1 ebenso)."""
    if beta == 1:
        return 2.0
    if beta != 0:
        raise ValueError("Grenze nur für beta = 0 und beta = 1 hergeleitet")
    cr = crossings_chain(k, p)
    return cr[0] if cr else math.inf


def crossings_dist(pmf) -> list[float]:
    return _crossings(_alpha_dist(pmf))


def critical_alpha_dist(pmf) -> float:
    """Grenze (beta = 0) für eine beliebige Lieferzeit-Verteilung."""
    cr = crossings_dist(pmf)
    return cr[0] if cr else math.inf


def critical_alpha_fixed(L: int) -> float:
    """Feste Lieferzeit L, beta = 0: 2 sin(pi / (4L - 2)) (eigene Herleitung in diesem Modell, bestätigt gegen die Eigenwerte)."""
    return 2.0 * math.sin(math.pi / (4 * L - 2))


def critical_continuous(k: int) -> float:
    """Stetiger Erlang-k-Kreis mit mittlerer Verzögerung 1: s (1 + s/k)^k + a = 0, kritisches a (inf für k = 1; fest: pi/2).

    Wurzelortskurve auf der imaginären Achse: a(w) = -i w (1 + i w / k)^k, kleinster positiver reeller Wert."""
    grid = np.unique(np.concatenate([np.geomspace(1e-4, 200.0, 8000), np.linspace(1e-3, 30.0, 8000)]))

    def a_of(w):
        w = np.asarray(w, float)
        return -1j * w * (1.0 + 1j * w / k) ** k

    im = a_of(grid).imag
    s = np.sign(im)
    idx = np.nonzero(s[:-1] * s[1:] < 0)[0]
    lo, hi = grid[idx], grid[idx + 1]
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        left = np.sign(a_of(mid).imag) == np.sign(a_of(lo).imag)
        lo = np.where(left, mid, lo)
        hi = np.where(left, hi, mid)
    vals = [v for v in a_of(0.5 * (lo + hi)).real if v > 1e-12] if len(idx) else []
    return min(vals) if vals else math.inf


def stability_returns(pmf, amax: float = 2.6, step: float = 0.01) -> tuple[float, int]:
    """(Grenze, Zahl der Rasterpunkte oberhalb der ersten Grenze mit Spektralradius < 1): kehrt die Stabilität zurück?

    Raster in α mit Schritt `step` bis `amax`. Nur Verteilungen mit mindestens zwei Kreuzungen unterhalb von amax können zurückkehren (zwischen zwei
    Wechseln braucht der Spektralradius eine Kreuzung des Einheitskreises); nur diese werden mit Eigenwerten gerastert."""
    cr = crossings_dist(pmf)
    first = cr[0] if cr else math.inf
    if len([c for c in cr if c <= amax + step]) < 2:
        return first, 0
    back = 0
    seen_unstable = False
    for a in np.arange(step, amax + 1e-9, step):
        r = spectral_radius(dist_matrix(pmf, float(a), 0.0)[0])
        if not seen_unstable:
            seen_unstable = r > 1 + 1e-9
        elif r < 1 - 1e-9:
            back += 1
    return first, back


# ---------------------------------------------------------------------------------------- Kennzahlen (anteilig)


def variances(A: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """(var(u)/var(e), var(i)/var(e)) bei weißem Rauschen e mit Varianz 1 (Lyapunov-Gleichung); e wirkt als -e auf i_{t+1}."""
    n = A.shape[0]
    b = np.zeros(n)
    b[0] = -1.0
    S = solve_discrete_lyapunov(A, np.outer(b, b))
    return float(c @ S @ c), float(S[0, 0])


def impulse_response(A: np.ndarray, c: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Bedarfsstoß e_0 = 1 (i_1 = -1): Bestand i_1..i_n und Bestellung u_1..u_n."""
    z = np.zeros(A.shape[0])
    z[0] = -1.0
    i_out = np.empty(n)
    u_out = np.empty(n)
    for t in range(n):
        i_out[t] = z[0]
        u_out[t] = c @ z
        z = A @ z
    return i_out, u_out


@dataclass(frozen=True)
class Decay:
    time: int | None      # Periode der letzten Überschreitung von frac * Spitze; None: nicht abgeklungen oder instabil
    peak_i: float
    peak_u: float


def decay_time(A: np.ndarray, c: np.ndarray, frac: float = 0.01, n_max: int = 100000) -> Decay:
    """Abklingzeit nach einem Bedarfsstoß: Periode t der letzten Überschreitung von frac * Spitze von |i| (Zustand ab t = 1)."""
    z = np.zeros(A.shape[0])
    z[0] = -1.0
    abs_i = [1.0]
    peak_u = 0.0
    for t in range(2, n_max):
        peak_u = max(peak_u, abs(float(c @ z)))
        z = A @ z
        abs_i.append(abs(z[0]))
        m = float(np.max(np.abs(z)))
        if m > 1e12:
            return Decay(None, max(abs_i), peak_u)
        if t > 20 and m < 1e-7 * frac:
            break
    else:
        return Decay(None, max(abs_i), peak_u)
    arr = np.array(abs_i)
    peak = float(arr.max())
    over = np.nonzero(arr > frac * peak)[0]
    return Decay(int(over.max()) + 1, peak, peak_u)


def rho_curve(k: int, p: float, beta: float, alphas) -> np.ndarray:
    return np.array([spectral_radius(chain_matrix(k, p, float(a), beta)[0]) for a in alphas])


def best_rho(k: int, p: float, beta: float, alphas) -> tuple[float, float]:
    """(kleinster Spektralradius, α dort): grobes α-Raster, danach zwei lokale Verfeinerungen (±1 Rasterschritt mit 81 Punkten, dann ±1/40 davon mit 41 Punkten).

    Der Spektralradius ist als Maximum von Eigenwertbeträgen nur stückweise glatt; die Verfeinerung findet das lokale Minimum auf etwa 10^-5 in α."""
    alphas = np.asarray(alphas, float)
    r = rho_curve(k, p, beta, alphas)
    j = int(np.argmin(r))
    a_best, r_best = float(alphas[j]), float(r[j])
    step = float(alphas[1] - alphas[0]) if len(alphas) > 1 else 0.01
    for n_pts, width in ((81, step), (41, step / 40.0)):
        fine = np.linspace(max(a_best - width, 1e-9), a_best + width, n_pts)
        rf = rho_curve(k, p, beta, fine)
        m = int(np.argmin(rf))
        if rf[m] < r_best:
            a_best, r_best = float(fine[m]), float(rf[m])
    return r_best, a_best


# -------------------------------------------------------------------------------------- Simulation (anteilig)


def simulate_fluid(k: int, p: float, alpha: float, beta: float, runs: int, periods: int, burn: int, seed: int) -> tuple[float, float]:
    """(var(i), var(u)) der anteiligen Kette bei weißem Normalrauschen (Varianz 1), über `runs` parallele Läufe und die Perioden nach `burn`."""
    A, c = chain_matrix(k, p, alpha, beta)
    n = k + 1
    e = R.normal_array(seed, runs * periods).reshape(periods, runs)
    z = np.zeros((runs, n))
    s_i = s_i2 = s_u = s_u2 = 0.0
    m = 0
    for t in range(periods):
        if t >= burn:
            u = z @ c
            s_i += float(z[:, 0].sum())
            s_i2 += float((z[:, 0] ** 2).sum())
            s_u += float(u.sum())
            s_u2 += float((u * u).sum())
            m += runs
        z = z @ A.T
        z[:, 0] -= e[t]
    return s_i2 / m - (s_i / m) ** 2, s_u2 / m - (s_u / m) ** 2


def fluid_path(k: int, p: float, alpha: float, beta: float, e_series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bestand i_1..i_T und Bestellung u_1..u_T des anteiligen Modells bei vorgegebener Bedarfsabweichung e_series (Länge T)."""
    A, c = chain_matrix(k, p, alpha, beta)
    z = np.zeros(k + 1)
    T = len(e_series)
    i_out = np.empty(T)
    u_out = np.empty(T)
    for t in range(T):
        i_out[t] = z[0]
        u_out[t] = c @ z
        z = A @ z
        z[0] -= e_series[t]
    return i_out, u_out


# ---------------------------------------------------------------------------------- Einzelbestellung (Simulation)


def lead_time_cap(tau: int) -> int:
    """Kappung der gezogenen Lieferzeit (Perioden): 30 tau, mindestens 40, höchstens 720; die Wahrscheinlichkeit darüber ist für k = 1 etwa e^-30."""
    return int(min(720, max(40, 30 * int(tau))))


def draw_lead_times(seed: int, k: int, p: float, shape: tuple[int, ...], cap: int) -> np.ndarray:
    """Lieferzeiten k + (Fehlversuche von k geometrischen Zeiten), auf `cap` gekappt; Form `shape`."""
    n = int(np.prod(shape))
    u = R.uniform_array(seed, n * k).reshape(*shape, k)
    g = R.geometric_failures(u, p).sum(axis=-1)
    return np.minimum(k + g, cap).reshape(shape)


def simulate_individual(k: int, p: float, alpha: float, beta: float, D: float, sigma: float, runs: int, periods: int, seed: int,
                        shock: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Einzelbestellung (linear, keine Klemmung): jede Bestellung O_t = D + u_t kommt als Ganzes nach einer gezogenen Lieferzeit.

    Regel u_t = -alpha (i_t + beta (Transit_t - D (tau - 1))); Transit_t = Summe der noch nicht angekommenen Bestellungen nach den Ankünften von t.
    Vorgeschichte: vor t = 0 wurde jede Periode D bestellt. Rückgabe: Bestand I (periods + 1, runs) mit I[t] = i_t und Bestellung U (periods, runs).
    Zufallsströme: je Zweck ein eigener `child_seed` (Rauschen, Lieferzeiten, Vorgeschichte)."""
    tau = k / p
    cap = lead_time_cap(int(round(tau)))
    cap = max(cap, k + 1)
    width = periods + cap + 5
    buf = np.zeros((runs, width))
    rows = np.arange(runs)
    transit = np.zeros(runs)
    pre = draw_lead_times(R.child_seed(seed, 3), k, p, (cap, runs), cap)       # Zeile j gehört zur Periode s = -cap + j
    for j in range(cap):
        arr = (-cap + j) + pre[j]
        m = arr >= 1
        if m.any():
            buf[rows[m], arr[m]] += D
            transit[m] += D
    ell = draw_lead_times(R.child_seed(seed, 2), k, p, (periods, runs), cap)
    noise = R.normal_array(R.child_seed(seed, 1), periods * runs).reshape(periods, runs) if sigma > 0 else None
    i = np.zeros(runs)
    I = np.zeros((periods + 1, runs))
    U = np.zeros((periods, runs))
    for t in range(periods):
        u = -alpha * (i + beta * (transit - D * (tau - 1.0)))
        O = D + u
        buf[rows, t + ell[t]] += O
        arrive = buf[:, t + 1]
        e = sigma * noise[t] if noise is not None else 0.0
        if t == 0 and shock:
            e = e + shock
        i = i + arrive - (D + e)
        transit = transit + O - arrive
        I[t + 1] = i
        U[t] = u
    return I, U


# ------------------------------------------------------------------------- Einzelbestellung: zweites Moment (exakt)


def _phi_matrix(k: int) -> np.ndarray:
    """Wirkung der Weitergabe aus Stufe j auf (i, x_1, ..., x_k): Stufe k an den Bestand, sonst an die nächste Stufe."""
    n = k + 1
    Phi = np.zeros((n, k))
    Phi[0, k - 1] += 1.0
    for j in range(1, k + 1):
        Phi[j, j - 1] -= 1.0
    for j in range(2, k + 1):
        Phi[j, j - 2] += 1.0
    return Phi


def ms_operator(k: int, p: float, alpha: float, beta: float) -> np.ndarray:
    """Lineare Abbildung (vec S, Q) -> (vec S', Q') des zweiten Moments; S = E[z z^T], Q_j = E[Summe der Quadrate der Bestellklumpen in Stufe j].

    Die Weitergabe aus Stufe j hat bedingt auf den Zustand den Mittelwert p x_j und die Varianz p q Q_j; verschiedene Stufen sind bedingt unkorreliert.
    Der lineare Teil hat Spektralradius < 1 genau dann, wenn das zweite Moment (bei D = 0) endlich bleibt. Größe (k+1)^2 + k."""
    q = 1.0 - p
    n = k + 1
    A, c = chain_matrix(k, p, alpha, beta)
    Phi = _phi_matrix(k)
    Kphi = np.stack([np.kron(Phi[:, j], Phi[:, j]) for j in range(k)], axis=1)       # n^2 x k
    cc = np.kron(c, c)
    T = q * np.eye(k)
    if k > 1:
        T[1:, :-1] += p * np.eye(k - 1)
    N = n * n + k
    M = np.zeros((N, N))
    M[:n * n, :n * n] = np.kron(A, A) + p * q * np.outer(Kphi[:, 0], cc)
    M[:n * n, n * n:] = Kphi * (p * q)
    M[n * n:, :n * n] = np.outer(T[:, 0], cc)
    M[n * n:, n * n:] = T
    return M


def ms_spectral_radius(k: int, p: float, alpha: float, beta: float) -> float:
    return spectral_radius(ms_operator(k, p, alpha, beta))


def ms_stationary(k: int, p: float, alpha: float, beta: float, D: float, sigma: float) -> tuple[float, float] | None:
    """Stationäre (var(i), var(u)) der Einzelbestellung bei Bedarf D und Normalrauschen sigma; None, wenn das zweite Moment divergiert."""
    n = k + 1
    q = 1.0 - p
    M = ms_operator(k, p, alpha, beta)
    if spectral_radius(M) >= 1 - 1e-12:
        return None
    _, c = chain_matrix(k, p, alpha, beta)
    Phi = _phi_matrix(k)
    Qs = np.zeros(k)
    Qs[0] = D ** 2
    b = np.zeros(n)
    b[0] = -1.0
    Ss = Phi @ np.diag(p * q * Qs) @ Phi.T + sigma ** 2 * np.outer(b, b)
    Qn = q * Qs
    Qn[1:] += p * Qs[:-1]
    y = np.linalg.solve(np.eye(len(Qn) + n * n) - M, np.concatenate([Ss.ravel(), Qn]))
    S = y[:n * n].reshape(n, n)
    return float(S[0, 0]), float(c @ S @ c)


def ms_critical(k: int, p: float, beta: float = 0.0, amax: float = 12.0, amin: float = 0.01, ratio: float = 1.2) -> float:
    """Grenze des zweiten Moments: erstes α, bei dem der Spektralradius des Operators 1 erreicht (inf, wenn bis amax keines).

    Geometrisches α-Raster mit Verhältnis `ratio` (20 % Schritt), danach Bisektion. Schmalere Instabilitätsfenster als ein Rasterschritt sind nicht ausgeschlossen."""
    lo = 0.0
    a = amin
    while a <= amax:
        if ms_spectral_radius(k, p, a, beta) >= 1 - 1e-12:
            hi = a
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if ms_spectral_radius(k, p, mid, beta) >= 1 - 1e-12:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi)
        lo = a
        a *= ratio
    return math.inf
