"""Orakel-Tests (ORACLE-PLAYBOOK §5, §6): der Kern gegen einen anderen Rechenweg.

Orakel: (1) feste Lieferzeit mit ausdrücklicher Bestellhistorie statt Kette, (2) Verteilungs-Matrix mit negativer Binomialverteilung statt Kette (Zeitreihen, nicht
Eigenwerte langer Matrizen), (3) Grenze über Eigenwerte (Raster und Bisektion) statt über die Wurzelortskurve, (4) Summe der Quadrate der Impulsantwort und Simulation
statt Lyapunov-Gleichung, (5) Moment-Operator durch Probevektoren statt Kronecker-Produkte, exaktes zweites Moment gegen Simulation, (6) Mittelwert der Einzelbestellung gegen
die anteilige Impulsantwort. Simulationen vergleichen Bänder (Plattform, Zufallsstichprobe), Rechnungen strenge Toleranzen. Parameter sind keine Zweierpotenzen (p = 1 ist die feste Lieferzeit)."""

import math

import numpy as np
import pytest
from refimpl import crit_by_eigenvalues, fixed_lag_path, ms_operator_by_probing, rho

import slz_dynamics as K
import slz_evaluation as E
import slz_rng as R

# --- (1) feste Lieferzeit ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("L,alpha,beta", [(1, 0.4, 0.0), (2, 1.2, 0.55), (3, 0.3, 0.0), (5, 0.7, 0.9), (8, 0.21, 1.0), (12, 0.13, 0.4)])
def test_chain_with_p_one_equals_the_fixed_delay_with_explicit_history(L, alpha, beta):
    e = 0.3 * R.normal_array(7, 120)
    i_ref, u_ref = fixed_lag_path(L, alpha, beta, e)
    i, u = K.fluid_path(L, 1.0, alpha, beta, e)
    assert np.max(np.abs(i - i_ref)) < 1e-9 and np.max(np.abs(u - u_ref)) < 1e-9


def test_charpoly_equals_matrix_polynomial():
    """Das hergeleitete Polynom (z − 1)(z − q)^k + α p^k z (β = 0) und (z − (1 − α))(z − q)^k (β = 1) gegen die Koeffizienten der Zustandsmatrix (numpy.poly), 70 Fälle."""
    worst = 0.0
    n = 0
    for k, p in ((1, 0.3), (2, 0.4), (3, 0.3), (4, 1 / 3), (5, 0.8), (6, 0.45), (3, 1.0)):
        for alpha in (0.13, 0.31, 0.47, 0.79, 1.3):
            for beta in (0, 1):
                worst = max(worst, float(np.max(np.abs(np.poly(K.chain_matrix(k, p, alpha, beta)[0]) - K.charpoly_chain(k, p, alpha, beta)))))
                n += 1
    assert n == 70 and worst < 1e-12


# --- (2) Verteilungs-Matrix ----------------------------------------------------------------------------------------------------------


def _series(A, n, i0=-1.0):
    z = np.zeros(A.shape[0])
    z[0] = i0
    out = []
    for _ in range(n):
        out.append(z[0])
        z = A @ z
    return np.array(out)


@pytest.mark.parametrize("k,p", [(1, 0.3), (1, 0.15), (2, 0.4), (2, 2 / 3), (3, 0.3), (3, 3 / 5), (4, 0.8)])
@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0])
def test_chain_equals_distribution_matrix_with_negative_binomial(k, p, beta):
    """Die Kette gegen die Matrix einer Verteilung mit f_l = C(l-1, k-1) p^k q^(l-k) (Länge 150): gleiche Bestandsfolge nach einem Stoß."""
    for alpha in (0.13, 0.31, 0.47):
        Ac, _ = K.chain_matrix(k, p, alpha, beta)
        Ad, _ = K.dist_matrix(K.erlang_pmf(k, p, 150), alpha, beta)
        assert np.max(np.abs(_series(Ac, 160) - _series(Ad, 160))) < 1e-8


def test_distribution_matrix_for_a_point_mass_equals_the_fixed_chain():
    for L in (1, 2, 3, 6):
        for alpha, beta in ((0.2, 0.0), (0.7, 1.0)):
            Ad, _ = K.dist_matrix(K.fixed_pmf(L), alpha, beta)
            Ac, _ = K.chain_matrix(L, 1.0, alpha, beta)
            assert np.max(np.abs(_series(Ad, 80) - _series(Ac, 80))) < 1e-12


# --- (3) Grenze ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k,p,amax", [(2, 2 / 3, 1.5), (2, 1 / 6, 1.0), (3, 3 / 5, 1.5), (3, 1.0, 1.5), (4, 1 / 3, 1.0), (6, 0.45, 1.0), (1, 1 / 3, 12.0), (1, 0.15, 28.0), (12, 1.0, 1.0)])
def test_limit_by_locus_equals_limit_by_eigenvalues(k, p, amax):
    by_locus = K.critical_alpha_chain(k, p)
    by_eig = crit_by_eigenvalues(lambda a: K.chain_matrix(k, p, a, 0.0)[0], amax=amax, step=0.01 if amax <= 2 else 0.05)
    assert by_locus == pytest.approx(by_eig, rel=1e-7)


def test_limit_formulas_k1_k2_fixed():
    for tau in (2, 3, 5, 12, 60, 600):
        assert K.critical_alpha_chain(1, 1.0 / tau) == pytest.approx(4 * tau - 2, rel=1e-9)
        assert K.critical_alpha_chain(2, 2.0 / tau) == pytest.approx((2.0 / tau) * (2 - 2.0 / tau), rel=1e-9)
    for L in (1, 2, 3, 6, 12, 24):
        assert K.critical_alpha_chain(L, 1.0) == pytest.approx(2 * math.sin(math.pi / (4 * L - 2)), rel=1e-9)
        assert K.critical_alpha_chain(L, 1.0) == pytest.approx(K.critical_alpha_fixed(L), rel=1e-12)


def test_limit_of_other_shapes_by_eigenvalues():
    """Zweipunkt, gleichverteilt, zufällig: die Grenze aus der Wurzelortskurve gegen den Eigenwert-Raster (Mittel 4 und 5)."""
    for tau in (4, 5):
        pmfs = [K.two_point_pmf(tau, 1), K.two_point_pmf(tau, 2), K.uniform_pmf(tau, 1), K.uniform_pmf(tau, 2)]
        pmfs += [E.frozen_pmf(20261115, tau, i) for i in range(12)]
        for f in pmfs:
            locus = K.critical_alpha_dist(f)
            by_eig = crit_by_eigenvalues(lambda a, f=f: K.dist_matrix(f, a, 0.0)[0], amax=2.9, step=0.01)
            if math.isfinite(by_eig):
                assert locus == pytest.approx(by_eig, rel=1e-6)
            else:
                assert locus > 2.9


def test_continuous_limit_against_roots_of_the_polynomial():
    """s (1 + s/k)^k + a hat bei a = a_krit eine Nullstelle auf der imaginären Achse; knapp darunter ist alles in der linken Halbebene."""
    for k in (2, 3, 4, 6, 10):
        a = K.critical_continuous(k)
        poly = np.polymul([1.0, 0.0], np.poly([-float(k)] * k))
        poly[-1] += a * k ** k                                              # s (s + k)^k + a k^k = k^k (s (1 + s/k)^k + a)
        assert abs(np.roots(poly).real.max()) < 5e-5 * k
        below = np.polymul([1.0, 0.0], np.poly([-float(k)] * k))
        below[-1] += 0.98 * a * k ** k
        assert np.roots(below).real.max() < 0


# --- (4) Varianzen -----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k,p,alpha,beta", [(2, 2 / 3, 0.4, 0), (1, 1 / 3, 0.55, 0), (3, 1.0, 0.45, 0), (2, 2 / 3, 0.6, 1), (1, 0.15, 1.1, 1), (3, 0.3, 0.15, 0)])
def test_lyapunov_equals_sum_of_squared_impulse_response(k, p, alpha, beta):
    A, c = K.chain_matrix(k, p, alpha, beta)
    i, u = K.impulse_response(A, c, 30000)
    vu, vi = K.variances(A, c)
    assert vi == pytest.approx(float(np.sum(i ** 2)), rel=1e-6) and vu == pytest.approx(float(np.sum(u ** 2)), rel=1e-6)


@pytest.mark.parametrize("k,p,alpha,beta", [(2, 2 / 3, 0.4, 0), (1, 1 / 3, 0.55, 0), (2, 2 / 3, 0.6, 1)])
def test_lyapunov_equals_simulation_within_a_band(k, p, alpha, beta):
    A, c = K.chain_matrix(k, p, alpha, beta)
    vu, vi = K.variances(A, c)
    si, su = K.simulate_fluid(k, p, alpha, beta, 200, 2400, 400, 35)
    assert si == pytest.approx(vi, rel=0.05) and su == pytest.approx(vu, rel=0.05)


def test_beta_one_order_variance_for_arbitrary_distributions():
    """Gegenprobe zur Absolutaussage var(u)/var(e) = α/(2 - α) für jede Verteilung: 60 zufällige Verteilungen (Mittel 3, 4, 5, 6), gelockerte Annahmen (beliebige Stützweite)."""
    worst = 0.0
    n = 0
    for tau in (3, 4, 5, 6):
        for i in range(15):
            f = E.frozen_pmf(20261115, tau, i)
            for alpha in (0.2, 0.9, 1.7):
                A, c = K.dist_matrix(f, alpha, 1.0)
                vu, _ = K.variances(A, c)
                worst = max(worst, abs(vu - alpha / (2 - alpha)))
                n += 1
    assert n == 180 and worst < 1e-8


def test_beta_one_limit_is_two_for_arbitrary_distributions():
    """Gegenprobe zur Absolutaussage „Grenze α = 2 bei β = 1 für jede Verteilung“ (ORACLE §6): 60 zufällige Verteilungen (Mittel 3, 4, 5, 6): ρ = 1 − α unter 1 für α = 0,3 und 1,5,
    über 1 für α = 2,2 (Eigenwerte der Verteilungsmatrix; die Nullwurzeln sind schlecht konditioniert, Toleranz 0,03)."""
    n = 0
    for tau in (3, 4, 5, 6):
        for i in range(15):
            f = E.frozen_pmf(20261115, tau, i)
            for alpha, expected in ((0.3, 0.7), (1.5, 0.5), (2.2, 1.2)):
                r = rho(K.dist_matrix(f, alpha, 1.0)[0])
                assert abs(r - expected) < 0.03, (tau, i, alpha, r)
                n += 1
    assert n == 180


# --- (5) zweites Moment -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k,p,alpha,beta", [(1, 1 / 3, 0.4, 0), (2, 2 / 3, 0.2, 0), (3, 0.3, 0.1, 1), (4, 1 / 3, 0.35, 0), (2, 1 / 6, 0.9, 1)])
def test_moment_operator_by_kronecker_equals_probing(k, p, alpha, beta):
    M = K.ms_operator(k, p, alpha, beta)
    M_ref = ms_operator_by_probing(k, p, alpha, beta, K.chain_matrix, K._phi_matrix)
    assert M.shape == ((k + 1) ** 2 + k,) * 2 and np.max(np.abs(M - M_ref)) < 1e-12


@pytest.mark.parametrize("k,p,alpha,beta,D,sigma", [(1, 1 / 3, 0.2, 0, 10.0, 0.0), (2, 2 / 3, 0.4, 0, 10.0, 0.0), (2, 2 / 3, 0.4, 0, 10.0, 1.0), (2, 2 / 3, 0.55, 1, 10.0, 1.0)])
def test_exact_second_moment_equals_simulation_within_a_band(k, p, alpha, beta, D, sigma):
    ex_i, ex_u = K.ms_stationary(k, p, alpha, beta, D, sigma)
    I, U = K.simulate_individual(k, p, alpha, beta, D, sigma, 800, 2200, 35)
    sim_i = float(np.var(I[500:2200]))
    assert sim_i == pytest.approx(ex_i, rel=0.04)
    if beta == 0:
        assert float(np.var(U[500:2200])) == pytest.approx(ex_u, rel=0.05)


def test_second_moment_limit_values_of_the_exact_system():
    """Grenze des zweiten Moments (β = 0) gegen die Raster-Suche mit Schritt 0,01 über den Spektralradius des Operators (Probevektoren)."""
    for k, p in ((1, 1 / 3), (2, 2 / 3), (2, 1 / 6)):
        step, lo, a, found = 0.01, 0.0, 0.01, None
        while a < 12.0 and found is None:
            if rho(ms_operator_by_probing(k, p, a, 0, K.chain_matrix, K._phi_matrix)) >= 1 - 1e-12:
                found = a
            lo, a = a, a + step
        assert found is not None
        assert abs(K.ms_critical(k, p) - found) <= step


def test_second_moment_diverges_between_the_limits_and_stays_finite_below():
    k, p = 2, 2 / 3
    a_mean, a_ms = K.critical_alpha_chain(k, p), K.ms_critical(k, p)
    assert a_ms < a_mean
    assert K.ms_stationary(k, p, 0.9 * a_ms, 0, 10.0, 1.0) is not None
    between = 0.5 * (a_ms + a_mean)
    assert K.spectral_radius(K.chain_matrix(k, p, between, 0)[0]) < 1                 # Mittelwert stabil
    assert K.ms_stationary(k, p, between, 0, 10.0, 1.0) is None                         # zweites Moment nicht


# --- (6) Mittelwert der Einzelbestellung -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k,p,alpha,beta", [(1, 1 / 3, 0.55, 0), (2, 2 / 3, 0.4, 0), (2, 2 / 3, 0.6, 1)])
def test_mean_of_the_individual_order_follows_the_fluid_impulse_response(k, p, alpha, beta):
    """Stoß e_0 = 1 bei Bedarf 0, 8000 Läufe: das Mittel von i_t liegt in keiner der 40 Perioden mehr als 4 Standardfehler von der anteiligen Impulsantwort."""
    R_RUNS, T = 8000, 40
    I, _ = K.simulate_individual(k, p, alpha, beta, 0.0, 0.0, R_RUNS, T, 11, shock=1.0)
    mean = I[1:].mean(axis=1)
    se = I[1:].std(axis=1, ddof=1) / math.sqrt(R_RUNS)
    A, c = K.chain_matrix(k, p, alpha, beta)
    fluid, _ = K.impulse_response(A, c, T)
    measurable = se > 1e-5
    z = np.abs(mean - fluid)[measurable] / se[measurable]
    assert measurable.sum() >= 5 and z.max() < 4.0
    assert (np.abs(mean - fluid)[~measurable] < 1e-4).all()


def test_individual_order_with_p_one_is_exactly_the_fluid_model():
    """Feste Lieferzeit: Einzelbestellung und anteilig sind dasselbe Modell, Pfad für Pfad (gleiche Bedarfsabweichung)."""
    k, alpha = 3, 0.35
    I, _ = K.simulate_individual(k, 1.0, alpha, 0, 10.0, 1.0, 4, 120, 35)
    noise = R.normal_array(R.child_seed(35, 1), 120 * 4).reshape(120, 4)
    for r in range(4):
        i_f, _ = K.fluid_path(k, 1.0, alpha, 0, 1.0 * noise[:, r])
        assert np.max(np.abs(I[:120, r] - i_f)) < 1e-9                      # I[t] = i_t, fluid_path[t] = i_t mit i_0 = 0


# --- Zeitreihen: Abklingzeit gegen die Schätzung --------------------------------------------------------------------------------------


@pytest.mark.parametrize("k,p,alpha", [(2, 2 / 3, 0.4), (2, 2 / 3, 0.6), (1, 1 / 3, 0.55), (3, 1.0, 0.45), (3, 1.0, 0.6)])
def test_decay_time_against_log_estimate_within_a_band(k, p, alpha):
    A, c = K.chain_matrix(k, p, alpha, 0)
    est = math.log(0.01) / math.log(K.spectral_radius(A))
    measured = K.decay_time(A, c).time
    assert abs(measured - est) <= max(6.0, 0.2 * est)


def test_best_spectral_radius_of_the_fixed_delay_is_the_double_root():
    """Bei fester Lieferzeit L ist das Minimum über α die Doppelwurzel: ρ = (L - 1)/L bei α = (1/L)((L - 1)/L)^(L - 1) (nachgerechnet am Polynom z^L - z^(L-1) + α)."""
    alphas = np.arange(0.01, 2.0001, 0.01)
    for L in (3, 5, 8):
        best, a_best = K.best_rho(L, 1.0, 0, alphas)
        a_star = (1.0 / L) * ((L - 1.0) / L) ** (L - 1)
        assert best == pytest.approx((L - 1.0) / L, abs=2e-4)
        assert abs(a_best - a_star) < 2e-3
        roots = np.roots([1.0, -1.0] + [0.0] * (L - 2) + [a_star])
        assert max(abs(roots)) == pytest.approx((L - 1.0) / L, abs=1e-3)
