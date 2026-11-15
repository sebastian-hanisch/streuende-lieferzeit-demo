"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, metamorph): Aussagen, die für das Modell gelten müssen, ohne das richtige Ergebnis zu kennen.

Jede Eigenschaft steht mit ihrer Voraussetzung da: linear, ohne Klemmung, Erwartungswert-Modell bzw. Einzelbestellung wie angegeben. Zweierpotenz-Fälle (p = 1/2, 1/4)
werden mit Rundungsrauschen (p mal 1 ± 1e-9) wiederholt: das Ergebnis darf sich nicht ändern."""

import math

import numpy as np
import pytest

import slz_dynamics as K
import slz_evaluation as E


def test_mean_of_the_chain_distribution_is_k_over_p():
    for k, p in ((1, 0.25), (2, 0.5), (3, 0.75), (4, 0.3), (2, 2 / 3)):
        assert K.mean_var(K.erlang_pmf(k, p, 800))[0] == pytest.approx(k / p, rel=1e-9)


def test_limit_grows_with_spread_in_the_measured_family():
    """Bei festem Mittel steigt α_krit, wenn die Stufenzahl k von τ (fest) auf 1 fällt, in jedem gemessenen Fall (τ = 6, 12, 24, 60; k Teiler von τ). Gilt nur für diese Familie."""
    for tau in (6, 12, 24, 60):
        ks = [k for k in range(tau, 0, -1) if tau % k == 0]
        vals = [K.critical_alpha_chain(k, k / tau) for k in ks]
        assert all(b > a for a, b in zip(vals, vals[1:])), (tau, vals)


def test_limit_of_the_fixed_delay_is_the_formula_for_every_length():
    for L in range(1, 41):
        assert K.critical_alpha_chain(L, 1.0) == pytest.approx(2 * math.sin(math.pi / (4 * L - 2)), rel=1e-9)


def test_scale_invariance_of_the_second_moment_in_the_demand():
    """Bei σ = 0 wächst die Varianz des Einzelbestellungs-Modells mit D²: var(i)/D² hängt nicht an D (exakt, k = 2, p = 2/3, α = 0,4)."""
    vals = [K.ms_stationary(2, 2 / 3, 0.4, 0, D, 0.0)[0] / D ** 2 for D in (1.0, 10.0, 100.0)]
    assert max(vals) - min(vals) < 1e-9 * max(vals)


def test_dyadic_p_gives_the_same_results_with_rounding_noise():
    """p = 1/4 (k = 3, τ = 12) und p = 1/2 (k = 6, τ = 12): Grenze, Spektralradius und Abklingzeit ändern sich bei p·(1 ± 1e-9) nicht."""
    for k, tau, alpha in ((3, 12, 0.15), (6, 12, 0.1), (2, 4, 0.5 * 1.04)):
        p = k / tau
        base_limit = K.critical_alpha_chain(k, p)
        base_rho = K.spectral_radius(K.chain_matrix(k, p, alpha, 0)[0])
        for eps in (-1e-9, 1e-9):
            q = p * (1 + eps)
            assert K.critical_alpha_chain(k, q) == pytest.approx(base_limit, rel=1e-6)
            assert K.spectral_radius(K.chain_matrix(k, q, alpha, 0)[0]) == pytest.approx(base_rho, rel=1e-6)
    A, c = K.chain_matrix(3, 0.25, 0.15, 0)
    A2, c2 = K.chain_matrix(3, 0.25 * (1 + 1e-9), 0.15, 0)
    assert abs(K.decay_time(A, c).time - K.decay_time(A2, c2).time) <= 1


def test_beta_one_spectral_radius_is_max_of_one_minus_alpha_and_q():
    """β = 1: ρ = max(|1 − α|, q) für jede Kette (Eigenwerte mit mehrfacher Wurzel q sind schlecht konditioniert: Toleranz 5e-3)."""
    for k, p in ((1, 0.25), (1, 1 / 3), (2, 0.4), (2, 2 / 3), (3, 0.3), (5, 0.8)):
        for alpha in (0.2, 0.9, 1.3, 1.7):
            r = K.spectral_radius(K.chain_matrix(k, p, alpha, 1)[0])
            assert r == pytest.approx(max(abs(1 - alpha), 1 - p), abs=5e-3)


def test_limit_times_tau_approaches_the_continuous_value_monotonically():
    """α_krit · τ nähert sich bei wachsendem τ dem stetigen Wert (k fest): die Folge der Abstände sinkt monoton (k = 2, 3, 4)."""
    for k in (2, 3, 4):
        cont = K.critical_continuous(k)
        gaps = [abs(K.critical_alpha_chain(k, k / tau) * tau - cont) for tau in (12, 60, 240, 1200)]
        assert all(b < a for a, b in zip(gaps, gaps[1:])) and gaps[-1] < 0.01


def test_same_seed_same_batch_different_seed_different_batch():
    a = E.random_batch(35, 4, 12)
    b = E.random_batch(35, 4, 12)
    c = E.random_batch(36, 4, 12)
    assert a == b and a != c


def test_variance_estimates_of_independent_samples_agree_with_the_exact_value():
    """Mehr Läufe ändern die Schätzung nur im Rahmen der Stichprobe: zwei halb so große Stichproben mit anderem Zufalls-Seed liegen beide innerhalb 8 % des exakten Werts."""
    exact = K.ms_stationary(2, 2 / 3, 0.4, 0, 10.0, 1.0)[0]
    for seed in (35, 36):
        I, _ = K.simulate_individual(2, 2 / 3, 0.4, 0, 10.0, 1.0, 400, 1800, seed)
        assert float(np.var(I[400:1800])) == pytest.approx(exact, rel=0.08)
