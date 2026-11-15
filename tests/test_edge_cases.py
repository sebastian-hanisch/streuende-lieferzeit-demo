"""Randfälle (DEMO-PLAYBOOK §12): kleinstes τ, k = 1, k = τ, k über τ, Reglergrenzen, σ = 0, kleinste Läufe-Zahl, Abstand über der zulässigen Breite, Randwerte der Verdikte."""

import math

import numpy as np
import pytest

import slz_constants as C
import slz_dynamics as K
import slz_evaluation as ev


def test_smallest_tau_two_all_stage_counts():
    """τ = 2: k = 1 (p = 1/2: geometrisch) hat die Grenze 6 = 4τ − 2; k ≥ 2 wird auf τ begrenzt, das ist die feste Lieferzeit 2 mit α_krit = 2 sin(π/6) = 1."""
    assert K.stages(2, 1) == (1, 0.5) and K.stages(2, 2) == (2, 1.0) and K.stages(2, 12) == (2, 1.0)
    assert K.critical_alpha_chain(1, 0.5) == pytest.approx(6.0, rel=1e-9)
    assert K.critical_alpha_chain(2, 1.0) == pytest.approx(1.0, rel=1e-9)
    a = ev.hand_analysis(2, 12, 0.9, 0)
    assert a["k"] == 2 and a["p"] == 1.0 and a["chosen"]["alpha_krit"] == pytest.approx(a["fixed"]["alpha_krit"], rel=1e-12)


def test_k_equal_tau_and_k_above_tau_are_the_same_fixed_delay():
    """Der k-Regler endet bei 12: für τ ≤ 12 ist „k = 12“ dieselbe Rechnung wie k = τ (fest); bei τ = 24 ist die feste Lieferzeit nur als Vergleich wählbar."""
    for tau in (2, 3, 12):
        a, b = ev.hand_analysis(tau, tau, 0.1, 0), ev.hand_analysis(tau, C.K_MAX, 0.1, 0)
        assert a["k"] == b["k"] == tau and a["chosen"]["rho"] == b["chosen"]["rho"] == a["fixed"]["rho"]
    assert ev.hand_analysis(24, C.K_MAX, 0.05, 0)["k"] == 12


def test_every_tau_and_alpha_edge_gives_finite_analyses():
    for tau in (C.TAU_MIN, 3, C.TAU_MAX):
        for alpha in (C.ALPHA_MIN, C.ALPHA_MAX):
            for beta in C.BETA_OPTIONS:
                a = ev.hand_analysis(tau, C.K_MAX, alpha, beta)
                assert math.isfinite(a["fixed"]["rho"]) and math.isfinite(a["chosen"]["rho"])
                kind, text = ev.verdict_hand(a)
                assert kind in ("success", "info", "warning") and text
                nz = ev.noise_analysis(tau, 1, alpha, beta, 35)
                assert ev.verdict_noise(nz)[1]


def test_alpha_at_the_limit_of_the_fixed_delay_is_unstable_or_marginal_never_stable_by_a_wide_margin():
    for tau in (2, 5, 12):
        a = K.critical_alpha_fixed(tau)
        rho = K.spectral_radius(K.chain_matrix(tau, 1.0, a, 0)[0])
        assert abs(rho - 1) < 1e-8                                           # auf der Grenze selbst: Band, nie ==


def test_limit_map_k1_row_and_ratio_with_fixed():
    lm = ev.limit_map(60)
    rows = {r["k"]: r for r in lm["rows"]}
    assert rows[1]["alpha_krit"] == pytest.approx(238.0) and rows[1]["continuous"] == math.inf and rows[60]["ratio"] == pytest.approx(1.0, rel=1e-9)
    assert all(r["ratio"] > 1 for k, r in rows.items() if k < 60)


def test_spread_is_clamped_to_tau_minus_one_in_the_forms_analysis():
    fa = ev.forms_analysis(2, C.SPREAD_MAX, "spitz", 35)
    assert fa["d"] == 1
    assert fa["rows"][1]["name"] == "Zweipunkt 1 / 3" and fa["rows"][2]["name"] == "gleichverteilt 1 bis 3"
    for r in fa["rows"]:
        assert r["mean"] == pytest.approx(2.0) and math.isfinite(r["alpha_krit"]) and r["ratio"] >= 1 - 1e-9


def test_spread_zero_equals_fixed():
    f = K.two_point_pmf(5, 0)
    assert K.critical_alpha_dist(f) == pytest.approx(K.critical_alpha_fixed(5), rel=1e-9)


def test_single_analysis_at_the_smallest_run_count_with_and_without_noise():
    for sigma in (0.0, 1.0):
        for beta in C.BETA_OPTIONS:
            sg = ev.single_analysis(3, 2, 0.3, beta, 10.0, sigma, C.RUNS_MIN, 35)
            assert sg["sim"]["finite"] and sg["exact"] is not None and np.isfinite(sg["path"]["i_single"]).all()
            assert sg["shock"] is not None
            assert ev.verdict_single(sg)[1]
    sg0 = ev.single_analysis(3, 3, 0.3, 0, 10.0, 0.0, C.RUNS_MIN, 35)                  # fest, konstanter Bedarf: nichts schwankt
    assert sg0["sim"]["var_i"] < 1e-12 and sg0["exact"]["var_i"] < 1e-9


def test_single_analysis_with_unstable_mean_stops_early_and_reports_no_shock():
    sg = ev.single_analysis(3, 3, 0.9, 0, 10.0, 1.0, 100, 35)                           # fest, α über der Grenze 0,618
    assert sg["rho"] > 1 and sg["shock"] is None and sg["sim"]["periods"] == 300 and sg["fluid"] is None
    assert np.isfinite(sg["path"]["i_single"]).all()
    assert ev.verdict_single(sg)[0] == "warning"


def test_single_analysis_for_k_above_the_exact_limit_uses_simulation_only():
    sg = ev.single_analysis(24, 12, 0.05, 0, 10.0, 1.0, C.RUNS_MIN, 35)
    assert not sg["ms_available"] and sg["exact"] is None and sg["alpha_ms"] is None
    assert "k > 8" in ev.verdict_single(sg)[1]


def test_decay_table_for_tau_two_and_for_the_chosen_stage_count():
    assert ev.decay_ks(2, 5) == [1, 2] and ev.decay_ks(12, 5) == [1, 2, 3, 4, 5, 6, 12] and ev.decay_ks(12, 12) == [1, 2, 3, 4, 6, 12]
    dc = ev.decay_analysis(2, 0.4, 1)
    assert [r["label"] for r in dc["rows"]] == ["k = 1", "fest"] and all(math.isfinite(r["rho0"]) for r in dc["rows"])


def test_frozen_json_has_all_means_and_consistent_counts():
    fz = ev.load_frozen()
    assert fz["seed"] == C.FROZEN_SEED and fz["n"] == C.FROZEN_N and set(fz["rows"]) == {str(m) for m in C.FROZEN_MEANS}
    for r in fz["rows"].values():
        assert r["above"] + r["below"] + r["equal"] == C.FROZEN_N and r["returns"] == len(r["return_indices"])
