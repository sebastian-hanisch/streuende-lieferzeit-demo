"""Auswertungen der Abschnitte der App und ihre Verdikte (reine Funktionen: (Art, Text)).

Jede Auswertung ruft nur den Kern `slz_dynamics` auf; die App legt sie hinter `st.cache_data`. Zahlen in den Verdikten kommen aus der Rechnung,
die vorgerechneten Zufallsverteilungen aus `data/slz_frozen.json` (erzeugt von `tools/make_frozen.py`)."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import slz_constants as C
import slz_dynamics as K
import slz_rng as R

ROOT = Path(__file__).resolve().parent
FROZEN_PATH = ROOT / "data" / "slz_frozen.json"


@dataclass(frozen=True)
class Settings:
    tau: int = C.DEFAULT_TAU
    k: int = C.DEFAULT_K
    alpha: float = C.DEFAULT_ALPHA
    beta: int = C.DEFAULT_BETA
    seed: int = C.DEFAULT_SEED
    tau_map: int = C.DEFAULT_TAU_MAP
    spread: int = C.DEFAULT_SPREAD
    random_form: str = C.DEFAULT_RANDOM_FORM
    d: int = C.DEFAULT_D
    sigma: float = C.DEFAULT_SIGMA
    runs: int = C.DEFAULT_RUNS


# --- Anzeige ---------------------------------------------------------------------------------------------------------------------------

def de(x: float, nd: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma (Anzeige); unendlich als „keine Grenze“ zu schreiben ist Sache des Aufrufers."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",")


def fmt_int(n: int) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner (3193 -> 3 193)."""
    return f"{int(n):,}".replace(",", " ")


def pct(x: float, nd: int = 0) -> str:
    return de(100.0 * x, nd) + " %"


def sig(x: float, nd: int = 3) -> str:
    """Zahl mit nd gültigen Stellen und Dezimalkomma (für Größen, die über mehrere Zehnerpotenzen streuen)."""
    if x == 0:
        return "0"
    digits = max(0, nd - 1 - int(math.floor(math.log10(abs(x)))))
    return de(x, digits)


def load_frozen() -> dict:
    with open(FROZEN_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def variance_of_lead_time(k: int, p: float) -> float:
    """Varianz der Lieferzeit der Kette: k q / p^2 (Summe aus k unabhängigen geometrischen Zeiten)."""
    return k * (1.0 - p) / p ** 2


# --- Experiment 1: von Hand ------------------------------------------------------------------------------------------------------------

def _model(name: str, k: int, p: float, alpha: float, beta: int, n_resp: int) -> dict:
    A, c = K.chain_matrix(k, p, alpha, beta)
    eigs = np.linalg.eigvals(A)
    rho = float(np.max(np.abs(eigs)))
    i_resp, u_resp = K.impulse_response(A, c, n_resp)
    return {"name": name, "k": k, "p": p, "rho": rho, "alpha_krit": K.critical_alpha_chain(k, p, beta), "eigs": eigs, "i": i_resp, "u": u_resp}


def hand_analysis(tau: int, k: int, alpha: float, beta: int) -> dict:
    """Drei Lieferzeit-Formen mit gleichem Mittel tau: fest (k = tau), die gewählte Stufenzahl und k = 1 (geometrisch)."""
    k_eff, p = K.stages(tau, k)
    n_resp = int(min(400, max(60, 10 * tau)))
    fixed = _model("fest", tau, 1.0, alpha, beta, n_resp)
    chosen = _model(f"k = {k_eff}", k_eff, p, alpha, beta, n_resp)
    one = _model("k = 1", 1, 1.0 / tau, alpha, beta, n_resp)
    return {
        "tau": tau, "k": k_eff, "p": p, "alpha": alpha, "beta": beta, "var_lead": variance_of_lead_time(k_eff, p),
        "pmf": K.erlang_pmf(k_eff, p, int(4 * tau + 12)), "fixed": fixed, "chosen": chosen, "one": one, "n_resp": n_resp,
    }


def verdict_hand(a: dict) -> tuple[str, str]:
    tau, alpha, beta = a["tau"], a["alpha"], a["beta"]
    rf, rc, r1 = a["fixed"]["rho"], a["chosen"]["rho"], a["one"]["rho"]
    k = a["k"]
    head = f"Mittlere Lieferzeit {tau}, α = {de(alpha)}"
    if k == tau:
        return "info", f"{head}: Mit k = τ ist die Lieferzeit fest; Spektralradius ρ = {de(rf, 4)} ({'instabil' if rf >= 1 else 'stabil'}), k = 1 (geometrisch) ρ = {de(r1, 4)}."
    tail = f"k = 1: ρ = {de(r1, 4)}"
    if beta == 1:
        tail += " (bei β = 1 gilt ρ = max(|1 − α|, q) mit q = 1 − p, für die Erlang-Kette; die Grenze α = 2 gilt für jede Verteilung)"
    if rf >= 1 and rc < 1:
        return "success", (f"{head}: Die feste Lieferzeit ist instabil (ρ = {de(rf, 3)}, der Bestand wächst um {de(100 * (rf - 1), 1)} % je Periode), "
                           f"die streuende Lieferzeit mit k = {k} ist stabil (ρ = {de(rc, 3)}); {tail}.")
    if rf < 1 and rc < 1:
        return "info", f"{head}: Beide stabil, fest ρ = {de(rf, 4)}, k = {k} ρ = {de(rc, 4)}; {tail}."
    if rf >= 1 and rc >= 1:
        return "warning", f"{head}: Beide instabil, fest ρ = {de(rf, 3)}, k = {k} ρ = {de(rc, 3)}; {tail}."
    return "warning", f"{head}: Die streuende Lieferzeit (k = {k}, ρ = {de(rc, 3)}) ist hier instabiler als die feste (ρ = {de(rf, 3)}); {tail}."


# --- Experiment 2: Grenzkarte ----------------------------------------------------------------------------------------------------------

MAP_KS = {12: tuple(range(1, 13)), 60: (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60), 240: (1, 2, 3, 4, 6, 10, 20, 30, 60), 1200: (1, 2, 3, 4, 6, 10, 20, 30, 60)}


def fixed_limit(tau: int) -> float:
    return K.critical_alpha_fixed(tau)


def limit_map(tau: int) -> dict:
    """α_krit (β = 0) der Erlang-Kette mit k Stufen bei gleichem Mittel tau; die feste Lieferzeit aus der Formel 2 sin(π/(4τ − 2))."""
    fixed = fixed_limit(tau)
    rows = []
    for k in MAP_KS[tau]:
        p = k / tau
        a = K.critical_alpha_chain(k, p, 0)
        rows.append({"k": k, "p": p, "alpha_krit": a, "product": a * tau, "ratio": a / fixed,
                     "continuous": K.critical_continuous(k) if k > 1 else math.inf})
    return {"tau": tau, "fixed": fixed, "fixed_product": fixed * tau, "rows": rows, "pi_half": math.pi / 2}


def k3_ratio(tau: int) -> float:
    """α_krit der Kette mit k = 3 geteilt durch α_krit der festen Lieferzeit (gleiches Mittel tau)."""
    return K.critical_alpha_chain(3, 3.0 / tau, 0) / fixed_limit(tau)


def verdict_limit(a: dict) -> tuple[str, str]:
    tau = a["tau"]
    r12, r60 = k3_ratio(12), k3_ratio(60)
    cont = K.critical_continuous(3) / (math.pi / 2)
    k1 = 4 * tau - 2
    return "success", (f"Bei gleichem Mittel liegt die Grenze der Erlang-Ketten mit k unter τ (β = 0, anteilig) über der festen Lieferzeit. Die Verzögerung dritter Ordnung (k = 3) liegt bei τ = 12 um "
                       f"{pct(r12 - 1)}, bei τ = 60 um {pct(r60 - 1)} über der festen Lieferzeit; stetig (τ → ∞) sind es {pct(cont - 1)}. "
                       f"k = 1: α_krit = 4τ − 2 = {k1} (bei τ = {tau}; die Grenze liegt bei z = −1, Periode 2, kein Hopf).")


# --- Experiment 3: andere Formen -------------------------------------------------------------------------------------------------------

def forms_analysis(tau: int, spread: int, random_form: str, seed: int) -> dict:
    """Vier Verteilungen mit Mittel tau (β = 0): fest, Zweipunkt (tau ± d), gleichverteilt (tau ± w), zufällig (Zufalls-Seed)."""
    d = max(0, min(int(spread), tau - 1))
    fixed_a = fixed_limit(tau)
    items = [
        ("fest", K.fixed_pmf(tau)),
        (f"Zweipunkt {tau - d} / {tau + d}", K.two_point_pmf(tau, d)),
        (f"gleichverteilt {tau - d} bis {tau + d}", K.uniform_pmf(tau, d)),
        (f"zufällig {random_form} (Zufalls-Seed {seed})", K.random_pmf(seed, tau, random_form)),
    ]
    rows = []
    for name, f in items:
        mean, var = K.mean_var(f)
        a = K.critical_alpha_dist(f)
        rows.append({"name": name, "pmf": f, "mean": mean, "var": var, "alpha_krit": a, "ratio": a / fixed_a})
    return {"tau": tau, "d": d, "fixed": fixed_a, "rows": rows}


def verdict_forms(fa: dict, frozen: dict) -> tuple[str, str]:
    row = frozen["rows"]["4"]
    n = frozen["n"]
    best = max(fa["rows"][1:], key=lambda r: r["ratio"])
    worst = min(fa["rows"][1:], key=lambda r: r["ratio"])
    return "info", (f"Bei Mittel {fa['tau']} liegt die Grenze der festen Lieferzeit bei {de(fa['fixed'], 3)}; die gewählten Formen liegen zwischen dem {de(worst['ratio'], 2)}fachen "
                    f"({worst['name']}) und dem {de(best['ratio'], 2)}fachen ({best['name']}). Vorgerechnet (Mittel 4, {fmt_int(n)} Zufallsverteilungen, Zufalls-Seed {frozen['seed']}): "
                    f"die Grenze liegt in {fmt_int(row['above'])} von {fmt_int(n)} über der festen (Median {de(row['ratio_median'], 2)}fach), "
                    f"unter ihr in {row['below']}; in {row['returns']} von {fmt_int(n)} kehrt die Stabilität oberhalb der ersten Grenze auf dem α-Raster zurück.")


# --- Experiment 4: Schwankung bei Rauschen (anteilig) ----------------------------------------------------------------------------------

def _noise_row(name: str, k: int, p: float, alpha: float, beta: int) -> dict:
    A, c = K.chain_matrix(k, p, alpha, beta)
    rho = K.spectral_radius(A)
    row = {"name": name, "k": k, "p": p, "rho": rho, "stable": rho < 1 - 1e-9, "var_u": None, "var_i": None}
    if row["stable"]:
        row["var_u"], row["var_i"] = K.variances(A, c)
    return row


def noise_analysis(tau: int, k: int, alpha: float, beta: int, seed: int) -> dict:
    """Bestell- und Bestandsvarianz bei weißem Rauschen (Varianz 1) für k = 1, die gewählte Stufenzahl und fest (Lyapunov-Gleichung);
    für die gewählte Stufenzahl dazu die Simulation (Zufalls-Seed `seed`)."""
    k_eff, p = K.stages(tau, k)
    rows = [_noise_row("k = 1", 1, 1.0 / tau, alpha, beta)]
    if k_eff not in (1, tau):
        rows.append(_noise_row(f"k = {k_eff}", k_eff, p, alpha, beta))
    rows.append(_noise_row("fest", tau, 1.0, alpha, beta))
    chosen = next(r for r in rows if r["name"] == (f"k = {k_eff}" if k_eff not in (1, tau) else ("k = 1" if k_eff == 1 else "fest")))
    sim = None
    if chosen["stable"] and chosen["rho"] < 0.995:
        vi, vu = K.simulate_fluid(chosen["k"], chosen["p"], alpha, beta, C.NOISE_RUNS, C.NOISE_PERIODS, C.NOISE_BURN, seed)
        sim = {"var_i": vi, "var_u": vu}
    return {"tau": tau, "k": k_eff, "alpha": alpha, "beta": beta, "rows": rows, "chosen": chosen, "sim": sim}


def verdict_noise(a: dict) -> tuple[str, str]:
    rows = {r["name"]: r for r in a["rows"]}
    fixed, chosen = rows["fest"], a["chosen"]
    head = f"τ = {a['tau']}, α = {de(a['alpha'])}, β = {a['beta']}"
    if not chosen["stable"]:
        return "warning", f"{head}: Die gewählte Lieferzeit (k = {a['k']}) ist hier instabil (ρ = {de(chosen['rho'], 3)}), es gibt keine stationäre Varianz."
    parts = [f"{r['name']}: {de(r['var_i'], 2)}" for r in a["rows"] if r["stable"]]
    text = f"{head}: Bestandsvarianz var(i)/var(e) " + ", ".join(parts) + "."
    if not fixed["stable"]:
        return "success", text + " Die feste Lieferzeit ist hier instabil, die streuende hat eine endliche Varianz."
    if a["beta"] == 1:
        return "info", text + f" Die Bestellvarianz var(u)/var(e) = α/(2 − α) = {de(a['alpha'] / (2 - a['alpha']), 3)} gilt für jede Verteilung; nur die Bestandsvarianz hängt an ihr."
    if chosen["var_i"] < fixed["var_i"]:
        return "info", text + f" Bei gleichem α ist die Schwankung der streuenden Lieferzeit kleiner (Bestellvarianz {de(chosen['var_u'], 2)} gegen {de(fixed['var_u'], 2)} bei fester Lieferzeit)."
    return "warning", text + f" Hier ist die Schwankung der streuenden Lieferzeit nicht kleiner (Bestellvarianz {de(chosen['var_u'], 2)} gegen {de(fixed['var_u'], 2)})."


# --- Experiment 5: Einzelbestellung gegen anteilig -------------------------------------------------------------------------------------

MS_K_MAX = 8      # das Moment-System hat (k+1)^2 + k Unbekannte; darüber wäre die Grenzensuche in der App zu langsam


def single_analysis(tau: int, k: int, alpha: float, beta: int, D: float, sigma: float, runs: int, seed: int) -> dict:
    k_eff, p = K.stages(tau, k)
    A, c = K.chain_matrix(k_eff, p, alpha, beta)
    rho = K.spectral_radius(A)
    out = {"tau": tau, "k": k_eff, "p": p, "alpha": alpha, "beta": beta, "D": D, "sigma": sigma, "runs": runs, "rho": rho,
           "alpha_fixed": fixed_limit(tau) if beta == 0 else 2.0, "alpha_mean": K.critical_alpha_chain(k_eff, p, beta), "alpha_ms": None,
           "exact": None, "fluid": None, "sim": None, "ms_available": k_eff <= MS_K_MAX}
    if out["ms_available"]:
        out["alpha_ms"] = K.ms_critical(k_eff, p, beta)
        ex = K.ms_stationary(k_eff, p, alpha, beta, float(D), float(sigma))
        out["exact"] = None if ex is None else {"var_i": ex[0], "var_u": ex[1]}
    if rho < 1 - 1e-9:
        vu, vi = K.variances(A, c)
        out["fluid"] = {"var_i": vi * sigma ** 2, "var_u": vu * sigma ** 2}
    stable = rho < 1 - 1e-9
    periods = C.SIM_PERIODS if stable else 300           # instabiler Mittelwert: kürzer rechnen, die Werte wachsen sonst über den Zahlenbereich
    burn = min(C.SIM_BURN, periods // 3)
    with np.errstate(all="ignore"):
        I, U = K.simulate_individual(k_eff, p, alpha, beta, float(D), float(sigma), runs, periods, seed)
        finite = bool(np.isfinite(I[burn:]).all() and float(np.abs(I[burn:]).max()) < 1e12)
        out["sim"] = {"var_i": float(np.var(I[burn:periods])), "var_u": float(np.var(U[burn:periods])), "finite": finite, "growth": _growth_ratio(I), "periods": periods, "burn": burn}
    noise = R.normal_array(R.child_seed(seed, 1), periods * runs).reshape(periods, runs)[:, 0] if sigma > 0 else np.zeros(periods)
    n_path = min(240, periods)
    i_f, u_f = K.fluid_path(k_eff, p, alpha, beta, sigma * noise[:n_path])
    lim = 1e9
    out["path"] = {"n": n_path, "i_fluid": np.clip(np.nan_to_num(i_f), -lim, lim), "u_fluid": np.clip(np.nan_to_num(u_f), -lim, lim),
                   "i_single": np.clip(np.nan_to_num(I[1:n_path + 1, 0]), -lim, lim), "u_single": np.clip(np.nan_to_num(U[:n_path, 0]), -lim, lim)}
    shock = None
    if rho < 1 - 1e-9:
        Is, _ = K.simulate_individual(k_eff, p, alpha, beta, 0.0, 0.0, C.SHOCK_RUNS, C.SHOCK_PERIODS, seed + 1, shock=1.0)
        mean = Is[1:].mean(axis=1)
        se = Is[1:].std(axis=1, ddof=1) / math.sqrt(C.SHOCK_RUNS)
        i_resp, _ = K.impulse_response(A, c, C.SHOCK_PERIODS)
        shock = {"mean": mean, "se": se, "fluid": i_resp}
    out["shock"] = shock
    return out


def _growth_ratio(I: np.ndarray) -> float:
    """Verhältnis der Varianz über die Läufe in der letzten Viertelzeit zu der in der zweiten Viertelzeit (über 1: Varianz wächst)."""
    T = I.shape[0]
    q2 = np.var(I[T // 4: T // 2], axis=1).mean()
    q4 = np.var(I[3 * T // 4:], axis=1).mean()
    return float(q4 / q2) if q2 > 0 else math.nan


def verdict_single(a: dict) -> tuple[str, str]:
    alpha = a["alpha"]
    head = f"τ = {a['tau']}, k = {a['k']}, α = {de(alpha)}, β = {a['beta']}"
    if not a["ms_available"]:
        return "info", f"{head}: Für k > {MS_K_MAX} wird das zweite Moment in der App nicht gerechnet; die Simulation steht unten."
    am, a2 = a["alpha_mean"], a["alpha_ms"]
    same = a2 >= am * (1 - 1e-6)
    lim = f"Mittelwert stabil bis α = {sig(am)}, zweites Moment {'ebenfalls' if same else 'nur'} bis α = {sig(a2)}"
    if alpha < a2 and a["exact"] is not None:
        ex, fl = a["exact"], a["fluid"]
        return "info", (f"{head}: {lim}; α liegt darunter. Bestandsvarianz bei Bedarf {de(a['D'], 0)} und Rauschen {de(a['sigma'], 1)}: "
                        f"Einzelbestellung {sig(ex['var_i'])} gegen {sig(fl['var_i'])} im anteiligen Modell (beide exakt gerechnet).")
    if alpha < am:
        return "warning", f"{head}: {lim}. α liegt zwischen den Grenzen: Der Mittelwert ist stabil, das zweite Moment divergiert, die Varianz wächst über die Zeit."
    return "warning", f"{head}: {lim}. α liegt über beiden Grenzen: auch der Mittelwert ist instabil."


# --- Experiment 6: stabil heißt nicht schnell ------------------------------------------------------------------------------------------

def decay_ks(tau: int, k: int) -> list[int]:
    """Stufenzahlen der Tabelle: 1, 2, 3, 4, 6 (soweit unter tau), die gewählte, und fest (k = tau)."""
    base = {x for x in (1, 2, 3, 4, 6) if x < tau}
    k_eff, _ = K.stages(tau, k)
    if k_eff < tau:
        base.add(k_eff)
    return sorted(base) + [tau]


def decay_analysis(tau: int, alpha: float, k: int) -> dict:
    alphas = np.arange(0.01, 2.0001, 0.01)
    rows = []
    for kk in decay_ks(tau, k):
        p = kk / tau
        row = {"k": kk, "p": p, "label": "fest" if kk == tau else f"k = {kk}"}
        for beta in (0, 1):
            A, c = K.chain_matrix(kk, p, alpha, beta)
            rho = K.spectral_radius(A)
            dec = K.decay_time(A, c, C.DECAY_FRAC) if rho < 1 - 1e-9 else None
            row[f"rho{beta}"] = rho
            row[f"decay{beta}"] = None if dec is None else dec.time
            row[f"decay{beta}_peak_i"] = None if dec is None else dec.peak_i
        best, a_best = K.best_rho(kk, p, 0, alphas)
        row["best_rho"], row["best_alpha"] = best, a_best
        row["curve"] = K.rho_curve(kk, p, 0, alphas)
        rows.append(row)
    return {"tau": tau, "alpha": alpha, "alphas": alphas, "rows": rows}


def verdict_decay(a: dict, beta: int = 0) -> tuple[str, str]:
    """Verdikt zu Experiment 5 für das gewählte β; „langsam“ heißt: eine stabile Form klingt nicht ab oder braucht über das Zehnfache der schnellsten stabilen Form."""
    rows = a["rows"]
    rk, dk = f"rho{beta}", f"decay{beta}"
    stable = [r for r in rows if r[rk] < 1 - 1e-9]
    unstable = [r for r in rows if r[rk] >= 1 - 1e-9]
    head = f"τ = {a['tau']}, α = {de(a['alpha'])}, β = {beta}"
    if not stable:
        return "warning", f"{head}: Alle gerechneten Formen sind instabil."
    parts = [f"{r['label']}: {fmt_int(r[dk])} Perioden (ρ = {de(r[rk], 4)})" if r[dk] is not None else f"{r['label']}: nicht abgeklungen (ρ = {de(r[rk], 5)})" for r in stable]
    text = f"{head}: Abklingzeit auf 1 % nach einem Bedarfsstoß, " + "; ".join(parts) + "."
    if unstable:
        text += " Instabil: " + ", ".join(f"{r['label']} (ρ = {de(r[rk], 3)})" for r in unstable) + "."
    if beta == 0:
        best_fixed = rows[-1]
        smallest = min(rows, key=lambda r: r["best_rho"])
        if smallest is best_fixed:
            text += f" Kleinster Spektralradius über α (Raster 0,01 bis 2): fest mit {de(best_fixed['best_rho'], 4)}, der kleinste aller gerechneten Formen."
        else:
            text += f" Kleinster Spektralradius über α (Raster 0,01 bis 2): {smallest['label']} mit {de(smallest['best_rho'], 4)} (fest: {de(best_fixed['best_rho'], 4)})."
    times = [r[dk] for r in stable if r[dk] is not None]
    slow = len(times) < len(stable) or (len(times) > 1 and max(times) > 10 * min(times))
    return ("warning" if slow else "info"), text


# --- vorgerechnete Reihe: Zufallsverteilungen --------------------------------------------------------------------------------------------

RANDOM_FORM_NAMES = tuple(K.RANDOM_FORMS)


def frozen_pmf(seed: int, mean: int, index: int) -> np.ndarray:
    """Die index-te Zufallsverteilung der vorgerechneten Reihe zum Mittel `mean` (Form reihum: glatt, gemischt, spitz; eigener Zufalls-Seed je Index)."""
    form = RANDOM_FORM_NAMES[index % 3]
    return K.random_pmf(R.child_seed(R.child_seed(seed, mean), index), mean, form)


def random_batch(seed: int, mean: int, n: int) -> dict:
    """Grenze (β = 0) von n Zufallsverteilungen mit Mittel `mean` gegen die feste Lieferzeit, mit der Zählung der Wiederkehr der Stabilität (α-Raster 0,01 bis 2,6)."""
    fixed = fixed_limit(mean)
    ratios, returns, no_cross = [], [], 0
    by_form = {name: [] for name in RANDOM_FORM_NAMES}
    for i in range(n):
        first, back = K.stability_returns(frozen_pmf(seed, mean, i))
        ratio = first / fixed
        ratios.append(ratio)
        by_form[RANDOM_FORM_NAMES[i % 3]].append(ratio)
        no_cross += not math.isfinite(first)
        if back > 0:
            returns.append(i)
    arr = np.array(ratios)
    finite = arr[np.isfinite(arr)]
    return {
        "mean": mean, "n": n, "above": int(np.sum(arr > 1 + 1e-6)), "below": int(np.sum(arr < 1 - 1e-6)), "equal": int(np.sum(np.abs(arr - 1) <= 1e-6)),
        "ratio_min": float(arr.min()), "ratio_median": float(np.median(arr)), "ratio_max": float(finite.max()) if len(finite) else math.inf,
        "no_crossing": no_cross, "returns": len(returns), "return_indices": returns,
        "median_by_form": {name: float(np.median(v)) for name, v in by_form.items()},
    }
