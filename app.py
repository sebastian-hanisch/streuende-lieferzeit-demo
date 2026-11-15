"""Streuende Lieferzeit - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein lineares Lager mit Bestellregel und Lieferzeit-Verteilung gleichen Mittels: Stabilitätsgrenze, Schwankung, Abklingzeit, Einzelbestellung.
Kern in slz_dynamics.py, Auswertungen und Verdikte in slz_evaluation.py; Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import slz_constants as C
import slz_dynamics as K
import slz_evaluation as E
from slz_layout import dunkles_design, kopf_css
from slz_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from slz_visualization import (
    build_eigs,
    build_forms_limits,
    build_forms_pmf,
    build_limit_axis,
    build_limit_map,
    build_noise,
    build_paths,
    build_pmf,
    build_response,
    build_rho_curves,
    build_shock,
)

st.set_page_config(page_title="Streuende Lieferzeit – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _hand(tau, k, alpha, beta):
    return E.hand_analysis(tau, k, alpha, beta)


@st.cache_data(show_spinner=False)
def _limit_map(tau):
    return E.limit_map(tau)


@st.cache_data(show_spinner=False)
def _k3_ratios():
    return {t: E.k3_ratio(t) for t in (6, 12, 24, 60, 120)}


@st.cache_data(show_spinner=False)
def _forms(tau, spread, random_form, seed):
    return E.forms_analysis(tau, spread, random_form, seed)


@st.cache_data(show_spinner=False)
def _noise(tau, k, alpha, beta, seed):
    return E.noise_analysis(tau, k, alpha, beta, seed)


@st.cache_data(show_spinner=False)
def _single(tau, k, alpha, beta, d, sigma, runs, seed):
    return E.single_analysis(tau, k, alpha, beta, d, sigma, runs, seed)


@st.cache_data(show_spinner=False)
def _decay(tau, alpha, k):
    return E.decay_analysis(tau, alpha, k)


@st.cache_data(show_spinner=False)
def _frozen():
    return E.load_frozen()


def _show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


def _num(x, nd=3):
    return "keine Grenze" if not math.isfinite(x) else E.de(x, nd)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in slz_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📦 Streuende Lieferzeit")
st.markdown(
    """
Ein Lager bestellt nach der Regel **u = −α · (Bestand − Soll)** nach, aber die Bestellung kommt nicht nach genau τ Perioden, sondern nach einer **streuenden Lieferzeit** mit demselben Mittel τ.
Die Demo fragt, was sich an der **Stabilitätsgrenze**, an der **Schwankung** und am **Abklingen** ändert, wenn nur die Form der Lieferzeit wechselt, und was passiert, wenn **jede Bestellung ihre eigene Lieferzeit zieht**.
Alles gilt für dieses lineare Modell ohne Klemmung.

Weiter unten: fünf Experimente (🔬), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Modell", expanded=True):
    st.markdown(
        """
1. **Regel:** In jeder Periode bestellt das Lager u = −α · (i + β · Leitung). i ist der Bestand minus Soll, die Leitung ist, was bestellt und noch nicht angekommen ist; **β = 0** rechnet sie nicht mit, **β = 1** rechnet sie voll mit.
2. **Lieferzeit:** Von jeder Bestellung kommt der Anteil f_l nach l Perioden an (**anteilig**, Erwartungswert-Modell, linear). Die Verteilung ist die Summe aus **k geometrischen Zeiten** (mindestens k Perioden, Mittel τ = k/p); **k = τ** ist die feste Lieferzeit (p = 1).
3. **Die Frage:** Bei fester Lieferzeit schwingt der Bestand ab **α_krit = 2·sin(π/(4τ − 2))** auf. Streut die Lieferzeit bei gleichem Mittel, verschiebt sich diese Grenze, und andere Größen reagieren anders: das zeigen die Experimente.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Lager und Lieferzeit**")
    tau = st.slider("Mittlere Lieferzeit τ (Perioden)", *bounds("tau_slider"), key="tau_slider", help="Mittel der Lieferzeit; die feste Lieferzeit ist genau τ Perioden.")
    k = st.slider("Stufen k der Lieferzeit", *bounds("k_slider"), key="k_slider",
                  help="Die Lieferzeit ist die Summe aus k geometrischen Zeiten. k = 1: breit gestreut, k = τ (oder größer): fest. Mehr als τ Stufen werden auf τ begrenzt.")
    alpha = st.slider("Anpassung α", *bounds("alpha_slider"), step=C.ALPHA_STEP, key="alpha_slider", help="Anteil des Bestandsfehlers, der je Periode nachbestellt wird.")
    beta = st.radio("Leitung mitrechnen (β)", C.BETA_OPTIONS, key="beta_radio", horizontal=True,
                    format_func=lambda b: "β = 0: nicht" if b == 0 else "β = 1: voll", help="β = 1 führt Bestand plus Leitung (die Bestandsposition) zurück.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Gilt für die Simulationen und die zufällige Verteilung.")
    st.button("🎲 Neuen Zufalls-Seed erzeugen", width="stretch", on_click=randomize_seed)

st.caption("Die Regler für einzelne Experimente stehen bei dem Experiment; die Seitenleiste enthält τ, k, α, β und den Zufalls-Seed.")

# --- Ergebnis: von Hand -----------------------------------------------------------------------------------------------------------------

a1 = _hand(int(tau), int(k), float(alpha), int(beta))
st.markdown("## 🎯 Ergebnis: dieselbe Regel, drei Formen der Lieferzeit")
st.caption(f"Mittlere Lieferzeit {a1['tau']}, gewählte Form: {a1['k']} Stufen mit Fortschrittswahrscheinlichkeit p = {E.de(a1['p'], 3)}, Varianz der Lieferzeit {E.de(a1['var_lead'], 2)} "
           f"(fest: 0). Spektralradius ρ: unter 1 stabil, über 1 wächst die Schwingung.")
m1, m2, m3 = st.columns(3)
fx, ch, one = a1["fixed"], a1["chosen"], a1["one"]
m1.metric("Spektralradius, feste Lieferzeit", E.de(fx["rho"], 4), delta="instabil" if fx["rho"] >= 1 else "stabil", delta_color="inverse" if fx["rho"] >= 1 else "off")
m2.metric(f"Spektralradius, {ch['name']}", E.de(ch["rho"], 4), delta=f"{E.de(ch['rho'] - fx['rho'], 4, sign=True)} gegen fest", delta_color="inverse")
m3.metric("Spektralradius, k = 1 (geometrisch)", E.de(one["rho"], 4), delta=f"{E.de(one['rho'] - fx['rho'], 4, sign=True)} gegen fest", delta_color="inverse")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Verteilung der Lieferzeit**")
    st.plotly_chart(build_pmf(a1), width="stretch", key="chart_pmf")
with c2:
    st.markdown("**Bestand nach einem Bedarfsstoß**")
    st.plotly_chart(build_response(a1), width="stretch", key="chart_response")
c3, c4 = st.columns([1, 1])
with c3:
    st.markdown("**Eigenwerte in der komplexen Ebene**")
    st.plotly_chart(build_eigs(a1), width="stretch", key="chart_eigs")
with c4:
    st.markdown("**Grenze α_krit bei diesem τ (β = " + str(int(beta)) + ")**")
    st.markdown(
        f"- fest: **{_num(fx['alpha_krit'])}**\n- {ch['name']}: **{_num(ch['alpha_krit'])}**\n- k = 1: **{_num(one['alpha_krit'])}**"
        + ("\n\nBei β = 1 liegt die Grenze für jede Verteilung bei α = 2 (siehe Formelteil)." if int(beta) == 1 else
           f"\n\nk = 1: α_krit = 4τ − 2 = {4 * a1['tau'] - 2} (Wurzel z = −1, Periode 2).")
    )
    st.caption("Die Grenze ist die kleinste positive Stelle, an der eine Wurzel des charakteristischen Polynoms den Einheitskreis trifft (Wurzelortskurve, unabhängig von den Eigenwerten gerechnet).")

st.markdown("## ❓ Kernfrage: Hält die streuende Lieferzeit stabil, was die feste nicht hält?")
_show(E.verdict_hand(a1))

st.markdown("---")

# --- Experiment: Grenzkarte -------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie hoch liegt die Grenze bei gleichem Mittel?")
st.caption("α_krit·τ über die Stufenzahl k (β = 0), mit der festen Lieferzeit und dem stetigen Grenzfall. k = 1 liegt mit α_krit = 4τ − 2 außerhalb des Bildes.")
tau_map = st.select_slider("Mittlere Lieferzeit τ der Karte", options=list(C.TAU_MAP_OPTIONS), key="tau_map_select")
lm = _limit_map(int(tau_map))
st.plotly_chart(build_limit_map(lm), width="stretch", key="chart_limit_map")
df = pd.DataFrame([{
    "k": r["k"], "p = k/τ": E.de(r["p"], 4), "α_krit": E.de(r["alpha_krit"], 5 if r["alpha_krit"] < 1 else 3), "α_krit · τ": E.de(r["product"], 4 if r["product"] < 10 else 1),
    "Verhältnis zu fest": E.de(r["ratio"], 3 if r["ratio"] < 10 else 1), "stetig (τ → ∞)": "keine Grenze" if not math.isfinite(r["continuous"]) else E.de(r["continuous"], 4),
} for r in lm["rows"]])
st.dataframe(df, hide_index=True, width="stretch")
st.caption(f"Feste Lieferzeit {lm['tau']}: α_krit = {E.de(lm['fixed'], 5)} (α_krit·τ = {E.de(lm['fixed_product'], 4)}); stetig π/2 = {E.de(math.pi / 2, 4)}.")
ratios = _k3_ratios()
st.markdown(
    "**Die Verzögerung dritter Ordnung (k = 3) gegen die feste Lieferzeit, Verhältnis der Grenzen bei gleichem Mittel:** "
    + ", ".join(f"τ = {t}: {E.de(r, 3)}" for t, r in ratios.items()) + f"; stetig {E.de(K.critical_continuous(3) / (math.pi / 2), 3)} (= (8/3)/(π/2), exakt aus dem Hurwitz-Kriterium)."
)
st.caption("DELAY3 in Vensim ist eine Verzögerung dritter Ordnung (Dokumentation von der unabhängigen Prüfung gelesen); ob sie die „Standardverzögerung“ der Systemdynamik-Software ist, ist nicht belegt. Die Dokumentation vergleicht keine Grenzen, die Zahlen sind diese Messung. Die „+70 %“ gelten nur im stetigen Grenzfall.")
_show(E.verdict_limit(lm))

st.markdown("---")

# --- Experiment: andere Formen ----------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Gilt „Streuung hebt die Grenze“ für jede Form?")
st.caption("Vier Verteilungen mit gleichem Mittel τ (Seitenleiste), β = 0: fest, Zweipunkt (τ ± d), gleichverteilt (τ ± w) und eine zufällige Verteilung (Zufalls-Seed der Seitenleiste).")
f1, f2 = st.columns(2)
spread = f1.slider("Abstand d (Zweipunkt) und Halbbreite w (gleichverteilt)", *bounds("dist_spread_slider"), key="dist_spread_slider",
                   help="Zweipunkt: Lieferzeit τ − d oder τ + d mit je 50 %; gleichverteilt auf τ − w bis τ + w. Auf höchstens τ − 1 begrenzt.")
rform = f2.selectbox("Form der zufälligen Verteilung", C.RANDOM_FORM_OPTIONS, key="random_form_select", help="glatt: gleichmäßigere Gewichte, spitz: wenige große Gewichte.")
fa = _forms(int(tau), int(spread), str(rform), int(seed))
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(build_forms_pmf(fa), width="stretch", key="chart_forms_pmf")
with g2:
    st.plotly_chart(build_forms_limits(fa, float(alpha)), width="stretch", key="chart_forms_limits")
dff = pd.DataFrame([{
    "Verteilung": r["name"], "Mittel": E.de(r["mean"], 3), "Varianz": E.de(r["var"], 2), "α_krit": E.de(r["alpha_krit"], 4), "Verhältnis zu fest": E.de(r["ratio"], 3),
    "bei gewähltem α": "stabil" if float(alpha) < r["alpha_krit"] else "instabil",
} for r in fa["rows"]])
st.dataframe(dff, hide_index=True, width="stretch")
frozen = _frozen()
st.markdown("**Vorgerechnet: viele Zufallsverteilungen** (`data/slz_frozen.json`, erzeugt von `tools/make_frozen.py`, Zufalls-Seed " + str(frozen["seed"]) + ", je " + E.fmt_int(frozen["n"]) + " Verteilungen auf 1 bis τ + 4):")
dfr = pd.DataFrame([{
    "Mittel": m, "Grenze über fest": f"{E.fmt_int(r['above'])} von {E.fmt_int(r['n'])}", "unter fest": r["below"], "gleich fest (±10⁻⁶)": r["equal"],
    "Median des Verhältnisses": E.de(r["ratio_median"], 3), "Stabilität kehrt zurück": r["returns"],
} for m, r in ((int(m), r) for m, r in frozen["rows"].items())])
st.dataframe(dfr, hide_index=True, width="stretch")
st.caption("Die Familie der zufälligen Verteilungen ist eigene Wahl (Exponentialgewichte, auf das Mittel geneigt); Aussagen gelten für diese Familie, nicht für alle Verteilungen. Rückkehr: Stabilität oberhalb der ersten Grenze auf dem α-Raster 0,01 bis 2,6, nicht näher untersucht.")
_show(E.verdict_forms(fa, frozen))

st.markdown("---")

# --- Experiment: Schwankung -------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wird auch die Schwankung kleiner? (weißes Rauschen im Bedarf, anteilig)")
st.caption("Varianz von Bestand und Bestellung je Varianz des Bedarfsrauschens (Lyapunov-Gleichung, exakt), dazu eine Simulation mit dem Zufalls-Seed der Seitenleiste. Bei β = 1 hängt die Bestellvarianz nicht an der Verteilung.")
nz = _noise(int(tau), int(k), float(alpha), int(beta), int(seed))
st.plotly_chart(build_noise(nz), width="stretch", key="chart_noise")
dfn = pd.DataFrame([{
    "Lieferzeit": r["name"], "Spektralradius ρ": E.de(r["rho"], 4), "var(i)/var(e)": E.de(r["var_i"], 3) if r["stable"] else "instabil", "var(u)/var(e)": E.de(r["var_u"], 4) if r["stable"] else "instabil",
} for r in nz["rows"]])
st.dataframe(dfn, hide_index=True, width="stretch")
if nz["sim"] is not None:
    cs = nz["chosen"]
    st.caption(f"Simulation ({C.NOISE_RUNS} Läufe zu je {C.NOISE_PERIODS - C.NOISE_BURN} Perioden nach {C.NOISE_BURN} Perioden Einschwingen, {nz['chosen']['name']}): "
               f"var(i) = {E.de(nz['sim']['var_i'], 3)} gegen exakt {E.de(cs['var_i'], 3)}; var(u) = {E.de(nz['sim']['var_u'], 4)} gegen exakt {E.de(cs['var_u'], 4)}.")
else:
    st.caption("Keine Simulation: die gewählte Lieferzeit ist instabil oder klingt zu langsam ab (ρ ≥ 0,995).")
_show(E.verdict_noise(nz))

st.markdown("---")

# --- Experiment: Einzelbestellung -------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Jede Bestellung zieht ihre eigene Lieferzeit")
st.caption("Einzelbestellung: jede Bestellung kommt als Ganzes nach einer gezogenen Lieferzeit, Bestellungen dürfen sich überholen. Der Mittelwert folgt dem anteiligen Modell, die Streuung nicht. "
           "Die exakte Varianz (zweites Moment) wird für k ≤ " + str(E.MS_K_MAX) + " gerechnet; die Simulation läuft immer.")
s1, s2, s3 = st.columns(3)
d_val = s1.slider("Mittlerer Bedarf D je Periode", *bounds("d_slider"), key="d_slider", help="Die Streuung der Lieferzeit wirkt mit D², das Bedarfsrauschen nur mit σ².")
sigma = s2.slider("Rauschen σ des Bedarfs", *bounds("sigma_slider"), step=C.SIGMA_STEP, key="sigma_slider", help="Standardabweichung der Bedarfsabweichung je Periode (Normalverteilung).")
runs = s3.slider("Läufe", *bounds("runs_slider"), step=C.RUNS_STEP, key="runs_slider", help="Zahl der parallelen Simulationsläufe; mehr Läufe brauchen mehr Rechenzeit.")
with st.spinner(f"Simuliere {int(runs)} Läufe zu je bis zu {C.SIM_PERIODS} Perioden …"):
    sg = _single(int(tau), int(k), float(alpha), int(beta), float(d_val), float(sigma), int(runs), int(seed))
st.plotly_chart(build_limit_axis(sg), width="stretch", key="chart_limit_axis")
st.caption("Die Marken zeigen, bis zu welchem α der Mittelwert (anteiliges Modell) und das zweite Moment (Einzelbestellung) stabil bleiben.")
e1, e2 = st.columns(2)
with e1:
    st.markdown("**Ein Lauf: Einzelbestellung gegen anteilig (gleiches Bedarfsrauschen)**")
    st.plotly_chart(build_paths(sg), width="stretch", key="chart_paths")
with e2:
    st.markdown("**Mittel über Läufe nach einem Bedarfsstoß gegen die anteilige Antwort**")
    if sg["shock"] is not None:
        st.plotly_chart(build_shock(sg), width="stretch", key="chart_shock")
        st.caption(f"Mittel über {C.SHOCK_RUNS} Läufe, Bedarf 0, Stoß 1 in Periode 1; der Mittelwert der Einzelbestellung folgt der anteiligen Antwort.")
    else:
        st.info("Der Mittelwert ist hier instabil (ρ ≥ 1): keine Stoßantwort.")
rows5 = [{"Größe": "var(i), Simulation", "Wert": E.sig(sg["sim"]["var_i"]) if sg["sim"]["finite"] else "wächst über alle Grenzen"}]
rows5.append({"Größe": "var(u), Simulation", "Wert": E.sig(sg["sim"]["var_u"]) if sg["sim"]["finite"] else "wächst über alle Grenzen"})
if sg["exact"] is not None:
    rows5 += [{"Größe": "var(i), exakt (zweites Moment)", "Wert": E.sig(sg["exact"]["var_i"])}, {"Größe": "var(u), exakt (zweites Moment)", "Wert": E.sig(sg["exact"]["var_u"])}]
if sg["fluid"] is not None:
    rows5 += [{"Größe": "var(i), anteilig (Lyapunov)", "Wert": E.sig(sg["fluid"]["var_i"])}, {"Größe": "var(u), anteilig (Lyapunov)", "Wert": E.sig(sg["fluid"]["var_u"])}]
st.dataframe(pd.DataFrame(rows5), hide_index=True, width="stretch")
st.caption(f"Simulation: {int(runs)} Läufe, {sg['sim']['periods']} Perioden, die ersten {sg['sim']['burn']} zählen nicht (ist der Mittelwert instabil, nur 300 Perioden); Zufalls-Seed {int(seed)}. Bei konstantem Bedarf (σ = 0) gibt das anteilige Modell Varianz 0, die Einzelbestellung nicht.")
_show(E.verdict_single(sg))

st.markdown("---")

# --- Experiment: stabil heißt nicht schnell ---------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Stabil heißt nicht schnell")
st.caption("Spektralradius ρ über α und Abklingzeit (Bestand unter 1 % seiner Spitze nach einem Bedarfsstoß) für verschiedene Stufenzahlen bei gleichem Mittel τ; α und β wie in der Seitenleiste.")
dc = _decay(int(tau), float(alpha), int(k))
st.plotly_chart(build_rho_curves(dc), width="stretch", key="chart_rho")


def _dec_txt(rho, t):
    if rho >= 1 - 1e-9:
        return "instabil"
    return "nicht abgeklungen" if t is None else E.fmt_int(t)


dfd = pd.DataFrame([{
    "Lieferzeit": r["label"], "ρ (β = 0)": E.de(r["rho0"], 4), "Abklingzeit β = 0": _dec_txt(r["rho0"], r["decay0"]),
    "ρ (β = 1)": E.de(r["rho1"], 4), "Abklingzeit β = 1": _dec_txt(r["rho1"], r["decay1"]),
    "kleinstes ρ über α (β = 0)": f"{E.de(r['best_rho'], 4)} bei α = {E.de(r['best_alpha'], 3)}",
} for r in dc["rows"]])
st.dataframe(dfd, hide_index=True, width="stretch")
st.caption("Abklingzeit: letzte Periode, in der |Bestand| noch über 1 % seiner Spitze liegt; Obergrenze der Rechnung 100 000 Perioden. Kleinstes ρ: α-Raster 0,01 bis 2 mit lokaler Verfeinerung.")
_show(E.verdict_decay(dc, int(beta)))

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Linear, ohne Klemmung** (die Bestellung darf negativ werden) | Mit Bestellung ≥ 0 bleibt die Schwingung hinter der Grenze beschränkt (Grenzzyklus, bei fester Lieferzeit auch Chaos); mit streuender Lieferzeit hier nicht gerechnet | [`lager-chaos-demo`](https://github.com/sebastian-hanisch/lager-chaos-demo) (feste Lieferzeit, β = 0) |
| **Anteilig (Erwartungswert-Modell)** | Kommt jede Bestellung als Ganzes, liegt die Grenze niedriger als im anteiligen Modell (Experiment 4) und das Lager schwankt auch bei konstantem Bedarf | Experiment 4 |
| **Lieferzeit = Summe geometrischer Zeiten** (eigene Wahl) | Andere Formen gehen in die gemessenen Fälle von Experiment 2; ob echte Lieferzeiten so verteilt sind, ist nicht untersucht | – |
| **Kein Meldeverzug** | Meldeverzug zusammen mit Streuung ist nicht gerechnet | [`meldeverzug-demo`](https://github.com/sebastian-hanisch/meldeverzug-demo) |
| **β nur 0 oder 1** | Zwischenwerte nicht gerechnet; bei fester Lieferzeit mit Schwelle bei β = ½ untersucht | [`leitung-mitrechnen-demo`](https://github.com/sebastian-hanisch/leitung-mitrechnen-demo) |
| **Weißes Rauschen, Normalverteilung** | Saison und Resonanz (Frequenzgang) nicht gerechnet | [`frequenzgang-lager-demo`](https://github.com/sebastian-hanisch/frequenzgang-lager-demo) |
| **Feste Lieferzeit als Grenzfall der Familie** | Die feste Verzögerung in stetiger Zeit (a·τ = π/2) steht in [`verzoegerung-hopf-demo`](https://github.com/sebastian-hanisch/verzoegerung-hopf-demo); der stetige Erlang-Kreis hier nur als Vergleichslinie | [`lager-lieferzeit-demo`](https://github.com/sebastian-hanisch/lager-lieferzeit-demo) (feste Lieferzeit) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben: Anpassung α, Leitungsgewicht β ∈ {0, 1}, Lieferzeit-Verteilung (hier die Summe aus k geometrischen Zeiten mit Mittel τ). Gesucht: ab welchem α die Schwingung wächst, wie stark der Bestand bei Rauschen schwankt, wie lange das Abklingen dauert.
Alle Größen sind Abweichungen vom Gleichgewicht, Zeit in Perioden. Die Formeln sind eigene Herleitungen in diesem Modell, geprüft gegen Eigenwerte, Simulation und Handrechnung (siehe README); eine Quelle zu Lieferzeit-Verteilungen im Lager wurde nicht gelesen.

| Symbol | Bedeutung |
|---|---|
| $i_t$ | Bestand minus Soll in Periode $t$ |
| $u_t$ | Bestellung minus Mittelbedarf |
| $e_t$ | Bedarfsabweichung (Rauschen) |
| $\alpha$ | Anpassung (Anteil des Bestandsfehlers, der nachbestellt wird) |
| $\beta$ | Leitungsgewicht: 0 nicht, 1 voll mitgerechnet |
| $f_l$ | Anteil einer Bestellung, der nach $l$ Perioden ankommt |
| $S_j$ | $P(\text{Lieferzeit} > j)$, Anteil der Bestellung, der nach $j$ Perioden noch unterwegs ist |
| $k,\ p,\ q$ | Zahl der Stufen, Fortschrittswahrscheinlichkeit je Periode, $q = 1 - p$; Mittel $\tau = k/p$ |
| $x_j$ | Inhalt der Stufe $j$ der Lieferkette |
| $\rho$ | Spektralradius der Zustandsmatrix (stabil genau bei $\rho < 1$) |

**Modell (anteilig).** Mit der Leitung $w_t = \sum_{j \ge 1} S_j\, u_{t-j}$ gilt
$$u_t = -\alpha\,(i_t + \beta\, w_t), \qquad i_{t+1} = i_t + \sum_{l \ge 1} f_l\, u_{t+1-l} - e_t .$$
Für $f_l = \binom{l-1}{k-1}\, p^k q^{\,l-k}$ ($l \ge k$) ist das gleichwertig zur Kette: $x_1 \leftarrow x_1 + u_t$, aus jeder Stufe $j$ geht der Anteil $p$ weiter, aus Stufe $k$ in den Bestand. Für $p = 1$ ist es die feste Lieferzeit $L = k$.

**Charakteristisches Polynom und Grenze.**
$$\beta = 0:\quad (z-1)(z-q)^k + \alpha\, p^k z = 0, \qquad \beta = 1:\quad \bigl(z-(1-\alpha)\bigr)(z-q)^k = 0 .$$
Für $\beta = 0$ ist die Grenze der kleinste positive reelle Wert von
$$\alpha(z) = \frac{(1-z)(z-q)^k}{p^k\, z}, \qquad |z| = 1 .$$
Daraus folgen $k = 1$: $\alpha_{\mathrm{krit}} = 4\tau - 2$ (Wurzel $z = -1$), $k = 2$: $\alpha_{\mathrm{krit}} = p(2-p)$, und für die feste Lieferzeit $\alpha_{\mathrm{krit}} = 2\sin\frac{\pi}{4L-2}$. Bei $\beta = 1$ gilt für jede Verteilung $\alpha_{\mathrm{krit}} = 2$; für die Erlang-Kette ist $\rho = \max(|1-\alpha|,\, q)$, denn die Bestandsposition $IP_t = i_t + w_t$ erfüllt $IP_{t+1} = (1-\alpha)\, IP_t - e_t$.
Stetiger Grenzfall (Erlang-$k$, mittlere Verzögerung 1): $s\,(1 + s/k)^k + a = 0$, Grenze $a\tau = 4$ für $k = 2$ und $8/3$ für $k = 3$ (Hurwitz), $\pi/2$ für die feste Verzögerung.

**Schwankung (weißes Rauschen, Varianz 1).** Mit Zustand $z$, Matrix $A$, $u = c^\top z$ und $b = (-1, 0, \dots, 0)^\top$ löst $S = A S A^\top + b b^\top$ die Varianzen: $\operatorname{var}(i) = S_{11}$, $\operatorname{var}(u) = c^\top S c$. Bei $\beta = 1$ ist $\operatorname{var}(u)/\operatorname{var}(e) = \alpha/(2-\alpha)$.

**Einzelbestellung.** Jede Bestellung $O_t = D + u_t$ kommt als Ganzes nach einer gezogenen Lieferzeit. Ihr Mittelwert folgt dem anteiligen Modell. Für das zweite Moment gilt: aus Stufe $j$ geht der Betrag $m_j$ weiter mit bedingtem Mittel $p\,x_j$ und bedingter Varianz $p\,q\,Q_j$, $Q_j$ die Summe der Quadrate der Bestellklumpen in Stufe $j$; $E[z z^\top]$ und $E[Q]$ erfüllen ein geschlossenes lineares System, das genau dann stationär ist, wenn sein Spektralradius unter 1 liegt (bei $D = 0$).

**Quellen.** Keine Quelle zu Lieferzeit-Verteilungen in Bestandsregeln wurde im Wortlaut gelesen; die Neuheit dieser Herleitungen wird nicht behauptet. DELAY3 in Vensim ist eine Verzögerung dritter Ordnung (Dokumentation von der unabhängigen Prüfung gelesen); ob sie die „Standardverzögerung“ der Systemdynamik-Software ist, ist nicht belegt.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)

sync_query_params({
    "tau_slider": int(tau), "k_slider": int(k), "alpha_slider": float(alpha), "beta_radio": int(beta), "seed_input": int(seed),
    "tau_map_select": int(tau_map), "dist_spread_slider": int(spread), "random_form_select": str(rform),
    "d_slider": int(d_val), "sigma_slider": float(sigma), "runs_slider": int(runs),
})
