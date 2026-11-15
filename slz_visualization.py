"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.
Beschriftungen stehen als Annotationen (nicht als Text-Spuren); Zahlen in den Hover-Texten mit Dezimalkomma."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import slz_constants as C

COLORS = {"fest": C.COLOR_FIXED, "streuend": C.COLOR_SPREAD, "k = 1": C.COLOR_ONE}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", y=-0.22))
    return lock_axes(fig)


def _dec(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


# --- Experiment 1 ----------------------------------------------------------------------------------------------------------------------

def build_pmf(a: dict):
    """Verteilung der Lieferzeit der gewählten Kette gegen die feste Lieferzeit (alles bei Periode tau)."""
    tau = a["tau"]
    f = a["pmf"]
    xmax = int(min(len(f), 4 * tau + 8))
    ls = np.arange(1, xmax + 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ls, y=f[:xmax], marker_color=C.COLOR_SPREAD, name=f"streuend (k = {a['k']})",
                         hovertemplate="Lieferzeit %{x}: %{y:.3f}<extra></extra>"))
    top = float(max(f[:xmax].max(), 1e-9)) * 1.25
    fig.add_shape(type="line", x0=tau, x1=tau, y0=0, y1=top, line=dict(color=C.COLOR_FIXED, width=3, dash="dash"))
    fig.add_annotation(x=tau, y=top, text=f"fest: 100 % bei {tau}", showarrow=False, yanchor="bottom", font=dict(color=C.COLOR_FIXED))
    fig.update_xaxes(title_text="Lieferzeit (Perioden)", range=[0, xmax + 1])
    fig.update_yaxes(title_text="Anteil der Bestellung", range=[0, top * 1.12])
    return _base(fig, 300)


def build_response(a: dict):
    """Bestand nach einem Bedarfsstoß (Bedarfsabweichung 1 in Periode 1): fest, gewählte Stufenzahl und k = 1."""
    fig = go.Figure()
    n = a["n_resp"]
    t = np.arange(1, n + 1)
    for key, label in (("fixed", "fest"), ("chosen", f"streuend ({a['chosen']['name']})"), ("one", "k = 1")):
        m = a[key]
        if key == "chosen" and a["k"] in (1, a["tau"]):
            continue
        if key == "one" and a["k"] == 1:
            continue
        color = COLORS["fest"] if key == "fixed" else (COLORS["streuend"] if key == "chosen" else COLORS["k = 1"])
        y = np.clip(m["i"], -3, 3)
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color=color, width=2.5), name=f"{label}, ρ = {_dec(m['rho'], 3)}",
                                 hovertemplate="Periode %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.6)", width=1))
    fig.update_xaxes(title_text="Periode nach dem Stoß")
    fig.update_yaxes(title_text="Bestand (Abweichung, bei ±3 abgeschnitten)", range=[-3.2, 3.2])
    return _base(fig, 320)


def build_eigs(a: dict):
    """Eigenwerte der Zustandsmatrix in der komplexen Ebene mit dem Einheitskreis: außerhalb heißt instabil."""
    fig = go.Figure()
    th = np.linspace(0, 2 * math.pi, 361)
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", line=dict(color="rgba(128,128,128,0.7)", width=1.5), name="Einheitskreis", hoverinfo="skip"))
    for key, label in (("fixed", "fest"), ("chosen", f"streuend ({a['chosen']['name']})"), ("one", "k = 1")):
        m = a[key]
        ev = m["eigs"]
        color = COLORS["fest"] if key == "fixed" else (COLORS["streuend"] if key == "chosen" else COLORS["k = 1"])
        fig.add_trace(go.Scatter(x=ev.real, y=ev.imag, mode="markers", marker=dict(color=color, size=9 if key != "fixed" else 7, symbol="circle" if key != "fixed" else "diamond"),
                                 name=label, hovertemplate="Re %{x:.3f}, Im %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Realteil", range=[-1.35, 1.35], scaleanchor="y", constrain="domain")
    fig.update_yaxes(title_text="Imaginärteil", range=[-1.35, 1.35], constrain="domain")
    return _base(fig, 340)


# --- Experiment 2 ----------------------------------------------------------------------------------------------------------------------

def build_limit_map(a: dict):
    """α_krit · τ über die Stufenzahl k (k = 1 steht außerhalb des Bildes: 4τ − 2), mit der festen Lieferzeit und dem stetigen Erlang-Kreis."""
    rows = [r for r in a["rows"] if r["k"] >= 2]
    ks = [r["k"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[r["product"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_SPREAD, width=3), name=f"diskret, τ = {a['tau']}",
                             hovertemplate="k = %{x}: α_krit·τ = %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ks, y=[r["continuous"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_ONE, width=2, dash="dot"), marker=dict(size=5),
                             name="stetig (τ → ∞)", hovertemplate="k = %{x}: a·τ = %{y:.4f}<extra></extra>"))
    fig.add_hline(y=a["fixed_product"], line=dict(color=C.COLOR_FIXED, width=2, dash="dash"),
                  annotation_text=f"feste Lieferzeit {a['tau']}: {_dec(a['fixed_product'], 3)}", annotation_position="top right")
    fig.add_hline(y=a["pi_half"], line=dict(color="rgba(128,128,128,0.8)", width=1, dash="dot"),
                  annotation_text="feste Verzögerung, stetig: π/2", annotation_position="bottom right")
    fig.update_xaxes(title_text="Stufenzahl k (logarithmisch)", type="log", tickvals=ks, ticktext=[str(k) for k in ks])
    fig.update_yaxes(title_text="α_krit · τ", range=[1.4, 4.3])
    return _base(fig, 360)


# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------

def build_forms_pmf(fa: dict):
    fig = go.Figure()
    xmax = max(len(r["pmf"]) for r in fa["rows"])
    colors = [C.COLOR_FIXED, C.COLOR_SPREAD, C.COLOR_ONE, "#8e6bb0"]
    for r, color in zip(fa["rows"], colors):
        f = r["pmf"]
        fig.add_trace(go.Bar(x=np.arange(1, len(f) + 1), y=f, name=r["name"], marker_color=color, hovertemplate="Lieferzeit %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Lieferzeit (Perioden)", range=[0, xmax + 1], dtick=1 if xmax <= 30 else None)
    fig.update_yaxes(title_text="Anteil der Bestellung", range=[0, 1.05])
    return _base(fig, 300)


def build_forms_limits(fa: dict, alpha: float):
    names = [r["name"] for r in fa["rows"]]
    vals = [r["alpha_krit"] for r in fa["rows"]]
    colors = [C.COLOR_FIXED, C.COLOR_SPREAD, C.COLOR_ONE, "#8e6bb0"]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors, hovertemplate="%{x}: α_krit = %{y:.3f}<extra></extra>"))
    fig.add_hline(y=alpha, line=dict(color=C.COLOR_BAD, width=2, dash="dash"), annotation_text=f"gewähltes α = {_dec(alpha)}", annotation_position="top left")
    fig.update_yaxes(title_text="α_krit (β = 0)", range=[0, max(vals + [alpha]) * 1.2])
    return _base(fig, 320, legend=False)


# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------

def build_noise(a: dict):
    rows = [r for r in a["rows"] if r["stable"]]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Bestandsvarianz var(i)/var(e)", "Bestellvarianz var(u)/var(e)"))
    for j, key in enumerate(("var_i", "var_u"), start=1):
        fig.add_trace(go.Bar(x=[r["name"] for r in rows], y=[r[key] for r in rows], marker_color=[C.COLOR_FIXED if r["name"] == "fest" else (C.COLOR_ONE if r["name"] == "k = 1" else C.COLOR_SPREAD) for r in rows],
                             hovertemplate="%{x}: %{y:.3f}<extra></extra>", showlegend=False), row=1, col=j)
    return _base(fig, 320, legend=False)


# --- Experiment 5 ----------------------------------------------------------------------------------------------------------------------

def build_paths(a: dict):
    p = a["path"]
    t = np.arange(1, p["n"] + 1)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Bestand (Abweichung vom Soll)", "Bestellung (Abweichung vom Mittelbedarf)"))
    fig.add_trace(go.Scatter(x=t, y=p["i_single"], mode="lines", line=dict(color=C.COLOR_BAD, width=1.8), name="Einzelbestellung (ein Lauf)",
                             hovertemplate="Periode %{x}: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=p["i_fluid"], mode="lines", line=dict(color=C.COLOR_FIXED, width=1.8), name="anteilig (gleiches Rauschen)",
                             hovertemplate="Periode %{x}: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=p["u_single"], mode="lines", line=dict(color=C.COLOR_BAD, width=1.8), name="Einzelbestellung", showlegend=False,
                             hovertemplate="Periode %{x}: %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=p["u_fluid"], mode="lines", line=dict(color=C.COLOR_FIXED, width=1.8), name="anteilig", showlegend=False,
                             hovertemplate="Periode %{x}: %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.update_xaxes(title_text="Periode", row=2, col=1)
    return _base(fig, 440)


def build_shock(a: dict):
    s = a["shock"]
    n = len(s["mean"])
    t = np.arange(1, n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([s["mean"] + 3 * s["se"], (s["mean"] - 3 * s["se"])[::-1]]), fill="toself",
                             fillcolor="rgba(181,69,58,0.2)", line=dict(width=0), hoverinfo="skip", name="Mittel der Läufe ± 3 Standardfehler"))
    fig.add_trace(go.Scatter(x=t, y=s["mean"], mode="lines", line=dict(color=C.COLOR_BAD, width=2), name="Einzelbestellung (Mittel über Läufe)",
                             hovertemplate="Periode %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=s["fluid"], mode="lines", line=dict(color=C.COLOR_FIXED, width=2, dash="dash"), name="anteilig (exakt)",
                             hovertemplate="Periode %{x}: %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Periode nach dem Stoß")
    fig.update_yaxes(title_text="Bestand nach einem Bedarfsstoß")
    return _base(fig, 320)


def build_limit_axis(a: dict):
    """Die α-Achse mit den Grenzen: feste Lieferzeit, Mittelwert, zweites Moment und das gewählte α."""
    marks = [("fest", a["alpha_fixed"], C.COLOR_FIXED), ("Mittelwert", a["alpha_mean"], C.COLOR_SPREAD)]
    if a["alpha_ms"] is not None:
        marks.append(("zweites Moment", a["alpha_ms"], C.COLOR_BAD))
    top = max(a["alpha"] * 1.6, min(max(v for _, v, _ in marks if math.isfinite(v)), 2.4) * 1.15, 0.3)
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, x1=top, y0=0, y1=0, line=dict(color="rgba(128,128,128,0.7)", width=2))
    for j, (name, val, color) in enumerate(marks):
        if not math.isfinite(val) or val > top:
            fig.add_annotation(x=top, y=0.25 + 0.2 * j, text=f"{name}: {_dec(val, 3) if math.isfinite(val) else 'keine'} →", showarrow=False, xanchor="right", font=dict(color=color))
            continue
        fig.add_trace(go.Scatter(x=[val], y=[0], mode="markers", marker=dict(color=color, size=13, symbol="diamond"), name=f"{name}: {_dec(val, 3)}",
                                 hovertemplate=f"{name}: {_dec(val, 3)}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[a["alpha"]], y=[0], mode="markers", marker=dict(color="black", size=11, symbol="line-ns-open", line=dict(width=3)), name=f"gewähltes α = {_dec(a['alpha'])}",
                             hovertemplate="gewähltes α<extra></extra>"))
    fig.update_xaxes(title_text="α", range=[0, top])
    fig.update_yaxes(range=[-0.5, 1.0], showticklabels=False, showgrid=False, zeroline=False)
    return _base(fig, 200)


# --- Experiment 6 ----------------------------------------------------------------------------------------------------------------------

def build_rho_curves(a: dict):
    fig = go.Figure()
    alphas = a["alphas"]
    for r in a["rows"]:
        color = C.COLOR_FIXED if r["label"] == "fest" else (C.COLOR_ONE if r["k"] == 1 else None)
        fig.add_trace(go.Scatter(x=alphas, y=np.minimum(r["curve"], 1.25), mode="lines", name=r["label"], line=dict(width=2.5 if r["label"] == "fest" else 1.8, color=color),
                                 hovertemplate=f"{r['label']}, α = %{{x:.2f}}: ρ = %{{y:.4f}}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color="rgba(128,128,128,0.8)", width=1.5, dash="dot"))
    fig.add_vline(x=a["alpha"], line=dict(color=C.COLOR_BAD, width=2, dash="dash"), annotation_text=f"α = {_dec(a['alpha'])}", annotation_position="top")
    fig.update_xaxes(title_text="α (logarithmisch)", type="log", range=[math.log10(0.01), math.log10(2.0)], tickvals=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2],
                     ticktext=["0,01", "0,02", "0,05", "0,1", "0,2", "0,5", "1", "2"])
    fig.update_yaxes(title_text="Spektralradius ρ (bei 1,25 abgeschnitten)", range=[0.6, 1.27])
    return _base(fig, 360)
