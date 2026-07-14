"""Plotly grafy pro D+1 plánování (staví se z plan DataFrame modelu)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.profiles import Profile, Site
from core.timegrid import TimeGrid

COLORS = {
    "kgj": "#e67e22", "boil": "#e74c3c", "ek": "#9b59b6", "imp": "#95a5a6",
    "tes": "#16a085", "pv": "#f1c40f", "bess": "#2ecc71",
    "pos": "#4fc3f7", "nom": "#f39c12", "dev": "#e74c3c",
    "da": "#2ecc71", "imb": "#e74c3c", "cons": "#7f8c8d",
}


def _x(grid: TimeGrid):
    return list(grid.index)


def price_preview_chart(grid: TimeGrid, da: np.ndarray, imb: np.ndarray,
                        da_actual: np.ndarray | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_x(grid), y=da, name="DA predikce",
                             line=dict(color=COLORS["da"], width=2)))
    fig.add_trace(go.Scatter(x=_x(grid), y=imb, name="Odchylka predikce",
                             line=dict(color=COLORS["imb"], width=1.5,
                                       dash="dot")))
    if da_actual is not None:
        fig.add_trace(go.Scatter(x=_x(grid), y=da_actual, name="DA skutečná",
                                 line=dict(color="#ffffff", width=2,
                                           dash="dash")))
    fig.update_layout(height=320, hovermode="x unified",
                      yaxis_title="€/MWh", margin=dict(t=30, b=10))
    return fig


def heat_chart(plan: pd.DataFrame, grid: TimeGrid, site: Site) -> go.Figure:
    """Skládané pokrytí poptávky tepla lokality."""
    fig = go.Figure()
    x = _x(grid)
    stacks = []
    for k in site.kgjs:
        stacks.append((f"KGJ {k.name}", f"{site.site_id}|{k.asset_id}|q_th_mw",
                       COLORS["kgj"]))
    for b in site.boilers:
        stacks.append((f"Kotel {b.name}", f"{site.site_id}|{b.asset_id}|q_th_mw",
                       COLORS["boil"]))
    for e in site.eks:
        stacks.append((f"EK {e.name}", f"{site.site_id}|{e.asset_id}|q_th_mw",
                       COLORS["ek"]))
    for im in site.heat_imports:
        stacks.append((f"Import {im.name}",
                       f"{site.site_id}|{im.asset_id}|q_th_mw", COLORS["imp"]))
    for label, col, color in stacks:
        if col in plan.columns:
            fig.add_trace(go.Scatter(
                x=x, y=plan[col], name=label, stackgroup="heat",
                mode="none", fillcolor=color))
    for t_ in site.tes_units:
        col_out = f"{site.site_id}|{t_.asset_id}|out_mw"
        col_in = f"{site.site_id}|{t_.asset_id}|in_mw"
        if col_out in plan.columns:
            net = plan[col_out] - plan[col_in]
            fig.add_trace(go.Scatter(
                x=x, y=net.clip(lower=0), name=f"TES {t_.name} výdej",
                stackgroup="heat", mode="none", fillcolor=COLORS["tes"]))
    dem_col = f"{site.site_id}|heat_dem_mw"
    if dem_col in plan.columns:
        fig.add_trace(go.Scatter(x=x, y=plan[dem_col], name="Poptávka",
                                 line=dict(color="#ffffff", width=2)))
    fig.update_layout(height=340, hovermode="x unified", yaxis_title="MW_th",
                      margin=dict(t=30, b=10))
    return fig


def portfolio_chart(plan: pd.DataFrame, grid: TimeGrid,
                    profile: Profile) -> go.Figure:
    """Pozice portfolia + export/import per lokalita."""
    fig = go.Figure()
    x = _x(grid)
    for s in profile.sites:
        net = plan[f"{s.site_id}|export_mw"] - plan[f"{s.site_id}|import_mw"]
        fig.add_trace(go.Scatter(x=x, y=net, name=f"{s.name} (netto)",
                                 stackgroup="pos", mode="none"))
    fig.add_trace(go.Scatter(x=x, y=plan["pos_mw"], name="Pozice portfolia",
                             line=dict(color=COLORS["pos"], width=2.5)))
    fig.update_layout(height=340, hovermode="x unified", yaxis_title="MW",
                      margin=dict(t=30, b=10))
    return fig


def soc_chart(plan: pd.DataFrame, grid: TimeGrid,
              profile: Profile) -> go.Figure:
    fig = go.Figure()
    x = _x(grid)
    for s in profile.sites:
        for t_ in s.tes_units:
            col = f"{s.site_id}|{t_.asset_id}|soc_mwh"
            if col in plan.columns:
                fig.add_trace(go.Scatter(
                    x=x, y=plan[col], name=f"TES {s.name}/{t_.name}",
                    line=dict(color=COLORS["tes"], width=2)))
        for b in s.bess_units:
            col = f"{s.site_id}|{b.asset_id}|soc_mwh"
            if col in plan.columns:
                fig.add_trace(go.Scatter(
                    x=x, y=plan[col], name=f"BESS {s.name}/{b.name}",
                    line=dict(color=COLORS["bess"], width=2)))
    fig.update_layout(height=300, hovermode="x unified", yaxis_title="MWh",
                      margin=dict(t=30, b=10))
    return fig


def pv_chart(plan: pd.DataFrame, grid: TimeGrid, profile: Profile) -> go.Figure:
    fig = go.Figure()
    x = _x(grid)
    for s in profile.sites:
        for pv in s.pvs:
            used = f"{s.site_id}|{pv.asset_id}|used_mw"
            fc = f"{s.site_id}|{pv.asset_id}|forecast_mw"
            if fc in plan.columns:
                fig.add_trace(go.Scatter(
                    x=x, y=plan[fc], name=f"{s.name}/{pv.name} predikce",
                    line=dict(color=COLORS["pv"], width=1, dash="dot")))
                fig.add_trace(go.Scatter(
                    x=x, y=plan[used], name=f"{s.name}/{pv.name} využito",
                    line=dict(color=COLORS["pv"], width=2)))
    fig.update_layout(height=300, hovermode="x unified", yaxis_title="MW",
                      margin=dict(t=30, b=10))
    return fig


def ladder_chart(ladder: pd.DataFrame) -> go.Figure:
    """Min. marginální cena vs. R per blok, zvlášť směr + a −."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("aFRR+", "aFRR−"),
                        shared_yaxes=True)
    palette = ["#4fc3f7", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c",
               "#9b59b6"]
    for col_i, direction in ((1, "up"), (2, "dn")):
        sub = ladder[(ladder["direction"] == direction) & ladder["feasible"]]
        for b, g in sub.groupby("block"):
            g = g.sort_values("r_mw")
            fig.add_trace(go.Scatter(
                x=g["r_mw"], y=g["min_price_marginal"],
                name=f"blok {g['block_label'].iloc[0]}",
                legendgroup=f"b{b}", showlegend=(col_i == 1),
                line=dict(color=palette[int(b) % 6], width=2),
                mode="lines+markers"), row=1, col=col_i)
            fig.add_trace(go.Scatter(
                x=g["r_mw"], y=g["predicted_price"],
                name=f"predikce {g['block_label'].iloc[0]}",
                legendgroup=f"b{b}", showlegend=False,
                line=dict(color=palette[int(b) % 6], width=1, dash="dot"),
                mode="lines"), row=1, col=col_i)
    fig.update_xaxes(title_text="R [MW]")
    fig.update_yaxes(title_text="€/MW/h", col=1)
    fig.update_layout(height=380, margin=dict(t=40, b=10))
    return fig


def deviation_chart(plan: pd.DataFrame, grid: TimeGrid) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.06)
    x = _x(grid)
    fig.add_trace(go.Scatter(x=x, y=plan["nomination_mw"], name="Nominace",
                             line=dict(color=COLORS["nom"], width=2,
                                       dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=plan["pos_mw"], name="Pozice re-dispatch",
                             line=dict(color=COLORS["pos"], width=2)),
                  row=1, col=1)
    dev = plan["deviation_mw"]
    fig.add_trace(go.Bar(x=x, y=dev, name="Odchylka",
                         marker_color=np.where(dev >= 0, "#2ecc71", "#e74c3c")),
                  row=2, col=1)
    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="Δ MW", row=2, col=1)
    fig.update_layout(height=430, hovermode="x unified",
                      margin=dict(t=30, b=10))
    return fig


def nomination_chart(nomination: pd.DataFrame, grid: TimeGrid) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=_x(grid), y=nomination["pos_mw"],
                         name="Nominace [MW]",
                         marker_color=np.where(nomination["pos_mw"] >= 0,
                                               "#4fc3f7", "#e67e22")))
    fig.update_layout(height=300, yaxis_title="MW", hovermode="x unified",
                      margin=dict(t=30, b=10))
    return fig
