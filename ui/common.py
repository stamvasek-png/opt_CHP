"""Společné UI utility: CSS, formátování, aplikační nastavení."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "settings.json"

DEFAULT_SETTINGS = {
    "fx_czk_eur": 25.0,
    "run1_time_limit_s": 15,
    "run1_gap_rel": 0.002,
    "solver_time_limit_s": 60,
    "solver_gap_rel": 0.003,
    "r_grid_frac": [0.25, 0.5, 0.75, 1.0],
    "workers": 0,          # 0 = auto (počet CPU)
}


def load_settings() -> dict:
    try:
        d = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return {**DEFAULT_SETTINGS, **d}
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def inject_css() -> None:
    """Vizuální styl (port z původní aplikace)."""
    st.markdown("""
<style>
/* KPI karty */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e2a3a 0%, #243447 100%);
    border: 1px solid #2d4a6b;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
div[data-testid="metric-container"] label {
    color: #8ab4d4 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
}
h3 { color: #e8f4fd; }
section[data-testid="stSidebar"] { background: #0f1923; }
section[data-testid="stSidebar"] label { color: #c5d8ea !important; }
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 { color: #4fc3f7 !important; }
</style>
""", unsafe_allow_html=True)


def fmt_eur(x: float) -> str:
    return f"{x:,.0f} €".replace(",", " ")


def fmt_mw(x: float) -> str:
    return f"{x:,.2f} MW".replace(",", " ")


def stage_badges(status: dict) -> str:
    """Řádek stavových odznaků dne pro sidebar."""
    items = [
        ("inputs_loaded", "Vstupy"),
        ("run1_done", "aFRR bidy"),
        ("auction_entered", "Aukce"),
        ("run2_done", "DA plán"),
        ("nomination_frozen", "Nominace 🔒"),
        ("actual_da_loaded", "Skut. DA"),
        ("run3_done", "Re-dispatch"),
    ]
    return "  \n".join(
        f"{'✅' if status.get(k) else '⬜'} {label}" for k, label in items)
