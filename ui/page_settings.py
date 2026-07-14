"""Stránka „Nastavení" — TDD koeficienty, FX, solver."""

from __future__ import annotations

import streamlit as st

from core.tdd import TddStore, parse_ote_tdd, synthetic_tdd_year
from ui.common import DEFAULT_SETTINGS, load_settings, save_settings

st.title("⚙️ Nastavení")

settings = load_settings()

# ── TDD ─────────────────────────────────────────────────────────────────────
st.subheader("TDD koeficienty (OTE)")
store = TddStore()
years = store.available_years()
st.caption("Nahrané roky: " + (", ".join(map(str, years)) if years else "žádné")
           + ". Potřebné pro lokality se spotřebou v režimu TDD.")

c1, c2 = st.columns(2)
with c1:
    tdd_year = st.number_input("Rok koeficientů", value=2026, step=1,
                               min_value=2020, max_value=2035)
    tdd_file = st.file_uploader("Soubor normalizovaných TDD z OTE (xlsx/csv)",
                                type=["xlsx", "csv"])
    if tdd_file is not None and st.button("📥 Nahrát TDD"):
        try:
            df = parse_ote_tdd(tdd_file, int(tdd_year))
            store.save(int(tdd_year), df)
            st.success(f"TDD {tdd_year} uloženo ({len(df.columns)} tříd, "
                       f"{len(df)} hodin).")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
with c2:
    st.markdown("**Syntetická TDD** (pro testování bez OTE souboru)")
    syn_year = st.number_input("Rok", value=2026, step=1, key="syn_year")
    if st.button("🧪 Vygenerovat syntetické TDD"):
        store.save(int(syn_year), synthetic_tdd_year(int(syn_year)))
        st.success(f"Syntetické TDD {syn_year} uloženo.")
        st.rerun()

st.divider()

# ── Obecná nastavení ────────────────────────────────────────────────────────
st.subheader("Výchozí hodnoty a solver")
c1, c2, c3 = st.columns(3)
with c1:
    fx = st.number_input("Výchozí kurz CZK/EUR",
                         value=float(settings["fx_czk_eur"]), step=0.1,
                         format="%.2f")
    workers = st.number_input("Paralelní procesy (0 = auto)",
                              value=int(settings["workers"]), min_value=0,
                              max_value=16)
with c2:
    r1_tl = st.number_input("Run 1: time limit / solve [s]",
                            value=int(settings["run1_time_limit_s"]),
                            min_value=5, max_value=120)
    r1_gap = st.number_input("Run 1: MIP gap [-]",
                             value=float(settings["run1_gap_rel"]),
                             format="%.4f", step=0.001)
with c3:
    s_tl = st.number_input("Run 2/3: time limit [s]",
                           value=int(settings["solver_time_limit_s"]),
                           min_value=10, max_value=600)
    s_gap = st.number_input("Run 2/3: MIP gap [-]",
                            value=float(settings["solver_gap_rel"]),
                            format="%.4f", step=0.001)

fr_opts = [0.1, 0.25, 0.5, 0.75, 1.0]
fracs = st.multiselect("Výchozí kroky R-gridu (zlomky maxima)",
                       options=fr_opts,
                       default=[f for f in settings["r_grid_frac"]
                                if f in fr_opts])

c1, c2 = st.columns(2)
with c1:
    if st.button("💾 Uložit nastavení", type="primary"):
        save_settings({
            "fx_czk_eur": float(fx),
            "workers": int(workers),
            "run1_time_limit_s": int(r1_tl),
            "run1_gap_rel": float(r1_gap),
            "solver_time_limit_s": int(s_tl),
            "solver_gap_rel": float(s_gap),
            "r_grid_frac": sorted(float(f) for f in fracs) or
                DEFAULT_SETTINGS["r_grid_frac"],
        })
        st.success("Nastavení uloženo.")
with c2:
    if st.button("↩️ Obnovit výchozí"):
        save_settings(dict(DEFAULT_SETTINGS))
        st.rerun()
