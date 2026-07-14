"""Exporty: nominace OTE (CSV/XLSX), tabulka aFRR nabídek, ekonomika dne.

Helpery formátování portovány z původního app.py (_wb_formats,
_safe_sheet, _write_sheet); přehled parametrů adaptován na Profile.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from .model import SolveResult
from .profiles import Profile
from .runs import Run1Result
from .timegrid import TimeGrid

# České popisky položek ekonomiky (sdílí UI i export)
ECO_LABELS = {
    "rev_heat": "Výnos teplo",
    "rev_export": "Výnos prodej EE (DA)",
    "rev_kgj_bonus": "Bonus fixní výkup KGJ",
    "rev_afrr_cap": "Výnos aFRR kapacita",
    "rev_nomination": "Zúčtování nominace (DA skutečná)",
    "rev_deviation": "Zúčtování odchylky",
    "cost_gas": "Náklad plyn (vč. distribuce)",
    "cost_ee_grid": "Náklad nákup EE z gridu",
    "cost_heat_import": "Náklad import tepla",
    "cost_start": "Náklady na starty KGJ",
    "cost_service": "Servis KGJ",
    "cost_bess_cycle": "Opotřebení BESS",
    "cost_shortfall": "Penalizace nedodání tepla",
    "cost_co2": "Náklad CO₂",
    "cost_dist_extra": "Extra distribuce (BESS/FVE)",
    "co2_total_t": "CO₂ celkem [t]",
    "profit_total": "Zisk celkem",
}


def _wb_formats(workbook):
    hdr = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
    num = workbook.add_format({'num_format': '#,##0.00'})
    txt = workbook.add_format({'num_format': '@'})
    grn = workbook.add_format({'bold': True, 'bg_color': '#C6EFCE', 'border': 1})
    return hdr, num, txt, grn


def _safe_sheet(name: str) -> str:
    """Odstraní znaky neplatné v Excel názvech listů, zkrátí na 31 znaků."""
    return re.sub(r'[/\\*?:\[\]]', '-', name)[:31]


def _write_sheet(writer, df, sheet_name, hdr_fmt, num_fmt, txt_fmt):
    safe = _safe_sheet(sheet_name)
    df.to_excel(writer, index=False, sheet_name=safe)
    ws = writer.sheets[safe]
    for col_idx, col_name in enumerate(df.columns):
        is_num = pd.api.types.is_numeric_dtype(df.iloc[:, col_idx])
        ws.set_column(col_idx, col_idx, 18, num_fmt if is_num else txt_fmt)
        ws.write(0, col_idx, col_name, hdr_fmt)


# ── Přehled parametrů profilu ────────────────────────────────────────────────

def build_parameters_df(profile: Profile) -> pd.DataFrame:
    """Přehled lokalit a parametrů profilu pro list 'Parametry'."""
    rows = []

    def add(site, cat, name, val):
        rows.append({"Lokalita": site, "Kategorie": cat,
                     "Parametr": name, "Hodnota": val})

    add("—", "Profil", "Název", profile.name)
    add("—", "Profil", "Trvání aktivace aFRR τ [h]", profile.afrr_activation_h)

    for s in profile.sites:
        add(s.name, "Síť", "Distribuce EE – nákup [€/MWh]", s.dist_ee_buy)
        add(s.name, "Síť", "Distribuce EE – prodej [€/MWh]", s.dist_ee_sell)
        add(s.name, "Síť", "Distribuce plyn [€/MWh]", s.gas_dist)
        add(s.name, "Síť", "Interní spotřeba bez distribuce",
            "ANO" if s.internal_ee_use else "NE")
        if s.grid_export_limit_mw is not None:
            add(s.name, "Síť", "Limit exportu [MW]", s.grid_export_limit_mw)
        if s.grid_import_limit_mw is not None:
            add(s.name, "Síť", "Limit importu [MW]", s.grid_import_limit_mw)
        if s.has_heat:
            add(s.name, "Teplo", "Prodejní cena tepla [€/MWh]", s.h_price)
            add(s.name, "Teplo", "Min. pokrytí poptávky [-]", s.h_cover)
            add(s.name, "Teplo", "Penalizace nedodání [€/MWh]",
                s.shortfall_penalty)
        for k in s.kgjs:
            cat = f"KGJ {k.name}"
            add(s.name, cat, "Jmenovitý tepelný výkon [MW]", k.k_th)
            add(s.name, cat, "Odvozený el. výkon [MW]", round(k.k_el, 4))
            add(s.name, cat, "η_th / η_el [-]", f"{k.k_eff_th} / {k.k_eff_el}")
            add(s.name, cat, "Min. zatížení [%]", round(k.k_min * 100, 1))
            add(s.name, cat, "Náklady na start [€/start]", k.k_start_cost)
            add(s.name, cat, "Min. doba běhu [h]", k.k_min_runtime_h)
            add(s.name, cat, "Servisní náklad [€/h]", k.k_service_cost)
            if k.var_eff:
                add(s.name, cat, "η_th / η_el při min. zátěži",
                    f"{k.eta_th_min} / {k.eta_el_min}")
            if k.gas_fix:
                add(s.name, cat, "Fixní cena plynu [€/MWh]", k.gas_fix_price)
            if k.ee_fix:
                add(s.name, cat, "Fixní výkupní cena EE [€/MWh]",
                    k.ee_fix_price)
            add(s.name, cat, "aFRR způsobilý", "ANO" if k.afrr_capable else "NE")
        for b in s.boilers:
            cat = f"Kotel {b.name}"
            add(s.name, cat, "Max. výkon [MW]", b.b_max)
            add(s.name, cat, "Účinnost [-]", b.boil_eff)
            if b.gas_fix:
                add(s.name, cat, "Fixní cena plynu [€/MWh]", b.gas_fix_price)
        for e in s.eks:
            cat = f"Elektrokotel {e.name}"
            add(s.name, cat, "Max. výkon [MW]", e.ek_max)
            add(s.name, cat, "Účinnost [-]", e.ek_eff)
            if e.ee_fix:
                add(s.name, cat, "Fixní cena EE [€/MWh]", e.ee_fix_price)
            add(s.name, cat, "aFRR způsobilý", "ANO" if e.afrr_capable else "NE")
        for x in s.tes_units:
            cat = f"TES {x.name}"
            add(s.name, cat, "Kapacita [MWh]", x.tes_cap)
            add(s.name, cat, "Ztráta [%/h]", x.tes_loss_pct_h)
            add(s.name, cat, "SoC start / min. konec [%]",
                f"{x.soc_start_frac * 100:.0f} / {x.soc_end_min_frac * 100:.0f}")
        for bb in s.bess_units:
            cat = f"BESS {bb.name}"
            add(s.name, cat, "Kapacita [MWh]", bb.bess_cap)
            add(s.name, cat, "Max. výkon [MW]", bb.bess_p)
            add(s.name, cat, "Účinnost [-]", bb.bess_eff)
            add(s.name, cat, "Opotřebení [€/MWh]", bb.bess_cycle_cost)
            add(s.name, cat, "SoC start / min. konec [%]",
                f"{bb.soc_start_frac * 100:.0f} / {bb.soc_end_min_frac * 100:.0f}")
            if bb.ee_fix:
                add(s.name, cat, "Fixní cena EE [€/MWh]", bb.ee_fix_price)
            add(s.name, cat, "aFRR způsobilý",
                "ANO" if bb.afrr_capable else "NE")
        for pv in s.pvs:
            cat = f"FVE {pv.name}"
            add(s.name, cat, "Instalovaný výkon [MW]", pv.installed_mw)
            add(s.name, cat, "Curtailment povolen",
                "ANO" if pv.allow_curtailment else "NE")
        for im in s.heat_imports:
            cat = f"Import tepla {im.name}"
            add(s.name, cat, "Max. výkon [MW]", im.imp_max)
            add(s.name, cat, "Cena [€/MWh]", im.imp_price)
        if s.consumption.mode == "tdd":
            add(s.name, "Spotřeba", "TDD třída", s.consumption.tdd_class)
            add(s.name, "Spotřeba", "Roční spotřeba [MWh]",
                s.consumption.annual_mwh)
        elif s.consumption.mode == "curve":
            add(s.name, "Spotřeba", "Zdroj", "vlastní křivka")
        if s.co2_price > 0:
            add(s.name, "CO₂", "Cena CO₂ [€/t]", s.co2_price)
    return pd.DataFrame(rows)


# ── Nominace ─────────────────────────────────────────────────────────────────

def nomination_csv(nomination: pd.DataFrame, delivery_date) -> bytes:
    """CSV nominace: MTU; čas od; pozice MW; energie MWh (UTF-8 BOM, ;)."""
    df = nomination.copy()
    df.insert(0, "den", str(delivery_date))
    return df.to_csv(index=False, sep=";", decimal=",",
                     float_format="%.3f").encode("utf-8-sig")


def nomination_xlsx(nomination: pd.DataFrame, delivery_date,
                    profile: Profile | None = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as wr:
        hdr, num, txt, _ = _wb_formats(wr.book)
        df = nomination.rename(columns={
            "mtu": "MTU", "cas_od": "Čas od", "pos_mw": "Pozice [MW]",
            "energie_mwh": "Energie [MWh]"})
        df.insert(0, "Den dodávky", str(delivery_date))
        _write_sheet(wr, df, "Nominace", hdr, num, txt)
        if profile is not None:
            _write_sheet(wr, build_parameters_df(profile), "Parametry",
                         hdr, num, txt)
    return buf.getvalue()


# ── aFRR nabídky ─────────────────────────────────────────────────────────────

def bids_xlsx(result: Run1Result, delivery_date, fx_czk_eur: float) -> bytes:
    """Tabulka doporučených nabídek do denní aukce ČEPS (EUR i CZK)."""
    lad = result.ladder.copy()
    df = pd.DataFrame({
        "Blok": lad["block_label"],
        "Směr": lad["direction_label"],
        "Kapacita R [MW]": lad["r_mw"],
        "Proveditelné": np.where(lad["feasible"], "ANO", "NE"),
        "Min. cena marginální [€/MW/h]": lad["min_price_marginal"],
        "Min. cena průměrná [€/MW/h]": lad["min_price_avg"],
        "Min. cena marginální [Kč/MW/h]":
            lad["min_price_marginal"] * fx_czk_eur,
        "Predikce clearing [€/MW/h]": lad["predicted_price"],
        "Doporučení": np.where(lad["recommend"], "NABÍDNOUT", "—"),
    })
    meta = pd.DataFrame({
        "Klíč": ["Den dodávky", "Zisk baseline (bez rezervace) [€]",
                 "Max. kapacita aFRR+ [MW]", "Max. kapacita aFRR− [MW]",
                 "Kurz CZK/EUR", "Počet solve", "Výpočet [s]"],
        "Hodnota": [str(delivery_date), round(result.baseline_profit, 2),
                    round(result.r_max_up, 3), round(result.r_max_dn, 3),
                    fx_czk_eur, result.solves, round(result.wall_s, 1)],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as wr:
        hdr, num, txt, grn = _wb_formats(wr.book)
        _write_sheet(wr, df, "Nabídky aFRR", hdr, num, txt)
        ws = wr.sheets["Nabídky aFRR"]
        for i, rec in enumerate(lad["recommend"].tolist()):
            if rec:
                ws.write(i + 1, len(df.columns) - 1, "NABÍDNOUT", grn)
        _write_sheet(wr, meta, "Info", hdr, num, txt)
    return buf.getvalue()


# ── Ekonomika a plán dne ─────────────────────────────────────────────────────

def economics_df(economics: dict) -> pd.DataFrame:
    """Breakdown ekonomiky s českými popisky (jen nenulové položky)."""
    rows = []
    order = list(ECO_LABELS)
    for key in order:
        if key not in economics:
            continue
        val = economics[key]
        if key != "profit_total" and abs(val) < 0.005:
            continue
        rows.append({"Položka": ECO_LABELS[key],
                     "Hodnota [€]": round(val, 2),
                     "Typ": ("výnos" if key.startswith("rev_") else
                             "náklad" if key.startswith("cost_") else "KPI")})
    return pd.DataFrame(rows)


def day_result_xlsx(result: SolveResult, grid: TimeGrid, profile: Profile,
                    title: str, compare_economics: dict | None = None,
                    ) -> bytes:
    """Workbook s plánem, ekonomikou a parametry (run 2 / run 3).

    compare_economics: ekonomika run2 pro srovnávací list v run3 exportu.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as wr:
        hdr, num, txt, _ = _wb_formats(wr.book)
        eco = economics_df(result.economics)
        _write_sheet(wr, eco, "Ekonomika", hdr, num, txt)
        plan = result.plan.reset_index()
        _write_sheet(wr, plan, _safe_sheet(title), hdr, num, txt)
        if result.reserve_alloc is not None:
            _write_sheet(wr, result.reserve_alloc.reset_index(),
                         "Rezervy aFRR", hdr, num, txt)
        if compare_economics:
            cmp_rows = []
            for key, label in ECO_LABELS.items():
                a = compare_economics.get(key)
                b = result.economics.get(key)
                if a is None and b is None:
                    continue
                cmp_rows.append({
                    "Položka": label,
                    "Plán 10:00 [€]": round(a, 2) if a is not None else None,
                    "Re-dispatch [€]": round(b, 2) if b is not None else None,
                    "Δ [€]": (round(b - a, 2)
                              if (a is not None and b is not None) else None)})
            _write_sheet(wr, pd.DataFrame(cmp_rows), "Srovnání", hdr, num, txt)
        _write_sheet(wr, build_parameters_df(profile), "Parametry",
                     hdr, num, txt)
    return buf.getvalue()
