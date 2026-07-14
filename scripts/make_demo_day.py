"""Vytvoří syntetická demo data pro ruční QA i testy.

- uloží syntetické TDD koeficienty pro rok dodávky,
- vygeneruje vyplněný workbook „Vstupy D+1" pro demo profil,
- volitelně vygeneruje soubor skutečných DA cen (run 3).

Použití:  python -m scripts.make_demo_day [YYYY-MM-DD] [výstupní_adresář]
"""

from __future__ import annotations

import datetime as dt
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.inputs import (BLOCK_LABELS, N_BLOCKS, SHEET_AFRR, SHEET_CONS,
                         SHEET_DA, SHEET_GAS, SHEET_HEAT, SHEET_IMB, SHEET_META,
                         SHEET_PV, build_template_xlsx, consumption_curve_sites,
                         heat_sites, pv_columns)
from core.profiles import Profile, load_profile
from core.tdd import TddStore, synthetic_tdd_year
from core.timegrid import TimeGrid, make_grid


def synthetic_curves(grid: TimeGrid, seed: int = 7) -> dict:
    """Syntetické křivky dne: DA cena s večerní špičkou a záporným polednem,
    odchylka kolem DA, FVE zvonovka, teplo, plyn."""
    rng = np.random.default_rng(seed)
    n = grid.n
    hours = np.array([ts.hour + ts.minute / 60 for ts in grid.index])

    da = 60 + 40 * np.exp(-0.5 * ((hours - 19.5) / 1.8) ** 2) \
        + 25 * np.exp(-0.5 * ((hours - 7.5) / 1.5) ** 2) \
        - 75 * np.exp(-0.5 * ((hours - 13.0) / 1.2) ** 2)
    da += rng.normal(0, 2, n)

    imb = da + rng.normal(0, 15, n)
    imb[(hours >= 19) & (hours < 20)] += 60   # večerní nedostatek

    pv_shape = np.clip(np.sin((hours - 6) / 14 * np.pi), 0, None) ** 1.5
    pv_shape[(hours < 6) | (hours > 20)] = 0.0

    heat = 1.8 + 0.8 * np.exp(-0.5 * ((hours - 6.5) / 2.0) ** 2) \
        + 0.6 * np.exp(-0.5 * ((hours - 20.0) / 2.5) ** 2)

    gas = np.full(n, 35.0)
    cons = 0.02 + 0.015 * np.exp(-0.5 * ((hours - 19.0) / 2.5) ** 2)

    afrr_up = np.array([8.0, 9.0, 12.0, 14.0, 18.0, 11.0])   # EUR/MW/h
    afrr_dn = np.array([5.0, 4.0, 7.0, 9.0, 10.0, 6.0])
    return {"da": da, "imb": imb, "pv_shape": pv_shape, "heat": heat,
            "gas": gas, "cons": cons, "afrr_up": afrr_up, "afrr_dn": afrr_dn}


def make_filled_workbook(profile: Profile, grid: TimeGrid,
                         curves: dict | None = None) -> bytes:
    """Vyplněný workbook Vstupy D+1 (ceny v defaultních měnách: odchylka
    a aFRR v CZK s kurzem 25.0)."""
    curves = curves or synthetic_curves(grid)
    fx = 25.0
    mtu = np.arange(1, grid.n + 1)
    times = grid.times_from()
    buf = io.BytesIO()

    def interval_df(data: dict) -> pd.DataFrame:
        return pd.DataFrame({"MTU": mtu, "Cas od": times, **data})

    with pd.ExcelWriter(buf, engine="xlsxwriter") as wr:
        pd.DataFrame({
            "Klíč": ["Den dodávky", "Profil", "Počet MTU"],
            "Hodnota": [grid.delivery_date.isoformat(), profile.profile_id, grid.n],
        }).to_excel(wr, sheet_name=SHEET_META, index=False)
        interval_df({"Cena DA": curves["da"]}).to_excel(
            wr, sheet_name=SHEET_DA, index=False)
        interval_df({"Zuctovaci cena": curves["imb"] * fx}).to_excel(
            wr, sheet_name=SHEET_IMB, index=False)
        pd.DataFrame({
            "Blok": BLOCK_LABELS,
            "aFRR+ [cena/MW/h]": curves["afrr_up"] * fx,
            "aFRR- [cena/MW/h]": curves["afrr_dn"] * fx,
        }).to_excel(wr, sheet_name=SHEET_AFRR, index=False)
        pv_data = {}
        for site_id, pv_id, col in pv_columns(profile):
            site = next(s for s in profile.sites if s.site_id == site_id)
            pv = next(a for a in site.pvs if a.asset_id == pv_id)
            pv_data[col] = curves["pv_shape"] * pv.installed_mw
        if pv_data:
            interval_df(pv_data).to_excel(wr, sheet_name=SHEET_PV, index=False)
        cons_data = {sid: curves["cons"] for sid in consumption_curve_sites(profile)}
        if cons_data:
            interval_df(cons_data).to_excel(wr, sheet_name=SHEET_CONS, index=False)
        heat_data = {sid: curves["heat"] for sid in heat_sites(profile)}
        if heat_data:
            interval_df(heat_data).to_excel(wr, sheet_name=SHEET_HEAT, index=False)
        interval_df({"Cena plyn": curves["gas"]}).to_excel(
            wr, sheet_name=SHEET_GAS, index=False)
    return buf.getvalue()


def make_actual_da_file(grid: TimeGrid, curves: dict | None = None,
                        shift_evening_eur: float = 30.0) -> bytes:
    """Soubor skutečných DA cen: predikce + posun večera (pro run 3 demo)."""
    curves = curves or synthetic_curves(grid)
    hours = np.array([ts.hour for ts in grid.index])
    actual = curves["da"].copy()
    actual[hours >= 17] += shift_evening_eur
    buf = io.BytesIO()
    pd.DataFrame({"MTU": np.arange(1, grid.n + 1),
                  "Cas od": grid.times_from(),
                  "Cena DA": actual}).to_excel(buf, index=False)
    return buf.getvalue()


def main(date_str: str | None = None, out_dir: str | None = None) -> None:
    date = dt.date.fromisoformat(date_str) if date_str else \
        dt.date.today() + dt.timedelta(days=1)
    out = Path(out_dir) if out_dir else Path(__file__).resolve().parent.parent / "data"
    out.mkdir(parents=True, exist_ok=True)

    store = TddStore()
    if date.year not in store.available_years():
        store.save(date.year, synthetic_tdd_year(date.year))
        print(f"Syntetické TDD {date.year} → data/tdd/")

    profile = load_profile("demo")
    grid = make_grid(date)
    wb = make_filled_workbook(profile, grid)
    wb_path = out / f"Vstupy_D1_{date.isoformat()}_demo.xlsx"
    wb_path.write_bytes(wb)
    print(f"Vyplněný workbook → {wb_path}")

    da_path = out / f"Skutecne_DA_{date.isoformat()}_demo.xlsx"
    da_path.write_bytes(make_actual_da_file(grid))
    print(f"Skutečné DA ceny → {da_path}")

    tpl_path = out / f"Vstupy_D1_{date.isoformat()}_sablona.xlsx"
    tpl_path.write_bytes(build_template_xlsx(profile, grid))
    print(f"Prázdná šablona → {tpl_path}")


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
