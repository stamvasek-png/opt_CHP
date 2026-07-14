import io

import numpy as np
import pandas as pd
import pytest

from core.export import (bids_xlsx, build_parameters_df, day_result_xlsx,
                         economics_df, nomination_csv, nomination_xlsx)
from core.inputs import parse_inputs_xlsx
from core.runs import AuctionResults, run1_afrr_bids, run2_da_plan

FX = 25.0


@pytest.fixture(scope="module")
def demo_series(demo_profile, grid, filled_workbook, tdd_store):
    filled_workbook.seek(0)
    series, _ = parse_inputs_xlsx(filled_workbook, grid, demo_profile,
                                  fx_czk_eur=FX, tdd_store=tdd_store)
    return series


@pytest.fixture(scope="module")
def res2(grid, demo_profile, demo_series):
    auction = AuctionResults()
    auction.up_mw[4] = 0.3
    auction.up_price[4] = 15.0
    res = run2_da_plan(grid, demo_profile, demo_series, auction,
                       time_limit_s=120)
    assert res is not None
    return res


def _nomination(res, grid):
    return pd.DataFrame({
        "mtu": np.arange(1, grid.n + 1),
        "cas_od": grid.times_from(),
        "pos_mw": res.position_mw,
        "energie_mwh": res.position_mw * grid.dt_h,
    })


def test_nomination_csv(grid, res2):
    data = nomination_csv(_nomination(res2, grid), grid.delivery_date)
    text = data.decode("utf-8-sig")
    lines = text.strip().splitlines()
    assert lines[0] == "den;mtu;cas_od;pos_mw;energie_mwh"
    assert len(lines) == grid.n + 1
    assert lines[1].startswith("2026-07-15;1;00:00;")


def test_nomination_xlsx(grid, res2, demo_profile):
    data = nomination_xlsx(_nomination(res2, grid), grid.delivery_date,
                           demo_profile)
    xls = pd.ExcelFile(io.BytesIO(data))
    assert {"Nominace", "Parametry"} <= set(xls.sheet_names)
    nom = xls.parse("Nominace")
    assert len(nom) == grid.n
    np.testing.assert_allclose(nom["Pozice [MW]"].to_numpy(),
                               res2.position_mw, atol=1e-6)


def test_bids_xlsx(grid, demo_profile, demo_series):
    r1 = run1_afrr_bids(grid, demo_profile, demo_series, r_grid_frac=(1.0,),
                        time_limit_s=15, workers=4)
    data = bids_xlsx(r1, grid.delivery_date, FX)
    xls = pd.ExcelFile(io.BytesIO(data))
    bids = xls.parse("Nabídky aFRR")
    assert len(bids) == 12  # 6 bloků × 2 směry × 1 krok
    assert "Min. cena marginální [Kč/MW/h]" in bids.columns
    feas = bids[bids["Proveditelné"] == "ANO"]
    np.testing.assert_allclose(
        feas["Min. cena marginální [Kč/MW/h]"].to_numpy(),
        feas["Min. cena marginální [€/MW/h]"].to_numpy() * FX, rtol=1e-9)
    info = xls.parse("Info")
    assert "Zisk baseline (bez rezervace) [€]" in info["Klíč"].tolist()


def test_economics_df(res2):
    df = economics_df(res2.economics)
    assert "Zisk celkem" in df["Položka"].tolist()
    assert "Výnos aFRR kapacita" in df["Položka"].tolist()
    profit = df.loc[df["Položka"] == "Zisk celkem", "Hodnota [€]"].iloc[0]
    assert profit == pytest.approx(res2.profit, abs=0.01)


def test_day_result_xlsx_with_compare(grid, demo_profile, res2):
    fake_run3_eco = dict(res2.economics)
    fake_run3_eco["rev_deviation"] = 12.34
    res2.economics, backup = fake_run3_eco, res2.economics
    try:
        data = day_result_xlsx(res2, grid, demo_profile, "Plán",
                               compare_economics=backup)
    finally:
        res2.economics = backup
    xls = pd.ExcelFile(io.BytesIO(data))
    assert {"Ekonomika", "Plán", "Srovnání", "Parametry",
            "Rezervy aFRR"} <= set(xls.sheet_names)
    cmp_df = xls.parse("Srovnání")
    assert {"Plán 10:00 [€]", "Re-dispatch [€]", "Δ [€]"} <= set(cmp_df.columns)


def test_parameters_df(demo_profile):
    df = build_parameters_df(demo_profile)
    assert (df["Lokalita"] == "Teplárna A").any()
    assert (df["Parametr"] == "TDD třída").any()
    assert (df["Kategorie"].str.startswith("KGJ")).any()
