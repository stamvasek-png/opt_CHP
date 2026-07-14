import io

import numpy as np
import pandas as pd
import pytest

from core.inputs import (InputError, InputSeries, build_template_xlsx,
                         parse_actual_da, parse_inputs_xlsx, pv_columns)

FX = 25.0


def test_template_sheets(demo_profile, grid):
    data = build_template_xlsx(demo_profile, grid)
    xls = pd.ExcelFile(io.BytesIO(data))
    assert {"Meta", "DA_ceny", "Odchylka", "aFRR_ceny", "FVE_vyroba",
            "Spotreba", "Teplo", "Plyn"} <= set(xls.sheet_names)
    pv = xls.parse("FVE_vyroba")
    # obě FVE lokality mají 1 FVE → sloupce = site_id
    assert {"fve_brno", "fve_zlin"} <= set(pv.columns)
    assert len(pv) == grid.n
    heat = xls.parse("Teplo")
    assert "teplarna_a" in heat.columns


def test_parse_filled_workbook(demo_profile, grid, filled_workbook, tdd_store,
                               curves):
    filled_workbook.seek(0)
    series, report = parse_inputs_xlsx(filled_workbook, grid, demo_profile,
                                       fx_czk_eur=FX, tdd_store=tdd_store)
    assert isinstance(series, InputSeries)
    np.testing.assert_allclose(series.da_price_pred, curves["da"])
    # odchylka a aFRR byly v CZK → převedeno na EUR
    np.testing.assert_allclose(series.imb_price_pred, curves["imb"], rtol=1e-9)
    np.testing.assert_allclose(series.afrr_cap_price_up, curves["afrr_up"])
    # FVE: 2 assety (brno, zlin)
    assert set(series.pv_forecast) == {("fve_brno", "fve1"), ("fve_zlin", "fve1")}
    np.testing.assert_allclose(series.pv_forecast[("fve_brno", "fve1")],
                               curves["pv_shape"] * 1.0)
    # spotřeba: zlin z křivky, brno z TDD (report to zmiňuje)
    assert "fve_zlin" in series.consumption and "fve_brno" in series.consumption
    assert any("TDD4" in r for r in report)
    assert "teplarna_a" in series.heat_demand


def test_parse_hourly_resample(demo_profile, grid, tdd_store):
    """Hodinové řady (24 řádků) se rozpadnou; FVE interpolací."""
    hours = np.arange(24, dtype=float)
    pv_hourly = np.clip(np.sin((hours - 6) / 14 * np.pi), 0, None)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as wr:
        base = {"MTU": np.arange(1, 25), "Cas od": [f"{h:02.0f}:00" for h in hours]}
        pd.DataFrame({**base, "Cena DA": np.full(24, 80.0)}).to_excel(
            wr, sheet_name="DA_ceny", index=False)
        pd.DataFrame({**base, "Zuctovaci cena": np.full(24, 2000.0)}).to_excel(
            wr, sheet_name="Odchylka", index=False)
        pd.DataFrame({"Blok": list(range(6)),
                      "aFRR+ [c]": np.full(6, 250.0),
                      "aFRR- [c]": np.full(6, 125.0)}).to_excel(
            wr, sheet_name="aFRR_ceny", index=False)
        pd.DataFrame({**base, "fve_brno": pv_hourly,
                      "fve_zlin": pv_hourly * 0.5}).to_excel(
            wr, sheet_name="FVE_vyroba", index=False)
        pd.DataFrame({**base, "fve_zlin": np.full(24, 0.02)}).to_excel(
            wr, sheet_name="Spotreba", index=False)
        pd.DataFrame({**base, "teplarna_a": np.full(24, 2.0)}).to_excel(
            wr, sheet_name="Teplo", index=False)
        pd.DataFrame({**base, "Cena plyn": np.full(24, 35.0)}).to_excel(
            wr, sheet_name="Plyn", index=False)
    buf.seek(0)
    series, report = parse_inputs_xlsx(buf, grid, demo_profile, fx_czk_eur=FX,
                                       tdd_store=tdd_store)
    assert len(series.da_price_pred) == 96
    assert series.da_price_pred[0] == 80.0
    assert series.imb_price_pred[0] == pytest.approx(80.0)   # 2000/25
    assert series.afrr_cap_price_up[0] == pytest.approx(10.0)  # 250/25
    # FVE interpolovaná — mezi hodinami hladký přechod (žádné schody)
    pv = series.pv_forecast[("fve_brno", "fve1")]
    mid_jump = np.abs(np.diff(pv))
    assert mid_jump.max() < 0.15
    assert any("interpolace" in r for r in report)
    assert any("opakování" in r for r in report)


def test_parse_missing_sheet_raises(demo_profile, grid, tdd_store):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as wr:
        pd.DataFrame({"Cena DA": [1.0]}).to_excel(wr, sheet_name="DA_ceny",
                                                  index=False)
    buf.seek(0)
    with pytest.raises(InputError):
        parse_inputs_xlsx(buf, grid, demo_profile, fx_czk_eur=FX,
                          tdd_store=tdd_store)


def test_parse_wrong_length_raises(demo_profile, grid, tdd_store,
                                   filled_workbook):
    """Workbook pro jiný den (jiný počet řádků) musí selhat srozumitelně."""
    import datetime as dt
    from core.timegrid import make_grid
    g_dst = make_grid(dt.date(2026, 3, 29))  # 92 MTU
    filled_workbook.seek(0)
    with pytest.raises(InputError, match="čekám 92"):
        parse_inputs_xlsx(filled_workbook, g_dst, demo_profile, fx_czk_eur=FX,
                          tdd_store=tdd_store)


def test_parse_actual_da(grid, actual_da_file, curves):
    actual_da_file.seek(0)
    actual_da_file.name = "skutecne.xlsx"
    vals = parse_actual_da(actual_da_file, grid, fx_czk_eur=FX, currency="EUR")
    assert len(vals) == grid.n
    hours = np.array([ts.hour for ts in grid.index])
    np.testing.assert_allclose(vals[hours >= 17],
                               curves["da"][hours >= 17] + 30.0)


def test_inputs_roundtrip_frame(demo_profile, grid, filled_workbook, tdd_store):
    filled_workbook.seek(0)
    series, _ = parse_inputs_xlsx(filled_workbook, grid, demo_profile,
                                  fx_czk_eur=FX, tdd_store=tdd_store)
    df = series.to_frame()
    back = InputSeries.from_frame(df, grid)
    np.testing.assert_allclose(back.da_price_pred, series.da_price_pred)
    np.testing.assert_allclose(back.afrr_cap_price_dn, series.afrr_cap_price_dn)
    assert set(back.pv_forecast) == set(series.pv_forecast)
    np.testing.assert_allclose(back.heat_demand["teplarna_a"],
                               series.heat_demand["teplarna_a"])
