"""End-to-end test celého denního workflow: run1 → aukce → run2 →
zmrazení nominace → (restart) → run3."""

import datetime as dt

import numpy as np
import pytest

from core.inputs import parse_actual_da, parse_inputs_xlsx
from core.runs import (AuctionResults, portfolio_reserve_capability,
                       run1_afrr_bids, run2_da_plan, run3_redispatch)
from core.timegrid import N_BLOCKS, make_grid
from core.trading_day import StageError, TradingDayStore
from scripts.make_demo_day import make_filled_workbook

DELIVERY = dt.date(2026, 7, 15)
FX = 25.0


@pytest.fixture(scope="module")
def demo_series(demo_profile, grid, filled_workbook, tdd_store):
    filled_workbook.seek(0)
    series, report = parse_inputs_xlsx(filled_workbook, grid, demo_profile,
                                       fx_czk_eur=FX, tdd_store=tdd_store)
    return series, report


@pytest.fixture(scope="module")
def run1_result(grid, demo_profile, demo_series):
    series, _ = demo_series
    return run1_afrr_bids(grid, demo_profile, series,
                          r_grid_frac=(0.5, 1.0), time_limit_s=20,
                          gap_rel=0.001, workers=4)


def test_run1_ladder_structure(run1_result, demo_profile):
    lad = run1_result.ladder
    r_up, r_dn = portfolio_reserve_capability(demo_profile)
    assert run1_result.r_max_up == pytest.approx(r_up)
    assert r_up > 0 and r_dn > r_up  # EK přidává jen záporný směr
    # 6 bloků × 2 směry × 2 kroky R
    assert len(lad) == 24
    # bloky 1–5 jsou plně proveditelné; blok 0 aFRR+ při plném R_max je
    # oprávněně nesplnitelný (BESS startuje s 20% SoC a do půlnoci se
    # nestihne nabít na energii pro plnou rezervu)
    assert lad[lad["block"] > 0]["feasible"].all()
    assert lad[(lad["block"] == 0) & (lad["direction"] == "dn")]["feasible"].all()
    feas = lad[lad["feasible"]]
    assert (feas["profit_eur"] <= run1_result.baseline_profit + 1e-6).all()


def test_run1_prices_sane(run1_result):
    lad = run1_result.ladder[run1_result.ladder["feasible"]]
    # opportunity cost nezáporný (rezervace nemůže zisk zvýšit)
    assert (lad["min_price_avg"] >= -0.5).all()
    # průměrná cena neklesá s R (konvexita) — tolerance na MIP gap
    for (b, d), g in lad.groupby(["block", "direction"]):
        g = g.sort_values("r_mw")
        avg = g["min_price_avg"].to_numpy()
        assert (np.diff(avg) >= -0.75).all(), (b, d, avg)


def test_run1_infeasible_r_flagged(grid, demo_profile, demo_series):
    series, _ = demo_series
    res = run1_afrr_bids(grid, demo_profile, series, r_grid_frac=(3.0,),
                         time_limit_s=10, workers=4)
    assert not res.ladder["feasible"].any()
    assert res.ladder["min_price_avg"].isna().all()


def test_full_day_workflow(tmp_path, grid, demo_profile, demo_series,
                           run1_result, actual_da_file, curves):
    series, report = demo_series
    store = TradingDayStore(base_dir=tmp_path)

    # založení dne + vstupy
    day = store.open(DELIVERY, profile=demo_profile, fx_czk_eur=FX)
    with pytest.raises(StageError):
        day.assert_stage("run1")
    day.save_inputs(series, report)
    day.assert_stage("run1")

    # run 1
    day.save_run1(run1_result)
    assert day.status["run1_done"]

    # aukce: výhra 0.3 MW aFRR+ v blocích 16–24 h za 15 EUR/MW/h
    auction = AuctionResults()
    auction.up_mw[4] = auction.up_mw[5] = 0.3
    auction.up_price[4] = auction.up_price[5] = 15.0
    day.set_auction(auction)

    # run 2 + nominace
    res2 = run2_da_plan(grid, demo_profile, series, auction, time_limit_s=120)
    assert res2 is not None and res2.status == "Optimal"
    assert res2.economics["rev_afrr_cap"] == pytest.approx(0.3 * 15.0 * 8)
    day.save_run2(res2)
    nom = day.freeze_nomination()
    np.testing.assert_allclose(nom["pos_mw"].to_numpy(), res2.position_mw,
                               atol=1e-9)
    np.testing.assert_allclose(nom["energie_mwh"].to_numpy(),
                               res2.position_mw * 0.25, atol=1e-9)
    with pytest.raises(StageError):
        day.save_run2(res2)   # zmrazeno → nelze přepsat

    # ── simulovaný restart aplikace ──
    store2 = TradingDayStore(base_dir=tmp_path)
    day2 = store2.open(DELIVERY)
    assert day2.status["nomination_frozen"]
    assert day2.profile.profile_id == demo_profile.profile_id
    assert day2.auction is not None and day2.auction.any_won
    nom2 = day2.load_nomination()
    np.testing.assert_allclose(nom2["pos_mw"].to_numpy(), res2.position_mw,
                               atol=1e-9)
    series2 = day2.load_inputs()
    np.testing.assert_allclose(series2.da_price_pred, series.da_price_pred)

    # run 3a: actual == pred, imb == pred DA, λ=0 → zisk ≈ run2
    s_eq = day2.load_inputs()
    s_eq.imb_price_pred = s_eq.da_price_pred.copy()
    s_eq.da_price_actual = s_eq.da_price_pred.copy()
    res3a = run3_redispatch(grid, day2.profile, s_eq, day2.auction,
                            nomination_mw=nom2["pos_mw"].to_numpy(),
                            lambda_dev=0.0, time_limit_s=120, gap_rel=0.001)
    assert res3a is not None and res3a.status == "Optimal"
    assert res3a.profit == pytest.approx(res2.profit, rel=2e-3)

    # run 3b: skutečné ceny s posunem večera, syntetická odchylka
    actual_da_file.seek(0)
    actual_da_file.name = "skutecne.xlsx"
    actual = parse_actual_da(actual_da_file, grid, fx_czk_eur=FX)
    day2.save_actual_da(actual)
    s3 = day2.load_inputs()
    s3.da_price_actual = day2.load_actual_da()
    res3b = run3_redispatch(grid, day2.profile, s3, day2.auction,
                            nomination_mw=nom2["pos_mw"].to_numpy(),
                            lambda_dev=5.0, time_limit_s=120, gap_rel=0.001)
    assert res3b is not None and res3b.status == "Optimal"
    day2.save_run3(res3b, lambda_dev=5.0)
    assert "deviation_mw" in res3b.plan.columns
    # re-optimalizace odchylek nesmí být horší než držení plánu bez odchylek
    # (λ=1e5 prakticky zakáže odchylku → ex-post ekonomika zmrazeného plánu)
    res3_frozen = run3_redispatch(grid, day2.profile, s3, day2.auction,
                                  nomination_mw=nom2["pos_mw"].to_numpy(),
                                  lambda_dev=1e5, time_limit_s=120,
                                  gap_rel=0.001)
    assert res3_frozen is not None
    assert np.abs(res3_frozen.plan["deviation_mw"].to_numpy()).sum() < 1e-3
    assert res3b.profit >= res3_frozen.profit - abs(res3_frozen.profit) * 3e-3

    # historie
    days = store2.list_days()
    assert len(days) == 1
    assert days[0]["status"]["run3_done"]
    assert days[0]["profit_run2"] == pytest.approx(res2.profit)


def test_dst_days_solve(demo_profile, tdd_store):
    """Run 2 na dnech přechodu času (92 a 100 MTU)."""
    import io
    for date in (dt.date(2026, 3, 29), dt.date(2026, 10, 25)):
        g = make_grid(date)
        wb = io.BytesIO(make_filled_workbook(demo_profile, g))
        series, _ = parse_inputs_xlsx(wb, g, demo_profile, fx_czk_eur=FX,
                                      tdd_store=tdd_store)
        res = run2_da_plan(g, demo_profile, series, AuctionResults(),
                           time_limit_s=120)
        assert res is not None and res.status == "Optimal"
        assert len(res.position_mw) == g.n
