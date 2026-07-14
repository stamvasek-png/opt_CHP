"""Numerická re-verifikace aFRR rezervačních omezení na vyřešeném plánu."""

import numpy as np
import pytest

from core.inputs import parse_inputs_xlsx
from core.model import (AfrrRequirement, RunMode, build_and_solve,
                        kgj_el_bounds)
from core.timegrid import N_BLOCKS

TOL = 1e-5


@pytest.fixture(scope="module")
def demo_series(demo_profile, grid, filled_workbook, tdd_store):
    filled_workbook.seek(0)
    series, _ = parse_inputs_xlsx(filled_workbook, grid, demo_profile,
                                  fx_czk_eur=25.0, tdd_store=tdd_store)
    return series


@pytest.fixture(scope="module")
def afrr_req():
    r_up = np.zeros(N_BLOCKS)
    r_dn = np.zeros(N_BLOCKS)
    r_up[4] = 0.3   # 16–20 h
    r_up[5] = 0.3   # 20–24 h
    r_dn[2] = 0.2   # 08–12 h
    return AfrrRequirement(r_up, r_dn, activation_h=1.0)


@pytest.fixture(scope="module")
def afrr_result(grid, demo_profile, demo_series, afrr_req):
    res = build_and_solve(grid, demo_profile, demo_series, afrr=afrr_req,
                          mode=RunMode.DA_PLAN, time_limit_s=120)
    assert res is not None and res.status == "Optimal"
    return res


def _reserve(res, site, aid, direction, n):
    col = f"{site}|{aid}|r_{direction}_mw"
    if res.reserve_alloc is not None and col in res.reserve_alloc.columns:
        return res.reserve_alloc[col].to_numpy()
    return np.zeros(n)


def test_coverage_equality(grid, demo_profile, afrr_result, afrr_req):
    n = grid.n
    up_tot = np.zeros(n)
    dn_tot = np.zeros(n)
    for s in demo_profile.sites:
        for k in s.kgjs:
            up_tot += _reserve(afrr_result, s.site_id, k.asset_id, "up", n)
            dn_tot += _reserve(afrr_result, s.site_id, k.asset_id, "dn", n)
        for b in s.bess_units:
            up_tot += _reserve(afrr_result, s.site_id, b.asset_id, "up", n)
            dn_tot += _reserve(afrr_result, s.site_id, b.asset_id, "dn", n)
        for e in s.eks:
            dn_tot += _reserve(afrr_result, s.site_id, e.asset_id, "dn", n)
    r_up_t = afrr_req.r_up_mw[grid.block_of]
    r_dn_t = afrr_req.r_dn_mw[grid.block_of]
    np.testing.assert_allclose(up_tot, r_up_t, atol=TOL)
    np.testing.assert_allclose(dn_tot, r_dn_t, atol=TOL)


def test_kgj_feasibility(grid, demo_profile, afrr_result):
    n = grid.n
    plan = afrr_result.plan
    for s in demo_profile.sites:
        for k in s.kgjs:
            e_min, e_max = kgj_el_bounds(k)
            e = plan[f"{s.site_id}|{k.asset_id}|e_el_mw"].to_numpy()
            on = plan[f"{s.site_id}|{k.asset_id}|on"].to_numpy()
            r_up = _reserve(afrr_result, s.site_id, k.asset_id, "up", n)
            r_dn = _reserve(afrr_result, s.site_id, k.asset_id, "dn", n)
            assert (e + r_up <= e_max * on + TOL).all()
            assert (e - r_dn >= e_min * on - TOL).all()
            # rezerva jen při běhu
            assert (r_up[on < 0.5] <= TOL).all()
            assert (r_dn[on < 0.5] <= TOL).all()


def test_bess_feasibility(grid, demo_profile, afrr_result, afrr_req):
    n = grid.n
    tau = afrr_req.activation_h
    plan = afrr_result.plan
    for s in demo_profile.sites:
        for b in s.bess_units:
            soc = plan[f"{s.site_id}|{b.asset_id}|soc_mwh"].to_numpy()
            cha = plan[f"{s.site_id}|{b.asset_id}|cha_mw"].to_numpy()
            dis = plan[f"{s.site_id}|{b.asset_id}|dis_mw"].to_numpy()
            r_up = _reserve(afrr_result, s.site_id, b.asset_id, "up", n)
            r_dn = _reserve(afrr_result, s.site_id, b.asset_id, "dn", n)
            assert ((dis - cha) + r_up <= b.bess_p + TOL).all()
            assert ((cha - dis) + r_dn <= b.bess_p + TOL).all()
            assert (soc >= r_up * tau / b.bess_eff - TOL).all()
            assert (b.bess_cap - soc >= r_dn * tau * b.bess_eff - TOL).all()


def test_ek_feasibility(grid, demo_profile, afrr_result):
    n = grid.n
    plan = afrr_result.plan
    for s in demo_profile.sites:
        for e in s.eks:
            ee_in = plan[f"{s.site_id}|{e.asset_id}|ee_in_mw"].to_numpy()
            r_dn = _reserve(afrr_result, s.site_id, e.asset_id, "dn", n)
            assert (r_dn <= e.ek_max / e.ek_eff - ee_in + TOL).all()


def test_reserved_capacity_lowers_profit(grid, demo_profile, demo_series,
                                         afrr_result):
    """Rezervace flexibility nesmí zisk zvýšit (opportunity cost ≥ 0)."""
    base = build_and_solve(grid, demo_profile, demo_series,
                           mode=RunMode.DA_PLAN, time_limit_s=120)
    assert base is not None
    assert afrr_result.profit <= base.profit + 1.0  # tolerance MIP gapu


def test_infeasible_reservation_returns_none(grid, demo_profile, demo_series):
    r_up = np.zeros(N_BLOCKS)
    r_up[0] = 50.0   # nesplnitelné (portfolio máx ~1.45 MW)
    res = build_and_solve(grid, demo_profile, demo_series,
                          afrr=AfrrRequirement(r_up, np.zeros(N_BLOCKS)),
                          mode=RunMode.DA_PLAN, time_limit_s=60)
    assert res is None


def test_redispatch_respects_reservation(grid, demo_profile, demo_series,
                                         afrr_req, afrr_result):
    """Rezervace platí i v re-dispatch režimu."""
    series = demo_series
    series.da_price_actual = series.da_price_pred.copy()
    res = build_and_solve(grid, demo_profile, series, afrr=afrr_req,
                          mode=RunMode.REDISPATCH,
                          nomination_mw=afrr_result.position_mw,
                          lambda_dev=0.0, time_limit_s=120)
    assert res is not None and res.status == "Optimal"
    assert res.reserve_alloc is not None
    r_up_t = afrr_req.r_up_mw[grid.block_of]
    up_cols = [c for c in res.reserve_alloc.columns if c.endswith("r_up_mw")]
    up_tot = res.reserve_alloc[up_cols].sum(axis=1).to_numpy()
    np.testing.assert_allclose(up_tot, r_up_t, atol=TOL)
