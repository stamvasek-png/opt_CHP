import numpy as np
import pytest

from core.inputs import InputSeries, parse_inputs_xlsx
from core.model import RunMode, build_and_solve, kgj_el_bounds
from core.profiles import BESSParams, Profile, Site
from core.timegrid import N_BLOCKS


@pytest.fixture(scope="module")
def demo_series(demo_profile, grid, filled_workbook, tdd_store):
    filled_workbook.seek(0)
    series, _ = parse_inputs_xlsx(filled_workbook, grid, demo_profile,
                                  fx_czk_eur=25.0, tdd_store=tdd_store)
    return series


@pytest.fixture(scope="module")
def da_result(grid, demo_profile, demo_series):
    res = build_and_solve(grid, demo_profile, demo_series,
                          mode=RunMode.DA_PLAN, time_limit_s=60)
    assert res is not None
    return res


def test_optimal_status(da_result):
    assert da_result.status == "Optimal"


def test_heat_coverage(grid, demo_profile, da_result):
    site = demo_profile.sites[0]
    plan = da_result.plan
    dem = plan[f"{site.site_id}|heat_dem_mw"].to_numpy()
    delivered = plan[f"{site.site_id}|heat_delivered_mw"].to_numpy()
    short = plan[f"{site.site_id}|heat_shortfall_mw"].to_numpy()
    assert (delivered + short >= dem * site.h_cover - 1e-5).all()
    # při rozumné penalizaci se teplo dodává
    assert short.sum() < 1e-4


def test_ee_balance_per_site(grid, demo_profile, da_result, demo_series):
    plan = da_result.plan
    for s in demo_profile.sites:
        e_kgj = np.zeros(grid.n)
        for k in s.kgjs:
            e_kgj += plan[f"{s.site_id}|{k.asset_id}|e_el_mw"].to_numpy()
        pv = np.zeros(grid.n)
        for p in s.pvs:
            pv += plan[f"{s.site_id}|{p.asset_id}|used_mw"].to_numpy()
        b_dis = np.zeros(grid.n)
        b_cha = np.zeros(grid.n)
        for b in s.bess_units:
            b_dis += plan[f"{s.site_id}|{b.asset_id}|dis_mw"].to_numpy()
            b_cha += plan[f"{s.site_id}|{b.asset_id}|cha_mw"].to_numpy()
        ee_ek = np.zeros(grid.n)
        for e in s.eks:
            ee_ek += plan[f"{s.site_id}|{e.asset_id}|ee_in_mw"].to_numpy()
        cons = plan[f"{s.site_id}|cons_mw"].to_numpy()
        exp = plan[f"{s.site_id}|export_mw"].to_numpy()
        imp = plan[f"{s.site_id}|import_mw"].to_numpy()
        np.testing.assert_allclose(
            e_kgj + pv + b_dis + imp, ee_ek + b_cha + cons + exp, atol=1e-5)


def test_position_is_sum_of_sites(demo_profile, da_result):
    plan = da_result.plan
    tot = np.zeros(len(plan))
    for s in demo_profile.sites:
        tot += plan[f"{s.site_id}|export_mw"].to_numpy()
        tot -= plan[f"{s.site_id}|import_mw"].to_numpy()
    np.testing.assert_allclose(da_result.position_mw, tot, atol=1e-6)


def test_economics_consistency(da_result):
    eco = da_result.economics
    revenue = sum(v for k, v in eco.items() if k.startswith("rev_"))
    costs = sum(v for k, v in eco.items() if k.startswith("cost_"))
    assert eco["profit_total"] == pytest.approx(revenue - costs)
    # objective (optimalizovaná část) ≈ profit v DA_PLAN režimu
    assert eco["profit_total"] == pytest.approx(da_result.objective, rel=1e-4)


def test_kgj_min_load_when_on(grid, demo_profile, da_result):
    s = demo_profile.sites[0]
    k = s.kgjs[0]
    plan = da_result.plan
    q = plan[f"{s.site_id}|{k.asset_id}|q_th_mw"].to_numpy()
    on = plan[f"{s.site_id}|{k.asset_id}|on"].to_numpy()
    assert ((q >= k.k_min * k.k_th * on - 1e-5)
            & (q <= k.k_th * on + 1e-5)).all()
    # min. doba běhu 4 h = 16 MTU: každý souvislý běh musí být >= 16 MTU
    runs, cur = [], 0
    for v in on > 0.5:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    interior = runs[:-1] if (cur and runs) else runs  # běh u konce dne může být kratší? ne — min runtime platí od startu
    assert all(r >= 16 for r in runs if r > 0) or runs == []


def toy_bess_profile() -> Profile:
    return Profile(
        profile_id="toy", name="Toy BESS",
        sites=[Site(site_id="s1", name="S1", kind="fve_bess",
                    dist_ee_buy=0.0, dist_ee_sell=0.0, gas_dist=0.0,
                    bess_units=[BESSParams(
                        asset_id="b1", bess_cap=2.0, bess_p=1.0, bess_eff=1.0,
                        bess_cycle_cost=0.0, soc_start_frac=0.0,
                        soc_end_min_frac=0.0)])])


def toy_series(grid) -> InputSeries:
    n = grid.n
    da = np.zeros(n)
    da[n // 2:] = 100.0
    return InputSeries(
        grid=grid, da_price_pred=da, imb_price_pred=da.copy(),
        gas_price=np.full(n, 35.0),
        afrr_cap_price_up=np.zeros(N_BLOCKS),
        afrr_cap_price_dn=np.zeros(N_BLOCKS))


def test_toy_bess_arbitrage_analytic(grid):
    """BESS 2 MWh / 1 MW, eff=1, ceny 0→100: analytický zisk = 200 EUR."""
    profile = toy_bess_profile()
    series = toy_series(grid)
    res = build_and_solve(grid, profile, series, mode=RunMode.DA_PLAN,
                          time_limit_s=30)
    assert res is not None and res.status == "Optimal"
    assert res.profit == pytest.approx(200.0, abs=1e-3)
    plan = res.plan
    soc = plan["s1|b1|soc_mwh"].to_numpy()
    assert soc.max() == pytest.approx(2.0, abs=1e-5)
    assert soc[-1] == pytest.approx(0.0, abs=1e-5)


def test_toy_redispatch_matches_daplan(grid):
    """REDISPATCH s actual==pred, imb==DA a nominací z DA plánu ⇒ stejný zisk."""
    profile = toy_bess_profile()
    series = toy_series(grid)
    da_res = build_and_solve(grid, profile, series, mode=RunMode.DA_PLAN)
    series.da_price_actual = series.da_price_pred.copy()
    re_res = build_and_solve(grid, profile, series, mode=RunMode.REDISPATCH,
                             nomination_mw=da_res.position_mw, lambda_dev=1.0)
    assert re_res is not None and re_res.status == "Optimal"
    assert re_res.profit == pytest.approx(da_res.profit, abs=1e-3)
    assert np.abs(re_res.plan["deviation_mw"].to_numpy()).sum() < 1e-4


def test_kgj_el_bounds(demo_profile):
    k = demo_profile.sites[0].kgjs[0]
    e_min, e_max = kgj_el_bounds(k)
    assert e_max == pytest.approx(k.k_el, rel=1e-9)
    assert e_min == pytest.approx(k.k_el * k.k_min, rel=1e-9)
