"""Testy správnosti výpočtu jádra — nezávislé na rampách.

Těžiště je na invariantech, které se dají snadno tiše porušit: reportovaný zisk
se počítá podruhé z výsledkového rámce, ne z LP, takže se může rozejít
s optimem, aniž by cokoli spadlo.
"""

import datetime as dt

import pytest

from conftest import (K_TH, SOLVER_TIME_LIMIT, make_df, make_params, make_uses,
                      with_ramp)
from opt_core import (compute_linear_fuel_params, create_profile_constraints,
                      get_kgj_fix_price, run_optimization_with_profile)


def solve(params, df=None, uses=None, **kw):
    r = run_optimization_with_profile(
        df=make_df() if df is None else df,
        params=params, uses=make_uses() if uses is None else uses,
        time_limit=SOLVER_TIME_LIMIT, **kw)
    assert r is not None, "solver nenašel řešení"
    return r


# ── Shoda reportu s optimem ──────────────────────────────────────────

@pytest.mark.parametrize('label, params, df', [
    ('bez rampy', make_params(), make_df()),
    ('s rampou', with_ramp(make_params(), 12.7, 8.9), make_df()),
    # KGJ nad poptávkou ⇒ přebytek tepla se zahazuje. Právě tady se report
    # rozcházel s objective, protože zahozené teplo se neodečítalo od tržby.
    ('se zahazováním tepla', make_params(), make_df(heat_demand=0.1 * K_TH)),
    ('zahazování + rampa', with_ramp(make_params(), 12.7, 8.9),
     make_df(heat_demand=0.1 * K_TH)),
])
def test_hourly_profit_matches_lp_objective(label, params, df):
    r = solve(params, df=df, profile_type='custom', custom_hours=[1, 2, 3])
    assert r['total_profit'] == pytest.approx(r['lp_objective'], abs=1e-6), label


def test_heat_revenue_excludes_dumped_heat():
    """Zahozené teplo se neprodá, takže nesmí vstoupit do tržby za teplo."""
    p = make_params()
    r = solve(p, df=make_df(heat_demand=0.1 * K_TH),
              profile_type='custom', custom_hours=[1, 2, 3])
    res = r['res']

    dumped = res['Zahozené teplo [MW]']
    assert dumped.sum() > 0, "test má smysl jen když se teplo opravdu zahazuje"

    expected = p['h_price'] * (res['Dodáno tepla [MW]'] - dumped)
    assert res['Rev teplo [€]'].values == pytest.approx(expected.values, abs=1e-9)


def test_gas_column_matches_affine_formula():
    """Sloupec s plynem musí sedět na afinní vzorec nad tvarovanými veličinami."""
    p = with_ramp(make_params(), 12.7, 8.9)
    res = solve(p, profile_type='custom', custom_hours=[1, 2, 3])['res']

    c1_th = 1.0 / p['k_eff_th']
    # při konstantní účinnosti je c0_th = 0, takže plyn = q_th / η_th
    expected = res['KGJ [MW_th]'] / p['k_eff_th']
    assert res['Plyn KGJ [MWh]'].values == pytest.approx(expected.values, rel=1e-9)
    assert c1_th > 0


# ── Tepelná bilance ──────────────────────────────────────────────────

def test_heat_balance_holds():
    p = with_ramp(make_params(h_cover=1.0, shortfall_penalty=5000.0), 12.7, 8.9)
    res = solve(p, df=make_df(heat_demand=0.5 * K_TH))['res']

    components = (res['KGJ [MW_th]'] + res['Kotel [MW_th]']
                  + res['Elektrokotel [MW_th]'] + res['Import tepla [MW_th]']
                  + res['TES netto [MW_th]'])
    assert res['Dodáno tepla [MW]'].values == pytest.approx(components.values, abs=1e-9)

    demand = res['Poptávka tepla [MW]']
    assert (res['Dodáno tepla [MW]'] + res['Shortfall [MW]']
            >= demand * p['h_cover'] - 1e-6).all()
    assert (res['Dodáno tepla [MW]']
            <= demand + res['Zahozené teplo [MW]'] + 1e-3 + 1e-6).all()


# ── Commitment logika ────────────────────────────────────────────────

def test_commitment_follows_the_profile_window():
    res = solve(make_params(), profile_type='custom', custom_hours=[1, 2, 3])['res']
    assert res['KGJ on'].round().tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    # bez zapnuté rampy se pomocné proměnné vůbec nezakládají
    assert (res['KGJ stop'] == 0).all()


def test_stop_indicator_matches_identity():
    """stop[t] = on[t-1] - on[t] + start[t] musí platit v každé hodině.

    Právě tahle identita dovoluje mít `stop` spojitý místo binárního.
    """
    res = solve(with_ramp(make_params(), 12.7, 8.9),
                profile_type='custom', custom_hours=[1, 2, 3])['res']
    on = res['KGJ on'].round().tolist()
    stop = res['KGJ stop'].round().tolist()

    expected = [0] + [max(0, on[t - 1] - on[t]) for t in range(1, len(on))]
    assert stop == expected
    assert sum(stop) == 1, "jeden běh ⇒ právě jedno odstavení"


def test_min_runtime_is_respected():
    """Minimální doba běhu drží jednotku zapnutou i po skončení výhodných hodin."""
    min_rt = 4
    p = make_params(k_min_runtime=min_rt, k_start_cost=0.0)
    # výhodná je jen hodina 1, dál je EE zadarmo a plyn drahý
    df = make_df(hours=12, ee_price=[1.0] * 12, gas_price=[200.0] * 12)
    df.loc[1, 'ee_price'] = 5000.0

    res = solve(p, df=df)['res']
    on = res['KGJ on'].round().tolist()

    runs, cur = [], 0
    for v in on + [0]:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    assert runs, "KGJ měla alespoň jednou naskočit"
    assert all(r >= min_rt for r in runs), f"běhy {runs} porušují min. dobu {min_rt} h"


# ── Linearizace účinnosti ────────────────────────────────────────────

def test_compute_linear_fuel_params_reproduces_both_endpoints():
    k_th, k_min = 0.975, 0.5
    eta_th_r, eta_th_m, eta_el_r, eta_el_m = 0.531, 0.478, 0.395, 0.356
    c0_th, c1_th, c0_el, c1_el = compute_linear_fuel_params(
        k_th, k_min, eta_th_r, eta_th_m, eta_el_r, eta_el_m)

    q_min, q_max = k_min * k_th, k_th
    assert c0_th + c1_th * q_max == pytest.approx(q_max / eta_th_r)
    assert c0_th + c1_th * q_min == pytest.approx(q_min / eta_th_m)
    assert c0_el + c1_el * q_max == pytest.approx(q_max * eta_el_r / eta_th_r)
    assert c0_el + c1_el * q_min == pytest.approx(q_min * eta_el_m / eta_th_m)


def test_compute_linear_fuel_params_handles_full_min_load():
    """k_min = 100 % splyne oba body — nesmí spadnout na dělení nulou."""
    c0_th, c1_th, c0_el, c1_el = compute_linear_fuel_params(
        0.975, 1.0, 0.531, 0.531, 0.395, 0.395)
    assert (c0_th, c0_el) == (0.0, 0.0)
    assert c1_th == pytest.approx(1.0 / 0.531)
    assert c1_el == pytest.approx(0.395 / 0.531)


# ── Profily ──────────────────────────────────────────────────────────

def test_peak_profile_covers_business_hours_only():
    df = make_df(hours=24, start='2026-01-05 00:00')  # pondělí
    c = create_profile_constraints(df, 'peak')
    assert [h for h, v in enumerate(c) if v == 0] == list(range(8, 20))
    assert all(v == -1 for h, v in enumerate(c) if h < 8 or h >= 20)


def test_base_profile_forces_everything_on():
    assert create_profile_constraints(make_df(hours=24), 'base') == [1] * 24


def test_free_profile_leaves_solver_unconstrained():
    assert create_profile_constraints(make_df(hours=24), 'free') == [0] * 24


def test_public_holiday_is_excluded_from_peak():
    """1. 1. 2026 je státní svátek, takže PEAK nesmí povolit ani jednu hodinu."""
    assert dt.date(2026, 1, 1) in __import__('opt_core').CZ_HOLIDAYS
    c = create_profile_constraints(make_df(hours=24, start='2026-01-01 00:00'), 'peak')
    assert c == [-1] * 24


def test_special_profile_differs_between_summer_and_rest():
    summer = create_profile_constraints(make_df(hours=24, start='2026-07-03 00:00'), 'special')
    winter = create_profile_constraints(make_df(hours=24, start='2026-01-09 00:00'), 'special')
    # oba dny jsou pátek; v létě SPECIAL jede jen Po–Čt, takže pátek je celý off
    assert summer == [-1] * 24
    assert any(v == 0 for v in winter)


# ── Fixní výkupní cena ───────────────────────────────────────────────

def test_get_kgj_fix_price():
    p = {'kgj_ee_fix_peak': True, 'kgj_ee_fix_price_peak': 130.0,
         'kgj_ee_fix_base': False, 'kgj_ee_fix_price_base': 106.0}
    assert get_kgj_fix_price(p, 'peak') == (True, 130.0)
    assert get_kgj_fix_price(p, 'base') == (False, None)
    assert get_kgj_fix_price(p, 'free') == (False, None)
    # custom profil fixní cenu nikdy nemá
    assert get_kgj_fix_price({'kgj_ee_fix_custom': True}, 'custom') == (False, None)
