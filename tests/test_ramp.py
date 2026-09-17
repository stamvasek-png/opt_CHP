"""Testy modelování nájezdu a sjezdu KGJ.

Referenční data (reálný provoz, 975 kW_th = 100 % zatížení):

    hodina 1: 872   nájezd ~12,7 min, pak plný výkon
    hodina 2: 975   plný výkon
    hodina 3: 975   plný výkon
    hodina 4:  72   sjezd ~8,9 min na začátku hodiny, pak nula

Lineární rampa délky τ minut ⇒ hodina startu dodá P·(1 − τ/120),
hodina po vypnutí ještě P·(τ/120).
"""

import pytest

from conftest import (K_TH, SOLVER_TIME_LIMIT, make_df, make_params, make_uses,
                      with_ramp)
from opt_core import run_optimization_with_profile

# Okno, ve kterém KGJ smí běžet. Hodina 4 je tím vynucená na off, takže se v ní
# může projevit jedině doběh.
RUN_HOURS = [1, 2, 3]


def solve(params, df=None, profile_type='custom', custom_hours=RUN_HOURS, **kw):
    df = make_df() if df is None else df
    r = run_optimization_with_profile(
        df=df, params=params, uses=kw.pop('uses', make_uses()),
        profile_type=profile_type, custom_hours=custom_hours,
        time_limit=SOLVER_TIME_LIMIT, **kw)
    assert r is not None, "solver nenašel řešení"
    return r


def kw_shape(res):
    """Skutečný výkon KGJ po hodinách v kW_th (zaokrouhleno jako v reálných datech)."""
    return [round(v * 1000) for v in res['KGJ [MW_th]']]


# ── Referenční tvar ──────────────────────────────────────────────────

@pytest.mark.parametrize('up_min, down_min, expected', [
    (12.7, 8.9, [872, 975, 975, 72]),   # rampa odhadnutá z reálných dat
    (10.0, 10.0, [894, 975, 975, 81]),  # kulatých 10 / 10 min
])
def test_ramp_shape_matches_real_data(up_min, down_min, expected):
    """Model reprodukuje naměřený tvar 872 / 975 / 975 / 72."""
    res = solve(with_ramp(make_params(), up_min, down_min))['res']
    assert kw_shape(res)[1:5] == expected


def test_ramp_off_is_baseline():
    """Vypnutá rampa nechá model v původním skokovém chování."""
    res = solve(make_params())['res']
    assert kw_shape(res)[1:5] == [975, 975, 975, 0]
    assert (res['KGJ [MW_th]'] == res['KGJ setpoint [MW_th]']).all()
    assert (res['KGJ nájezd ztráta [MW_th]'] == 0).all()
    assert (res['KGJ doběh [MW_th]'] == 0).all()


@pytest.mark.parametrize('tau, first_pct, tail_pct', [
    (0.0, 1.00, 0.00),   # nulová rampa = skok
    (60.0, 0.50, 0.50),  # rampa přes celou hodinu
])
def test_ramp_bounds(tau, first_pct, tail_pct):
    res = solve(with_ramp(make_params(), tau, tau))['res']
    assert res['KGJ [MW_th]'][1] == pytest.approx(K_TH * first_pct, abs=1e-9)
    assert res['KGJ [MW_th]'][4] == pytest.approx(K_TH * tail_pct, abs=1e-9)


# ── Umístění a velikost doběhu ───────────────────────────────────────

def test_tail_lands_in_hour_after_last_on_hour():
    """Doběh padá do hodiny, kde je KGJ už formálně vypnutá."""
    res = solve(with_ramp(make_params(), 12.7, 8.9))['res']
    last_on, tail = 3, 4

    assert res['KGJ on'][last_on] == 1
    assert res['KGJ on'][tail] == 0
    assert res['KGJ stop'][tail] == 1
    assert res['KGJ [MW_th]'][tail] > 0

    a_dn = 8.9 / 120.0
    assert res['KGJ [MW_th]'][tail] == pytest.approx(
        a_dn * res['KGJ setpoint [MW_th]'][last_on], rel=1e-9)


def test_ramp_scales_with_setpoint_not_rated_power():
    """Při částečném zatížení škáluje ztráta i doběh podle setpointu, ne podle k_th.

    Toto je přímý test správnosti McCormickovy linearizace: kdyby se rampa
    vztahovala k jmenovitému výkonu, vyšly by obě hodnoty výrazně vyšší.
    """
    a_up, a_dn = 12.7 / 120.0, 8.9 / 120.0
    p = with_ramp(make_params(k_min=0.3), 12.7, 8.9)
    # Nízká cena EE a drahý plyn ⇒ výroba nad poptávku se nevyplatí,
    # takže solver drží setpoint právě na poptávce (60 % jmenovitého výkonu).
    df = make_df(ee_price=1.0, gas_price=120.0, heat_demand=0.6 * K_TH)
    p['h_cover'], p['shortfall_penalty'] = 1.0, 5000.0

    res = solve(p, df=df)['res']
    setpoint = res['KGJ setpoint [MW_th]'][1]

    assert setpoint < K_TH * 0.99, "test má smysl jen při částečném zatížení"
    assert res['KGJ nájezd ztráta [MW_th]'][1] == pytest.approx(a_up * setpoint, rel=1e-6)
    assert res['KGJ doběh [MW_th]'][4] == pytest.approx(
        a_dn * res['KGJ setpoint [MW_th]'][3], rel=1e-6)
    # a zároveň jasně méně, než kdyby se počítalo z k_th
    assert res['KGJ nájezd ztráta [MW_th]'][1] < a_up * K_TH * 0.95


# ── Energetická bilance ──────────────────────────────────────────────

def test_symmetric_ramp_conserves_energy():
    """Při τ_up == τ_dn se energie jen rozmaže přes hranice hodin."""
    res = solve(with_ramp(make_params(), 10.0, 10.0))['res']
    delivered = res['KGJ [MW_th]'].sum()
    setpoint = res['KGJ setpoint [MW_th]'].sum()
    assert delivered == pytest.approx(setpoint, abs=1e-9)


def test_asymmetric_ramp_loses_exactly_the_difference():
    """Pomalejší nájezd než sjezd znamená reálnou ztrátu (α_up − α_dn)·P."""
    up_min, down_min = 12.7, 8.9
    res = solve(with_ramp(make_params(), up_min, down_min))['res']
    expected_loss = (up_min - down_min) / 120.0 * K_TH

    loss = res['KGJ setpoint [MW_th]'].sum() - res['KGJ [MW_th]'].sum()
    assert loss == pytest.approx(expected_loss, abs=1e-9)
    # kontrola proti reálným datům: 872+975+975+72 = 2894 proti 3×975 = 2925
    assert round(loss * 1000) == 31


# ── Interakce s ostatními omezeními ──────────────────────────────────

def test_min_load_constraint_stays_satisfied():
    """Rampa deratuje jen dodanou energii, setpoint zůstává nad minimem."""
    k_min = 0.6
    # τ = 60 min ⇒ nájezdová hodina dodá polovinu setpointu, tedy spolehlivě
    # pod hranici minimálního zatížení — to je právě ten případ, který by model
    # shodil, kdyby se rampa aplikovala přímo na q_kgj.
    p = with_ramp(make_params(k_min=k_min), 60.0, 60.0)
    res = solve(p)['res']
    running = res[res['KGJ on'] > 0.5]

    assert not running.empty
    assert (running['KGJ setpoint [MW_th]'] >= k_min * K_TH - 1e-6).all()
    # dodaný výkon v nájezdové hodině naopak pod minimum klesnout smí a má
    assert res['KGJ [MW_th]'][1] < k_min * K_TH


def test_base_profile_has_no_tail():
    """BASE vynucuje on==1 všude, takže po t=0 nenastane start ani stop."""
    res = solve(with_ramp(make_params(), 12.7, 8.9),
                profile_type='base', custom_hours=None)['res']

    assert (res['KGJ on'] == 1).all()
    assert (res['KGJ stop'] == 0).all()
    assert res['KGJ nájezd ztráta [MW_th]'][0] > 0          # jediný nájezd, v t=0
    assert (res['KGJ nájezd ztráta [MW_th]'][1:] == 0).all()
    assert (res['KGJ doběh [MW_th]'] == 0).all()


def test_tail_outside_profile_window_is_settled_at_spot():
    """Doběh mimo obchodní pásmo profilu se vykupuje za spot, ne za PPA profilu.

    PEAK končí ve 20:00; doběh spadne do hodiny 20, která už do produktu nepatří.
    """
    spot, ppa = 100.0, 130.0
    p = with_ramp(make_params(kgj_ee_fix_peak=True, kgj_ee_fix_price_peak=ppa), 12.7, 8.9)
    df = make_df(hours=24, ee_price=spot)

    res = solve(p, df=df, profile_type='peak', custom_hours=None)['res']
    price = res['Cena výkupu EE z KGJ [€/MWh]']

    assert res['KGJ stop'][20] == 1, "PEAK má končit ve 20:00"
    assert res['KGJ doběh [MW_th]'][20] > 0
    assert price[20] == pytest.approx(spot), "doběh mimo PEAK musí jít za spot"
    assert price[10] == pytest.approx(ppa), "hodina uvnitř PEAK drží PPA cenu"


def test_profile_pruning_does_not_change_result():
    """Strukturální prořezání pomocných proměnných nesmí změnit výsledek.

    PEAK okno se prořeže zhruba na čtvrtinu; FREE se neprořezává vůbec,
    takže stejný rozvrh musí přes obě cesty dát stejná čísla.
    """
    p = with_ramp(make_params(), 12.7, 8.9)
    df = make_df(hours=24)

    peak = solve(p, df=df, profile_type='peak', custom_hours=None)['res']
    # stejné okno vyjádřené jako custom (jiná maska, stejná fyzika)
    free = solve(p, df=df, profile_type='custom',
                 custom_hours=list(range(8, 20)))['res']

    assert kw_shape(peak) == kw_shape(free)

# ── Oddělená rampa pro teplo a elektřinu ─────────────────────────────

# Jednotka nastavená tak, aby jmenovity ELEKTRICKY vykon byl 975 kW —
# tak si uzivatel jednotku odecita.
ETA_TH, ETA_EL = 0.531, 0.395
K_EL = 0.975
K_TH_FOR_EL = K_EL * ETA_TH / ETA_EL


def el_params(**over):
    p = make_params(k_th=K_TH_FOR_EL, k_eff_th=ETA_TH, k_eff_el=ETA_EL, k_min=1.0)
    p.update(over)
    return p


def test_ramp_shape_matches_real_data_electrical():
    """Zlatý test na reálná data, tentokrát elektricky.

    975 kW_el, nájezd 12,7 min a sjezd 8,9 min musí dát naměřený odečet.
    """
    res = solve(with_ramp(el_params(), 12.7, 8.9))['res']
    ee_kw = [round(v * 1000) for v in res['EE z KGJ [MW]']]
    assert ee_kw[1:5] == [872, 975, 975, 72]


def test_thermal_tail_outlasts_electrical():
    """S delší tepelnou rampou nese doběhová hodina víc tepla než elektřiny."""
    # elektricky sjezd 9 min, tepelny 30 min (dochlazeni)
    p = with_ramp(el_params(), 12.7, 9.0, th_up_min=30.0, th_down_min=30.0)
    res = solve(p)['res']
    tail = 4

    el_frac = res['EE z KGJ [MW]'][tail] / K_EL
    th_frac = res['KGJ [MW_th]'][tail] / K_TH_FOR_EL

    assert el_frac == pytest.approx(9.0 / 120.0, rel=1e-6)
    assert th_frac == pytest.approx(30.0 / 120.0, rel=1e-6)
    assert th_frac > el_frac * 3, "tepelny dobeh ma byt vyrazne delsi"


def test_gas_follows_electrical_ramp():
    """Plyn jde s motorem, ne s teplem — jádro celé změny.

    V doběhové hodině se dodá zbytkové teplo z bloku, které už žádné palivo
    nespotřebovává, takže poměr plynu musí sedět na elektrickou rampu.
    """
    a_dn_el, a_dn_th = 9.0 / 120.0, 30.0 / 120.0
    p = with_ramp(el_params(), 12.7, 9.0, th_up_min=30.0, th_down_min=30.0)
    res = solve(p)['res']
    tail, last_on = 4, 3

    gas_full = res['Plyn KGJ [MWh]'][last_on]
    gas_tail = res['Plyn KGJ [MWh]'][tail]

    assert gas_tail / gas_full == pytest.approx(a_dn_el, rel=1e-6)
    assert gas_tail / gas_full != pytest.approx(a_dn_th, rel=1e-3)
    # a teplo v te hodine je naopak podle tepelne rampy
    assert res['KGJ [MW_th]'][tail] / res['KGJ [MW_th]'][last_on] == pytest.approx(
        a_dn_th, rel=1e-6)


def test_thermal_split_off_is_previous_behaviour():
    """Bez zapnutého rozlišení se teplo i elektřina deratují stejně."""
    res = solve(with_ramp(el_params(), 12.7, 8.9))['res']
    for t in (1, 4):
        el = res['EE z KGJ [MW]'][t] / res['EE z KGJ setpoint [MW]'].max()
        th = res['KGJ [MW_th]'][t] / res['KGJ setpoint [MW_th]'].max()
        assert el == pytest.approx(th, rel=1e-9)


def test_ramp_parts_sum_up():
    """setpoint − ztráta + doběh musí dát skutečnou hodnotu, u obou veličin."""
    p = with_ramp(el_params(), 12.7, 9.0, th_up_min=25.0, th_down_min=25.0)
    res = solve(p)['res']

    ee = (res['EE z KGJ setpoint [MW]'] - res['EE z KGJ nájezd ztráta [MW]']
          + res['EE z KGJ doběh [MW]'])
    th = (res['KGJ setpoint [MW_th]'] - res['KGJ nájezd ztráta [MW_th]']
          + res['KGJ doběh [MW_th]'])

    assert ee.values == pytest.approx(res['EE z KGJ [MW]'].values, abs=1e-9)
    assert th.values == pytest.approx(res['KGJ [MW_th]'].values, abs=1e-9)
    # a ztrata/dobeh jsou nenulove tam, kde maji byt
    assert res['EE z KGJ nájezd ztráta [MW]'][1] > 0
    assert res['EE z KGJ doběh [MW]'][4] > 0
