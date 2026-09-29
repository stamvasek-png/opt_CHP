"""Profily po celých měsících — v měsíci buď všechny hodiny profilu, nebo žádná.

Parametr `kgj_whole_months` (checkbox v UI): měsíce vybírá solver sám, jedna
binárka na kalendářní měsíc. S limitem hodin tak třeba PEAK do 3300 h jede
v nejvýnosnějších celých měsících, místo aby sbíral jednotlivé hodiny a dělal
různě dlouhá okna. Co celé měsíce z limitu nevyčerpají, smí jít do jednoho
dalšího měsíce — ten je neúplný a hodiny v něm se vybírají volně.

Bez tepla (poptávka 0) vydělává KGJ jen na elektřině: marže je zhruba
0,725 × cena − 55 €/h, bod zvratu ~76 €/MWh.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

import opt_core
from conftest import SOLVER_TIME_LIMIT, make_df, make_params, make_uses
from opt_core import create_profile_constraints, run_optimization_with_profile

APP = Path(opt_core.__file__).parent / 'app.py'


def solve(df, profile, whole, custom_hours=None, max_starts=None, **params):
    """Hodiny provozu KGJ; bez `whole` se parametr vůbec nezadá (výchozí stav)."""
    if whole:
        params['kgj_whole_months'] = True
    r = run_optimization_with_profile(df, make_params(**params), make_uses(),
                                      profile_type=profile,
                                      custom_hours=custom_hours,
                                      max_starts_per_month=max_starts,
                                      time_limit=SOLVER_TIME_LIMIT)
    assert r is not None
    return (r['res']['KGJ on'] > 0.5).values


def window(df, profile, custom_hours=None):
    return np.array(create_profile_constraints(df, profile, custom_hours)) == 0


def per_month(df, mask):
    return pd.Series(mask).groupby(df['datetime'].dt.month.values).sum().to_dict()


def runs(mask):
    """Délky souvislých bloků provozu."""
    out, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    return out + ([cur] if cur else [])


def february(prices):
    """2.–15. 2. 2026 (od pondělí), 60 h peaku v každém týdnu."""
    return make_df(hours=24 * 14, ee_price=prices, heat_demand=0.0,
                   start='2026-02-02 00:00')


def january_february(price_of):
    """26. 1.–8. 2. 2026: 60 h peaku v lednu (26.–30.) i v únoru (2.–6.)."""
    idx = pd.date_range('2026-01-26', periods=24 * 14, freq='h')
    return make_df(hours=len(idx), ee_price=[price_of(t) for t in idx],
                   heat_demand=0.0, start='2026-01-26 00:00')


# ── model ───────────────────────────────────────────────────────────

def test_whole_month_runs_through_a_losing_week():
    """Drahý týden a ztrátový týden: měsíc jako celek vydělává, jede celý."""
    df = february([200.0] * 168 + [40.0] * 168)
    free_on = solve(df, 'peak', whole=False)
    assert free_on[:168].sum() == 60 and free_on[168:].sum() == 0
    on = solve(df, 'peak', whole=True)
    assert (on == window(df, 'peak')).all()


def test_hour_limit_picks_the_best_whole_month():
    """Limit 60 h: bez režimu nejlepší hodiny z obou měsíců, v režimu celý únor.

    Leden má 36 h peaku za 300 a 24 h za 100 €/MWh, únor 60 h za 240 €/MWh.
    Nejlepší hodiny jsou lednové, ale jako celek vydělá víc únor.
    """
    df = january_february(lambda t: 300.0 if t.day in (26, 27, 28)
                          else 100.0 if t.month == 1 else 240.0)
    win = window(df, 'peak')
    assert per_month(df, win) == {1: 60, 2: 60}

    limit = dict(kgj_hour_limit_on=True, kgj_hour_limit=60)
    assert per_month(df, solve(df, 'peak', whole=False, **limit)) == {1: 36, 2: 24}
    on = solve(df, 'peak', whole=True, **limit)
    assert per_month(df, on) == {1: 0, 2: 60}
    assert (on == (win & (df['datetime'].dt.month == 2).values)).all()


def test_leftover_hours_go_into_one_more_month():
    """Limit 90 h: celý únor (60 h) a zbylých 30 h do ledna, ten je neúplný."""
    df = january_february(lambda t: 300.0 if t.day in (26, 27, 28)
                          else 100.0 if t.month == 1 else 240.0)
    on = solve(df, 'peak', whole=True, kgj_hour_limit_on=True, kgj_hour_limit=90)
    assert per_month(df, on) == {1: 30, 2: 60}
    feb = (df['datetime'].dt.month == 2).values
    assert (on[feb] == window(df, 'peak')[feb]).all()
    assert (df['datetime'][on & ~feb].dt.day <= 28).all()      # nejdrazsi dny


def test_month_that_fits_stays_whole():
    """Limit 200 h pojme oba měsíce: leden jede celý i se ztrátovými dny.

    Neúplný měsíc je jen pro zbytek limitu v nevybraném měsíci — solver z něj
    nesmí udělat neúplný měsíc, aby vybral jen lednové drahé dny.
    """
    df = january_february(lambda t: 200.0 if t.day in (26, 27, 28)
                          else 60.0 if t.month == 1 else 240.0)
    on = solve(df, 'peak', whole=True, kgj_hour_limit_on=True, kgj_hour_limit=200)
    assert (on == window(df, 'peak')).all()


def test_leftover_goes_into_a_single_month():
    """Limit 40 h je menší než celý měsíc: zbytek smí jít jen do jednoho.

    Bez režimu vezme solver nejlepší hodiny z obou měsíců. V režimu jen únor,
    jehož nejlepších 40 h vydělá víc než nejlepších 40 h ledna.
    """
    df = january_february(lambda t: {26: 300.0, 27: 300.0}.get(t.day, 100.0)
                          if t.month == 1
                          else 250.0 if t.day in (2, 3, 4) else 100.0)
    limit = dict(kgj_hour_limit_on=True, kgj_hour_limit=40)
    assert per_month(df, solve(df, 'peak', whole=False, **limit)) == {1: 24, 2: 16}
    assert per_month(df, solve(df, 'peak', whole=True, **limit)) == {1: 0, 2: 40}


def test_min_runtime_holds_in_the_partial_month():
    """V neúplném lednu se hodiny vybírají volně — ale s min. dobou běhu.

    Lednový peak střídá drahé (liché) a ztrátové (sudé) hodiny. S min. dobou
    běhu 1 h vezme solver 6 osamocených drahých hodin, se 4 h jen bloky ≥ 4 h.
    """
    df = january_february(lambda t: (300.0 if t.hour % 2 else 40.0)
                          if t.month == 1 else 240.0)
    jan = (df['datetime'].dt.month == 1).values
    limit = dict(kgj_hour_limit_on=True, kgj_hour_limit=66)
    loose = solve(df, 'peak', whole=True, k_min_runtime=1, **limit)
    assert runs(loose[jan]) == [1] * 6
    strict = solve(df, 'peak', whole=True, k_min_runtime=4, **limit)
    assert strict[jan].sum() > 0 and min(runs(strict[jan])) >= 4
    assert strict[~jan].sum() == 60                            # unor cely


def test_short_profile_block_runs_despite_min_runtime():
    """Okno 18–21 (3 h) a min. doba běhu 4 h: bez režimu KGJ nenastartuje."""
    df = make_df(hours=24 * 7, ee_price=200.0, heat_demand=0.0,
                 start='2026-02-02 00:00')
    hours = [18, 19, 20]
    assert solve(df, 'custom', whole=False, custom_hours=hours,
                 k_min_runtime=4).sum() == 0
    on = solve(df, 'custom', whole=True, custom_hours=hours, k_min_runtime=4)
    assert (on == window(df, 'custom', hours)).all()


def test_month_start_limit_does_not_apply():
    df = february(200.0)
    win = window(df, 'peak')
    assert solve(df, 'peak', whole=False, max_starts=1).sum() < win.sum()
    assert (solve(df, 'peak', whole=True, max_starts=1) == win).all()


def test_losing_month_stays_off():
    df = february(40.0)
    assert solve(df, 'peak', whole=True).sum() == 0


def test_free_means_whole_months_around_the_clock():
    df = february([200.0] * 300 + [40.0] * 36)
    assert solve(df, 'free', whole=True).all()


def test_base_is_unchanged():
    assert solve(february(40.0), 'base', whole=True).all()


# ── UI ──────────────────────────────────────────────────────────────

TREE = ast.parse(APP.read_text(encoding='utf-8'))


def test_checkbox_defaults_to_off():
    for n in ast.walk(TREE):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'checkbox'):
            kw = {k.arg: k.value for k in n.keywords}
            key = kw.get('key')
            if isinstance(key, ast.Constant) and key.value == 'cb_whole_months':
                val = kw.get('value')
                assert isinstance(val, ast.Constant) and val.value is False
                return
    raise AssertionError('checkbox cb_whole_months nenalezen')


def test_checkbox_reaches_solver_params():
    """Hodnota musí skončit v `p`, který dostávají všechny běhy solveru."""
    for n in ast.walk(TREE):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Subscript)
                and isinstance(n.targets[0].value, ast.Name)
                and n.targets[0].value.id == 'p'
                and isinstance(n.targets[0].slice, ast.Constant)
                and n.targets[0].slice.value == 'kgj_whole_months'):
            assert isinstance(n.value, ast.Name) and n.value.id == 'whole_months'
            return
    raise AssertionError("p['kgj_whole_months'] se v app.py nenastavuje")


def test_parameters_sheet_records_the_mode():
    from app_extract import load_app
    build = load_app().build_parameters_df
    on = build(make_params(kgj_whole_months=True), make_uses())
    assert list(on.loc[on['Parametr'] == 'Profily po celých měsících',
                       'Hodnota']) == ['ANO']
    off = build(make_params(), make_uses())
    assert 'Profily po celých měsících' not in set(off['Parametr'])
