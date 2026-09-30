"""Zákaz maření tepla — KGJ vyrobí jen teplo, které soustava odebere.

Parametr `kgj_no_heat_dump` (checkbox v sekci KGJ). Bez zákazu jede KGJ při
drahé elektřině naplno a přebytek tepla zahodí. Se zákazem sleduje poptávku
(nebo nabíjí TES) a pod min. zatížením stojí.

Výkon 0,975 MW, min. zatížení 50 % = 0,4875 MW.
"""

import ast
from pathlib import Path

import opt_core
from conftest import SOLVER_TIME_LIMIT, make_df, make_params, make_uses
from opt_core import run_optimization_with_profile

APP = Path(opt_core.__file__).parent / 'app.py'


def solve(df, profile='free', **params):
    """Hodinové výsledky, nebo None, když řešení neexistuje."""
    r = run_optimization_with_profile(df, make_params(k_min=0.5, **params),
                                      make_uses(), profile_type=profile,
                                      time_limit=SOLVER_TIME_LIMIT)
    return r['res'] if r is not None else None


def two_levels():
    """Drahá elektřina; 4 h poptávka 0,8 MW, pak 4 h jen 0,3 MW (pod min.)."""
    return make_df(hours=8, ee_price=300.0, heat_demand=[0.8] * 4 + [0.3] * 4)


def test_without_ban_kgj_runs_full_and_dumps():
    res = solve(two_levels())
    assert (res['KGJ on'] > 0.5).all()
    assert (res['KGJ [MW_th]'] > 0.97).all()
    assert res['Zahozené teplo [MW]'].sum() > 2.0


def test_ban_follows_demand_and_stops_below_min_load():
    res = solve(two_levels(), kgj_no_heat_dump=True)
    on = (res['KGJ on'] > 0.5).values
    assert on[:4].all() and not on[4:].any()
    assert res['Zahozené teplo [MW]'].abs().max() < 1e-6
    assert (res['KGJ [MW_th]'][:4] - 0.8).abs().max() < 2e-3   # jede na poptávku


def test_ban_makes_base_infeasible_below_min_load():
    """BASE drží KGJ v provozu i v hodině, kdy teplo nemá kam jít."""
    assert solve(two_levels(), 'base') is not None
    assert solve(two_levels(), 'base', kgj_no_heat_dump=True) is None


def test_ban_skips_whole_month_that_would_dump():
    """Únor by se vyplatil celý, ale v jedné hodině peaku soustava bere 0,1 MW.

    Celý měsíc bez maření odjet nejde, a bez limitu hodin není ani neúplný
    měsíc — nejede nic.
    """
    df = make_df(hours=24 * 14, ee_price=200.0, heat_demand=1.0,
                 start='2026-02-02 00:00')
    df.loc[10, 'Poptávka po teple (MW)'] = 0.1          # po 2. 2. 10:00
    full = solve(df, 'peak', kgj_whole_months=True)
    assert (full['KGJ on'] > 0.5).sum() == 120           # 2 týdny × 60 h
    banned = solve(df, 'peak', kgj_whole_months=True, kgj_no_heat_dump=True)
    assert (banned['KGJ on'] > 0.5).sum() == 0


# ── UI ──────────────────────────────────────────────────────────────

TREE = ast.parse(APP.read_text(encoding='utf-8'))


def test_checkbox_defaults_to_off_and_reaches_params():
    """`p['kgj_no_heat_dump'] = st.checkbox(..., value=False)` v sekci KGJ."""
    for n in ast.walk(TREE):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Subscript)
                and isinstance(n.targets[0].value, ast.Name)
                and n.targets[0].value.id == 'p'
                and isinstance(n.targets[0].slice, ast.Constant)
                and n.targets[0].slice.value == 'kgj_no_heat_dump'):
            call = n.value
            assert isinstance(call, ast.Call) and call.func.attr == 'checkbox'
            kw = {k.arg: k.value for k in call.keywords}
            assert isinstance(kw['value'], ast.Constant) and kw['value'].value is False
            return
    raise AssertionError("p['kgj_no_heat_dump'] se v app.py nenastavuje")


def test_parameters_sheet_records_the_ban():
    from app_extract import load_app
    build = load_app().build_parameters_df
    on = build(make_params(kgj_no_heat_dump=True), make_uses())
    assert list(on.loc[on['Parametr'] == 'Zákaz maření tepla',
                       'Hodnota']) == ['ANO']
    off = build(make_params(), make_uses())
    assert 'Zákaz maření tepla' not in set(off['Parametr'])
