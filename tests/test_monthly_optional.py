"""Měsíční optimalizace se smí spustit jen po zaškrtnutí checkboxu.

Je to zdaleka nejdelší část výpočtu (profil × měsíc je samostatná úloha),
takže ji nesmí spustit nic jiného než vědomé zaškrtnutí. Testuje se přes
AST `app.py`, protože jeho import vyžaduje běžící Streamlit.
"""

import ast
from pathlib import Path

import opt_core

APP = Path(opt_core.__file__).parent / 'app.py'
TREE = ast.parse(APP.read_text(encoding='utf-8'))
SRC = APP.read_text(encoding='utf-8')


def _calls_named(name):
    return [n for n in ast.walk(TREE)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]


def _guards_of(node):
    """Podminky vsech `if`, uvnitr kterych dany uzel lezi."""
    out = []
    for n in ast.walk(TREE):
        if isinstance(n, ast.If):
            body = [x for b in n.body for x in ast.walk(b)]
            if node in body:
                out.append(n.test)
    return out


def test_checkbox_exists():
    assert 'cb_run_monthly' in SRC, 'chybi checkbox pro mesicni optimalizaci'


def test_checkbox_defaults_to_off():
    """Vychozi stav musi byt vypnuto, jinak se nic neusetri."""
    for n in ast.walk(TREE):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'checkbox'):
            keys = {k.arg: k.value for k in n.keywords}
            key = keys.get('key')
            if isinstance(key, ast.Constant) and key.value == 'cb_run_monthly':
                val = keys.get('value')
                assert isinstance(val, ast.Constant) and val.value is False, \
                    'checkbox ma byt vychozi vypnuty'
                return
    raise AssertionError('checkbox cb_run_monthly nenalezen')


def test_monthly_analysis_is_guarded_by_the_checkbox():
    calls = _calls_named('run_monthly_profile_analysis')
    assert calls, 'v app.py se mesicni analyza vubec nevola'
    for call in calls:
        names = {g.id for g in _guards_of(call) if isinstance(g, ast.Name)}
        assert 'run_monthly' in names, \
            'volani mesicni analyzy neni podminene promennou run_monthly'


def test_results_are_cleared_when_unchecked():
    """Bez zaskrtnuti se ma ulozit None, aby nezustal viset stary vysledek."""
    assert 'monthly_res = None' in SRC, \
        'pri vypnutem checkboxu se musi vysledek vynulovat'


def test_view_is_hidden_without_results():
    """Pohled na mesicni analyzu se kresli jen kdyz vysledek existuje."""
    assert 'st.session_state.monthly_profile_results is not None' in SRC
