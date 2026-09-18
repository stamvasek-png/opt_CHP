"""Kontrola, že je profil skutečně zaregistrovaný v UI.

Samotný constraint nestačí — pokud profil chybí v nabídce, v barvách nebo
v popisech, uživatel se k němu vůbec nedostane. Testujeme přes zdroják
`app.py`, protože jeho import vyžaduje běžící Streamlit.

Všechno se hledá přes AST, ne přes hledání textu. Jinak by test hlídal
i to, kde je zalomený řádek, a rozbil by se při každém přidání profilu.
"""

import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / 'app.py'


def app_source():
    return APP.read_text(encoding='utf-8')


def _strings(node):
    """Řetězcové konstanty ze seznamu / n-tice; jinak prázdno."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    return [e.value for e in node.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)]


def _tree():
    return ast.parse(app_source())


def multiselect_options():
    """Profily z `st.multiselect(..., options=[...])`."""
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                vals = _strings(kw.value) if kw.arg == 'options' else []
                if 'free' in vals and 'custom' in vals:
                    return vals
    raise AssertionError('v app.py neni multiselect s nabidkou profilu')


def profile_colors():
    """Klíče slovníku PROFILE_COLORS."""
    for node in ast.walk(_tree()):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(isinstance(t, ast.Name) and t.id == 'PROFILE_COLORS'
                        for t in node.targets)):
            return [k.value for k in node.value.keys
                    if isinstance(k, ast.Constant)]
    raise AssertionError('v app.py neni PROFILE_COLORS')


def profile_definitions():
    """Klíče slovníku profile_definitions."""
    for node in ast.walk(_tree()):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(isinstance(t, ast.Name) and t.id == 'profile_definitions'
                        for t in node.targets)):
            return [k.value for k in node.value.keys
                    if isinstance(k, ast.Constant)]
    raise AssertionError('v app.py neni profile_definitions')


def export_profiles():
    """Profily, které se vypisují do listu Parametry (`for prof in (...)`)."""
    for node in ast.walk(_tree()):
        if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == 'prof'):
            vals = _strings(node.iter)
            if vals:
                return vals
    raise AssertionError('v app.py neni vypis fixnich cen do parametru')


def assert_profile_registered(profile):
    """Profil má barvu, je v nabídce, má popis, fixní cenu a jde do exportu."""
    assert profile in profile_colors(), f'{profile}: chybi barva profilu'
    assert profile in multiselect_options(), f'{profile}: chybi v nabidce'
    assert profile in profile_definitions(), f'{profile}: chybi popis profilu'
    assert f'kgj_ee_fix_{profile}' in app_source(), \
        f'{profile}: chybi fixni vykupni cena'
    assert profile in export_profiles(), \
        f'{profile}: chybi ve vypisu parametru do exportu'
