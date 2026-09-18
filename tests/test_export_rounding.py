"""Testy zaokrouhlování při exportu.

`DataFrame.round()` na sloupci s časem nic nedělá a pandas na to u každého
exportu upozorní varováním. Na cílovém stroji je `run.log` jediný diagnostický
kanál, takže takový šum ztěžuje hledání skutečných chyb.
"""

import ast
from pathlib import Path
import types
import warnings

import pandas as pd
import pytest

import opt_core

REPO = opt_core.__file__.rsplit('/', 1)[0]


def _round_numeric():
    """Vytáhne helper z app.py — Streamlit skript nejde importovat."""
    src = Path(f'{REPO}/app.py').read_text(encoding='utf-8')
    mod = types.ModuleType('app_round')
    mod.__dict__['pd'] = pd
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == '_round_numeric':
            exec(compile(ast.Module([node], []), '<app>', 'exec'), mod.__dict__)
            return mod.__dict__['_round_numeric']
    pytest.fail('v app.py chybí _round_numeric')


@pytest.fixture
def frame():
    return pd.DataFrame({
        'Čas': pd.date_range('2026-01-01', periods=3, freq='h'),
        'Výkon [MW]': [1.234567, 2.345678, 3.456789],
        'Počet': [1, 2, 3],
        'Popis': ['a', 'b', 'c'],
    })


def test_no_warning_on_datetime_column(frame):
    """Tohle je ten warning, který uživatel viděl v konzoli."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')       # jakékoli varování shodí test
        _round_numeric()(frame)


def test_numbers_are_rounded(frame):
    out = _round_numeric()(frame)
    assert out['Výkon [MW]'].tolist() == [1.2346, 2.3457, 3.4568]


def test_other_columns_survive(frame):
    out = _round_numeric()(frame)
    assert out['Čas'].tolist() == frame['Čas'].tolist()
    assert out['Popis'].tolist() == ['a', 'b', 'c']
    assert out['Počet'].tolist() == [1, 2, 3]


def test_input_is_not_modified(frame):
    before = frame['Výkon [MW]'].tolist()
    _round_numeric()(frame)
    assert frame['Výkon [MW]'].tolist() == before, 'helper nesmí měnit vstup'


def test_no_raw_round_left_in_exports():
    """Všechny exporty musí jít přes helper, jinak se warning vrátí."""
    src = Path(f'{REPO}/app.py').read_text(encoding='utf-8')
    lines = [(i + 1, l) for i, l in enumerate(src.split('\n'))
             if '.round(4)' in l and '_round_numeric' not in l
             and 'ndigits' not in l]
    assert not lines, f'přímé .round(4) mimo helper: {lines}'
