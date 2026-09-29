"""Testy profilu W — okno po měsících zvlášť pro pracovní den a sobotu.

Zadání (tabulka po měsících): pracovní dny a soboty mají vlastní okno,
neděle a státní svátky stojí. Svátek se chová jako neděle i tehdy, když
připadne na sobotu.
"""

import ast
import datetime as dt

import pandas as pd
import pytest

import opt_core
from conftest import make_df
from opt_core import (CZ_HOLIDAYS, W_WINDOWS, create_profile_constraints,
                      short_blocks)

# mesic -> (Po-Pa, sobota)
EXPECTED = {
    1: ((6, 24), (7, 22)), 2: ((6, 24), (7, 22)),
    3: ((16, 22), (16, 22)), 4: ((17, 23), (17, 23)),
    5: ((18, 24), (18, 24)), 6: ((18, 24), (18, 24)),
    7: ((18, 24), (18, 24)), 8: ((18, 24), (18, 24)),
    9: ((18, 24), (18, 24)), 10: ((17, 23), (17, 23)),
    11: ((6, 24), (7, 22)), 12: ((6, 24), (7, 22)),
}
MONTH_HOURS_2027 = {1: 435, 2: 420, 3: 150, 4: 156, 5: 144, 6: 156, 7: 150,
                    8: 156, 9: 150, 10: 150, 11: 438, 12: 441}
HOLIDAYS_2027 = sorted(d for d in CZ_HOLIDAYS if d.year == 2027)


def allowed(day):
    df = make_df(hours=24, start=f'{day} 00:00')
    return {h for h, v in enumerate(create_profile_constraints(df, 'w')) if v == 0}


def first(month, weekday):
    """První den v měsíci roku 2027 s daným dnem v týdnu, který není svátek."""
    day = dt.date(2027, month, 1)
    while day.weekday() != weekday or day in CZ_HOLIDAYS:
        day += dt.timedelta(days=1)
    return day


def year_constraints(year=2027):
    df = make_df(hours=8760, start=f'{year}-01-01 00:00')
    return df, create_profile_constraints(df, 'w')


# ── okna ────────────────────────────────────────────────────────────

def test_windows_match_spec():
    assert W_WINDOWS == EXPECTED


@pytest.mark.parametrize('month', range(1, 13))
def test_workday_window(month):
    lo, hi = EXPECTED[month][0]
    assert allowed(first(month, 2)) == set(range(lo, hi))       # streda


@pytest.mark.parametrize('month', range(1, 13))
def test_saturday_window(month):
    lo, hi = EXPECTED[month][1]
    assert allowed(first(month, 5)) == set(range(lo, hi))


@pytest.mark.parametrize('month', range(1, 13))
def test_sunday_is_off(month):
    assert allowed(first(month, 6)) == set()


@pytest.mark.parametrize('day', HOLIDAYS_2027, ids=str)
def test_holidays_are_off(day):
    assert allowed(day) == set()


def test_saturday_holiday_is_off():
    """1. 5. 2027 je sobota a svátek — stojí; další sobota už jede."""
    assert dt.date(2027, 5, 1).weekday() == 5
    assert allowed(dt.date(2027, 5, 1)) == set()
    assert allowed(dt.date(2027, 5, 15)) == set(range(18, 24))


# ── roční součty ────────────────────────────────────────────────────

def test_year_2027_hours():
    df, c = year_constraints()
    on = pd.Series([v == 0 for v in c])
    assert int(on.sum()) == 2946
    assert on.groupby(df['datetime'].dt.month.values).sum().to_dict() \
        == MONTH_HOURS_2027


def test_never_forces_operation():
    _, c = year_constraints()
    assert set(c) == {0, -1}


# ── minimální doba běhu ─────────────────────────────────────────────

def test_default_min_runtime_fits_everywhere():
    assert short_blocks('w', 4) == {}


def test_short_block_check_sees_each_window_once():
    """III–X má sobota stejné okno jako Po–Pá — v hlášení jen jednou."""
    got = short_blocks('w', 7)
    assert set(got) == set(range(3, 11))
    assert got[3] == [(16, 22)] and got[5] == [(18, 24)]


# ── registrace v UI ─────────────────────────────────────────────────

def test_profile_is_registered_in_ui():
    from ui_source import assert_profile_registered
    assert_profile_registered('w')


def test_holiday_calendar_warning_covers_w():
    """W závisí na kalendáři svátků — mimo pokryté roky musí UI varovat."""
    from pathlib import Path
    tree = ast.parse((Path(opt_core.__file__).parent / 'app.py')
                     .read_text(encoding='utf-8'))
    for n in ast.walk(tree):
        if isinstance(n, ast.Set):
            vals = {e.value for e in n.elts if isinstance(e, ast.Constant)}
            if {'peak', 'offpeak'} <= vals:
                assert 'w' in vals
                return
    raise AssertionError('v app.py neni mnozina profilu se svatky')
