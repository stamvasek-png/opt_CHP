"""Testy profilu PROM26 — dodaná hodinová maska na rok 2026.

Předloha je 8760 hodin 0/1. Beze zbytku se rozkládá na okna po měsících
a pět výjimečných dní; tady se hlídá, že rozklad dává přesně předlohu.
"""

import calendar

import pytest

from conftest import make_df
from opt_core import (PROM26_EXCEPTIONS, PROM26_WINDOWS,
                      create_profile_constraints, short_blocks)

# Soucet hodin 1 v predloze
YEAR_HOURS = 3420
# Hodin s 1 po mesicich v predloze
MONTH_HOURS = {1: 480, 2: 448, 3: 310, 4: 240, 5: 186, 6: 90, 7: 93,
               8: 93, 9: 210, 10: 307, 11: 480, 12: 483}


def allowed(day):
    df = make_df(hours=24, start=f'{day} 00:00')
    c = create_profile_constraints(df, 'prom26')
    return {h for h, v in enumerate(c) if v == 0}


def hours(*blocks):
    return {h for lo, hi in blocks for h in range(lo, hi)}


@pytest.fixture(scope='module')
def year():
    df = make_df(hours=8760, start='2026-01-01 00:00')
    return df['datetime'], create_profile_constraints(df, 'prom26')


# ── celkove soucty proti predloze ────────────────────────────────────

def test_year_total_matches_source(year):
    _, c = year
    assert sum(1 for v in c if v == 0) == YEAR_HOURS


def test_month_totals_match_source(year):
    t, c = year
    got = {}
    for ts, v in zip(t, c):
        if v == 0:
            got[ts.month] = got.get(ts.month, 0) + 1
    assert got == MONTH_HOURS


# ── okna po mesicich ─────────────────────────────────────────────────

@pytest.mark.parametrize('month, expected', [
    (2, hours((6, 22))),
    (3, hours((6, 10), (16, 22))),
    (4, hours((6, 9), (17, 22))),
    (5, hours((6, 9), (19, 22))),
    (7, hours((6, 9))),
    (9, hours((6, 9), (18, 22))),
    (11, hours((6, 22))),
])
def test_typical_day_of_month(month, expected):
    # 15. v mesici neni v zadnem mesici vyjimka
    assert allowed(f'2026-{month:02d}-15') == expected


def test_shoulder_months_have_two_blocks():
    """Na rozdíl od ostatních profilů: ráno i večer, tedy dva starty."""
    for m in (3, 4, 5, 9, 10):
        assert len(PROM26_WINDOWS[m]) == 2, m


def test_summer_is_morning_only():
    for m in (6, 7, 8):
        assert PROM26_WINDOWS[m] == ((6, 9),), m


def test_every_month_defined():
    assert set(PROM26_WINDOWS) == set(range(1, 13))


# ── vyjimecne dny ────────────────────────────────────────────────────

@pytest.mark.parametrize('day, expected', [
    ('2026-01-01', set()),                          # Novy rok
    ('2026-10-01', hours((6, 9), (18, 22))),        # jeste zarijove okno
    ('2026-12-24', hours((13, 24))),
    ('2026-12-25', hours((13, 24))),
    ('2026-12-26', hours((13, 24))),
    ('2026-12-31', hours((6, 24))),
])
def test_exception_days(day, expected):
    assert allowed(day) == expected


def test_neighbours_of_exceptions_use_month_window():
    """Výjimka nesmí přetéct na sousední den."""
    assert allowed('2026-01-02') == hours((6, 22))
    assert allowed('2026-10-02') == hours((6, 10), (16, 22))
    assert allowed('2026-12-23') == hours((6, 22))
    assert allowed('2026-12-27') == hours((6, 22))
    assert allowed('2026-09-30') == hours((6, 9), (18, 22))


def test_exceptions_are_only_for_2026():
    """Jinde se použijí samotná okna — 1. 1. 2027 není klid."""
    assert all(d.year == 2026 for d in PROM26_EXCEPTIONS)
    assert allowed('2027-01-01') == hours((6, 22))


# ── nezavislost na typu dne ──────────────────────────────────────────

def test_weekends_follow_the_month_window():
    """Předloha jede i o víkendech — sobota i neděle mají okno měsíce."""
    assert allowed('2026-02-07') == hours((6, 22))      # sobota
    assert allowed('2026-02-08') == hours((6, 22))      # nedele


def test_other_holidays_are_not_special():
    """Svátky mimo výjimky se chovají jako každý jiný den."""
    assert allowed('2026-05-01') == hours((6, 9), (19, 22))    # Svatek prace
    assert allowed('2026-07-06') == hours((6, 9))               # Hus


# ── minimalni doba behu ──────────────────────────────────────────────

def test_short_blocks_with_default_min_runtime():
    """S výchozí min. dobou běhu 4 h jsou tříhodinová okna nedosažitelná."""
    sb = short_blocks('prom26', 4)
    assert set(sb) == {4, 5, 6, 7, 8, 9}
    assert sb[5] == [(6, 9), (19, 22)], 'v kvetnu jsou kratke oba bloky'
    assert sb[7] == [(6, 9)]


def test_no_short_blocks_with_three_hours():
    assert short_blocks('prom26', 3) == {}


def test_short_blocks_other_profiles():
    """Kontrola funguje i pro okna po měsících, jiné profily nic nevrací."""
    assert short_blocks('season', 7) == {m: [(17, 23)] for m in (5, 6, 7, 8, 9)}
    assert short_blocks('peak', 24) == {}
    assert short_blocks('prom26', 1) == {}


def test_year_2026_hours_lost_to_short_blocks():
    """Kolik hodin masky při min. době běhu 4 h nejde použít.

    Počítá se z oken měsíců; výjimečné dny to posunou jen o pár hodin
    (1. 10. má ještě zářijové ranní okno 06–09).
    """
    lost = sum((hi - lo) * calendar.monthrange(2026, m)[1]
               for m, bl in short_blocks('prom26', 4).items()
               for lo, hi in bl)
    assert lost == 642


def test_profile_is_registered_in_ui():
    from ui_source import assert_profile_registered
    assert_profile_registered('prom26')
