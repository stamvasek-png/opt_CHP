"""Testy profilu X — vzor z cenové analýzy FWD křivky 2027.

Zadání:
  · žádný blok delší než 48 h,
  · zima má nejvíc hodin, jaro a podzim méně, léto nejméně,
  · rozpočet zhruba 5000 h za rok.

Ověřuje se na skutečné ose 2027 v místním čase (28. 3. chybí 02:00,
31. 10. je dvakrát). Osa začíná 1. 1. o půlnoci uprostřed zimního bloku,
takže první blok je useknutý — kontrola minimální délky ho vynechává.
"""

import numpy as np
import pandas as pd
import pytest

from opt_core import (WEEKDAY_ABBR, X_WEEK_BLOCKS, X_WINTER_BLOCKS,
                      create_profile_constraints, daily_blocks, short_blocks,
                      week_hours)

YEAR_HOURS = 5040
MONTH_HOURS = {1: 672, 2: 608, 3: 498, 4: 414, 5: 410, 6: 240, 7: 248,
               8: 248, 9: 268, 10: 397, 11: 510, 12: 527}
WINTER, MID, SUMMER = (12, 1, 2), (3, 4, 5, 9, 10, 11), (6, 7, 8)


def local_axis(year=2027):
    t = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h',
                      tz='Europe/Prague')
    return pd.Series(t.tz_localize(None))


@pytest.fixture(scope='module')
def axis():
    return local_axis()


@pytest.fixture(scope='module')
def on(axis):
    c = create_profile_constraints(pd.DataFrame({'datetime': axis}), 'x')
    return np.array([v == 0 for v in c])


def runs_of(on):
    """Délky souvislých bloků okna; první blok, pokud začíná s osou, zvlášť."""
    runs, cur = [], 0
    for v in on:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return (runs[1:], runs[0]) if on[0] else (runs, None)


# ── zápis šablony ───────────────────────────────────────────────────

def test_daily_blocks_on_selected_days():
    blocks = daily_blocks(18, 9, range(5))
    assert blocks[0] == ('Po 18', 'Út 09') and blocks[-1] == ('Pá 18', 'So 09')
    assert len(blocks) == 5
    assert len(daily_blocks(18, 2)) == 7                 # vychozi = vsechny dny


def test_every_month_has_a_template():
    assert set(X_WEEK_BLOCKS) == set(range(1, 13))
    for m, blocks in X_WEEK_BLOCKS.items():
        for start, end in blocks:
            assert start.split()[0] in WEEKDAY_ABBR, (m, start)
            assert end.split()[0] in WEEKDAY_ABBR, (m, end)


def test_winter_runs_around_the_clock_with_four_night_pauses():
    for m in (1, 2):
        assert X_WEEK_BLOCKS[m] == X_WINTER_BLOCKS
    assert len(week_hours(X_WINTER_BLOCKS)) == 168 - 4 * 4


# ── pravidla na kalendáři 2027 ──────────────────────────────────────

def test_no_block_longer_than_48h(on):
    runs, first = runs_of(on)
    assert max(runs) <= 48 and first <= 48


def test_no_block_shorter_than_4h(on):
    runs, _ = runs_of(on)
    assert min(runs) >= 4, min(runs)


def test_year_2027_totals(axis, on):
    assert int(on.sum()) == YEAR_HOURS
    got = pd.Series(on).groupby(axis.dt.month.values).sum().to_dict()
    assert got == MONTH_HOURS


def test_seasons_ordered_month_by_month():
    """Každý zimní měsíc ≥ každý jarní/podzimní ≥ každý letní."""
    h = MONTH_HOURS
    assert min(h[m] for m in WINTER) >= max(h[m] for m in MID)
    assert min(h[m] for m in MID) >= max(h[m] for m in SUMMER)


def test_consistent_within_month(axis, on):
    """Každý stejný den v týdnu má v rámci měsíce stejné hodiny okna."""
    df = pd.DataFrame({'m': axis.dt.month, 'w': axis.dt.weekday,
                       'd': axis.dt.date, 'h': axis.dt.hour, 'on': on})
    # 28. 3. a 31. 10. maji o hodinu min / vic - zmena casu, ne sablona
    df = df[~df['d'].isin({pd.Timestamp('2027-03-28').date(),
                           pd.Timestamp('2027-10-31').date()})]
    for (m, w), g in df.groupby(['m', 'w']):
        days = {d: frozenset(gg.loc[gg['on'], 'h']) for d, gg in g.groupby('d')}
        assert len(set(days.values())) == 1, (m, w)


def test_never_forces_operation(axis):
    c = create_profile_constraints(pd.DataFrame({'datetime': axis}), 'x')
    assert set(c) == {0, -1}


def test_short_block_check_does_not_apply():
    """Týdenní šablona nemá okna po měsících — kontrola krátkých bloků mlčí."""
    assert short_blocks('x', 4) == {}


# ── ukázky šablon ───────────────────────────────────────────────────

def test_january_pause_every_second_night(axis, on):
    """Noc Po→Út se jede, noc Út→St má pauzu 01–05."""
    s = pd.Series(on, index=axis)
    assert s['2027-01-12 01:00':'2027-01-12 04:00'].all()       # utery
    assert not s['2027-01-13 01:00':'2027-01-13 04:00'].any()   # streda
    assert s['2027-01-13 05:00'] and s['2027-01-12 12:00']


def test_april_skips_midday(axis, on):
    """Duben: Po–Pá 18 → 09 přes noc, poledne stojí, neděle přes den taky."""
    s = pd.Series(on, index=axis)
    assert s['2027-04-14 03:00'] and s['2027-04-14 20:00']      # streda
    assert not s['2027-04-14 09:00'] and not s['2027-04-14 12:00']
    assert not s['2027-04-18 12:00']                             # nedele


def test_september_has_morning_and_evening_peak(axis, on):
    s = pd.Series(on, index=axis)
    wed = s['2027-09-15 00:00':'2027-09-15 23:00']
    assert set(wed[wed].index.hour) == set(range(6, 10)) | set(range(17, 23))
    sat = s['2027-09-18 00:00':'2027-09-18 23:00']
    assert set(sat[sat].index.hour) == set(range(17, 23))


# ── registrace v UI ─────────────────────────────────────────────────

def test_profile_is_registered_in_ui():
    from ui_source import assert_profile_registered
    assert_profile_registered('x')
