"""Testy profilu V — plán pro dispečink nad FWD křivkou 2027.

Pravidla oken (zadání + technická nutnost):
  · blok nejvýš 96 h, mezi bloky aspoň 16 h pauza,
  · blok aspoň 4 h (jinak by do něj KGJ s min. dobou běhu 4 h nenastartovala),
  · nejvýš 1 start za kalendářní den,
  · září bez provozu.

Všechno se ověřuje na skutečné ose 2027 v místním čase, stejné jako ve
FWD souboru: 28. 3. chybí hodina 02:00, 31. 10. je dvakrát. Pravidla musí
platit i přes přechody mezi měsíci, kde se šablona mění.
"""

import numpy as np
import pandas as pd
import pytest

from opt_core import (V_WEEK_BLOCKS, V_WINTER_BLOCKS, WEEKDAY_ABBR,
                      create_profile_constraints, daily_blocks, short_blocks,
                      week_hours)

YEAR_HOURS = 4136
MONTH_HOURS = {1: 600, 2: 544, 3: 217, 4: 240, 5: 186, 6: 240, 7: 248,
               8: 248, 9: 0, 10: 421, 11: 592, 12: 600}
# Měsíce s jedním oknem každý den: hodiny okna podle kalendářního dne.
DAILY_HOURS = {
    3: set(range(17, 24)),
    4: set(range(18, 24)) | {0, 1},
    5: set(range(19, 24)) | {0},
    6: set(range(18, 24)) | {0, 1},
    7: set(range(18, 24)) | {0, 1},
    8: set(range(18, 24)) | {0, 1},
}


def local_axis(year=2027):
    t = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h',
                      tz='Europe/Prague')
    return pd.Series(t.tz_localize(None))


@pytest.fixture(scope='module')
def axis():
    return local_axis()


@pytest.fixture(scope='module')
def on(axis):
    c = create_profile_constraints(pd.DataFrame({'datetime': axis}), 'v')
    return np.array([v == 0 for v in c])


def runs_and_gaps(on):
    """Délky souvislých bloků okna a pauz mezi nimi (bez okrajů roku)."""
    runs, gaps, cur, gap, seen = [], [], 0, 0, False
    for v in on:
        if v:
            if gap and seen:
                gaps.append(gap)
            gap, cur, seen = 0, cur + 1, True
        else:
            if cur:
                runs.append(cur)
                cur = 0
            gap += 1
    if cur:
        runs.append(cur)
    return runs, gaps


# ── zápis šablony ───────────────────────────────────────────────────

def test_daily_blocks_over_midnight():
    """18 → 02 končí druhý den, neděle přechází do pondělí."""
    blocks = daily_blocks(18, 2)
    assert len(blocks) == 7
    assert blocks[0] == ('Po 18', 'Út 02')
    assert blocks[-1] == ('Ne 18', 'Po 02')
    assert len(week_hours(blocks)) == 7 * 8


def test_daily_blocks_until_midnight():
    assert daily_blocks(17, 0)[0] == ('Po 17', 'Út 00')
    assert len(week_hours(daily_blocks(17, 0))) == 7 * 7


def test_daily_blocks_within_one_day():
    assert daily_blocks(10, 14)[2] == ('St 10', 'St 14')
    assert len(week_hours(daily_blocks(10, 14))) == 7 * 4


def test_every_month_has_a_template():
    assert set(V_WEEK_BLOCKS) == set(range(1, 13))
    for m, blocks in V_WEEK_BLOCKS.items():
        for start, end in blocks:
            assert start.split()[0] in WEEKDAY_ABBR, (m, start)
            assert end.split()[0] in WEEKDAY_ABBR, (m, end)


def test_winter_months_share_one_template():
    """XI–II: dva bloky a dvě pauzy po 16 h = 136 h týdně."""
    for m in (11, 12, 1, 2):
        assert V_WEEK_BLOCKS[m] == V_WINTER_BLOCKS, m
    assert len(week_hours(V_WINTER_BLOCKS)) == 168 - 2 * 16


# ── pravidla na kalendáři 2027 ──────────────────────────────────────

def test_axis_has_dst_like_the_fwd_file(axis):
    assert len(axis) == 8760
    assert pd.Timestamp('2027-03-28 02:00') not in set(axis)
    assert (axis == pd.Timestamp('2027-10-31 02:00')).sum() == 2


def test_no_block_longer_than_96h(on):
    runs, _ = runs_and_gaps(on)
    assert max(runs) <= 96, max(runs)


def test_pause_at_least_16h(on):
    _, gaps = runs_and_gaps(on)
    assert min(gaps) >= 16, min(gaps)


def test_no_block_shorter_than_4h(on):
    runs, _ = runs_and_gaps(on)
    assert min(runs) >= 4, min(runs)


def test_at_most_one_start_per_day(axis, on):
    starts = on & ~np.concatenate(([False], on[:-1]))
    per_day = pd.Series(starts).groupby(axis.dt.date.values).sum()
    assert per_day.max() <= 1, per_day[per_day > 1].head()


def test_year_2027_totals(axis, on):
    assert int(on.sum()) == YEAR_HOURS
    got = pd.Series(on).groupby(axis.dt.month.values).sum().to_dict()
    assert got == MONTH_HOURS


def test_starts_per_year(on):
    runs, _ = runs_and_gaps(on)
    assert len(runs) == 234


def test_september_is_off(axis, on):
    """Poptávka 0,338 MW by znamenala mařit 44 % tepla KGJ."""
    assert V_WEEK_BLOCKS[9] == ()
    assert not on[(axis.dt.month == 9).values].any()


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


def test_same_window_every_day_march_to_august(axis, on):
    """III–VIII: každý den v měsíci stejné hodiny, bez ohledu na den v týdnu."""
    df = pd.DataFrame({'m': axis.dt.month, 'd': axis.dt.date,
                       'h': axis.dt.hour, 'on': on})
    df = df[df['d'] != pd.Timestamp('2027-03-28').date()]
    for m, hours in DAILY_HOURS.items():
        g = df[df['m'] == m]
        for d, gg in g.groupby('d'):
            assert set(gg.loc[gg['on'], 'h']) == hours, (m, d)


def test_never_forces_operation(axis):
    c = create_profile_constraints(pd.DataFrame({'datetime': axis}), 'v')
    assert set(c) == {0, -1}


def test_short_block_check_does_not_apply():
    """Týdenní šablona nemá okna po měsících — kontrola krátkých bloků mlčí."""
    assert short_blocks('v', 4) == {}


# ── ukázky šablon ───────────────────────────────────────────────────

def test_january_two_blocks_a_week(axis, on):
    """Leden: Ne 15 → St 21 (78 h), pauza 16 h, pak Čt 13 → So 23."""
    s = pd.Series(on, index=axis)
    assert s['2027-01-10 15:00':'2027-01-13 20:00'].all()      # Ne-St
    assert not s['2027-01-10 14:00']
    assert not s['2027-01-13 21:00':'2027-01-14 12:00'].any()
    assert s['2027-01-14 13:00':'2027-01-16 22:00'].all()      # Ct-So
    assert not s['2027-01-16 23:00']


def test_summer_runs_evening_to_2am(axis, on):
    """V červnu každý den 18:00–02:00, poledne ne."""
    s = pd.Series(on, index=axis)
    assert s['2027-06-16 18:00':'2027-06-17 01:00'].all()      # streda
    assert not s['2027-06-16 17:00'] and not s['2027-06-17 02:00']
    assert not s['2027-06-16 12:00']


# ── registrace v UI ─────────────────────────────────────────────────

def test_profile_is_registered_in_ui():
    from ui_source import assert_profile_registered
    assert_profile_registered('v')
