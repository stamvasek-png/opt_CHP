"""Testy profilu U — týdenní šablona po měsících nad FWD křivkou 2027.

Pravidla oken (zadání + technická nutnost):
  · blok nejvýš 96 h, mezi bloky aspoň 8 h pauza,
  · blok aspoň 4 h (jinak by do něj KGJ s min. dobou běhu 4 h nenastartovala),
  · nejvýš 1 start za kalendářní den.

Všechno se ověřuje na skutečné ose 2027 v místním čase, stejné jako ve
FWD souboru: 28. 3. chybí hodina 02:00, 31. 10. je dvakrát. Pravidla musí
platit i přes přechody mezi měsíci, kde se šablona mění.
"""

import numpy as np
import pandas as pd
import pytest

from opt_core import (U_WEEK_BLOCKS, WEEKDAY_ABBR, create_profile_constraints,
                      short_blocks, week_hours)

YEAR_HOURS = 7032
MONTH_HOURS = {1: 672, 2: 576, 3: 667, 4: 532, 5: 498, 6: 492, 7: 640,
               8: 518, 9: 436, 10: 673, 11: 656, 12: 672}


def local_axis(year=2027):
    t = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h',
                      tz='Europe/Prague')
    return pd.Series(t.tz_localize(None))


@pytest.fixture(scope='module')
def axis():
    return local_axis()


@pytest.fixture(scope='module')
def on(axis):
    c = create_profile_constraints(pd.DataFrame({'datetime': axis}), 'u')
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

def test_week_hours_simple_block():
    assert week_hours((('Út 06', 'Út 10'),)) == {30, 31, 32, 33}


def test_week_hours_wraps_over_sunday():
    """So 07 → Po 22 přechází přes neděli do pondělí."""
    hrs = week_hours((('So 07', 'Po 22'),))
    assert len(hrs) == 17 + 24 + 22
    assert 5 * 24 + 7 in hrs and 6 * 24 + 23 in hrs and 21 in hrs
    assert 22 not in hrs and 5 * 24 + 6 not in hrs


def test_week_hours_end_is_exclusive():
    """Čt 06 → Ne 00 končí o půlnoci ze soboty na neděli."""
    hrs = week_hours((('Čt 06', 'Ne 00'),))
    assert max(hrs) == 6 * 24 - 1 and min(hrs) == 3 * 24 + 6


def test_every_month_has_a_template():
    assert set(U_WEEK_BLOCKS) == set(range(1, 13))
    for m, blocks in U_WEEK_BLOCKS.items():
        for start, end in blocks:
            assert start.split()[0] in WEEKDAY_ABBR, (m, start)
            assert end.split()[0] in WEEKDAY_ABBR, (m, end)


# ── pravidla na kalendáři 2027 ──────────────────────────────────────

def test_axis_has_dst_like_the_fwd_file(axis):
    assert len(axis) == 8760
    assert pd.Timestamp('2027-03-28 02:00') not in set(axis)
    assert (axis == pd.Timestamp('2027-10-31 02:00')).sum() == 2


def test_no_block_longer_than_96h(on):
    runs, _ = runs_and_gaps(on)
    assert max(runs) <= 96, max(runs)


def test_pause_at_least_8h(on):
    _, gaps = runs_and_gaps(on)
    assert min(gaps) >= 8, min(gaps)


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
    assert len(runs) == 198


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
    c = create_profile_constraints(pd.DataFrame({'datetime': axis}), 'u')
    assert set(c) == {0, -1}


def test_short_block_check_does_not_apply():
    """Týdenní šablona nemá okna po měsících — kontrola krátkých bloků mlčí."""
    assert short_blocks('u', 4) == {}


# ── ukázky šablon ───────────────────────────────────────────────────

def test_january_runs_through_the_working_week(axis, on):
    """Leden Út 06 → Pá 23 = 89 h v kuse."""
    s = pd.Series(on, index=axis)
    week = s['2027-01-12 00:00':'2027-01-16 23:00']     # Ut-So
    assert week['2027-01-12 06:00':'2027-01-15 22:00'].all()
    assert not week['2027-01-12 05:00']
    assert not week['2027-01-15 23:00']


def test_summer_skips_midday_on_regular_days(axis, on):
    """V červnu jedou noční bloky 17/18 → 07–09, poledne ne."""
    s = pd.Series(on, index=axis)
    day = s['2027-06-16 00:00':'2027-06-16 23:00']       # streda
    assert not day['2027-06-16 12:00'] and not day['2027-06-16 14:00']
    assert day['2027-06-16 20:00'] and day['2027-06-16 03:00']


# ── registrace v UI ─────────────────────────────────────────────────

def test_profile_is_registered_in_ui():
    from ui_source import assert_profile_registered
    assert_profile_registered('u')
