"""Testy profilu Y — X s letními rány.

Mimo léto je Y stejný jako X. Ve VI–VIII jede Po–Pá 05–09 a 18–23 (dva
starty), o víkendu 18–24. Pravidla X platí dál: blok nejvýš 48 h, zima má
nejvíc hodin, jaro a podzim méně, léto nejméně — i měsíc po měsíci.

Ověřuje se na skutečné ose 2027 v místním čase. Osa začíná 1. 1. o půlnoci
uprostřed zimního bloku, takže první blok je useknutý a kontrola minimální
délky ho vynechává.
"""

import numpy as np
import pandas as pd
import pytest

from opt_core import (X_WEEK_BLOCKS, Y_SUMMER_BLOCKS, Y_WEEK_BLOCKS,
                      create_profile_constraints, short_blocks, week_hours)

YEAR_HOURS = 5054
MONTH_HOURS = {1: 672, 2: 608, 3: 498, 4: 414, 5: 410, 6: 246, 7: 252,
               8: 252, 9: 268, 10: 397, 11: 510, 12: 527}
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
    c = create_profile_constraints(pd.DataFrame({'datetime': axis}), 'y')
    return np.array([v == 0 for v in c])


def runs_and_gaps(on):
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


def day_hours(on, axis, day):
    s = pd.Series(on, index=axis)[f'{day} 00:00':f'{day} 23:00']
    return set(s[s].index.hour)


# ── šablona ─────────────────────────────────────────────────────────

def test_same_as_x_outside_summer():
    for m in set(range(1, 13)) - set(SUMMER):
        assert Y_WEEK_BLOCKS[m] == X_WEEK_BLOCKS[m], m


def test_summer_template():
    """Po–Pá 05–09 a 18–23, So a Ne 18–24 — 57 h týdně."""
    for m in SUMMER:
        assert Y_WEEK_BLOCKS[m] == Y_SUMMER_BLOCKS
    hrs = week_hours(Y_SUMMER_BLOCKS)
    for w in range(5):
        assert {h for h in range(24) if w * 24 + h in hrs} == \
            set(range(5, 9)) | set(range(18, 23))
    for w in (5, 6):
        assert {h for h in range(24) if w * 24 + h in hrs} == set(range(18, 24))
    assert len(hrs) == 57


# ── pravidla na kalendáři 2027 ──────────────────────────────────────

def test_year_2027_totals(axis, on):
    assert int(on.sum()) == YEAR_HOURS
    got = pd.Series(on).groupby(axis.dt.month.values).sum().to_dict()
    assert got == MONTH_HOURS


def test_seasons_ordered_month_by_month():
    h = MONTH_HOURS
    assert min(h[m] for m in WINTER) >= max(h[m] for m in MID)
    assert min(h[m] for m in MID) >= max(h[m] for m in SUMMER)


def test_blocks_and_pauses(on):
    runs, gaps = runs_and_gaps(on)
    assert max(runs) <= 48
    assert min(runs[1:]) >= 4                  # prvni blok je useknuty osou
    assert min(gaps) >= 4


def test_at_most_two_starts_per_day(axis, on):
    starts = on & ~np.concatenate(([False], on[:-1]))
    per_day = pd.Series(starts).groupby(axis.dt.date.values).sum()
    assert per_day.max() <= 2


def test_summer_weekday_and_weekend(axis, on):
    assert day_hours(on, axis, '2027-07-14') == set(range(5, 9)) | set(range(18, 23))
    assert day_hours(on, axis, '2027-07-17') == set(range(18, 24))


def test_may_to_june_transition(axis, on):
    """31. 5. (Po) jede květnový večer do půlnoci, 1. 6. už červnové ráno."""
    s = pd.Series(on, index=axis)
    assert s['2027-05-31 18:00':'2027-05-31 23:00'].all()
    assert not s['2027-06-01 00:00':'2027-06-01 04:00'].any()
    assert s['2027-06-01 05:00':'2027-06-01 08:00'].all()


def test_never_forces_operation(axis):
    c = create_profile_constraints(pd.DataFrame({'datetime': axis}), 'y')
    assert set(c) == {0, -1}


def test_short_block_check_does_not_apply():
    assert short_blocks('y', 4) == {}


# ── registrace v UI ─────────────────────────────────────────────────

def test_profile_is_registered_in_ui():
    from ui_source import assert_profile_registered
    assert_profile_registered('y')
