"""Testy profilu P — pásmo z nejlepších FWD hodin ceny EE.

Okna vznikla exaktním výběrem (batoh přes měsíce, strop 16 h na blok,
rozpočet ~4000 h), takže je testujeme proti zadaným hodnotám. Kdyby je
někdo posunul, roční rozpočet i tvar se tu rozsvítí dřív, než se pustí
roční optimalizace.
"""

import pandas as pd
import pytest

from conftest import make_df
from opt_core import P_WINDOWS, create_profile_constraints, is_business_day

# mesic -> (okno, hodin/den)
EXPECTED = {
    1: ((7, 23), 16),
    2: ((6, 22), 16), 3: ((6, 22), 16),
    4: ((18, 24), 6), 5: ((18, 24), 6),
    6: ((17, 24), 7), 7: ((17, 24), 7), 8: ((17, 24), 7),
    9: ((17, 23), 6),
    10: ((6, 22), 16), 11: ((6, 22), 16), 12: ((6, 22), 16),
}
YEAR_HOURS = 4102


def allowed(day):
    df = make_df(hours=24, start=f'{day} 00:00')
    c = create_profile_constraints(df, 'p')
    return {h for h, v in enumerate(c) if v == 0}


def test_windows_match_spec():
    assert P_WINDOWS == {m: w for m, (w, _) in EXPECTED.items()}


@pytest.mark.parametrize('month', range(1, 13))
def test_calendar_matches_window(month):
    (lo, hi), size = EXPECTED[month]
    assert allowed(f'2026-{month:02d}-15') == set(range(lo, hi))
    assert hi - lo == size


@pytest.mark.parametrize('month', range(1, 13))
def test_window_is_one_unbroken_block(month):
    """Jeden blok denně = jeden start denně."""
    hrs = sorted(allowed(f'2026-{month:02d}-15'))
    assert hrs == list(range(hrs[0], hrs[-1] + 1))


@pytest.mark.parametrize('month', range(1, 13))
def test_block_never_exceeds_16h(month):
    """Strop 16 h na blok je součást zadání, ze kterého okna vyšla."""
    assert len(allowed(f'2026-{month:02d}-15')) <= 16


def test_september_is_included():
    """Září se bere výslovně, přestože při plném výkonu teplo nestačí."""
    assert allowed('2026-09-15') == set(range(17, 23))


def test_summer_skips_midday_and_pv():
    """V létě se nesmí jet přes poledne ani v době provozu FVE."""
    for month in (4, 5, 6, 7, 8, 9):
        assert not (allowed(f'2026-{month:02d}-15') & set(range(0, 17)))


def test_summer_skips_morning_peak():
    """Ranní špička se nebere — vyžadovala by druhý start za den."""
    for month in (4, 5, 6, 7, 8, 9):
        assert not (allowed(f'2026-{month:02d}-15') & set(range(4, 12)))


def test_runs_seven_days_a_week():
    """Svátky ani víkendy okno nemění."""
    workday = allowed('2026-07-13')            # pondeli
    for day in ('2026-07-11', '2026-07-12', '2026-07-05'):
        assert allowed(day) == workday, day
    assert not is_business_day(pd.Timestamp('2026-07-05')), 'ma byt svatek'


def test_year_2026_hour_budget():
    df = make_df(hours=8760, start='2026-01-01 00:00')
    c = create_profile_constraints(df, 'p')
    assert sum(1 for v in c if v == 0) == YEAR_HOURS


def test_one_block_per_day_over_a_year():
    df = make_df(hours=8760, start='2026-01-01 00:00')
    c = create_profile_constraints(df, 'p')
    blocks, prev = 0, -1
    for v in c:
        if v == 0 and prev != 0:
            blocks += 1
        prev = v
    assert blocks == 365, f'{blocks} bloku misto 365'


def test_profile_is_registered_in_ui():
    from ui_source import assert_profile_registered
    assert_profile_registered('p')
