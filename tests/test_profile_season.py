"""Testy profilů SEASON a SEASONPLUS — sezónní okno, jeden blok denně.

Zadání vzniklo z rozboru reálné forwardové křivky a poptávky po teple:
 · jedno souvislé okno na den (= jeden start denně),
 · 7 dní v týdnu, svátky se neuplatňují,
 · v létě mimo poledne a dobu provozu FVE,
 · ranní letní špička se nebere, protože druhý start se z ní nezaplatí.
"""

import pandas as pd
import pytest

from conftest import make_df
from opt_core import (SEASON_WINDOWS, SEASONPLUS_WINDOWS,
                      create_profile_constraints, is_business_day)

SEASON_HOURS = {  # mesic -> pocet povolenych hodin za den
    1: 16, 2: 16, 11: 16, 12: 16,
    3: 10, 10: 10,
    4: 8,
    5: 6, 6: 6, 7: 6, 8: 6, 9: 6,
}
SEASONPLUS_HOURS = {
    1: 18, 2: 18, 11: 18, 12: 18,
    3: 12, 10: 12,
    4: 11,
    5: 8, 6: 8, 7: 8, 8: 8, 9: 8,
}

PROFILES = [('season', SEASON_WINDOWS, SEASON_HOURS),
            ('seasonplus', SEASONPLUS_WINDOWS, SEASONPLUS_HOURS)]
NAMES = ['season', 'seasonplus']


def allowed(profile, day):
    """Povolené hodiny dne — indexy, kde constraint == 0 (volno pro solver)."""
    df = make_df(hours=24, start=f'{day} 00:00')
    c = create_profile_constraints(df, profile)
    return {h for h, v in enumerate(c) if v == 0}


# ── okna ────────────────────────────────────────────────────────────

@pytest.mark.parametrize('profile, windows, sizes', PROFILES, ids=NAMES)
def test_every_month_has_a_window(profile, windows, sizes):
    """Každý měsíc musí mít okno, jinak by KGJ v tom měsíci vůbec nejela."""
    assert set(windows) == set(range(1, 13))
    for month, (lo, hi) in windows.items():
        assert 0 <= lo < hi <= 24, month
        assert hi - lo == sizes[month], month


@pytest.mark.parametrize('profile, windows, sizes', PROFILES, ids=NAMES)
def test_window_matches_calendar(profile, windows, sizes):
    """Constraint pro 15. každého měsíce sedí na definované okno."""
    for month, (lo, hi) in windows.items():
        day = f'2026-{month:02d}-15'
        assert allowed(profile, day) == set(range(lo, hi)), day


@pytest.mark.parametrize('profile, windows, sizes', PROFILES, ids=NAMES)
def test_window_is_one_unbroken_block(profile, windows, sizes):
    """Jeden blok denně = jeden start denně. Díra v okně by ho zdvojila."""
    for month in windows:
        hrs = sorted(allowed(profile, f'2026-{month:02d}-15'))
        assert hrs == list(range(hrs[0], hrs[-1] + 1)), month


# ── zadání uživatele ────────────────────────────────────────────────

@pytest.mark.parametrize('profile, windows, sizes', PROFILES, ids=NAMES)
def test_summer_skips_midday_and_pv(profile, windows, sizes):
    """V létě (V–IX) se nesmí jet přes poledne ani v době provozu FVE."""
    for month in (5, 6, 7, 8, 9):
        hrs = allowed(profile, f'2026-{month:02d}-15')
        assert not (hrs & set(range(0, 16))), f'mesic {month}: zasahuje do FVE'


@pytest.mark.parametrize('profile, windows, sizes', PROFILES, ids=NAMES)
def test_summer_skips_morning_peak(profile, windows, sizes):
    """Ranní letní špička (04–10) se nebere — druhý start se nezaplatí."""
    for month in (5, 6, 7, 8, 9):
        hrs = allowed(profile, f'2026-{month:02d}-15')
        assert not (hrs & set(range(4, 11))), f'mesic {month}: ranni spicka'


@pytest.mark.parametrize('profile, windows, sizes', PROFILES, ids=NAMES)
def test_window_narrows_towards_summer(profile, windows, sizes):
    """Okno se s ubývající poptávkou po teple zužuje: zima > přechod > léto."""
    assert sizes[1] > sizes[3] > sizes[4] > sizes[7]


@pytest.mark.parametrize('profile, windows, sizes', PROFILES, ids=NAMES)
def test_runs_seven_days_a_week(profile, windows, sizes):
    """Teplo se topí i o víkendu a o svátku — okno se tím nemění."""
    # 2026: 11. 7. sobota, 12. 7. nedele, 13. 7. pondeli; 5. 7. statni svatek.
    workday = allowed(profile, '2026-07-13')
    for day in ('2026-07-11', '2026-07-12', '2026-07-05'):
        assert allowed(profile, day) == workday, day
    assert not is_business_day(pd.Timestamp('2026-07-05')), 'ma byt svatek'
    assert is_business_day(pd.Timestamp('2026-07-13')), 'ma byt pracovni den'


# ── vztah obou profilů ──────────────────────────────────────────────

def test_seasonplus_is_strictly_wider():
    """SEASONPLUS je nadmnožina SEASON — jen přidává hodiny na okrajích."""
    for month in range(1, 13):
        day = f'2026-{month:02d}-15'
        s, sp = allowed('season', day), allowed('seasonplus', day)
        assert s < sp, f'mesic {month}: SEASON+ neni sirsi'


def test_seasonplus_widens_both_ends():
    """Rozšiřuje se na obou koncích, ne jen jedním směrem."""
    for month, (lo, hi) in SEASON_WINDOWS.items():
        plo, phi = SEASONPLUS_WINDOWS[month]
        assert plo < lo and phi > hi, month


# ── roční součty ────────────────────────────────────────────────────
# Rozpočet 4000 provozních hodin je duvod, proc profil vznikl. Kdyby
# nekdo posunul okno, tenhle test to chytne drive nez rocni optimalizace.

@pytest.mark.parametrize('profile, expected', [('season', 3698),
                                               ('seasonplus', 4458)])
def test_year_2026_hour_budget(profile, expected):
    df = make_df(hours=8760, start='2026-01-01 00:00')
    c = create_profile_constraints(df, profile)
    assert sum(1 for v in c if v == 0) == expected


@pytest.mark.parametrize('profile', NAMES)
def test_one_start_per_day_over_a_year(profile):
    """Přes celý rok nesmí vzniknout ani jeden den se dvěma bloky."""
    df = make_df(hours=8760, start='2026-01-01 00:00')
    c = create_profile_constraints(df, profile)
    blocks, prev = 0, -1
    for v in c:
        if v == 0 and prev != 0:
            blocks += 1
        prev = v
    assert blocks == 365, f'{blocks} bloku misto 365'


# ── registrace v UI ─────────────────────────────────────────────────

@pytest.mark.parametrize('profile', NAMES)
def test_profile_is_registered_in_ui(profile):
    from ui_source import assert_profile_registered
    assert_profile_registered(profile)
