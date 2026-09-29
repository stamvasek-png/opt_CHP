"""Testy profilů S a T — pevné pásmo nad FWD křivkou 2027.

Okna vzešla z exaktního výběru (batoh přes měsíce, pevná velikost pásma
~3300 h) s podmínkou dostatečné poptávky po teple; září 18–23 bylo
doplněno dodatečně. S má strop 16 h na blok, T strop nemá. Hlídá se tu,
že se okna nezměnila a že roční součet sedí na zadání.
"""

import pytest

from conftest import make_df
from opt_core import (S_WINDOWS, T_WINDOWS, create_profile_constraints,
                      short_blocks)

# mesic -> okno
EXPECTED = {
    's': {1: (6, 22), 2: (6, 22), 3: (16, 24), 4: (18, 24), 5: (18, 24),
          6: (18, 24), 7: (18, 24), 8: (18, 24), 9: (18, 23), 10: (15, 22),
          11: (6, 22), 12: (7, 23)},
    't': {1: (5, 24), 2: (2, 24), 3: (17, 23), 4: (18, 23), 5: (19, 23),
          6: (18, 24), 7: (18, 24), 8: (18, 23), 9: (18, 23), 10: (16, 21),
          11: (6, 23), 12: (7, 21)},
}
YEAR_2027 = {'s': 3453, 't': 3435}
WINDOWS = {'s': S_WINDOWS, 't': T_WINDOWS}
PROFILES = ('s', 't')


def allowed(profile, day):
    df = make_df(hours=24, start=f'{day} 00:00')
    c = create_profile_constraints(df, profile)
    return {h for h, v in enumerate(c) if v == 0}


def year_constraints(profile, year=2027):
    df = make_df(hours=8760, start=f'{year}-01-01 00:00')
    return create_profile_constraints(df, profile)


# ── okna ────────────────────────────────────────────────────────────

@pytest.mark.parametrize('profile', PROFILES)
def test_windows_match_spec(profile):
    assert WINDOWS[profile] == EXPECTED[profile]


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('month', range(1, 13))
def test_calendar_matches_window(profile, month):
    lo, hi = EXPECTED[profile][month]
    assert allowed(profile, f'2027-{month:02d}-15') == set(range(lo, hi))


@pytest.mark.parametrize('profile', PROFILES)
def test_september_is_evening_only(profile):
    """Září 18–23 u obou profilů, každý den v měsíci stejně.

    Poptávka po teple je tam 0,338 MW — na plný výkon KGJ nestačí, ale na
    minimální zatížení 50 % (0,3025 MW) ano.
    """
    for day in (1, 15, 30):
        assert allowed(profile, f'2027-09-{day:02d}') == set(range(18, 23))


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('month', range(1, 13))
def test_window_is_one_unbroken_block(profile, month):
    """Jeden blok denně = jeden start denně."""
    hrs = sorted(allowed(profile, f'2027-{month:02d}-15'))
    assert hrs == list(range(hrs[0], hrs[-1] + 1))


@pytest.mark.parametrize('profile', PROFILES)
def test_runs_seven_days_a_week(profile):
    """Víkend ani svátek okno nemění."""
    workday = allowed(profile, '2027-01-13')                    # streda
    for day in ('2027-01-16', '2027-01-17', '2027-01-01'):      # So, Ne, svatek
        assert allowed(profile, day) == workday, day


@pytest.mark.parametrize('profile', PROFILES)
def test_summer_skips_midday(profile):
    """Hluboký solární propad (IV–VIII, 10–16 h) zůstává mimo pásmo.

    V těchto měsících padá FWD kolem poledne na 6–60 €/MWh a do záporu.
    V březnu a říjnu je propad mírný — S tam okno 15–22 začíná v 15:00
    (říjen 15 h ≈ 130 €/MWh), což je v pořádku.
    """
    for month in range(4, 9):
        assert not (allowed(profile, f'2027-{month:02d}-15') & set(range(10, 16)))


def test_s_block_never_exceeds_16h():
    for month in range(1, 13):
        assert len(allowed('s', f'2027-{month:02d}-15')) <= 16, month


def test_t_has_longer_winter_blocks():
    """T se od S liší právě delšími zimními bloky."""
    assert len(allowed('t', '2027-01-15')) == 19
    assert len(allowed('t', '2027-02-15')) == 22
    assert len(allowed('s', '2027-01-15')) == 16


# ── roční součty ────────────────────────────────────────────────────

@pytest.mark.parametrize('profile', PROFILES)
def test_year_2027_hour_budget(profile):
    c = year_constraints(profile)
    assert sum(1 for v in c if v == 0) == YEAR_2027[profile]


@pytest.mark.parametrize('profile', PROFILES)
def test_one_block_per_day(profile):
    """365 bloků za rok — každý den jeden, žádný den se dvěma."""
    blocks, prev = 0, -1
    for v in year_constraints(profile):
        if v == 0 and prev != 0:
            blocks += 1
        prev = v
    assert blocks == 365


@pytest.mark.parametrize('profile', PROFILES)
def test_never_forces_operation(profile):
    """Profil je jen povolené okno — nikde nevynucuje provoz (žádné 1)."""
    assert set(year_constraints(profile)) == {0, -1}


# ── minimální doba běhu ─────────────────────────────────────────────

def test_empty_window_is_not_a_short_block(monkeypatch):
    """Měsíc bez provozu (okno (0, 0)) nesmí vyjít jako krátký blok.

    Jinak by UI u takového profilu hlásilo falešné varování, že se do
    bloku nevejde minimální doba běhu.
    """
    import opt_core
    monkeypatch.setitem(opt_core.MONTH_WINDOW_PROFILES, '_test',
                        {**S_WINDOWS, 9: (0, 0)})
    assert 9 not in short_blocks('_test', 4)
    assert 9 not in short_blocks('_test', 24)


@pytest.mark.parametrize('profile', PROFILES)
def test_default_min_runtime_fits_everywhere(profile):
    """S výchozí min. dobou běhu 4 h jde nastartovat do každého bloku."""
    assert short_blocks(profile, 4) == {}


def test_s_shortest_block_is_september():
    """Nejkratší blok S je září 18–23 (5 h) — s min. dobou běhu 6 h by
    zůstalo nevyužité jako jediné."""
    assert short_blocks('s', 6) == {9: [(18, 23)]}


def test_t_shortest_block_is_may():
    """Nejkratší blok T je květen 19–23 (4 h) — s 5 h by nešel použít."""
    assert short_blocks('t', 5) == {5: [(19, 23)]}


# ── registrace v UI ─────────────────────────────────────────────────

@pytest.mark.parametrize('profile', PROFILES)
def test_profile_is_registered_in_ui(profile):
    from ui_source import assert_profile_registered
    assert_profile_registered(profile)
