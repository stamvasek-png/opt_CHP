"""Testy profilu EXTPSUM — ExtPeak s letní úpravou.

Zadání: stejné jako EXTPEAK, ale od 1. 6. do 30. 9. včetně nejde brát
11:00–17:00, zato jdou navíc 04:00–06:00 a 22:00–24:00.
"""

from pathlib import Path

import pytest

from conftest import make_df
from opt_core import create_profile_constraints

# Hodina h pokryva interval [h, h+1), takze zakaz 11:00-17:00 = hodiny 11..16
# a hodina 17 (17:00-18:00) uz je povolena.
SUMMER_ALLOWED = {4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 23}
WINTER_ALLOWED = set(range(6, 22))

# 2026: 15. 1. je ctvrtek, 15. 7. streda, 30. 9. streda - vsechno pracovni dny.
# 1. 6. 2026 je pondeli, 4. 7. sobota.
WINTER_DAY, SUMMER_DAY = '2026-01-15', '2026-07-15'


def _allowed_hours(day):
    df = make_df(hours=24, start=f'{day} 00:00')
    c = create_profile_constraints(df, 'extpsum')
    return {h for h, v in enumerate(c) if v == 0}


def test_summer_blocks_midday_and_adds_edges():
    """Jádro zadání: v létě zmizí poledne a přibudou okraje dne."""
    assert _allowed_hours(SUMMER_DAY) == SUMMER_ALLOWED


def test_summer_has_fourteen_hours():
    assert len(_allowed_hours(SUMMER_DAY)) == 14


def test_hour_seventeen_is_allowed_in_summer():
    """Zákaz končí v 17:00, takže hodina 17:00–18:00 už se brát smí.

    Off-by-one by tu stál 122 hodin ročně.
    """
    allowed = _allowed_hours(SUMMER_DAY)
    assert 16 not in allowed, '16:00-17:00 je posledni zakazana'
    assert 17 in allowed, '17:00-18:00 uz je povolena'


def test_outside_summer_is_plain_extpeak():
    """Mimo VI–IX se profil chová přesně jako EXTPEAK."""
    assert _allowed_hours(WINTER_DAY) == WINTER_ALLOWED
    df = make_df(hours=24, start=f'{WINTER_DAY} 00:00')
    assert (create_profile_constraints(df, 'extpsum')
            == create_profile_constraints(df, 'extpeak'))


# Vsechny ctyri musi byt pracovni dny, jinak by test meril vikend misto hranice.
@pytest.mark.parametrize('day, summer', [
    ('2026-05-29', False),   # patek pred zacatkem (31. 5. je nedele)
    ('2026-06-01', True),    # pondeli, prvni den letni upravy
    ('2026-09-30', True),    # streda, posledni den letni upravy
    ('2026-10-01', False),   # ctvrtek, den po konci
])
def test_summer_window_boundaries(day, summer):
    """Úprava platí od 1. 6. do 30. 9. včetně."""
    expected = SUMMER_ALLOWED if summer else WINTER_ALLOWED
    assert _allowed_hours(day) == expected, day


def test_weekends_and_holidays_stay_off():
    """Základ je EXTPEAK, takže mimo pracovní dny se neběží ani v létě."""
    assert _allowed_hours('2026-07-04') == set(), 'sobota'
    assert _allowed_hours('2026-07-05') == set(), 'statni svatek'
    assert _allowed_hours('2026-01-01') == set(), 'Novy rok'


def test_summer_is_subset_of_nothing_silly():
    """Letní okno není jen podmnožinou zimního — okraje dne přibyly."""
    s, w = _allowed_hours(SUMMER_DAY), _allowed_hours(WINTER_DAY)
    assert s - w == {4, 5, 22, 23}, 'nove hodiny na okrajich dne'
    assert w - s == {11, 12, 13, 14, 15, 16}, 'vypadle poledne'


def test_profile_is_registered_in_ui():
    """Profil musí být k výběru a mít barvu, jinak se k němu uživatel nedostane."""
    from ui_source import assert_profile_registered
    assert_profile_registered('extpsum')


def test_boundary_dates_are_business_days():
    """Pojistka k testu výše: hraniční data musí být pracovní dny.

    Jinak by test měřil víkend a prošel by i se špatně nastavenou hranicí.
    """
    from opt_core import is_business_day
    import pandas as pd
    for day in ('2026-05-29', '2026-06-01', '2026-09-30', '2026-10-01'):
        assert is_business_day(pd.Timestamp(day)), day
