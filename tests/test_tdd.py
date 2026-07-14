import datetime as dt

import numpy as np
import pandas as pd
import pytest

from core.tdd import (TddStore, consumption_curve, is_business_day,
                      parse_ote_tdd, synthetic_tdd_year)
from core.timegrid import make_grid


def test_is_business_day():
    assert is_business_day(pd.Timestamp("2026-07-15"))       # středa
    assert not is_business_day(pd.Timestamp("2026-07-18"))   # sobota
    assert not is_business_day(pd.Timestamp("2026-07-06"))   # svátek


def test_synthetic_year_shape():
    df = synthetic_tdd_year(2026)
    assert len(df) == 8760
    assert list(df.columns) == [f"TDD{i}" for i in range(1, 9)]
    assert (df.to_numpy() > 0).all()


def test_consumption_curve_annual_energy(tdd_store):
    """Součet spotřeby přes celý rok musí dát annual_mwh."""
    annual = 120.0
    total = 0.0
    d = dt.date(2026, 1, 1)
    while d.year == 2026:
        g = make_grid(d)
        mw = consumption_curve("TDD4", annual, g, store=tdd_store)
        assert len(mw) == g.n
        total += mw.sum() * g.dt_h
        d += dt.timedelta(days=32)
        d = d.replace(day=1)
    # jen vzorek měsíců → kontrola řádu na jednom dni:
    g = make_grid(dt.date(2026, 7, 15))
    mw = consumption_curve("TDD4", annual, g, store=tdd_store)
    daily = mw.sum() * g.dt_h
    assert 0.1 < daily < 1.5  # ~120/365 ≈ 0.33 MWh/den, tvarové odchylky OK
    assert (mw >= 0).all()


def test_consumption_curve_full_year_sums(tdd_store):
    """Přesná kontrola: Σ přes všechny dny roku == annual_mwh."""
    annual = 50.0
    total = 0.0
    d = dt.date(2026, 1, 1)
    while d.year == 2026:
        g = make_grid(d)
        total += consumption_curve("TDD1", annual, g, store=tdd_store).sum() * g.dt_h
        d += dt.timedelta(days=1)
    assert total == pytest.approx(annual, rel=1e-9)


def test_consumption_dst_days(tdd_store):
    tdd_store.save(2026, synthetic_tdd_year(2026))
    for date, n in [(dt.date(2026, 3, 29), 92), (dt.date(2026, 10, 25), 100)]:
        g = make_grid(date)
        mw = consumption_curve("TDD4", 100.0, g, store=tdd_store)
        assert len(mw) == n


def test_parse_ote_tdd_roundtrip(tmp_path):
    df = synthetic_tdd_year(2026)
    raw = df.reset_index().rename(columns={
        "index": "Datum a čas",
        **{f"TDD{i}": f"TDD č. {i} přepočtený" for i in range(1, 9)}})
    import io
    buf = io.BytesIO()
    raw.to_excel(buf, index=False)
    buf.seek(0)
    buf.name = "tdd.xlsx"
    parsed = parse_ote_tdd(buf, 2026)
    assert list(parsed.columns) == [f"TDD{i}" for i in range(1, 9)]
    assert len(parsed) == 8760
    store = TddStore(base_dir=tmp_path)
    store.save(2026, parsed)
    assert store.available_years() == [2026]
    np.testing.assert_allclose(store.load(2026)["TDD4"].values,
                               df["TDD4"].values)
