import datetime as dt

import numpy as np
import pytest

from core.timegrid import make_grid, resample_hourly_to_quarter


def test_normal_day_96():
    g = make_grid(dt.date(2026, 7, 15))
    assert g.n == 96
    assert g.n_hours == 24
    assert g.index[0].strftime("%H:%M") == "00:00"
    assert g.index[-1].strftime("%H:%M") == "23:45"
    # 6 bloků po 16 MTU
    for b in range(6):
        assert len(g.block_intervals(b)) == 16


def test_spring_dst_92():
    g = make_grid(dt.date(2026, 3, 29))
    assert g.n == 92
    assert g.n_hours == 23
    # blok 0 přijde o hodinu 02:00 → 12 MTU
    assert len(g.block_intervals(0)) == 12
    for b in range(1, 6):
        assert len(g.block_intervals(b)) == 16


def test_autumn_dst_100():
    g = make_grid(dt.date(2026, 10, 25))
    assert g.n == 100
    assert g.n_hours == 25
    # blok 0 má 02:00 dvakrát → 20 MTU
    assert len(g.block_intervals(0)) == 20
    for b in range(1, 6):
        assert len(g.block_intervals(b)) == 16


def test_mtu_labels_and_times():
    g = make_grid(dt.date(2026, 7, 15))
    labels = g.mtu_labels()
    assert labels[0] == "1 (00:00–00:15)"
    assert labels[-1] == "96 (23:45–00:00)"
    assert g.times_from()[4] == "01:00"


def test_resample_repeat():
    g = make_grid(dt.date(2026, 7, 15))
    hourly = np.arange(24, dtype=float)
    q = resample_hourly_to_quarter(hourly, g, kind="repeat")
    assert len(q) == 96
    assert (q[:4] == 0).all() and (q[-4:] == 23).all()


def test_resample_interpolate_preserves_shape():
    g = make_grid(dt.date(2026, 7, 15))
    hourly = np.zeros(24)
    hourly[12] = 1.0  # poledne
    q = resample_hourly_to_quarter(hourly, g, kind="interpolate")
    assert len(q) == 96
    # vrchol mezi středy hodin: MTU nejblíž 12:30 mají hodnotu 0.875
    assert q.max() == pytest.approx(0.875)
    assert np.argmax(q) in (49, 50)
    # hladký nárůst, žádné schody
    assert q[47] < q[48] < q[49]


def test_resample_wrong_length_raises():
    g = make_grid(dt.date(2026, 3, 29))  # 23 hodin
    with pytest.raises(ValueError):
        resample_hourly_to_quarter(np.zeros(24), g)
    # správná délka projde
    assert len(resample_hourly_to_quarter(np.zeros(23), g)) == 92
