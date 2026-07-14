import datetime as dt
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.profiles import load_profile
from core.tdd import TddStore, synthetic_tdd_year
from core.timegrid import make_grid
from scripts.make_demo_day import (make_actual_da_file, make_filled_workbook,
                                   synthetic_curves)

DELIVERY = dt.date(2026, 7, 15)


@pytest.fixture(scope="session")
def grid():
    return make_grid(DELIVERY)


@pytest.fixture(scope="session")
def demo_profile():
    return load_profile("demo")


@pytest.fixture(scope="session")
def tdd_store(tmp_path_factory):
    base = tmp_path_factory.mktemp("tdd")
    store = TddStore(base_dir=base)
    store.save(DELIVERY.year, synthetic_tdd_year(DELIVERY.year))
    return store


@pytest.fixture(scope="session")
def curves(grid):
    return synthetic_curves(grid)


@pytest.fixture(scope="session")
def filled_workbook(demo_profile, grid, curves):
    return io.BytesIO(make_filled_workbook(demo_profile, grid, curves))


@pytest.fixture(scope="session")
def actual_da_file(grid, curves):
    return io.BytesIO(make_actual_da_file(grid, curves))
