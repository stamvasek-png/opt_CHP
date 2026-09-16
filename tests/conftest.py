"""Sdílené fixtury pro testy výpočetního jádra.

Horizonty jsou schválně krátké (8–48 h), aby celá sada běžela v řádu sekund —
delší limit solveru z `DEFAULT_SOLVER_TIME_LIMIT` je určený pro UI, ne pro testy.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Jmenovitý tepelný výkon zvolený tak, aby 100 % zatížení odpovídalo 975 kW_th
# z reálných dat, na kterých je feature kalibrovaná.
K_TH = 0.975

SOLVER_TIME_LIMIT = 30


def make_df(hours=8, ee_price=100.0, gas_price=30.0, heat_demand=1.0,
            start='2026-01-05 00:00'):
    """Minimální vstupní rámec ve tvaru, jaký očekává solver.

    2026-01-05 je pondělí a není státní svátek, takže PEAK/EXTPEAK okna platí.
    """
    idx = pd.date_range(start, periods=hours, freq='h')
    as_list = lambda v: list(v) if isinstance(v, (list, tuple)) else [float(v)] * hours
    return pd.DataFrame({
        'datetime': idx,
        'ee_price': as_list(ee_price),
        'gas_price': as_list(gas_price),
        'Poptávka po teple (MW)': as_list(heat_demand),
        'FVE (MW)': [0.0] * hours,
    })


def make_params(**overrides):
    """Parametry s jedinou zapnutou technologií (KGJ) a bez rampy."""
    p = {
        'k_th': K_TH, 'k_eff_th': 0.531, 'k_eff_el': 0.395, 'k_min': 1.0,
        'k_start_cost': 1.0, 'k_min_runtime': 1, 'k_service_cost': 0.0,
        'h_price': 95.0, 'h_cover': 0.0, 'shortfall_penalty': 0.0,
        'dist_ee_buy': 0.0, 'dist_ee_sell': 0.0, 'gas_dist': 0.0,
        'internal_ee_use': True,
        'kgj_ramp_on': False, 'k_ramp_up_min': 0.0, 'k_ramp_down_min': 0.0,
    }
    p.update(overrides)
    return p


def make_uses(**overrides):
    u = dict(kgj=True, boil=False, ek=False, tes=False,
             bess=False, fve=False, ext_heat=False)
    u.update(overrides)
    return u


def with_ramp(p, up_min, down_min):
    p = dict(p)
    p.update(kgj_ramp_on=True, k_ramp_up_min=up_min, k_ramp_down_min=down_min)
    return p


@pytest.fixture
def df():
    return make_df()


@pytest.fixture
def params():
    return make_params()


@pytest.fixture
def uses():
    return make_uses()
