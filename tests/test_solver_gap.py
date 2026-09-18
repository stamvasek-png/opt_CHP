"""Testy tolerance od optima (gapRel) předávané solveru."""

import inspect

import pulp
import pytest

import opt_core
from conftest import SOLVER_TIME_LIMIT, make_df, make_params, make_uses
from opt_core import run_optimization_with_profile


def test_solver_has_gap_parameter():
    sig = inspect.signature(run_optimization_with_profile)
    assert 'gap_rel' in sig.parameters
    assert sig.parameters['gap_rel'].default == 0.01, 'výchozí tolerance 1 %'


def _captured_kwargs(monkeypatch):
    """Odchytí, s čím se volá PULP_CBC_CMD."""
    seen = {}
    real = pulp.PULP_CBC_CMD

    def spy(**kw):
        seen.update(kw)
        return real(**{k: v for k, v in kw.items() if k != 'gapRel'} |
                    ({'gapRel': kw['gapRel']} if kw.get('gapRel') else {}))

    monkeypatch.setattr(opt_core.pulp, 'PULP_CBC_CMD', spy)
    return seen


@pytest.mark.parametrize('gap, expected', [
    (0.01, 0.01),
    (0.05, 0.05),
    (0, None),       # 0 znamena dokazovat optimalitu, volba se vubec neposila
    (None, None),
])
def test_gap_is_passed_to_cbc(monkeypatch, gap, expected):
    seen = _captured_kwargs(monkeypatch)
    run_optimization_with_profile(
        make_df(), make_params(), make_uses(),
        time_limit=SOLVER_TIME_LIMIT, gap_rel=gap)
    assert seen.get('gapRel') == expected


def test_gap_does_not_break_the_result(monkeypatch):
    """S tolerancí musí model dál vracet konzistentní výsledek."""
    r = run_optimization_with_profile(
        make_df(hours=48), make_params(), make_uses(),
        time_limit=SOLVER_TIME_LIMIT, gap_rel=0.01)
    assert r is not None
    # reportovany zisk dal sedi na objective, i kdyz reseni neni dokazane optimum
    assert r['total_profit'] == pytest.approx(r['lp_objective'], abs=1e-6)


def test_app_threads_gap_through_wrappers():
    """Wrappery v app.py musí mít gap_rel, jinak by ho UI nemělo kam předat."""
    src = open(f'{opt_core.__file__.rsplit("/", 1)[0]}/app.py', encoding='utf-8').read()
    assert 'DEFAULT_SOLVER_GAP_REL = 0.01' in src
    for fn in ('run_scenario_analysis', 'run_monthly_profile_analysis',
               'run_sensitivity_analysis'):
        start = src.index(f'def {fn}(')
        head = src[start:src.index('):', start)]
        assert 'gap_rel' in head, f'{fn} nebere gap_rel'
