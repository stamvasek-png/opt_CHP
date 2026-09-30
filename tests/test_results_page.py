"""Stránka s výsledky se musí vejít do fronty zpráv Streamlitu.

Server Streamlitu (verze se Starlette) drží pro každý prohlížeč frontu
nejvýš 500 zpráv k odeslání. Skript vyrobí celou stránku za pár sekund,
velké hodinové grafy ale jdou do prohlížeče desítky sekund, takže se do
fronty dostane skoro celá stránka najednou. Když přeteče, server spojení
potichu zahodí: stránka zůstane viset ve stavu běhu a tlačítka pod
detailem se už nevykreslí. Přesně tak to dopadlo se 16 profily, kdy měl
každý vlastní záložku s metrikami a třemi hodinovými grafy — přes 1100
zpráv na stránku.

Každý prvek stránky (i sloupec nebo jiný kontejner) je jedna zpráva, takže
počet uzlů ve stromu AppTest odpovídá počtu zpráv. Testuje se nejhorší
případ: všechny profily v nabídce, měsíční analýza i roční plán.
"""

import copy
import pickle
from pathlib import Path

import pytest

from conftest import make_df, make_params, make_uses
from opt_core import calculate_smoothness_metrics, run_optimization_with_profile
from ui_source import multiselect_options

AppTest = pytest.importorskip('streamlit.testing.v1').AppTest

APP = Path(__file__).resolve().parent.parent / 'app.py'

# WEBSOCKET_MAX_SEND_QUEUE_SIZE ve streamlit/web/server/starlette
SEND_QUEUE_LIMIT = 500


def fake_cache(profiles):
    """Uložená data jako po běhu všech profilů (48 h, ať je test rychlý).

    Každý profil má jiný zisk, aby šlo poznat, který se v detailu kreslí.
    """
    df = make_df(hours=48)
    r = run_optimization_with_profile(df, make_params(), make_uses(),
                                      profile_type='free', time_limit=30)
    sm = calculate_smoothness_metrics(r['res'])
    scenarios = {}
    for i, pr in enumerate(profiles):
        result = copy.deepcopy(r)
        result['total_profit'] = 1000.0 * (i + 1)
        scenarios[pr] = {'result': result, 'smoothness': dict(sm),
                         'profile_name': pr.upper()}
    monthly = {m: {pr: {'profit': 1.0, 'profit_per_h': 1.0,
                        'smoothness': dict(sm), 'total_co2': 0.0}
                   for pr in profiles}
               for m in range(1, 13)}
    annual = r['res'].copy()
    annual['_best_profile'] = 'FREE'
    fwd = df[['datetime']].assign(ee_original=df['ee_price'],
                                  gas_original=df['gas_price'],
                                  ee_price=df['ee_price'],
                                  gas_price=df['gas_price'])
    return {'scenario_results': scenarios, 'monthly_profile_results': monthly,
            'annual_plan_result': annual, 'df_main': df, 'uses': make_uses(),
            'fwd_data': fwd}


@pytest.fixture(scope='module')
def page(tmp_path_factory):
    profiles = multiselect_options()
    data_dir = tmp_path_factory.mktemp('data')
    with open(data_dir / 'last_run.pkl', 'wb') as f:
        pickle.dump(fake_cache(profiles), f)

    mp = pytest.MonkeyPatch()
    mp.setenv('OPT_CHP_DATA_DIR', str(data_dir))
    try:
        at = AppTest.from_file(str(APP), default_timeout=120)
        at.run()
        choice = next(m for m in at.multiselect
                      if m.label == 'Které profily testovat?')
        choice.set_value(profiles).run()
        yield at, profiles
    finally:
        mp.undo()


def n_nodes(node):
    return 1 + sum(n_nodes(c) for c in getattr(node, 'children', {}).values())


def detail_switch(at):
    return next(r for r in at.radio if r.label == 'Profil')


def test_page_renders_to_the_end(page):
    at, _ = page
    assert not at.exception
    labels = [b.label for b in at.button]
    assert any('Připravit Excel scénářů' in lb for lb in labels)
    assert any('Připravit provozní plán' in lb for lb in labels)


def test_page_fits_streamlit_send_queue(page):
    """Rezerva 20 % — fronta se mezitím i vyprazdňuje, ale ne spolehlivě."""
    at, _ = page
    n = n_nodes(at.main) + n_nodes(at.sidebar)
    assert n < 0.8 * SEND_QUEUE_LIMIT, n


def test_detail_offers_every_profile(page):
    at, profiles = page
    assert detail_switch(at).options == [pr.upper() for pr in profiles]


def test_switching_profile_redraws_detail(page):
    at, profiles = page
    target = profiles.index('x')
    detail_switch(at).set_value('x').run()
    assert not at.exception
    assert detail_switch(at).value == 'x'
    profit = next(m for m in at.metric if m.label == 'Celkový zisk')
    assert profit.value == f"{1000.0 * (target + 1):,.0f} €"
