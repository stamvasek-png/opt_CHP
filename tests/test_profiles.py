import json

import pytest

from core.profiles import (
    BESSParams, ConsumptionSpec, KGJParams, PVParams, Profile, Site,
    delete_profile, duplicate_profile, list_profiles, load_profile,
    profile_from_dict, profile_to_dict, save_profile, slugify,
)


def make_profile() -> Profile:
    return Profile(
        profile_id="test_p",
        name="Testovací profil",
        sites=[
            Site(site_id="loc1", name="Lokalita 1", kind="fve_bess",
                 pvs=[PVParams(asset_id="fve1", installed_mw=1.0),
                      PVParams(asset_id="fve2", installed_mw=0.5)],
                 bess_units=[BESSParams(asset_id="b1"), BESSParams(asset_id="b2")],
                 consumption=ConsumptionSpec(mode="tdd", tdd_class="TDD4",
                                             annual_mwh=100.0)),
            Site(site_id="loc2", name="Lokalita 2", kind="heat",
                 kgjs=[KGJParams(asset_id="kgj1"), KGJParams(asset_id="kgj2",
                                                             k_th=1.2)]),
        ],
    )


def test_roundtrip_dict():
    p = make_profile()
    d = profile_to_dict(p)
    p2 = profile_from_dict(json.loads(json.dumps(d)))
    assert profile_to_dict(p2) == d
    assert p2.sites[0].pvs[1].installed_mw == 0.5
    assert p2.sites[1].kgjs[1].k_th == 1.2


def test_save_load_list_delete(tmp_path):
    p = make_profile()
    save_profile(p, base_dir=tmp_path)
    metas = list_profiles(base_dir=tmp_path)
    assert len(metas) == 1 and metas[0]["profile_id"] == "test_p"
    assert metas[0]["n_sites"] == 2
    p2 = load_profile("test_p", base_dir=tmp_path)
    assert p2.name == "Testovací profil"
    assert p2.updated_at != ""
    dup = duplicate_profile("test_p", "Kopie profilu", base_dir=tmp_path)
    assert dup.profile_id == "kopie_profilu"
    assert len(list_profiles(base_dir=tmp_path)) == 2
    delete_profile("test_p", base_dir=tmp_path)
    assert len(list_profiles(base_dir=tmp_path)) == 1


def test_validation_errors(tmp_path):
    p = Profile(profile_id="bad", name="Bad", sites=[])
    with pytest.raises(ValueError):
        save_profile(p, base_dir=tmp_path)

    p = make_profile()
    p.sites[0].pvs[1].asset_id = "fve1"  # duplicita
    assert any("duplicitní asset_id" in e.lower() for e in p.validate())

    p = make_profile()
    p.sites[0].consumption = ConsumptionSpec(mode="tdd", tdd_class="TDD9",
                                             annual_mwh=10)
    assert any("TDD" in e for e in p.validate())

    p = make_profile()
    p.sites[1].kgjs[0].var_eff = True
    assert any("var_eff" in e for e in p.validate())


def test_unknown_keys_ignored():
    d = profile_to_dict(make_profile())
    d["future_field"] = 123
    d["sites"][0]["another_new"] = "x"
    d["sites"][0]["pvs"][0]["new_pv_param"] = 1
    p = profile_from_dict(d)
    assert p.sites[0].pvs[0].asset_id == "fve1"


def test_demo_profile_loads():
    p = load_profile("demo")
    assert p.validate() == []
    assert len(p.sites) == 3
    heat = p.sites[0]
    assert heat.has_heat and heat.kgjs[0].k_el == pytest.approx(0.45, abs=0.01)
    assert p.sites[1].consumption.mode == "tdd"


def test_slugify():
    assert slugify("Kopie profilu č. 2") == "kopie_profilu_c_2"
    assert slugify("Teplárna Žižkov") == "teplarna_zizkov"
