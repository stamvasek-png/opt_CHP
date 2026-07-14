"""Stránka „Šablony zdrojů" — CRUD editor profilů portfolia."""

from __future__ import annotations

import streamlit as st

from core.profiles import (BESSParams, BoilerParams, ConsumptionSpec, EKParams,
                           HeatImportParams, KGJParams, PVParams, Profile,
                           Site, TDD_CLASSES, TESParams, delete_profile,
                           duplicate_profile, list_profiles, load_profile,
                           save_profile, slugify)

st.title("🧩 Šablony zdrojů")

ASSET_TYPES = [
    ("kgjs", "KGJ (kogenerace)", KGJParams, "kgj"),
    ("boilers", "Plynový kotel", BoilerParams, "kotel"),
    ("eks", "Elektrokotel", EKParams, "ek"),
    ("tes_units", "Nádrž TES", TESParams, "tes"),
    ("bess_units", "Baterie BESS", BESSParams, "bess"),
    ("pvs", "Fotovoltaika FVE", PVParams, "fve"),
    ("heat_imports", "Import tepla", HeatImportParams, "imp"),
]


def _bump():
    st.session_state["edit_rev"] = st.session_state.get("edit_rev", 0) + 1


def _key(*parts) -> str:
    return "pe_" + str(st.session_state.get("edit_rev", 0)) + "_" + \
        "_".join(str(p) for p in parts)


def _num(obj, attr, label, key, fmt="%.3f", step=0.01, minv=None):
    val = st.number_input(label, value=float(getattr(obj, attr)), key=key,
                          format=fmt, step=step,
                          min_value=minv)
    setattr(obj, attr, float(val))


def _bool(obj, attr, label, key):
    setattr(obj, attr, bool(st.checkbox(label, value=bool(getattr(obj, attr)),
                                        key=key)))


def _text(obj, attr, label, key):
    setattr(obj, attr, st.text_input(label, value=str(getattr(obj, attr)),
                                     key=key))


def _optprice(obj, flag_attr, price_attr, label, key, default=40.0):
    on = st.checkbox(label, value=bool(getattr(obj, flag_attr)), key=key + "f")
    setattr(obj, flag_attr, on)
    if on:
        cur = getattr(obj, price_attr)
        val = st.number_input("cena [€/MWh]", key=key + "p",
                              value=float(cur if cur is not None else default))
        setattr(obj, price_attr, float(val))


def _render_asset(site_i: int, atype: str, a, i: int):
    k = lambda f: _key(site_i, atype, i, f)  # noqa: E731
    c1, c2 = st.columns(2)
    with c1:
        _text(a, "asset_id", "ID (slug)", k("id"))
        _text(a, "name", "Název", k("nm"))
    if atype == "kgjs":
        with c2:
            _num(a, "k_th", "Jmenovitý tepelný výkon [MW]", k("kth"))
            _num(a, "k_min", "Min. zatížení [zlomek]", k("kmin"), step=0.05)
        c3, c4 = st.columns(2)
        with c3:
            _num(a, "k_eff_th", "η_th [-]", k("eth"))
            _num(a, "k_eff_el", "η_el [-]", k("eel"))
            _num(a, "k_min_runtime_h", "Min. doba běhu [h]", k("mrt"),
                 fmt="%.2f", step=0.25)
        with c4:
            _num(a, "k_start_cost", "Náklad na start [€]", k("sc"), fmt="%.1f",
                 step=10.0)
            _num(a, "k_service_cost", "Servis [€/h provozu]", k("svc"),
                 fmt="%.1f", step=1.0)
            _bool(a, "afrr_capable", "aFRR způsobilý", k("af"))
        _bool(a, "var_eff", "Proměnná účinnost dle zatížení", k("ve"))
        if a.var_eff:
            c5, c6 = st.columns(2)
            with c5:
                if a.eta_th_min is None:
                    a.eta_th_min = round(a.k_eff_th * 0.9, 3)
                _num(a, "eta_th_min", "η_th při min. zátěži", k("etm"))
            with c6:
                if a.eta_el_min is None:
                    a.eta_el_min = round(a.k_eff_el * 0.9, 3)
                _num(a, "eta_el_min", "η_el při min. zátěži", k("eem"))
        _optprice(a, "gas_fix", "gas_fix_price", "Fixní cena plynu", k("gf"))
        _optprice(a, "ee_fix", "ee_fix_price",
                  "Fixní výkupní cena EE (PPA/bonus)", k("ef"), default=100.0)
        st.caption(f"Odvozený el. výkon: {a.k_el:.3f} MW")
    elif atype == "boilers":
        with c2:
            _num(a, "b_max", "Max. výkon [MW]", k("bm"))
            _num(a, "boil_eff", "Účinnost [-]", k("be"))
        _optprice(a, "gas_fix", "gas_fix_price", "Fixní cena plynu", k("gf"))
    elif atype == "eks":
        with c2:
            _num(a, "ek_max", "Max. výkon [MW]", k("em"))
            _num(a, "ek_eff", "Účinnost [-]", k("ee"))
        _bool(a, "afrr_capable", "aFRR způsobilý (záporná regulace)", k("af"))
        _optprice(a, "ee_fix", "ee_fix_price", "Fixní cena EE", k("ef"),
                  default=80.0)
    elif atype == "tes_units":
        with c2:
            _num(a, "tes_cap", "Kapacita [MWh]", k("tc"))
            _num(a, "tes_loss_pct_h", "Ztráta [%/h]", k("tl"))
        c3, c4 = st.columns(2)
        with c3:
            _num(a, "soc_start_frac", "SoC na začátku dne [zlomek]", k("ss"),
                 step=0.05)
        with c4:
            _num(a, "soc_end_min_frac", "Min. SoC na konci dne [zlomek]",
                 k("se"), step=0.05)
    elif atype == "bess_units":
        with c2:
            _num(a, "bess_cap", "Kapacita [MWh]", k("bc"))
            _num(a, "bess_p", "Max. výkon [MW]", k("bp"))
        c3, c4 = st.columns(2)
        with c3:
            _num(a, "bess_eff", "Účinnost nab/vyb [-]", k("be"))
            _num(a, "bess_cycle_cost", "Opotřebení [€/MWh]", k("bcc"),
                 fmt="%.1f", step=1.0)
            _bool(a, "afrr_capable", "aFRR způsobilý", k("af"))
        with c4:
            _num(a, "soc_start_frac", "SoC na začátku dne [zlomek]", k("ss"),
                 step=0.05)
            _num(a, "soc_end_min_frac", "Min. SoC na konci dne [zlomek]",
                 k("se"), step=0.05)
        _bool(a, "dist_buy_extra", "Extra distribuce na nabíjení", k("db"))
        _bool(a, "dist_sell_extra", "Extra distribuce na vybíjení", k("ds"))
        _optprice(a, "ee_fix", "ee_fix_price", "Fixní cena EE", k("ef"),
                  default=80.0)
    elif atype == "pvs":
        with c2:
            _num(a, "installed_mw", "Instalovaný výkon [MW]", k("im"))
        _bool(a, "allow_curtailment", "Povolit curtailment (omezení výroby)",
              k("cu"))
        _bool(a, "dist_sell", "Extra distribuce při prodeji", k("ds"))
    elif atype == "heat_imports":
        with c2:
            _num(a, "imp_max", "Max. výkon [MW]", k("im"))
            _num(a, "imp_price", "Cena [€/MWh]", k("ip"), fmt="%.1f", step=5.0)


def _render_site(p: Profile, site_i: int):
    s = p.sites[site_i]
    k = lambda f: _key(site_i, f)  # noqa: E731
    with st.expander(f"📍 **{s.name}** (`{s.site_id}`)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            _text(s, "site_id", "ID lokality (slug)", k("sid"))
            _text(s, "name", "Název lokality", k("snm"))
            s.kind = st.selectbox(
                "Typ (jen UI šablona)", ["heat", "fve_bess"],
                index=0 if s.kind == "heat" else 1, key=k("knd"))
        with c2:
            _num(s, "dist_ee_buy", "Distribuce EE nákup [€/MWh]", k("db"),
                 fmt="%.1f", step=1.0)
            _num(s, "dist_ee_sell", "Distribuce EE prodej [€/MWh]", k("dsl"),
                 fmt="%.1f", step=1.0)
            _num(s, "gas_dist", "Distribuce plyn [€/MWh]", k("gd"),
                 fmt="%.1f", step=1.0)
            _bool(s, "internal_ee_use",
                  "Interní spotřeba EE bez distribuce", k("iu"))
        with c3:
            lim_e = st.checkbox("Limit exportu",
                                value=s.grid_export_limit_mw is not None,
                                key=k("lef"))
            if lim_e:
                s.grid_export_limit_mw = float(st.number_input(
                    "Limit exportu [MW]", key=k("lev"),
                    value=float(s.grid_export_limit_mw or 1.0)))
            else:
                s.grid_export_limit_mw = None
            lim_i = st.checkbox("Limit importu",
                                value=s.grid_import_limit_mw is not None,
                                key=k("lif"))
            if lim_i:
                s.grid_import_limit_mw = float(st.number_input(
                    "Limit importu [MW]", key=k("liv"),
                    value=float(s.grid_import_limit_mw or 1.0)))
            else:
                s.grid_import_limit_mw = None

        if s.has_heat:
            st.markdown("**Teplo**")
            c4, c5, c6 = st.columns(3)
            with c4:
                _num(s, "h_price", "Cena tepla [€/MWh]", k("hp"), fmt="%.1f",
                     step=1.0)
            with c5:
                _num(s, "h_cover", "Min. pokrytí poptávky [-]", k("hc"),
                     step=0.01)
            with c6:
                _num(s, "shortfall_penalty", "Penalizace nedodání [€/MWh]",
                     k("sp"), fmt="%.0f", step=50.0)

        st.markdown("**Spotřeba lokality**")
        c7, c8, c9 = st.columns(3)
        with c7:
            mode = st.selectbox(
                "Zdroj spotřeby", ["none", "tdd", "curve"],
                index=["none", "tdd", "curve"].index(s.consumption.mode),
                format_func={"none": "žádná", "tdd": "TDD × roční spotřeba",
                             "curve": "vlastní křivka (workbook)"}.get,
                key=k("cm"))
        if mode == "tdd":
            with c8:
                cls = st.selectbox(
                    "TDD třída", TDD_CLASSES,
                    index=(TDD_CLASSES.index(s.consumption.tdd_class)
                           if s.consumption.tdd_class in TDD_CLASSES else 3),
                    key=k("ct"))
            with c9:
                annual = st.number_input(
                    "Roční spotřeba [MWh]", key=k("ca"),
                    value=float(s.consumption.annual_mwh or 100.0),
                    min_value=0.0)
            s.consumption = ConsumptionSpec("tdd", cls, float(annual))
        else:
            s.consumption = ConsumptionSpec(mode)

        st.divider()
        for atype, label, cls, prefix in ASSET_TYPES:
            lst = getattr(s, atype)
            st.markdown(f"**{label}** ({len(lst)}×)")
            for i, a in enumerate(list(lst)):
                with st.container(border=True):
                    _render_asset(site_i, atype, a, i)
                    if st.button(f"🗑️ Odebrat {a.asset_id}",
                                 key=_key(site_i, atype, i, "rm")):
                        lst.remove(a)
                        _bump()
                        st.rerun()
            if st.button(f"➕ Přidat: {label}", key=k("add_" + atype)):
                new_id = f"{prefix}{len(lst) + 1}"
                while new_id in s.all_asset_ids():
                    new_id += "x"
                lst.append(cls(asset_id=new_id))
                _bump()
                st.rerun()

        st.divider()
        if st.button(f"🗑️ Odebrat lokalitu {s.name}", key=k("rm_site")):
            p.sites.remove(s)
            _bump()
            st.rerun()


# ── Výběr / správa profilů ───────────────────────────────────────────────────
metas = list_profiles()
names = {f'{m["name"]} ({m["profile_id"]})': m["profile_id"] for m in metas}

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    sel = st.selectbox("Uložené profily", list(names) or ["— žádné —"])
with c2:
    if names and st.button("✏️ Otevřít v editoru", width="stretch"):
        st.session_state["edit_profile"] = load_profile(names[sel])
        _bump()
        st.rerun()
with c3:
    if names and st.button("📋 Duplikovat", width="stretch"):
        dup = duplicate_profile(names[sel],
                                f"Kopie {sel.rsplit(' (', 1)[0]}")
        st.session_state["edit_profile"] = dup
        _bump()
        st.rerun()
with c4:
    if names:
        confirm_del = st.checkbox("potvrdit", key="del_confirm")
        if st.button("🗑️ Smazat", width="stretch",
                     disabled=not confirm_del):
            delete_profile(names[sel])
            st.session_state.pop("edit_profile", None)
            st.rerun()

if st.button("➕ Nový prázdný profil"):
    st.session_state["edit_profile"] = Profile(
        profile_id="novy_profil", name="Nový profil",
        sites=[Site(site_id="lokalita_1", name="Lokalita 1")])
    _bump()
    st.rerun()

# ── Editor ───────────────────────────────────────────────────────────────────
p: Profile | None = st.session_state.get("edit_profile")
if p is None:
    st.info("Otevřete existující profil v editoru, nebo založte nový.")
    st.stop()

st.divider()
st.subheader(f"Editor: {p.name}")
c1, c2, c3 = st.columns(3)
with c1:
    p.name = st.text_input("Název profilu", value=p.name, key=_key("pname"))
    default_id = p.profile_id or slugify(p.name)
    p.profile_id = st.text_input("ID profilu (slug)", value=default_id,
                                 key=_key("pid"))
with c2:
    p.description = st.text_input("Popis", value=p.description,
                                  key=_key("pdesc"))
with c3:
    p.afrr_activation_h = float(st.number_input(
        "Trvání aFRR aktivace τ [h]", value=float(p.afrr_activation_h),
        min_value=0.25, step=0.25, key=_key("ptau"),
        help="Kryté trvání plné aktivace pro energetické podmínky BESS."))

for i in range(len(p.sites)):
    _render_site(p, i)

if st.button("➕ Přidat lokalitu"):
    n = len(p.sites) + 1
    p.sites.append(Site(site_id=f"lokalita_{n}", name=f"Lokalita {n}"))
    _bump()
    st.rerun()

st.divider()
errs = p.validate()
if errs:
    for e in errs:
        st.warning(e)
if st.button("💾 Uložit profil", type="primary", disabled=bool(errs)):
    try:
        save_profile(p)
        st.success(f"Profil „{p.name}“ uložen jako {p.profile_id}.json.")
        st.rerun()
    except ValueError as e:
        st.error(str(e))
