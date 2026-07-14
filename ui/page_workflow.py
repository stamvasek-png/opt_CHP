"""Stránka „Obchodní den" — třífázový denní workflow D+1."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from core.export import (bids_xlsx, day_result_xlsx, economics_df,
                         nomination_csv, nomination_xlsx)
from core.inputs import InputError, build_template_xlsx, parse_actual_da, \
    parse_inputs_xlsx
from core.model import SolveResult
from core.profiles import list_profiles, load_profile
from core.runs import (AuctionResults, Run1Result, run1_afrr_bids,
                       run2_da_plan, run3_redispatch)
from core.timegrid import BLOCK_LABELS, N_BLOCKS
from core.trading_day import StageError, TradingDayStore
from ui.charts import (deviation_chart, heat_chart, ladder_chart,
                       nomination_chart, portfolio_chart, price_preview_chart,
                       pv_chart, soc_chart)
from ui.common import fmt_eur, load_settings, stage_badges

settings = load_settings()
store = TradingDayStore()

st.title("📅 Obchodní den — plánování D+1")

# ── Sidebar: den, profil, FX ────────────────────────────────────────────────
with st.sidebar:
    st.header("Obchodní den")
    date = st.date_input("Den dodávky (D+1)",
                         value=dt.date.today() + dt.timedelta(days=1))
    day = None
    if store.exists(date):
        day = store.open(date)
        st.caption(f"Profil dne: **{day.profile.name}** (snapshot)")
        fx = st.number_input("Kurz CZK/EUR", value=float(day.fx_czk_eur),
                             step=0.1, format="%.2f")
        if fx != day.fx_czk_eur:
            day.fx_czk_eur = fx
            day.save()
    else:
        metas = list_profiles()
        if not metas:
            st.warning("Nejdřív vytvořte profil zdrojů (stránka Šablony "
                       "zdrojů).")
        else:
            names = {m["name"]: m["profile_id"] for m in metas}
            sel = st.selectbox("Profil zdrojů", list(names))
            fx = st.number_input("Kurz CZK/EUR",
                                 value=float(settings["fx_czk_eur"]),
                                 step=0.1, format="%.2f")
            if st.button("➕ Založit obchodní den", width="stretch"):
                store.open(date, profile=load_profile(names[sel]),
                           fx_czk_eur=fx)
                st.rerun()
    if day is not None:
        st.divider()
        st.markdown(stage_badges(day.status))

if day is None:
    st.info("Vyberte den dodávky a založte obchodní den v postranním panelu.")
    st.stop()

grid = day.grid
profile = day.profile
st.caption(f"Den dodávky **{date.isoformat()}** · {grid.n} MTU à 15 min · "
           f"profil **{profile.name}** · kurz {day.fx_czk_eur:.2f} CZK/EUR")


def _series():
    if "series_cache" not in st.session_state \
            or st.session_state.get("series_date") != date:
        st.session_state.series_cache = day.load_inputs()
        st.session_state.series_date = date
    return st.session_state.series_cache


def _result_from_store(plan, economics, reserves) -> SolveResult:
    return SolveResult(status="Optimal", objective=0.0, plan=plan,
                       position_mw=plan["pos_mw"].to_numpy(),
                       economics=economics, reserve_alloc=reserves)


# ═════════════════════ KROK 0 — VSTUPY ══════════════════════════════════════
with st.expander("**Krok 0 — Vstupy D+1** (predikce cen, výroby a spotřeby)",
                 expanded=not day.status["inputs_loaded"]):
    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "⬇️ Stáhnout prázdnou šablonu vstupů",
            data=build_template_xlsx(profile, grid),
            file_name=f"Vstupy_D1_{date.isoformat()}.xlsx",
            width="stretch")
        cur_imb = st.selectbox("Měna listu Odchylka", ["CZK", "EUR"], index=0)
        cur_afrr = st.selectbox("Měna listu aFRR_ceny", ["CZK", "EUR"], index=0)
    with c2:
        up = st.file_uploader("Nahrát vyplněný workbook", type=["xlsx"],
                              key="inputs_upload")
        if up is not None and st.button("📥 Načíst vstupy",
                                        width="stretch"):
            try:
                series, report = parse_inputs_xlsx(
                    up, grid, profile, fx_czk_eur=day.fx_czk_eur,
                    currencies={"imb": cur_imb, "afrr": cur_afrr})
                day.save_inputs(series, report)
                st.session_state.pop("series_cache", None)
                st.rerun()
            except (InputError, FileNotFoundError, ValueError) as e:
                st.error(f"Vstupy se nepodařilo načíst: {e}")

    if day.status["inputs_loaded"]:
        st.success("Vstupy načteny.")
        for r in day.input_report:
            st.caption(f"ℹ️ {r}")
        s = _series()
        st.plotly_chart(price_preview_chart(grid, s.da_price_pred,
                                            s.imb_price_pred),
                        width="stretch")
        afrr_df = pd.DataFrame({
            "Blok": BLOCK_LABELS,
            "aFRR+ predikce [€/MW/h]": s.afrr_cap_price_up,
            "aFRR− predikce [€/MW/h]": s.afrr_cap_price_dn,
        })
        st.dataframe(afrr_df, hide_index=True, width="stretch")

# ═════════════════════ KROK 1 — aFRR NABÍDKY (~08:00) ═══════════════════════
with st.expander("**Krok 1 — aFRR nabídky do denní aukce ČEPS** (~08:00)",
                 expanded=(day.status["inputs_loaded"]
                           and not day.status["run1_done"])):
    if not day.status["inputs_loaded"]:
        st.info("Nejdřív načtěte vstupy (krok 0).")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            fracs = st.multiselect(
                "Kroky kapacity (zlomek maxima portfolia)",
                options=[0.1, 0.25, 0.5, 0.75, 1.0],
                default=[float(f) for f in settings["r_grid_frac"]])
        with c2:
            st.write("")
            compute = st.button("⚡ Spočítat nabídky", width="stretch",
                                disabled=not fracs)
            skip = st.button("Přeskočit (nenabízím SVR)",
                             width="stretch")
        if skip:
            day.skip_run1()
            st.rerun()
        if compute:
            prog = st.progress(0.0, text="Počítám žebřík opportunity cost…")

            def _cb(done, total):
                prog.progress(done / total,
                              text=f"Solve {done}/{total} (paralelně)")

            try:
                r1 = run1_afrr_bids(
                    grid, profile, _series(),
                    r_grid_frac=tuple(sorted(fracs)),
                    time_limit_s=int(settings["run1_time_limit_s"]),
                    gap_rel=float(settings["run1_gap_rel"]),
                    workers=(int(settings["workers"]) or None),
                    progress_cb=_cb)
                day.save_run1(r1)
                st.session_state["run1_obj"] = r1
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))

    if day.status["run1_done"]:
        if day.run1_meta.get("skipped"):
            st.info("Krok přeskočen — SVR se nenabízí.")
        else:
            ladder = day.load_run1_ladder()
            meta = day.run1_meta
            m1, m2, m3 = st.columns(3)
            m1.metric("Zisk baseline (bez SVR)",
                      fmt_eur(meta.get("baseline_profit", 0)))
            m2.metric("Max. aFRR+ portfolia",
                      f"{meta.get('r_max_up', 0):.2f} MW")
            m3.metric("Max. aFRR− portfolia",
                      f"{meta.get('r_max_dn', 0):.2f} MW")
            show = ladder[ladder["feasible"]].copy()
            rec_n = int(show["recommend"].sum())
            if rec_n:
                st.success(f"Doporučeno nabídnout {rec_n} kombinací "
                           f"(marginální cena < predikce clearingu).")
            else:
                st.warning("Žádná kombinace nevychází výhodně vůči predikci "
                           "cen aFRR.")
            tbl = pd.DataFrame({
                "Blok": show["block_label"],
                "Směr": show["direction_label"],
                "R [MW]": show["r_mw"],
                "Min. cena marg. [€/MW/h]": show["min_price_marginal"].round(2),
                "Min. cena prům. [€/MW/h]": show["min_price_avg"].round(2),
                "Predikce [€/MW/h]": show["predicted_price"].round(2),
                "Nabídnout": np.where(show["recommend"], "✅", "—"),
            })
            st.dataframe(tbl, hide_index=True, width="stretch",
                         height=330)
            st.plotly_chart(ladder_chart(ladder), width="stretch")
            r1_obj = st.session_state.get("run1_obj")
            if r1_obj is None:
                r1_obj = Run1Result(
                    ladder=ladder,
                    baseline_profit=meta.get("baseline_profit", 0.0),
                    r_max_up=meta.get("r_max_up", 0.0),
                    r_max_dn=meta.get("r_max_dn", 0.0),
                    solves=meta.get("solves", 0),
                    wall_s=meta.get("wall_s", 0.0),
                    r_grid_frac=tuple(meta.get("r_grid_frac", [])))
            st.download_button(
                "⬇️ Export nabídek (XLSX)",
                data=bids_xlsx(r1_obj, date, day.fx_czk_eur),
                file_name=f"aFRR_nabidky_{date.isoformat()}.xlsx")

# ═════════════════════ KROK 2 — AUKCE + DA PLÁN + NOMINACE (~10:00) ═════════
with st.expander("**Krok 2 — Výsledky aukce, DA plán a nominace OTE** (~10:00)",
                 expanded=(day.status["run1_done"]
                           and not day.status["nomination_frozen"])):
    if not day.status["inputs_loaded"]:
        st.info("Nejdřív načtěte vstupy (krok 0).")
    else:
        st.markdown("**Vysoutěžená aFRR kapacita** (0 = nevyhráno)")
        cur_auc = st.selectbox("Měna kapacitních cen", ["CZK", "EUR"], index=0)
        factor = day.fx_czk_eur if cur_auc == "CZK" else 1.0
        auc = day.auction or AuctionResults()
        edit_df = pd.DataFrame({
            "Blok": BLOCK_LABELS,
            "aFRR+ [MW]": np.asarray(auc.up_mw, dtype=float),
            f"aFRR+ cena [{cur_auc}/MW/h]":
                np.asarray(auc.up_price, dtype=float) * factor,
            "aFRR− [MW]": np.asarray(auc.dn_mw, dtype=float),
            f"aFRR− cena [{cur_auc}/MW/h]":
                np.asarray(auc.dn_price, dtype=float) * factor,
        })
        edited = st.data_editor(edit_df, hide_index=True,
                                width="stretch",
                                disabled=["Blok"], key="auction_editor")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Uložit výsledky aukce", width="stretch"):
                day.set_auction(AuctionResults(
                    up_mw=edited["aFRR+ [MW]"].to_numpy(dtype=float),
                    up_price=edited[f"aFRR+ cena [{cur_auc}/MW/h]"]
                        .to_numpy(dtype=float) / factor,
                    dn_mw=edited["aFRR− [MW]"].to_numpy(dtype=float),
                    dn_price=edited[f"aFRR− cena [{cur_auc}/MW/h]"]
                        .to_numpy(dtype=float) / factor))
                st.rerun()
        with c2:
            build_plan = st.button(
                "🧮 Sestavit plán a nominaci", width="stretch",
                disabled=day.status["nomination_frozen"])
        if build_plan:
            try:
                day.assert_stage("run2")
                if day.auction is None:
                    day.set_auction(AuctionResults())
                with st.spinner("Optimalizuji plán provozu…"):
                    res2 = run2_da_plan(
                        grid, profile, _series(), day.auction,
                        time_limit_s=int(settings["solver_time_limit_s"]),
                        gap_rel=float(settings["solver_gap_rel"]))
                if res2 is None:
                    st.error("Plán je nesplnitelný — vysoutěžená kapacita "
                             "zřejmě přesahuje možnosti portfolia, nebo "
                             "nelze pokrýt teplo.")
                else:
                    day.save_run2(res2)
                    st.rerun()
            except StageError as e:
                st.error(str(e))

    if day.status["run2_done"]:
        plan2 = day.load_run2_plan()
        eco2 = day.run2_economics
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Očekávaný zisk", fmt_eur(eco2.get("profit_total", 0)))
        m2.metric("Výnos aFRR kapacita", fmt_eur(eco2.get("rev_afrr_cap", 0)))
        m3.metric("Max. export", f"{plan2['pos_mw'].max():.2f} MW")
        m4.metric("Max. odběr", f"{-plan2['pos_mw'].min():.2f} MW")

        tab_pos, tab_heat, tab_soc, tab_pv, tab_eco = st.tabs(
            ["Pozice EE", "Teplo", "SoC zásobníků", "FVE", "Ekonomika"])
        with tab_pos:
            st.plotly_chart(portfolio_chart(plan2, grid, profile),
                            width="stretch")
        with tab_heat:
            heat_sites_ = [s for s in profile.sites if s.has_heat]
            if heat_sites_:
                for s_ in heat_sites_:
                    st.markdown(f"**{s_.name}**")
                    st.plotly_chart(heat_chart(plan2, grid, s_),
                                    width="stretch")
            else:
                st.caption("Profil nemá tepelné lokality.")
        with tab_soc:
            st.plotly_chart(soc_chart(plan2, grid, profile),
                            width="stretch")
        with tab_pv:
            if any(s.pvs for s in profile.sites):
                st.plotly_chart(pv_chart(plan2, grid, profile),
                                width="stretch")
            else:
                st.caption("Profil nemá FVE.")
        with tab_eco:
            st.dataframe(economics_df(eco2), hide_index=True,
                         width="stretch")

        # nominace
        st.markdown("### Nominace pozice na OTE")
        nom = day.load_nomination()
        if nom is None:
            nom_preview = pd.DataFrame({
                "mtu": np.arange(1, grid.n + 1),
                "cas_od": grid.times_from(),
                "pos_mw": plan2["pos_mw"].to_numpy(),
                "energie_mwh": plan2["pos_mw"].to_numpy() * grid.dt_h,
            })
            st.plotly_chart(nomination_chart(nom_preview, grid),
                            width="stretch")
            if st.button("🔒 Zmrazit nominaci (odesláno na OTE)",
                         type="primary"):
                day.freeze_nomination()
                st.rerun()
        else:
            st.success("Nominace je zmrazená.")
            st.plotly_chart(nomination_chart(nom, grid),
                            width="stretch")
            with st.popover("Zobrazit tabulku nominace"):
                st.dataframe(nom, hide_index=True, height=400)
            d1, d2, d3 = st.columns(3)
            d1.download_button("⬇️ Nominace CSV",
                               data=nomination_csv(nom, date),
                               file_name=f"nominace_{date.isoformat()}.csv")
            d2.download_button("⬇️ Nominace XLSX",
                               data=nomination_xlsx(nom, date, profile),
                               file_name=f"nominace_{date.isoformat()}.xlsx")
            reserves2 = None
            p_res = day.dir / "run2_reserves.parquet"
            if p_res.exists():
                reserves2 = pd.read_parquet(p_res)
            d3.download_button(
                "⬇️ Plán dne XLSX",
                data=day_result_xlsx(
                    _result_from_store(plan2, eco2, reserves2), grid, profile,
                    "Plán DA"),
                file_name=f"plan_{date.isoformat()}.xlsx")
            confirm = st.checkbox("Rozumím, že nominace na OTE už mohla být "
                                  "odeslána", key="unfreeze_confirm")
            if st.button("🔓 Odemknout nominaci", disabled=not confirm):
                day.unfreeze_nomination()
                st.rerun()

# ═════════════════════ KROK 3 — RE-DISPATCH (po 14:00) ══════════════════════
with st.expander("**Krok 3 — Re-dispatch po zveřejnění cen OTE** (po 14:00)",
                 expanded=(day.status["nomination_frozen"]
                           and not day.status["run3_done"])):
    if not day.status["nomination_frozen"]:
        st.info("Nejdřív zmrazte nominaci (krok 2).")
    else:
        c1, c2 = st.columns(2)
        with c1:
            up_da = st.file_uploader("Skutečné ceny DA (xlsx/csv)",
                                     type=["xlsx", "csv"], key="actual_upload")
            cur_da = st.selectbox("Měna skutečných cen", ["EUR", "CZK"],
                                  index=0)
            if up_da is not None and st.button("📥 Načíst skutečné ceny"):
                try:
                    day.save_actual_da(parse_actual_da(
                        up_da, grid, fx_czk_eur=day.fx_czk_eur,
                        currency=cur_da))
                    st.rerun()
                except (InputError, ValueError) as e:
                    st.error(f"Ceny se nepodařilo načíst: {e}")
        with c2:
            lam = st.number_input(
                "Riziková přirážka λ [€/MWh odchylky]",
                value=float(day.lambda_dev), min_value=0.0, step=1.0,
                help="Penalizuje |odchylku| v objective — vyšší λ = "
                     "konzervativnější držení nominace. Do ekonomiky "
                     "nevstupuje.")
            recompute = st.button("🧮 Přepočítat re-dispatch",
                                  width="stretch",
                                  disabled=not day.status["actual_da_loaded"])
        if day.status["actual_da_loaded"]:
            s = _series()
            actual = day.load_actual_da()
            st.plotly_chart(price_preview_chart(grid, s.da_price_pred,
                                                s.imb_price_pred, actual),
                            width="stretch")
        if recompute:
            try:
                day.assert_stage("run3")
                s3 = day.load_inputs()
                s3.da_price_actual = day.load_actual_da()
                nom = day.load_nomination()
                with st.spinner("Optimalizuji re-dispatch…"):
                    res3 = run3_redispatch(
                        grid, profile, s3, day.auction or AuctionResults(),
                        nomination_mw=nom["pos_mw"].to_numpy(),
                        lambda_dev=lam,
                        time_limit_s=int(settings["solver_time_limit_s"]),
                        gap_rel=float(settings["solver_gap_rel"]))
                if res3 is None:
                    st.error("Re-dispatch je nesplnitelný.")
                else:
                    day.save_run3(res3, lambda_dev=lam)
                    st.rerun()
            except StageError as e:
                st.error(str(e))

    if day.status["run3_done"]:
        plan3 = day.load_run3_plan()
        eco3 = day.run3_economics
        eco2 = day.run2_economics
        st.divider()
        delta = eco3.get("profit_total", 0) - eco2.get("profit_total", 0)
        m1, m2, m3 = st.columns(3)
        m1.metric("Zisk re-dispatch", fmt_eur(eco3.get("profit_total", 0)),
                  delta=f"{delta:+,.0f} € vs. plán".replace(",", " "))
        m2.metric("Zúčtování odchylky",
                  fmt_eur(eco3.get("rev_deviation", 0)))
        dev_abs = np.abs(plan3["deviation_mw"].to_numpy()).sum() * grid.dt_h
        m3.metric("Objem odchylky", f"{dev_abs:.2f} MWh")
        st.plotly_chart(deviation_chart(plan3, grid), width="stretch")

        cmp_rows = []
        from core.export import ECO_LABELS
        for key, label in ECO_LABELS.items():
            a, b = eco2.get(key), eco3.get(key)
            if a is None and b is None:
                continue
            cmp_rows.append({"Položka": label,
                             "Plán 10:00 [€]": round(a, 2) if a is not None else None,
                             "Re-dispatch [€]": round(b, 2) if b is not None else None,
                             "Δ [€]": round((b or 0) - (a or 0), 2)})
        st.dataframe(pd.DataFrame(cmp_rows), hide_index=True,
                     width="stretch")
        reserves3 = None
        p_res3 = day.dir / "run3_reserves.parquet"
        if p_res3.exists():
            reserves3 = pd.read_parquet(p_res3)
        st.download_button(
            "⬇️ Re-dispatch XLSX (vč. srovnání)",
            data=day_result_xlsx(_result_from_store(plan3, eco3, reserves3),
                                 grid, profile, "Re-dispatch",
                                 compare_economics=eco2),
            file_name=f"redispatch_{date.isoformat()}.xlsx")
