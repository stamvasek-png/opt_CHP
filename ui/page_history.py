"""Stránka „Historie dní" — archiv obchodních dní (read-only)."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from core.export import (day_result_xlsx, economics_df, nomination_csv,
                         nomination_xlsx)
from core.model import SolveResult
from core.trading_day import TradingDayStore
from ui.charts import deviation_chart, nomination_chart, portfolio_chart
from ui.common import fmt_eur, stage_badges

st.title("🗂️ Historie obchodních dní")

store = TradingDayStore()
days = store.list_days()

if not days:
    st.info("Zatím žádné uložené obchodní dny.")
    st.stop()

overview = pd.DataFrame([{
    "Den dodávky": d["delivery_date"],
    "Profil": d["profile_name"],
    "Nominace": "🔒" if d["status"].get("nomination_frozen") else "—",
    "Re-dispatch": "✅" if d["status"].get("run3_done") else "—",
    "Zisk plán [€]": (round(d["profit_run2"], 0)
                      if d["profit_run2"] is not None else None),
    "Zisk re-dispatch [€]": (round(d["profit_run3"], 0)
                             if d["profit_run3"] is not None else None),
} for d in days])
st.dataframe(overview, hide_index=True, width="stretch")

sel = st.selectbox("Detail dne", [d["delivery_date"] for d in days])
day = store.open(dt.date.fromisoformat(sel))
grid = day.grid
profile = day.profile

st.markdown(stage_badges(day.status))
st.caption(f"Profil (snapshot): **{profile.name}** · "
           f"kurz {day.fx_czk_eur:.2f} CZK/EUR")


def _shim(plan, eco, reserves=None) -> SolveResult:
    return SolveResult(status="Optimal", objective=0.0, plan=plan,
                       position_mw=plan["pos_mw"].to_numpy(), economics=eco,
                       reserve_alloc=reserves)


tab2, tab3 = st.tabs(["Plán 10:00 + nominace", "Re-dispatch"])

with tab2:
    plan2 = day.load_run2_plan()
    if plan2 is None:
        st.caption("Plán nebyl sestaven.")
    else:
        st.metric("Očekávaný zisk",
                  fmt_eur(day.run2_economics.get("profit_total", 0)))
        st.plotly_chart(portfolio_chart(plan2, grid, profile),
                        width="stretch")
        st.dataframe(economics_df(day.run2_economics), hide_index=True,
                     width="stretch")
        nom = day.load_nomination()
        if nom is not None:
            st.plotly_chart(nomination_chart(nom, grid),
                            width="stretch")
            c1, c2, c3 = st.columns(3)
            c1.download_button("⬇️ Nominace CSV",
                               data=nomination_csv(nom, day.delivery_date),
                               file_name=f"nominace_{sel}.csv")
            c2.download_button("⬇️ Nominace XLSX",
                               data=nomination_xlsx(nom, day.delivery_date,
                                                    profile),
                               file_name=f"nominace_{sel}.xlsx")
            c3.download_button(
                "⬇️ Plán dne XLSX",
                data=day_result_xlsx(_shim(plan2, day.run2_economics), grid,
                                     profile, "Plán DA"),
                file_name=f"plan_{sel}.xlsx")

with tab3:
    plan3 = day.load_run3_plan()
    if plan3 is None:
        st.caption("Re-dispatch nebyl proveden.")
    else:
        delta = (day.run3_economics.get("profit_total", 0)
                 - day.run2_economics.get("profit_total", 0))
        st.metric("Zisk re-dispatch",
                  fmt_eur(day.run3_economics.get("profit_total", 0)),
                  delta=f"{delta:+,.0f} € vs. plán".replace(",", " "))
        st.plotly_chart(deviation_chart(plan3, grid),
                        width="stretch")
        st.dataframe(economics_df(day.run3_economics), hide_index=True,
                     width="stretch")
        st.download_button(
            "⬇️ Re-dispatch XLSX",
            data=day_result_xlsx(_shim(plan3, day.run3_economics), grid,
                                 profile, "Re-dispatch",
                                 compare_economics=day.run2_economics),
            file_name=f"redispatch_{sel}.xlsx")
