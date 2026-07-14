"""Jeden parametrizovaný MILP pro všechny běhy D+1 plánování.

Režimy:
- DA_PLAN    — plán proti (predikovaným) cenám DA; slouží běhu 1 (výpočet
               opportunity cost aFRR) i běhu 2 (plán + nominace). Tržní
               hodnota: export/nákup za cenu DA ± distribuce.
- REDISPATCH — nominace N_t je fixní (zúčtuje se skutečnou cenou DA jako
               konstanta v ekonomice), odchylka pozice od nominace se
               oceňuje predikovanou zúčtovací cenou odchylky, volitelně
               s rizikovou přirážkou λ na |odchylku|. Fyzické toky nesou
               jen distribuci, palivo a delty fixních kontraktů — energie
               se oceňuje přes nominaci + odchylku (žádné dvojí započtení).

Fyzikální formulace assetů jsou portem ověřeného modelu z původního
app.py (run_optimization_with_profile), přeškálované na 15min krok
(dt = 0.25 h) a zobecněné na více lokalit a více assetů stejného typu.

Znaménková konvence odchylky: dev = pozice − nominace [MW];
peněžní tok = zúčtovací cena × dev (kladná cena ⇒ přebytek vydělává).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import pulp

from .inputs import InputSeries
from .profiles import Profile
from .timegrid import N_BLOCKS, TimeGrid


class RunMode(Enum):
    DA_PLAN = 1
    REDISPATCH = 2


@dataclass
class AfrrRequirement:
    """Rezervovaná aFRR kapacita per 4h blok [MW] + trvání aktivace τ [h]."""
    r_up_mw: np.ndarray = field(default_factory=lambda: np.zeros(N_BLOCKS))
    r_dn_mw: np.ndarray = field(default_factory=lambda: np.zeros(N_BLOCKS))
    activation_h: float = 1.0

    @classmethod
    def none(cls, activation_h: float = 1.0) -> "AfrrRequirement":
        return cls(np.zeros(N_BLOCKS), np.zeros(N_BLOCKS), activation_h)

    @property
    def any_reserved(self) -> bool:
        return bool((self.r_up_mw > 1e-9).any() or (self.r_dn_mw > 1e-9).any())


@dataclass
class SolveResult:
    status: str
    objective: float                 # hodnota optimalizované části [EUR]
    plan: pd.DataFrame               # per-MTU plán (sloupce 'site|asset|vel.')
    position_mw: np.ndarray          # čistá pozice portfolia per MTU [MW]
    economics: dict                  # breakdown reálných peněžních toků [EUR]
    reserve_alloc: pd.DataFrame | None = None   # alokace rezerv per asset/MTU

    @property
    def profit(self) -> float:
        return self.economics.get("profit_total", float("nan"))


def compute_linear_fuel_params(k_th, k_min, eta_th_rated, eta_th_min,
                               eta_el_rated, eta_el_min):
    """2-bodová linearizace paliva a el. výkonu KGJ podle tepelného výkonu.

    (port z původního app.py — beze změny logiky)
    gas(q) ≈ c0_th·on + c1_th·q  [MW paliva];  el(q) ≈ c0_el·on + c1_el·q [MW]
    """
    q_min = k_min * k_th
    f_rated = k_th / eta_th_rated
    f_min = q_min / eta_th_min if eta_th_min > 0 else q_min / eta_th_rated
    if k_th - q_min > 1e-9:
        c1_th = (f_rated - f_min) / (k_th - q_min)
        c0_th = f_rated - c1_th * k_th
    else:
        c1_th, c0_th = 1.0 / eta_th_rated, 0.0
    e_rated = f_rated * eta_el_rated
    e_min = f_min * eta_el_min
    if k_th - q_min > 1e-9:
        c1_el = (e_rated - e_min) / (k_th - q_min)
        c0_el = e_rated - c1_el * k_th
    else:
        c1_el, c0_el = eta_el_rated / eta_th_rated, 0.0
    return c0_th, c1_th, c0_el, c1_el


def kgj_fuel_el_coefs(k) -> tuple[float, float, float, float]:
    """Koeficienty (c0_th, c1_th, c0_el, c1_el) pro KGJParams."""
    if k.var_eff and k.eta_th_min and k.eta_el_min:
        return compute_linear_fuel_params(
            k.k_th, k.k_min, k.k_eff_th, k.eta_th_min, k.k_eff_el, k.eta_el_min)
    return 0.0, 1.0 / k.k_eff_th, 0.0, k.k_eff_el / k.k_eff_th


def kgj_el_bounds(k) -> tuple[float, float]:
    """(e_min, e_max) elektrického výkonu při běhu [MW]."""
    c0_th, c1_th, c0_el, c1_el = kgj_fuel_el_coefs(k)
    e_max = c0_el + c1_el * k.k_th
    e_min = c0_el + c1_el * (k.k_min * k.k_th)
    return e_min, e_max


# ─────────────────────────────────────────────────────────────────────────────
# Stavba modelu
# ─────────────────────────────────────────────────────────────────────────────

def _v(name: str, *idx) -> str:
    return name + "_" + "_".join(str(i) for i in idx)


def build_and_solve(grid: TimeGrid, profile: Profile, series: InputSeries,
                    afrr: AfrrRequirement | None = None,
                    mode: RunMode = RunMode.DA_PLAN,
                    nomination_mw: np.ndarray | None = None,
                    lambda_dev: float = 0.0,
                    time_limit_s: int = 60, gap_rel: float = 0.003,
                    msg: bool = False) -> SolveResult | None:
    """Sestaví a vyřeší MILP. Vrací SolveResult, None při infeasibilitě."""
    afrr = afrr or AfrrRequirement.none(profile.afrr_activation_h)
    model, V = _build(grid, profile, series, afrr, mode, nomination_mw,
                      lambda_dev)
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_s, gapRel=gap_rel)
    status = model.solve(solver)
    if status != pulp.LpStatusOptimal:
        return None
    return _extract(grid, profile, series, afrr, mode, nomination_mw, model, V)


def _purchase_unit_price(mode: RunMode, p_ref_t: float, p_da_t: float,
                         fix: bool, fix_price: float | None,
                         dist_buy: float) -> float:
    """Jednotková cena nákupu EE z gridu pro daný MTU.

    DA_PLAN:    (kontrakt nebo DA) + distribuce.
    REDISPATCH: jen distribuce + delta fixního kontraktu vůči referenční
                ceně (energie se oceňuje přes nominaci/odchylku).
    """
    if mode is RunMode.DA_PLAN:
        p_e = fix_price if (fix and fix_price is not None) else p_da_t
        return p_e + dist_buy
    delta = (fix_price - p_ref_t) if (fix and fix_price is not None) else 0.0
    return dist_buy + delta


def _build(grid: TimeGrid, profile: Profile, series: InputSeries,
           afrr: AfrrRequirement, mode: RunMode,
           nomination_mw: np.ndarray | None, lambda_dev: float):
    n, dt = grid.n, grid.dt_h
    T = range(n)
    model = pulp.LpProblem("D1_dispatch", pulp.LpMaximize)
    V: dict = {"sites": {}}
    obj = []

    da = series.da_price_pred
    gas = series.gas_price
    # referenční cena energie pro kontraktní delty v REDISPATCH
    p_ref = (series.da_price_actual
             if (mode is RunMode.REDISPATCH and series.da_price_actual is not None)
             else da)

    for s in profile.sites:
        sv: dict = {}
        V["sites"][s.site_id] = sv
        heat_dem = series.heat_demand.get(s.site_id, np.zeros(n))
        cons = series.consumption.get(s.site_id, np.zeros(n))

        # ── proměnné assetů ──────────────────────────────────────────────
        for k in s.kgjs:
            q = pulp.LpVariable.dicts(_v("qKGJ", s.site_id, k.asset_id), T,
                                      0, k.k_th)
            on = pulp.LpVariable.dicts(_v("on", s.site_id, k.asset_id), T,
                                       0, 1, "Binary")
            st = pulp.LpVariable.dicts(_v("start", s.site_id, k.asset_id), T,
                                       0, 1, "Binary")
            sv[("kgj", k.asset_id)] = {"q": q, "on": on, "start": st,
                                       "coefs": kgj_fuel_el_coefs(k),
                                       "params": k}
            for t in T:
                model += q[t] <= k.k_th * on[t]
                model += q[t] >= k.k_min * k.k_th * on[t]
            model += st[0] == on[0]
            for t in range(1, n):
                model += st[t] >= on[t] - on[t - 1]
                model += st[t] <= on[t]
                model += st[t] <= 1 - on[t - 1]
            min_rt = int(round(k.k_min_runtime_h / dt))
            for t in T:
                for d_ in range(1, min_rt):
                    if t + d_ < n:
                        model += on[t + d_] >= st[t]

        for b in s.boilers:
            sv[("boil", b.asset_id)] = {
                "q": pulp.LpVariable.dicts(_v("qBoil", s.site_id, b.asset_id),
                                           T, 0, b.b_max),
                "params": b}

        for e in s.eks:
            sv[("ek", e.asset_id)] = {
                "q": pulp.LpVariable.dicts(_v("qEK", s.site_id, e.asset_id),
                                           T, 0, e.ek_max),
                "params": e}

        for x in s.tes_units:
            soc = pulp.LpVariable.dicts(_v("tesSOC", s.site_id, x.asset_id),
                                        range(n + 1), 0, x.tes_cap)
            tin = pulp.LpVariable.dicts(_v("tesIn", s.site_id, x.asset_id), T, 0)
            tout = pulp.LpVariable.dicts(_v("tesOut", s.site_id, x.asset_id), T, 0)
            model += soc[0] == x.tes_cap * x.soc_start_frac
            model += soc[n] >= x.tes_cap * x.soc_end_min_frac
            loss = x.tes_loss_pct_h / 100.0
            for t in T:
                model += soc[t + 1] == soc[t] * (1 - loss * dt) \
                    + (tin[t] - tout[t]) * dt
            sv[("tes", x.asset_id)] = {"soc": soc, "in": tin, "out": tout,
                                       "params": x}

        for bb in s.bess_units:
            soc = pulp.LpVariable.dicts(_v("bSOC", s.site_id, bb.asset_id),
                                        range(n + 1), 0, bb.bess_cap)
            cha = pulp.LpVariable.dicts(_v("bCha", s.site_id, bb.asset_id),
                                        T, 0, bb.bess_p)
            dis = pulp.LpVariable.dicts(_v("bDis", s.site_id, bb.asset_id),
                                        T, 0, bb.bess_p)
            model += soc[0] == bb.bess_cap * bb.soc_start_frac
            model += soc[n] >= bb.bess_cap * bb.soc_end_min_frac
            for t in T:
                model += soc[t + 1] == soc[t] \
                    + (cha[t] * bb.bess_eff - dis[t] / bb.bess_eff) * dt
            sv[("bess", bb.asset_id)] = {"soc": soc, "cha": cha, "dis": dis,
                                         "params": bb}

        for pv in s.pvs:
            fc = series.pv_forecast.get((s.site_id, pv.asset_id), np.zeros(n))
            used = pulp.LpVariable.dicts(_v("pv", s.site_id, pv.asset_id), T, 0)
            for t in T:
                model += used[t] <= float(fc[t])
                if not pv.allow_curtailment:
                    model += used[t] >= float(fc[t])
            sv[("pv", pv.asset_id)] = {"used": used, "forecast": fc,
                                       "params": pv}

        for im in s.heat_imports:
            sv[("imp", im.asset_id)] = {
                "q": pulp.LpVariable.dicts(_v("qImp", s.site_id, im.asset_id),
                                           T, 0, im.imp_max),
                "params": im}

        # ── síť lokality ─────────────────────────────────────────────────
        exp = pulp.LpVariable.dicts(_v("exp", s.site_id), T, 0)
        imp = pulp.LpVariable.dicts(_v("imp", s.site_id), T, 0)
        sv["export"], sv["import"] = exp, imp
        sv["cons"] = cons
        sv["heat_dem"] = heat_dem

        if s.has_heat:
            sv["shortfall"] = pulp.LpVariable.dicts(_v("short", s.site_id), T, 0)
            sv["dump"] = pulp.LpVariable.dicts(_v("dump", s.site_id), T, 0)

        # lokál/grid split spotřebičů EE (EK, BESS nabíjení, spotřeba)
        splits: dict = {}
        for key in [k for k in sv if isinstance(k, tuple)]:
            typ, aid = key
            if typ == "ek":
                splits[key] = (
                    pulp.LpVariable.dicts(_v("ekLoc", s.site_id, aid), T, 0),
                    pulp.LpVariable.dicts(_v("ekGrid", s.site_id, aid), T, 0))
            elif typ == "bess":
                splits[key] = (
                    pulp.LpVariable.dicts(_v("bLoc", s.site_id, aid), T, 0),
                    pulp.LpVariable.dicts(_v("bGrid", s.site_id, aid), T, 0))
        cons_loc = pulp.LpVariable.dicts(_v("cLoc", s.site_id), T, 0)
        cons_grid = pulp.LpVariable.dicts(_v("cGrid", s.site_id), T, 0)
        sv["splits"] = splits
        sv["cons_split"] = (cons_loc, cons_grid)

        def heat_delivered_expr(t):
            return (pulp.lpSum(sv[k]["q"][t] for k in sv
                               if isinstance(k, tuple)
                               and k[0] in ("kgj", "boil", "ek", "imp"))
                    + pulp.lpSum(sv[k]["out"][t] - sv[k]["in"][t] for k in sv
                                 if isinstance(k, tuple) and k[0] == "tes"))

        # ── rovnice per MTU ──────────────────────────────────────────────
        for t in T:
            if s.has_heat:
                hd = heat_delivered_expr(t)
                model += hd + sv["shortfall"][t] >= float(heat_dem[t]) * s.h_cover
                model += hd <= float(heat_dem[t]) + sv["dump"][t] + 1e-3

            e_kgj = pulp.lpSum(
                sv[k]["coefs"][2] * sv[k]["on"][t]
                + sv[k]["coefs"][3] * sv[k]["q"][t]
                for k in sv if isinstance(k, tuple) and k[0] == "kgj")
            pv_used = pulp.lpSum(sv[k]["used"][t] for k in sv
                                 if isinstance(k, tuple) and k[0] == "pv")
            b_dis = pulp.lpSum(sv[k]["dis"][t] for k in sv
                               if isinstance(k, tuple) and k[0] == "bess")
            b_cha = pulp.lpSum(sv[k]["cha"][t] for k in sv
                               if isinstance(k, tuple) and k[0] == "bess")
            ee_ek = pulp.lpSum(sv[k]["q"][t] / sv[k]["params"].ek_eff
                               for k in sv if isinstance(k, tuple)
                               and k[0] == "ek")
            model += e_kgj + pv_used + b_dis + imp[t] \
                == ee_ek + b_cha + float(cons[t]) + exp[t]

            # split: total = lokál + grid; import kryje výhradně grid strany
            grid_parts = [cons_grid[t]]
            model += cons_loc[t] + cons_grid[t] == float(cons[t])
            for key, (loc, gr) in splits.items():
                if key[0] == "ek":
                    total = sv[key]["q"][t] / sv[key]["params"].ek_eff
                else:
                    total = sv[key]["cha"][t]
                model += loc[t] + gr[t] == total
                grid_parts.append(gr[t])
            model += imp[t] == pulp.lpSum(grid_parts)
            if not s.internal_ee_use:
                model += cons_loc[t] == 0
                for loc, _ in splits.values():
                    model += loc[t] == 0

            if s.grid_export_limit_mw is not None:
                model += exp[t] <= s.grid_export_limit_mw
            if s.grid_import_limit_mw is not None:
                model += imp[t] <= s.grid_import_limit_mw

        # ── ekonomické členy objective ───────────────────────────────────
        for t in T:
            p_da_t = float(da[t])
            p_gas_t = float(gas[t])
            p_ref_t = float(p_ref[t])
            terms = []
            if s.has_heat:
                hd = heat_delivered_expr(t)
                terms.append(s.h_price * (hd - sv["dump"][t]))
                terms.append(-s.shortfall_penalty * sv["shortfall"][t])

            co2_gas_flows = []
            for key in [k for k in sv if isinstance(k, tuple)]:
                typ, aid = key
                a = sv[key]
                if typ == "kgj":
                    k = a["params"]
                    c0_th, c1_th, c0_el, c1_el = a["coefs"]
                    fuel = c0_th * a["on"][t] + c1_th * a["q"][t]
                    p_g = k.gas_fix_price if (k.gas_fix and k.gas_fix_price
                                              is not None) else p_gas_t
                    terms.append(-(p_g + s.gas_dist) * fuel)
                    co2_gas_flows.append(fuel)
                    terms.append(-k.k_start_cost / dt * a["start"][t])
                    terms.append(-k.k_service_cost * a["on"][t])
                    if k.ee_fix and k.ee_fix_price is not None:
                        e_out = c0_el * a["on"][t] + c1_el * a["q"][t]
                        terms.append((k.ee_fix_price - p_ref_t) * e_out)
                elif typ == "boil":
                    b = a["params"]
                    fuel = a["q"][t] * (1.0 / b.boil_eff)
                    p_g = b.gas_fix_price if (b.gas_fix and b.gas_fix_price
                                              is not None) else p_gas_t
                    terms.append(-(p_g + s.gas_dist) * fuel)
                    co2_gas_flows.append(fuel)
                elif typ == "ek":
                    e = a["params"]
                    unit = _purchase_unit_price(mode, p_ref_t, p_da_t,
                                                e.ee_fix, e.ee_fix_price,
                                                s.dist_ee_buy)
                    terms.append(-unit * splits[key][1][t])
                elif typ == "bess":
                    bb = a["params"]
                    unit = _purchase_unit_price(mode, p_ref_t, p_da_t,
                                                bb.ee_fix, bb.ee_fix_price,
                                                s.dist_ee_buy)
                    terms.append(-unit * splits[key][1][t])
                    terms.append(-bb.bess_cycle_cost * (a["cha"][t] + a["dis"][t]))
                    if bb.dist_buy_extra:
                        terms.append(-s.dist_ee_buy * a["cha"][t])
                    if bb.dist_sell_extra:
                        terms.append(-s.dist_ee_sell * a["dis"][t])
                elif typ == "pv":
                    if a["params"].dist_sell:
                        terms.append(-s.dist_ee_sell * a["used"][t])
                elif typ == "imp":
                    terms.append(-a["params"].imp_price * a["q"][t])

            # spotřeba krytá z gridu
            unit_c = _purchase_unit_price(mode, p_ref_t, p_da_t, False, None,
                                          s.dist_ee_buy)
            terms.append(-unit_c * cons_grid[t])

            if s.co2_price > 0:
                terms.append(-s.co2_price * (
                    s.co2_gas_factor * pulp.lpSum(co2_gas_flows)
                    + s.co2_grid_factor * imp[t]
                    - s.co2_grid_factor * exp[t]))

            if mode is RunMode.DA_PLAN:
                terms.append((p_da_t - s.dist_ee_sell) * exp[t])
            else:
                terms.append(-s.dist_ee_sell * exp[t])

            obj.append(pulp.lpSum(terms) * dt)

    # ── portfolio: pozice a odchylka ─────────────────────────────────────────
    pos = pulp.LpVariable.dicts("pos", T)
    for t in T:
        model += pos[t] == pulp.lpSum(
            V["sites"][s.site_id]["export"][t]
            - V["sites"][s.site_id]["import"][t]
            for s in profile.sites)
    V["pos"] = pos

    if mode is RunMode.REDISPATCH:
        if nomination_mw is None:
            raise ValueError("REDISPATCH vyžaduje nomination_mw.")
        dev_p = pulp.LpVariable.dicts("devP", T, 0)
        dev_n = pulp.LpVariable.dicts("devN", T, 0)
        V["dev_p"], V["dev_n"] = dev_p, dev_n
        imb = series.imb_price_pred
        for t in T:
            model += pos[t] - float(nomination_mw[t]) == dev_p[t] - dev_n[t]
            obj.append((float(imb[t]) * (dev_p[t] - dev_n[t])
                        - lambda_dev * (dev_p[t] + dev_n[t])) * grid.dt_h)

    _add_afrr_constraints(model, V, grid, profile, afrr)

    model += pulp.lpSum(obj)
    return model, V


def _add_afrr_constraints(model, V, grid: TimeGrid, profile: Profile,
                          afrr: AfrrRequirement) -> None:
    """aFRR rezervační omezení — doplní WP5. Bez rezerv no-op; s rezervami
    zatím vyvolá NotImplementedError (implementace v dalším balíčku)."""
    if afrr.any_reserved:
        raise NotImplementedError("aFRR rezervace přijdou ve WP5.")


# ─────────────────────────────────────────────────────────────────────────────
# Extrakce výsledku
# ─────────────────────────────────────────────────────────────────────────────

def _val(x) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    v = pulp.value(x)
    return float(v) if v is not None else 0.0


def _extract(grid: TimeGrid, profile: Profile, series: InputSeries,
             afrr: AfrrRequirement, mode: RunMode,
             nomination_mw: np.ndarray | None, model, V) -> SolveResult:
    n, dt = grid.n, grid.dt_h
    T = range(n)
    cols: dict[str, np.ndarray] = {}
    eco: dict[str, float] = {k: 0.0 for k in (
        "rev_heat", "rev_export", "rev_kgj_bonus", "rev_deviation",
        "cost_gas", "cost_ee_grid", "cost_heat_import", "cost_start",
        "cost_service", "cost_bess_cycle", "cost_shortfall", "cost_co2",
        "cost_dist_extra")}
    co2_total = 0.0

    da = series.da_price_pred
    gas = series.gas_price
    p_ref = (series.da_price_actual
             if (mode is RunMode.REDISPATCH and series.da_price_actual is not None)
             else da)

    pos = np.array([_val(V["pos"][t]) for t in T])

    for s in profile.sites:
        sv = V["sites"][s.site_id]
        pre = s.site_id
        exp = np.array([_val(sv["export"][t]) for t in T])
        imp = np.array([_val(sv["import"][t]) for t in T])
        cols[f"{pre}|export_mw"] = exp
        cols[f"{pre}|import_mw"] = imp
        cols[f"{pre}|cons_mw"] = np.asarray(sv["cons"], dtype=float)
        _, cons_grid = sv["cons_split"]
        cg = np.array([_val(cons_grid[t]) for t in T])

        if s.has_heat:
            cols[f"{pre}|heat_dem_mw"] = np.asarray(sv["heat_dem"], dtype=float)
            sh = np.array([_val(sv["shortfall"][t]) for t in T])
            du = np.array([_val(sv["dump"][t]) for t in T])
            cols[f"{pre}|heat_shortfall_mw"] = sh
            cols[f"{pre}|heat_dump_mw"] = du
            eco["cost_shortfall"] += float((s.shortfall_penalty * sh).sum() * dt)

        heat_delivered = np.zeros(n)
        gas_flow_site = np.zeros(n)

        def _grid_cost(key, params) -> float:
            _, gr = sv["splits"][key]
            grv = np.array([_val(gr[t]) for t in T])
            unit = np.array([_purchase_unit_price(
                mode, float(p_ref[t]), float(da[t]),
                getattr(params, "ee_fix", False),
                getattr(params, "ee_fix_price", None),
                s.dist_ee_buy) for t in T])
            return float((unit * grv).sum() * dt)

        for key in [k for k in sv if isinstance(k, tuple)]:
            typ, aid = key
            a = sv[key]
            apre = f"{pre}|{aid}"
            if typ == "kgj":
                k = a["params"]
                c0_th, c1_th, c0_el, c1_el = a["coefs"]
                q = np.array([_val(a["q"][t]) for t in T])
                on = np.array([_val(a["on"][t]) for t in T])
                st = np.array([_val(a["start"][t]) for t in T])
                e_out = c0_el * on + c1_el * q
                fuel = c0_th * on + c1_th * q
                cols[f"{apre}|q_th_mw"] = q
                cols[f"{apre}|e_el_mw"] = e_out
                cols[f"{apre}|on"] = on
                heat_delivered += q
                gas_flow_site += fuel
                p_g = (np.full(n, k.gas_fix_price)
                       if (k.gas_fix and k.gas_fix_price is not None) else gas)
                eco["cost_gas"] += float(((p_g + s.gas_dist) * fuel).sum() * dt)
                eco["cost_start"] += float(k.k_start_cost * st.sum())
                eco["cost_service"] += float(k.k_service_cost * on.sum() * dt)
                if k.ee_fix and k.ee_fix_price is not None:
                    eco["rev_kgj_bonus"] += float(
                        ((k.ee_fix_price - p_ref) * e_out).sum() * dt)
            elif typ == "boil":
                b = a["params"]
                q = np.array([_val(a["q"][t]) for t in T])
                fuel = q / b.boil_eff
                cols[f"{apre}|q_th_mw"] = q
                heat_delivered += q
                gas_flow_site += fuel
                p_g = (np.full(n, b.gas_fix_price)
                       if (b.gas_fix and b.gas_fix_price is not None) else gas)
                eco["cost_gas"] += float(((p_g + s.gas_dist) * fuel).sum() * dt)
            elif typ == "ek":
                e = a["params"]
                q = np.array([_val(a["q"][t]) for t in T])
                cols[f"{apre}|q_th_mw"] = q
                cols[f"{apre}|ee_in_mw"] = q / e.ek_eff
                heat_delivered += q
                eco["cost_ee_grid"] += _grid_cost(key, e)
            elif typ == "tes":
                soc = np.array([_val(a["soc"][t + 1]) for t in T])
                ti = np.array([_val(a["in"][t]) for t in T])
                to = np.array([_val(a["out"][t]) for t in T])
                cols[f"{apre}|soc_mwh"] = soc
                cols[f"{apre}|in_mw"] = ti
                cols[f"{apre}|out_mw"] = to
                heat_delivered += to - ti
            elif typ == "bess":
                bb = a["params"]
                soc = np.array([_val(a["soc"][t + 1]) for t in T])
                cha = np.array([_val(a["cha"][t]) for t in T])
                dis = np.array([_val(a["dis"][t]) for t in T])
                cols[f"{apre}|soc_mwh"] = soc
                cols[f"{apre}|cha_mw"] = cha
                cols[f"{apre}|dis_mw"] = dis
                eco["cost_ee_grid"] += _grid_cost(key, bb)
                eco["cost_bess_cycle"] += float(
                    (bb.bess_cycle_cost * (cha + dis)).sum() * dt)
                if bb.dist_buy_extra:
                    eco["cost_dist_extra"] += float(
                        (s.dist_ee_buy * cha).sum() * dt)
                if bb.dist_sell_extra:
                    eco["cost_dist_extra"] += float(
                        (s.dist_ee_sell * dis).sum() * dt)
            elif typ == "pv":
                used = np.array([_val(a["used"][t]) for t in T])
                fc = np.asarray(a["forecast"], dtype=float)
                cols[f"{apre}|used_mw"] = used
                cols[f"{apre}|forecast_mw"] = fc
                if a["params"].dist_sell:
                    eco["cost_dist_extra"] += float(
                        (s.dist_ee_sell * used).sum() * dt)
            elif typ == "imp":
                q = np.array([_val(a["q"][t]) for t in T])
                cols[f"{apre}|q_th_mw"] = q
                heat_delivered += q
                eco["cost_heat_import"] += float(
                    (a["params"].imp_price * q).sum() * dt)

        if s.has_heat:
            cols[f"{pre}|heat_delivered_mw"] = heat_delivered
            du = cols[f"{pre}|heat_dump_mw"]
            eco["rev_heat"] += float(
                (s.h_price * (heat_delivered - du)).sum() * dt)

        # nákup spotřeby z gridu
        unit_c = np.array([_purchase_unit_price(
            mode, float(p_ref[t]), float(da[t]), False, None,
            s.dist_ee_buy) for t in T])
        eco["cost_ee_grid"] += float((unit_c * cg).sum() * dt)

        # CO2 KPI + náklad
        co2_site = (s.co2_gas_factor * gas_flow_site
                    + s.co2_grid_factor * imp - s.co2_grid_factor * exp)
        co2_total += float(co2_site.sum() * dt)
        if s.co2_price > 0:
            eco["cost_co2"] += float(s.co2_price * co2_site.sum() * dt)

        if mode is RunMode.DA_PLAN:
            eco["rev_export"] += float(((da - s.dist_ee_sell) * exp).sum() * dt)
        else:
            eco["rev_export"] += float((-s.dist_ee_sell * exp).sum() * dt)

    cols["pos_mw"] = pos
    cols["da_price_eur"] = da
    cols["imb_price_eur"] = series.imb_price_pred
    if mode is RunMode.REDISPATCH:
        dev = pos - np.asarray(nomination_mw, dtype=float)
        cols["nomination_mw"] = np.asarray(nomination_mw, dtype=float)
        cols["deviation_mw"] = dev
        eco["rev_deviation"] = float((series.imb_price_pred * dev).sum() * dt)
        if series.da_price_actual is not None:
            eco["rev_nomination"] = float(
                (series.da_price_actual
                 * np.asarray(nomination_mw, dtype=float)).sum() * dt)
            cols["da_price_actual_eur"] = series.da_price_actual

    eco["co2_total_t"] = co2_total
    revenue = sum(v for k, v in eco.items() if k.startswith("rev_"))
    costs = sum(v for k, v in eco.items() if k.startswith("cost_"))
    eco["profit_total"] = revenue - costs

    plan = pd.DataFrame(cols, index=np.arange(1, n + 1))
    plan.index.name = "MTU"
    plan.insert(0, "cas_od", grid.times_from())

    return SolveResult(
        status=pulp.LpStatus[model.status],
        objective=float(pulp.value(model.objective)),
        plan=plan, position_mw=pos, economics=eco,
        reserve_alloc=_extract_reserves(grid, profile, V))


def _extract_reserves(grid: TimeGrid, profile: Profile, V) -> pd.DataFrame | None:
    """Alokace aFRR rezerv per asset — doplní WP5 (bez rezerv None)."""
    return None
