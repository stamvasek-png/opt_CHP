"""Orchestrace tří běhů obchodního dne.

Run 1 (~08:00) — aFRR nabídky: žebřík opportunity cost per (blok, směr, R)
Run 2 (~10:00) — DA plán + nominace s vysoutěženou kapacitou jako omezením
Run 3 (po 14:00) — re-dispatch s fixní nominací proti ceně odchylky
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .inputs import InputSeries
from .model import (AfrrRequirement, RunMode, SolveResult, build_and_solve,
                    kgj_el_bounds)
from .profiles import Profile
from .timegrid import BLOCK_LABELS, N_BLOCKS, TimeGrid

DIRECTIONS = ("up", "dn")
DIR_LABELS = {"up": "aFRR+", "dn": "aFRR−"}


@dataclass
class AuctionResults:
    """Výsledek denní aukce ČEPS: vysoutěžené MW a kapacitní ceny per blok."""
    up_mw: np.ndarray = field(default_factory=lambda: np.zeros(N_BLOCKS))
    up_price: np.ndarray = field(default_factory=lambda: np.zeros(N_BLOCKS))
    dn_mw: np.ndarray = field(default_factory=lambda: np.zeros(N_BLOCKS))
    dn_price: np.ndarray = field(default_factory=lambda: np.zeros(N_BLOCKS))

    def to_requirement(self, activation_h: float) -> AfrrRequirement:
        return AfrrRequirement(np.asarray(self.up_mw, dtype=float),
                               np.asarray(self.dn_mw, dtype=float),
                               activation_h)

    def capacity_revenue(self, grid: TimeGrid) -> float:
        """Σ_b (cena_up·R_up + cena_dn·R_dn) × skutečné hodiny bloku [EUR]."""
        rev = 0.0
        for b in range(N_BLOCKS):
            hours = len(grid.block_intervals(b)) * grid.dt_h
            rev += (float(self.up_price[b]) * float(self.up_mw[b])
                    + float(self.dn_price[b]) * float(self.dn_mw[b])) * hours
        return rev

    def to_dict(self) -> dict:
        return {"up_mw": list(map(float, self.up_mw)),
                "up_price": list(map(float, self.up_price)),
                "dn_mw": list(map(float, self.dn_mw)),
                "dn_price": list(map(float, self.dn_price))}

    @classmethod
    def from_dict(cls, d: dict) -> "AuctionResults":
        return cls(np.array(d["up_mw"]), np.array(d["up_price"]),
                   np.array(d["dn_mw"]), np.array(d["dn_price"]))

    @property
    def any_won(self) -> bool:
        return bool((np.asarray(self.up_mw) > 1e-9).any()
                    or (np.asarray(self.dn_mw) > 1e-9).any())


def portfolio_reserve_capability(profile: Profile) -> tuple[float, float]:
    """(R_max_up, R_max_dn) [MW] — horní odhad nabídnutelné kapacity."""
    r_up = r_dn = 0.0
    for s in profile.sites:
        for k in s.kgjs:
            if k.afrr_capable:
                e_min, e_max = kgj_el_bounds(k)
                r_up += e_max - e_min
                r_dn += e_max - e_min
        for b in s.bess_units:
            if b.afrr_capable:
                r_up += b.bess_p
                r_dn += b.bess_p
        for e in s.eks:
            if e.afrr_capable:
                r_dn += e.ek_max / e.ek_eff
    return r_up, r_dn


@dataclass
class Run1Result:
    ladder: pd.DataFrame
    baseline_profit: float
    r_max_up: float
    r_max_dn: float
    solves: int
    wall_s: float
    r_grid_frac: tuple


def _solve_job(args):
    """Worker pro ProcessPool: jeden solve s rezervou v jednom bloku."""
    (grid, profile, series, block, direction, r_mw, tau,
     time_limit_s, gap_rel) = args
    r_up = np.zeros(N_BLOCKS)
    r_dn = np.zeros(N_BLOCKS)
    (r_up if direction == "up" else r_dn)[block] = r_mw
    res = build_and_solve(grid, profile, series,
                          afrr=AfrrRequirement(r_up, r_dn, tau),
                          mode=RunMode.DA_PLAN,
                          time_limit_s=time_limit_s, gap_rel=gap_rel)
    profit = res.profit if res is not None else None
    return block, direction, r_mw, profit


def run1_afrr_bids(grid: TimeGrid, profile: Profile, series: InputSeries,
                   r_grid_frac: tuple = (0.25, 0.5, 0.75, 1.0),
                   time_limit_s: int = 15, gap_rel: float = 0.002,
                   workers: int | None = None,
                   progress_cb=None) -> Run1Result:
    """Žebřík aFRR nabídek z opportunity cost.

    Pro každý (blok, směr, R) full-day solve s rezervou jen v tom bloku
    (SoC váže bloky mezi sebou, per-blok dekompozice by náklad podcenila).
    min_price_avg      = (π0 − π(R)) / (R · hodiny bloku)
    min_price_marginal = (π(R_{k−1}) − π(R_k)) / ((R_k − R_{k−1}) · hodiny)
    Doporučení: nabídnout, pokud marginal < predikce clearing ceny.
    """
    t0 = time.time()
    tau = profile.afrr_activation_h
    r_max_up, r_max_dn = portfolio_reserve_capability(profile)

    baseline = build_and_solve(grid, profile, series, mode=RunMode.DA_PLAN,
                               time_limit_s=max(time_limit_s * 4, 60),
                               gap_rel=gap_rel)
    if baseline is None:
        raise RuntimeError("Baseline plán je nesplnitelný — zkontrolujte "
                           "vstupy a profil (pokrytí tepla?).")
    pi0 = baseline.profit

    jobs = []
    for direction, r_max in (("up", r_max_up), ("dn", r_max_dn)):
        r_values = sorted({round(f * r_max, 1) for f in r_grid_frac
                           if f * r_max > 0.05})
        for b in range(N_BLOCKS):
            for r in r_values:
                jobs.append((grid, profile, series, b, direction, r, tau,
                             time_limit_s, gap_rel))

    results: dict[tuple, float | None] = {}
    total = len(jobs) + 1
    done = 1
    if progress_cb:
        progress_cb(done, total)
    workers = workers or min(os.cpu_count() or 2, 8)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_solve_job, j) for j in jobs]
        for fut in as_completed(futures):
            block, direction, r_mw, profit = fut.result()
            results[(block, direction, r_mw)] = profit
            done += 1
            if progress_cb:
                progress_cb(done, total)

    rows = []
    for direction, r_max, pred in (("up", r_max_up, series.afrr_cap_price_up),
                                   ("dn", r_max_dn, series.afrr_cap_price_dn)):
        r_values = sorted({round(f * r_max, 1) for f in r_grid_frac
                           if f * r_max > 0.05})
        for b in range(N_BLOCKS):
            hours = len(grid.block_intervals(b)) * grid.dt_h
            prev_r, prev_pi = 0.0, pi0
            for r in r_values:
                pi = results.get((b, direction, r))
                feasible = pi is not None
                if feasible:
                    avg = (pi0 - pi) / (r * hours)
                    marg = ((prev_pi - pi) / ((r - prev_r) * hours)
                            if prev_pi is not None else np.nan)
                else:
                    avg = marg = np.nan
                rows.append({
                    "block": b, "block_label": BLOCK_LABELS[b],
                    "direction": direction,
                    "direction_label": DIR_LABELS[direction],
                    "r_mw": r, "profit_eur": pi,
                    "min_price_avg": avg, "min_price_marginal": marg,
                    "predicted_price": float(pred[b]),
                    "feasible": feasible,
                    "recommend": bool(feasible and not np.isnan(marg)
                                      and marg < float(pred[b])),
                })
                if feasible:
                    prev_r, prev_pi = r, pi
                else:
                    prev_pi = None
    ladder = pd.DataFrame(rows)
    return Run1Result(ladder=ladder, baseline_profit=pi0,
                      r_max_up=r_max_up, r_max_dn=r_max_dn,
                      solves=total, wall_s=time.time() - t0,
                      r_grid_frac=tuple(r_grid_frac))


def run2_da_plan(grid: TimeGrid, profile: Profile, series: InputSeries,
                 auction: AuctionResults,
                 time_limit_s: int = 60, gap_rel: float = 0.003,
                 ) -> SolveResult | None:
    """DA plán s vysoutěženou kapacitou; nominace = pozice portfolia."""
    res = build_and_solve(grid, profile, series,
                          afrr=auction.to_requirement(profile.afrr_activation_h),
                          mode=RunMode.DA_PLAN,
                          time_limit_s=time_limit_s, gap_rel=gap_rel)
    if res is None:
        return None
    _add_capacity_revenue(res, auction, grid)
    return res


def run3_redispatch(grid: TimeGrid, profile: Profile, series: InputSeries,
                    auction: AuctionResults, nomination_mw: np.ndarray,
                    lambda_dev: float = 0.0,
                    time_limit_s: int = 60, gap_rel: float = 0.003,
                    ) -> SolveResult | None:
    """Re-dispatch s fixní nominací; vyžaduje series.da_price_actual."""
    if series.da_price_actual is None:
        raise ValueError("Run 3 vyžaduje skutečné ceny DA "
                         "(series.da_price_actual).")
    res = build_and_solve(grid, profile, series,
                          afrr=auction.to_requirement(profile.afrr_activation_h),
                          mode=RunMode.REDISPATCH,
                          nomination_mw=nomination_mw, lambda_dev=lambda_dev,
                          time_limit_s=time_limit_s, gap_rel=gap_rel)
    if res is None:
        return None
    _add_capacity_revenue(res, auction, grid)
    return res


def _add_capacity_revenue(res: SolveResult, auction: AuctionResults,
                          grid: TimeGrid) -> None:
    rev = auction.capacity_revenue(grid)
    res.economics["rev_afrr_cap"] = rev
    res.economics["profit_total"] += rev
