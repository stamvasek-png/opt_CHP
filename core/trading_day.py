"""Persistence obchodního dne: data/runs/YYYY-MM-DD/.

Nahrazuje pickle cache původní aplikace. Každý dodávkový den má vlastní
adresář s day.json (metadata, snapshot profilu, stavové flagy, aukce,
ekonomiky) a parquet artefakty (vstupy, plány, nominace). Snapshot
profilu se ukládá při založení dne — pozdější úpravy šablony historii
neovlivní. Nominace tak přežije restart aplikace.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .inputs import InputSeries
from .model import SolveResult
from .profiles import Profile, profile_from_dict, profile_to_dict
from .runs import AuctionResults, Run1Result
from .timegrid import TimeGrid, make_grid

RUNS_DIR = Path(__file__).resolve().parent.parent / "data" / "runs"

STAGE_FLAGS = ("inputs_loaded", "run1_done", "auction_entered", "run2_done",
               "nomination_frozen", "actual_da_loaded", "run3_done")

# co musí být hotové, než smí začít daný krok
STAGE_REQUIRES = {
    "run1": ("inputs_loaded",),
    "run2": ("inputs_loaded",),
    "freeze": ("run2_done",),
    "run3": ("nomination_frozen", "actual_da_loaded"),
}


class StageError(RuntimeError):
    pass


@dataclass
class TradingDay:
    delivery_date: dt.date
    profile: Profile                     # snapshot z okamžiku založení
    fx_czk_eur: float = 25.0
    lambda_dev: float = 0.0
    status: dict = field(default_factory=lambda: {f: False for f in STAGE_FLAGS})
    auction: AuctionResults | None = None
    run1_meta: dict = field(default_factory=dict)
    run2_economics: dict = field(default_factory=dict)
    run3_economics: dict = field(default_factory=dict)
    timestamps: dict = field(default_factory=dict)
    input_report: list = field(default_factory=list)
    base_dir: Path = RUNS_DIR

    # ── cesty ────────────────────────────────────────────────────────────
    @property
    def dir(self) -> Path:
        return self.base_dir / self.delivery_date.isoformat()

    def _p(self, name: str) -> Path:
        return self.dir / name

    @property
    def grid(self) -> TimeGrid:
        return make_grid(self.delivery_date)

    # ── guardy ───────────────────────────────────────────────────────────
    def assert_stage(self, stage: str) -> None:
        for flag in STAGE_REQUIRES.get(stage, ()):
            if not self.status.get(flag):
                raise StageError(
                    f"Krok '{stage}' vyžaduje nejdřív dokončit: {flag}.")

    def _stamp(self, flag: str) -> None:
        self.status[flag] = True
        self.timestamps[flag] = dt.datetime.now().astimezone().isoformat(
            timespec="seconds")

    # ── persistence day.json ─────────────────────────────────────────────
    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        d = {
            "schema_version": 1,
            "delivery_date": self.delivery_date.isoformat(),
            "profile_snapshot": profile_to_dict(self.profile),
            "fx_czk_eur": self.fx_czk_eur,
            "lambda_dev": self.lambda_dev,
            "status": self.status,
            "auction_results": self.auction.to_dict() if self.auction else None,
            "run1_meta": self.run1_meta,
            "run2_economics": self.run2_economics,
            "run3_economics": self.run3_economics,
            "timestamps": self.timestamps,
            "input_report": self.input_report,
        }
        tmp = self._p("day.json.tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, self._p("day.json"))

    # ── artefakty ────────────────────────────────────────────────────────
    def save_inputs(self, series: InputSeries, report: list[str]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        series.to_frame().to_parquet(self._p("inputs.parquet"))
        self.input_report = list(report)
        self._stamp("inputs_loaded")
        # nové vstupy zneplatní navazující kroky
        for f in ("run1_done", "run2_done", "run3_done"):
            self.status[f] = False
        self.save()

    def load_inputs(self) -> InputSeries:
        df = pd.read_parquet(self._p("inputs.parquet"))
        return InputSeries.from_frame(df, self.grid)

    def save_run1(self, result: Run1Result) -> None:
        self.assert_stage("run1")
        result.ladder.to_parquet(self._p("run1_ladder.parquet"))
        self.run1_meta = {
            "baseline_profit": result.baseline_profit,
            "r_max_up": result.r_max_up, "r_max_dn": result.r_max_dn,
            "solves": result.solves, "wall_s": round(result.wall_s, 1),
            "r_grid_frac": list(result.r_grid_frac),
        }
        self._stamp("run1_done")
        self.save()

    def skip_run1(self) -> None:
        self.assert_stage("run1")
        self.run1_meta = {"skipped": True}
        self._stamp("run1_done")
        self.save()

    def load_run1_ladder(self) -> pd.DataFrame | None:
        p = self._p("run1_ladder.parquet")
        return pd.read_parquet(p) if p.exists() else None

    def set_auction(self, auction: AuctionResults) -> None:
        self.auction = auction
        self._stamp("auction_entered")
        self.save()

    def save_run2(self, result: SolveResult) -> None:
        self.assert_stage("run2")
        if self.status.get("nomination_frozen"):
            raise StageError("Nominace je zmrazená — nejdřív ji odemkněte.")
        result.plan.to_parquet(self._p("run2_plan.parquet"))
        if result.reserve_alloc is not None:
            result.reserve_alloc.to_parquet(self._p("run2_reserves.parquet"))
        self.run2_economics = dict(result.economics)
        self._stamp("run2_done")
        self.save()

    def load_run2_plan(self) -> pd.DataFrame | None:
        p = self._p("run2_plan.parquet")
        return pd.read_parquet(p) if p.exists() else None

    def freeze_nomination(self) -> pd.DataFrame:
        """Zmrazí nominaci z pozice run2 plánu; vrací nominační tabulku."""
        self.assert_stage("freeze")
        plan = self.load_run2_plan()
        grid = self.grid
        nom = pd.DataFrame({
            "mtu": np.arange(1, grid.n + 1),
            "cas_od": grid.times_from(),
            "pos_mw": plan["pos_mw"].to_numpy(),
            "energie_mwh": plan["pos_mw"].to_numpy() * grid.dt_h,
        })
        nom.to_parquet(self._p("nomination.parquet"))
        self._stamp("nomination_frozen")
        self.save()
        return nom

    def unfreeze_nomination(self) -> None:
        self.status["nomination_frozen"] = False
        self.save()

    def load_nomination(self) -> pd.DataFrame | None:
        p = self._p("nomination.parquet")
        return pd.read_parquet(p) if p.exists() else None

    def save_actual_da(self, prices_eur: np.ndarray) -> None:
        pd.DataFrame({"da_price_actual": prices_eur}).to_parquet(
            self._p("actual_da.parquet"))
        self._stamp("actual_da_loaded")
        self.save()

    def load_actual_da(self) -> np.ndarray | None:
        p = self._p("actual_da.parquet")
        if not p.exists():
            return None
        return pd.read_parquet(p)["da_price_actual"].to_numpy()

    def save_run3(self, result: SolveResult, lambda_dev: float) -> None:
        self.assert_stage("run3")
        result.plan.to_parquet(self._p("run3_plan.parquet"))
        if result.reserve_alloc is not None:
            result.reserve_alloc.to_parquet(self._p("run3_reserves.parquet"))
        self.run3_economics = dict(result.economics)
        self.lambda_dev = lambda_dev
        self._stamp("run3_done")
        self.save()

    def load_run3_plan(self) -> pd.DataFrame | None:
        p = self._p("run3_plan.parquet")
        return pd.read_parquet(p) if p.exists() else None


class TradingDayStore:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else RUNS_DIR

    def exists(self, date: dt.date) -> bool:
        return (self.base_dir / date.isoformat() / "day.json").exists()

    def open(self, date: dt.date, profile: Profile | None = None,
             fx_czk_eur: float | None = None) -> TradingDay:
        """Načte existující den, nebo založí nový (vyžaduje profile)."""
        if self.exists(date):
            return self._load(date)
        if profile is None:
            raise ValueError(f"Den {date} neexistuje — pro založení předejte "
                             f"profil.")
        day = TradingDay(delivery_date=date, profile=profile,
                         fx_czk_eur=fx_czk_eur or 25.0,
                         base_dir=self.base_dir)
        day.save()
        return day

    def _load(self, date: dt.date) -> TradingDay:
        path = self.base_dir / date.isoformat() / "day.json"
        d = json.loads(path.read_text(encoding="utf-8"))
        day = TradingDay(
            delivery_date=dt.date.fromisoformat(d["delivery_date"]),
            profile=profile_from_dict(d["profile_snapshot"]),
            fx_czk_eur=d.get("fx_czk_eur", 25.0),
            lambda_dev=d.get("lambda_dev", 0.0),
            status={**{f: False for f in STAGE_FLAGS}, **d.get("status", {})},
            auction=(AuctionResults.from_dict(d["auction_results"])
                     if d.get("auction_results") else None),
            run1_meta=d.get("run1_meta", {}),
            run2_economics=d.get("run2_economics", {}),
            run3_economics=d.get("run3_economics", {}),
            timestamps=d.get("timestamps", {}),
            input_report=d.get("input_report", []),
            base_dir=self.base_dir,
        )
        return day

    def list_days(self) -> list[dict]:
        """Metadata všech uložených dní (nejnovější první)."""
        out = []
        if not self.base_dir.exists():
            return out
        for p in sorted(self.base_dir.iterdir(), reverse=True):
            f = p / "day.json"
            if not f.is_file():
                continue
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                out.append({
                    "delivery_date": d["delivery_date"],
                    "profile_id": d["profile_snapshot"].get("profile_id", "?"),
                    "profile_name": d["profile_snapshot"].get("name", "?"),
                    "status": d.get("status", {}),
                    "profit_run2": d.get("run2_economics", {}).get("profit_total"),
                    "profit_run3": d.get("run3_economics", {}).get("profit_total"),
                })
            except (json.JSONDecodeError, OSError, KeyError):
                continue
        return out
