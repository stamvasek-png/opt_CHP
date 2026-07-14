"""Časová osa dodávkového dne D+1 v 15minutové granularitě (MTU).

Osa se staví v lokálním čase Europe/Prague, takže dny přechodu času mají
92 (jaro) nebo 100 (podzim) intervalů. Vstupní řady se klíčují číslem MTU
1..n — nikoli wall-clock časem — což jednoznačně rozliší duplicitní
hodinu 02:00 při podzimním přechodu.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TZ = "Europe/Prague"
DT_H = 0.25          # délka MTU v hodinách
N_BLOCKS = 6         # aFRR 4hodinové bloky (ALPACA): 00-04, 04-08, ... 20-24
BLOCK_LABELS = ["00–04", "04–08", "08–12", "12–16", "16–20", "20–24"]


@dataclass(frozen=True)
class TimeGrid:
    delivery_date: dt.date
    index: pd.DatetimeIndex          # tz-aware, délka n
    n: int                           # 92 / 96 / 100
    dt_h: float = DT_H
    block_of: np.ndarray = field(default=None, repr=False)  # len n, hodnoty 0..5

    @property
    def n_hours(self) -> int:
        """Počet hodin dne (23/24/25) — pro validaci hodinových vstupů."""
        return round(self.n * self.dt_h)

    def mtu_labels(self) -> list[str]:
        """Popisky '1 (00:00–00:15)', … pro tabulky a exporty."""
        out = []
        for i, ts in enumerate(self.index):
            end = ts + pd.Timedelta(minutes=15)
            out.append(f"{i + 1} ({ts.strftime('%H:%M')}–{end.strftime('%H:%M')})")
        return out

    def times_from(self) -> list[str]:
        """Časy 'HH:MM' začátků MTU (lokální čas)."""
        return [ts.strftime("%H:%M") for ts in self.index]

    def block_intervals(self, block: int) -> np.ndarray:
        """Indexy MTU patřící do daného 4h bloku."""
        return np.flatnonzero(self.block_of == block)


def make_grid(delivery_date: dt.date) -> TimeGrid:
    """Sestaví 15min osu dodávkového dne v tz Europe/Prague.

    n = 96 běžný den, 92 jarní přechod (chybí 02:00–03:00),
    100 podzimní přechod (02:00–03:00 dvakrát).
    """
    start = pd.Timestamp(delivery_date, tz=TZ)
    end = pd.Timestamp(delivery_date + dt.timedelta(days=1), tz=TZ)
    index = pd.date_range(start, end, freq="15min", inclusive="left")
    n = len(index)
    # Blok podle lokální hodiny; na podzim mají obě 02:00 hodinu blok 0
    # (blok 0 pak má 20 MTU, na jaře 12 MTU) — odpovídá praxi ČEPS/OTE.
    block_of = np.array([ts.hour // 4 for ts in index], dtype=int)
    return TimeGrid(delivery_date=delivery_date, index=index, n=n, block_of=block_of)


def resample_hourly_to_quarter(values: np.ndarray, grid: TimeGrid,
                               kind: str = "repeat") -> np.ndarray:
    """Rozpad hodinové řady (délky grid.n_hours) na 15min osu.

    kind='repeat'      — poziční opakování 4× (ceny, teplo, spotřeba, TDD);
                         DST-safe, protože jde po pozicích, ne po wall-clocku.
    kind='interpolate' — lineární interpolace pro hladké průběhy (FVE osvit):
                         hodinová hodnota = hodnota ve středu hodiny, krajní
                         MTU extrapolace konstantou. Zachovává tvar křivky.
    """
    values = np.asarray(values, dtype=float)
    if len(values) != grid.n_hours:
        raise ValueError(
            f"Hodinová řada má {len(values)} hodnot, den {grid.delivery_date} "
            f"vyžaduje {grid.n_hours} hodin."
        )
    if kind == "repeat":
        return np.repeat(values, 4)[: grid.n]
    if kind == "interpolate":
        # Středy hodin v "pozičních" hodinách: 0.5, 1.5, …; MTU středy: 0.125, 0.375, …
        hour_mid = np.arange(grid.n_hours) + 0.5
        mtu_mid = (np.arange(grid.n) + 0.5) * grid.dt_h
        return np.interp(mtu_mid, hour_mid, values)
    raise ValueError(f"Neznámý kind: {kind!r}")
