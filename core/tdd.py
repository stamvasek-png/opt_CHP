"""TDD — typové diagramy dodávky (OTE) a kalendář CZ svátků.

Spotřeba lokality v režimu 'tdd' se staví z normalizovaných TDD koeficientů
OTE pro daný rok:  mw[t] = coef[class, hodina(t)] * annual_mwh / Σ_rok coef.
Koeficienty se nahrávají ze souboru OTE (Nastavení) a ukládají do
data/tdd/tdd_<rok>.parquet; hodinové hodnoty se na 15min osu rozpadají
pozičním opakováním 4× (DST-safe).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

from .profiles import TDD_CLASSES
from .timegrid import TimeGrid, resample_hourly_to_quarter

TDD_DIR = Path(__file__).resolve().parent.parent / "data" / "tdd"

# ── České státní svátky 2026-2030 (zákon č. 245/2000 Sb.) ─────────────
# Pevné svátky + Velký pátek a Velikonoční pondělí podle data Velikonoc.
CZ_HOLIDAYS = {
    # 2026
    _dt.date(2026, 1, 1),   _dt.date(2026, 4, 3),   _dt.date(2026, 4, 6),
    _dt.date(2026, 5, 1),   _dt.date(2026, 5, 8),   _dt.date(2026, 7, 5),
    _dt.date(2026, 7, 6),   _dt.date(2026, 9, 28),  _dt.date(2026, 10, 28),
    _dt.date(2026, 11, 17), _dt.date(2026, 12, 24), _dt.date(2026, 12, 25),
    _dt.date(2026, 12, 26),
    # 2027
    _dt.date(2027, 1, 1),   _dt.date(2027, 3, 26),  _dt.date(2027, 3, 29),
    _dt.date(2027, 5, 1),   _dt.date(2027, 5, 8),   _dt.date(2027, 7, 5),
    _dt.date(2027, 7, 6),   _dt.date(2027, 9, 28),  _dt.date(2027, 10, 28),
    _dt.date(2027, 11, 17), _dt.date(2027, 12, 24), _dt.date(2027, 12, 25),
    _dt.date(2027, 12, 26),
    # 2028
    _dt.date(2028, 1, 1),   _dt.date(2028, 4, 14),  _dt.date(2028, 4, 17),
    _dt.date(2028, 5, 1),   _dt.date(2028, 5, 8),   _dt.date(2028, 7, 5),
    _dt.date(2028, 7, 6),   _dt.date(2028, 9, 28),  _dt.date(2028, 10, 28),
    _dt.date(2028, 11, 17), _dt.date(2028, 12, 24), _dt.date(2028, 12, 25),
    _dt.date(2028, 12, 26),
    # 2029
    _dt.date(2029, 1, 1),   _dt.date(2029, 3, 30),  _dt.date(2029, 4, 2),
    _dt.date(2029, 5, 1),   _dt.date(2029, 5, 8),   _dt.date(2029, 7, 5),
    _dt.date(2029, 7, 6),   _dt.date(2029, 9, 28),  _dt.date(2029, 10, 28),
    _dt.date(2029, 11, 17), _dt.date(2029, 12, 24), _dt.date(2029, 12, 25),
    _dt.date(2029, 12, 26),
    # 2030
    _dt.date(2030, 1, 1),   _dt.date(2030, 4, 19),  _dt.date(2030, 4, 22),
    _dt.date(2030, 5, 1),   _dt.date(2030, 5, 8),   _dt.date(2030, 7, 5),
    _dt.date(2030, 7, 6),   _dt.date(2030, 9, 28),  _dt.date(2030, 10, 28),
    _dt.date(2030, 11, 17), _dt.date(2030, 12, 24), _dt.date(2030, 12, 25),
    _dt.date(2030, 12, 26),
}
CZ_HOLIDAYS_COVERED_YEARS = (2026, 2030)


def is_business_day(ts) -> bool:
    """True = pracovní den po–pá, který NENÍ státní svátek (CZ)."""
    ts = pd.Timestamp(ts)
    return ts.weekday() < 5 and ts.date() not in CZ_HOLIDAYS


# ── TDD store ────────────────────────────────────────────────────────────────

class TddStore:
    """Uložené roční TDD koeficienty: DataFrame index=datetime (lokální,
    hodinový, 8760/8784 řádků vč. DST specifik dle OTE), sloupce TDD1..TDD8."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or TDD_DIR

    def _path(self, year: int) -> Path:
        return self.base_dir / f"tdd_{year}.parquet"

    def available_years(self) -> list[int]:
        if not self.base_dir.exists():
            return []
        return sorted(int(f.stem.split("_")[1]) for f in self.base_dir.glob("tdd_*.parquet"))

    def save(self, year: int, df: pd.DataFrame) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(year)
        df.to_parquet(path)
        return path

    def load(self, year: int) -> pd.DataFrame:
        path = self._path(year)
        if not path.exists():
            raise FileNotFoundError(
                f"TDD koeficienty pro rok {year} nejsou nahrané "
                f"(stránka Nastavení → TDD).")
        return pd.read_parquet(path)


def parse_ote_tdd(file, year: int) -> pd.DataFrame:
    """Parsuje soubor normalizovaných TDD z OTE (xlsx/csv).

    Očekává první sloupec datum+čas (hodinově, lokální čas) a sloupce
    s koeficienty tříd; názvy sloupců se normalizují na TDD1..TDD8
    (hledá se číslice v názvu). Vrací DataFrame index=datetime,
    sloupce = dostupné třídy.
    """
    name = getattr(file, "name", str(file))
    if str(name).lower().endswith(".csv"):
        raw = pd.read_csv(file)
    else:
        raw = pd.read_excel(file)
    raw.columns = [str(c).strip() for c in raw.columns]
    dt_col = raw.columns[0]
    raw[dt_col] = pd.to_datetime(raw[dt_col], dayfirst=True)
    raw = raw[raw[dt_col].dt.year == year]
    if raw.empty:
        raise ValueError(f"Soubor neobsahuje žádná data pro rok {year}.")
    out = pd.DataFrame(index=raw[dt_col].values)
    import re
    for c in raw.columns[1:]:
        m = re.search(r"(\d+)", c)
        if not m:
            continue
        cls = f"TDD{int(m.group(1))}"
        if cls in TDD_CLASSES:
            out[cls] = pd.to_numeric(raw[c], errors="coerce").values
    if out.empty or out.isna().all().all():
        raise ValueError("V souboru se nepodařilo najít sloupce TDD tříd.")
    n = len(out)
    if n not in (8760, 8784, 8759, 8761, 8783, 8785):
        raise ValueError(f"Neočekávaný počet hodinových řádků: {n} "
                         f"(čekám celý rok ~8760/8784).")
    return out


def consumption_curve(tdd_class: str, annual_mwh: float, grid: TimeGrid,
                      store: TddStore | None = None) -> np.ndarray:
    """15min křivka spotřeby [MW] pro daný den z ročních TDD koeficientů."""
    store = store or TddStore()
    year_df = store.load(grid.delivery_date.year)
    if tdd_class not in year_df.columns:
        raise ValueError(f"TDD třída {tdd_class} v nahraných koeficientech chybí "
                         f"(dostupné: {list(year_df.columns)}).")
    coef = year_df[tdd_class].astype(float)
    total = coef.sum()  # Σ_rok coef · 1h → normalizace na roční energii
    if not total > 0:
        raise ValueError(f"TDD {tdd_class}: součet koeficientů není kladný.")
    idx = pd.DatetimeIndex(year_df.index)
    day_mask = (idx.year == grid.delivery_date.year) & \
               (idx.month == grid.delivery_date.month) & \
               (idx.day == grid.delivery_date.day)
    day_coef = coef[day_mask].to_numpy()
    if len(day_coef) != grid.n_hours:
        # tolerance: soubor bez DST specifik (vždy 24 h/den) — dorovnej
        if len(day_coef) == 24 and grid.n_hours != 24:
            if grid.n_hours == 23:
                day_coef = np.delete(day_coef, 2)      # vynech 02:00
            else:
                day_coef = np.insert(day_coef, 3, day_coef[2])  # zdvoj 02:00
        else:
            raise ValueError(
                f"TDD: pro den {grid.delivery_date} nalezeno {len(day_coef)} "
                f"hodin, čekám {grid.n_hours}.")
    mw_hourly = day_coef * annual_mwh / total
    return resample_hourly_to_quarter(mw_hourly, grid, kind="repeat")


def synthetic_tdd_year(year: int) -> pd.DataFrame:
    """Syntetické TDD koeficienty (testy, demo bez OTE souboru).

    Jednoduchý tvar: noční útlum, ranní a večerní špička; víkend/svátek
    plošší. Všech 8 tříd = stejný tvar s posunem špičky.
    """
    idx = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h",
                        inclusive="left")
    base_shape = np.array(
        [0.5, 0.45, 0.42, 0.4, 0.42, 0.5, 0.7, 0.9, 1.0, 0.95, 0.9, 0.9,
         0.92, 0.9, 0.88, 0.9, 1.0, 1.15, 1.25, 1.2, 1.1, 0.95, 0.75, 0.6])
    out = {}
    for i, cls in enumerate(TDD_CLASSES):
        shape = np.roll(base_shape, i % 3)
        vals = np.empty(len(idx))
        for j, ts in enumerate(idx):
            v = shape[ts.hour]
            if not is_business_day(ts):
                v = 0.8 * v + 0.15
            vals[j] = v
        out[cls] = vals
    return pd.DataFrame(out, index=idx)
