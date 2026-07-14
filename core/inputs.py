"""Vstupní workbook „Vstupy D+1" — generování šablony, parsování, FX, validace.

Všechny intervalové listy mají sloupce `MTU` (1..n) a `Cas od` (HH:MM);
řady se klíčují číslem MTU, ne wall-clock časem (DST). Parser přijme
i hodinové řady (24/23/25 řádků) a rozpadne je na 15 min — FVE výrobu
lineární interpolací (hladký osvit), ostatní pozičním opakováním.
Interní měna je EUR; listy Odchylka a aFRR_ceny jsou default CZK a
převádí se kurzem FX (CZK/EUR).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profiles import Profile
from .tdd import TddStore, consumption_curve
from .timegrid import BLOCK_LABELS, N_BLOCKS, TimeGrid, resample_hourly_to_quarter

SHEET_META = "Meta"
SHEET_DA = "DA_ceny"
SHEET_IMB = "Odchylka"
SHEET_AFRR = "aFRR_ceny"
SHEET_PV = "FVE_vyroba"
SHEET_CONS = "Spotreba"
SHEET_HEAT = "Teplo"
SHEET_GAS = "Plyn"

# defaultní měny listů; přepínatelné v UI
DEFAULT_CURRENCIES = {"da": "EUR", "imb": "CZK", "afrr": "CZK", "gas": "EUR"}


@dataclass
class InputSeries:
    """Parsované vstupy pro jeden dodávkový den — vše EUR / MW, délka grid.n."""
    grid: TimeGrid
    da_price_pred: np.ndarray
    imb_price_pred: np.ndarray
    gas_price: np.ndarray
    afrr_cap_price_up: np.ndarray       # délka 6 [EUR/MW/h]
    afrr_cap_price_dn: np.ndarray       # délka 6
    pv_forecast: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    consumption: dict[str, np.ndarray] = field(default_factory=dict)
    heat_demand: dict[str, np.ndarray] = field(default_factory=dict)
    da_price_actual: np.ndarray | None = None

    def to_frame(self) -> pd.DataFrame:
        """Ploché uložení do parquet (sloupce s prefixy)."""
        d = {
            "da_price_pred": self.da_price_pred,
            "imb_price_pred": self.imb_price_pred,
            "gas_price": self.gas_price,
        }
        for (s, a), v in self.pv_forecast.items():
            d[f"pv__{s}__{a}"] = v
        for s, v in self.consumption.items():
            d[f"cons__{s}"] = v
        for s, v in self.heat_demand.items():
            d[f"heat__{s}"] = v
        if self.da_price_actual is not None:
            d["da_price_actual"] = self.da_price_actual
        df = pd.DataFrame(d)
        # aFRR ceny (délka 6) uložit jako atributy přes první řádky pomocného sloupce
        for b in range(N_BLOCKS):
            df.loc[b, "afrr_up"] = self.afrr_cap_price_up[b]
            df.loc[b, "afrr_dn"] = self.afrr_cap_price_dn[b]
        return df

    @classmethod
    def from_frame(cls, df: pd.DataFrame, grid: TimeGrid) -> "InputSeries":
        pv, cons, heat = {}, {}, {}
        for c in df.columns:
            if c.startswith("pv__"):
                _, s, a = c.split("__", 2)
                pv[(s, a)] = df[c].to_numpy()
            elif c.startswith("cons__"):
                cons[c[6:]] = df[c].to_numpy()
            elif c.startswith("heat__"):
                heat[c[6:]] = df[c].to_numpy()
        return cls(
            grid=grid,
            da_price_pred=df["da_price_pred"].to_numpy(),
            imb_price_pred=df["imb_price_pred"].to_numpy(),
            gas_price=df["gas_price"].to_numpy(),
            afrr_cap_price_up=df["afrr_up"].to_numpy()[:N_BLOCKS],
            afrr_cap_price_dn=df["afrr_dn"].to_numpy()[:N_BLOCKS],
            pv_forecast=pv, consumption=cons, heat_demand=heat,
            da_price_actual=(df["da_price_actual"].to_numpy()
                             if "da_price_actual" in df.columns else None),
        )


# ── Pomocné mapování sloupců podle profilu ──────────────────────────────────

def pv_columns(profile: Profile) -> list[tuple[str, str, str]]:
    """[(site_id, pv_id, název_sloupce)] — u jediné FVE na lokalitě stačí site_id."""
    out = []
    for s in profile.sites:
        for pv in s.pvs:
            col = s.site_id if len(s.pvs) == 1 else f"{s.site_id}__{pv.asset_id}"
            out.append((s.site_id, pv.asset_id, col))
    return out


def consumption_curve_sites(profile: Profile) -> list[str]:
    return [s.site_id for s in profile.sites if s.consumption.mode == "curve"]


def heat_sites(profile: Profile) -> list[str]:
    return [s.site_id for s in profile.sites if s.has_heat]


# ── Generátor šablony ────────────────────────────────────────────────────────

def build_template_xlsx(profile: Profile, grid: TimeGrid) -> bytes:
    """Prázdný workbook s předvyplněnými MTU/časy a sloupci dle profilu."""
    buf = io.BytesIO()
    mtu = np.arange(1, grid.n + 1)
    times = grid.times_from()

    def interval_df(cols: list[str]) -> pd.DataFrame:
        d = {"MTU": mtu, "Cas od": times}
        for c in cols:
            d[c] = np.nan
        return pd.DataFrame(d)

    with pd.ExcelWriter(buf, engine="xlsxwriter") as wr:
        meta = pd.DataFrame({
            "Klíč": ["Den dodávky", "Profil", "Počet MTU",
                     "Měna DA", "Měna odchylka", "Měna aFRR", "Měna plyn"],
            "Hodnota": [grid.delivery_date.isoformat(), profile.profile_id,
                        grid.n, DEFAULT_CURRENCIES["da"], DEFAULT_CURRENCIES["imb"],
                        DEFAULT_CURRENCIES["afrr"], DEFAULT_CURRENCIES["gas"]],
        })
        meta.to_excel(wr, sheet_name=SHEET_META, index=False)
        interval_df(["Cena DA"]).to_excel(wr, sheet_name=SHEET_DA, index=False)
        interval_df(["Zuctovaci cena"]).to_excel(wr, sheet_name=SHEET_IMB, index=False)
        pd.DataFrame({
            "Blok": BLOCK_LABELS,
            "aFRR+ [cena/MW/h]": [np.nan] * N_BLOCKS,
            "aFRR- [cena/MW/h]": [np.nan] * N_BLOCKS,
        }).to_excel(wr, sheet_name=SHEET_AFRR, index=False)
        pv_cols = [c for _, _, c in pv_columns(profile)]
        if pv_cols:
            interval_df(pv_cols).to_excel(wr, sheet_name=SHEET_PV, index=False)
        cons_cols = consumption_curve_sites(profile)
        if cons_cols:
            interval_df(cons_cols).to_excel(wr, sheet_name=SHEET_CONS, index=False)
        h_cols = heat_sites(profile)
        if h_cols:
            interval_df(h_cols).to_excel(wr, sheet_name=SHEET_HEAT, index=False)
        interval_df(["Cena plyn"]).to_excel(wr, sheet_name=SHEET_GAS, index=False)
    return buf.getvalue()


# ── Parser ───────────────────────────────────────────────────────────────────

class InputError(ValueError):
    pass


def _series_from_sheet(df: pd.DataFrame, col: str, grid: TimeGrid, sheet: str,
                       report: list[str], kind: str = "repeat") -> np.ndarray:
    """Vytáhne číselnou řadu; přijme grid.n řádků, nebo hodinový počet
    (grid.n_hours) s automatickým rozpadem na 15 min."""
    if col not in df.columns:
        raise InputError(f"List {sheet}: chybí sloupec {col!r}.")
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    vals = vals[: _last_valid(vals) + 1] if len(vals) else vals
    if np.isnan(vals).any():
        bad = int(np.isnan(vals).sum())
        raise InputError(f"List {sheet}, sloupec {col!r}: {bad} chybějících/"
                         f"nečíselných hodnot.")
    if len(vals) == grid.n:
        return vals
    if len(vals) == grid.n_hours:
        report.append(f"List {sheet}/{col}: hodinová řada ({len(vals)}) "
                      f"rozpadnuta na 15 min ({'interpolace' if kind == 'interpolate' else 'opakování'}).")
        return resample_hourly_to_quarter(vals, grid, kind=kind)
    raise InputError(f"List {sheet}, sloupec {col!r}: {len(vals)} hodnot, "
                     f"čekám {grid.n} (15min) nebo {grid.n_hours} (hodinově).")


def _last_valid(vals: np.ndarray) -> int:
    """Index poslední ne-NaN hodnoty (ořez prázdných řádků na konci listu)."""
    idx = np.flatnonzero(~np.isnan(vals))
    return int(idx[-1]) if len(idx) else -1


def _to_eur(vals: np.ndarray, currency: str, fx_czk_eur: float) -> np.ndarray:
    if currency.upper() == "CZK":
        return vals / fx_czk_eur
    return vals


def parse_inputs_xlsx(file, grid: TimeGrid, profile: Profile,
                      fx_czk_eur: float,
                      currencies: dict[str, str] | None = None,
                      tdd_store: TddStore | None = None,
                      ) -> tuple[InputSeries, list[str]]:
    """Parsuje vyplněný workbook. Vrací (InputSeries, validační report)."""
    cur = {**DEFAULT_CURRENCIES, **(currencies or {})}
    report: list[str] = []
    xls = pd.ExcelFile(file)

    def sheet(name: str, required: bool = True) -> pd.DataFrame | None:
        if name not in xls.sheet_names:
            if required:
                raise InputError(f"Ve workbooku chybí list {name!r}.")
            return None
        df = xls.parse(name)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    da = _to_eur(_series_from_sheet(sheet(SHEET_DA), "Cena DA", grid, SHEET_DA,
                                    report), cur["da"], fx_czk_eur)
    imb = _to_eur(_series_from_sheet(sheet(SHEET_IMB), "Zuctovaci cena", grid,
                                     SHEET_IMB, report), cur["imb"], fx_czk_eur)
    gas = _to_eur(_series_from_sheet(sheet(SHEET_GAS), "Cena plyn", grid,
                                     SHEET_GAS, report), cur["gas"], fx_czk_eur)

    afrr_df = sheet(SHEET_AFRR)
    up_col = next((c for c in afrr_df.columns if c.lower().startswith("afrr+")), None)
    dn_col = next((c for c in afrr_df.columns if c.lower().startswith("afrr-")), None)
    if up_col is None or dn_col is None:
        raise InputError(f"List {SHEET_AFRR}: čekám sloupce 'aFRR+ …' a 'aFRR- …'.")
    up = pd.to_numeric(afrr_df[up_col], errors="coerce").to_numpy(dtype=float)[:N_BLOCKS]
    dn = pd.to_numeric(afrr_df[dn_col], errors="coerce").to_numpy(dtype=float)[:N_BLOCKS]
    if len(up) < N_BLOCKS or np.isnan(up).any() or np.isnan(dn).any():
        raise InputError(f"List {SHEET_AFRR}: vyplňte všech {N_BLOCKS} bloků "
                         f"pro oba směry (0 = nenabízím).")
    up = _to_eur(up, cur["afrr"], fx_czk_eur)
    dn = _to_eur(dn, cur["afrr"], fx_czk_eur)

    # FVE výroba — sloupec per asset, interpolace hodinových řad
    pv: dict[tuple[str, str], np.ndarray] = {}
    pv_cols = pv_columns(profile)
    if pv_cols:
        pv_df = sheet(SHEET_PV)
        for site_id, pv_id, col in pv_cols:
            use_col = col if col in pv_df.columns else (
                site_id if site_id in pv_df.columns else col)
            arr = _series_from_sheet(pv_df, use_col, grid, SHEET_PV, report,
                                     kind="interpolate")
            if (arr < -1e-9).any():
                raise InputError(f"List {SHEET_PV}/{use_col}: záporná výroba.")
            pv[(site_id, pv_id)] = np.clip(arr, 0.0, None)

    # Spotřeba — curve lokality z workbooku, tdd lokality dopočíst
    cons: dict[str, np.ndarray] = {}
    curve_sites = consumption_curve_sites(profile)
    if curve_sites:
        cons_df = sheet(SHEET_CONS)
        for sid in curve_sites:
            cons[sid] = np.clip(
                _series_from_sheet(cons_df, sid, grid, SHEET_CONS, report),
                0.0, None)
    for s in profile.sites:
        if s.consumption.mode == "tdd":
            cons[s.site_id] = consumption_curve(
                s.consumption.tdd_class, s.consumption.annual_mwh, grid,
                store=tdd_store)
            report.append(f"Spotřeba {s.site_id}: dopočtena z "
                          f"{s.consumption.tdd_class} × {s.consumption.annual_mwh} MWh/rok.")

    # Teplo per heat lokalita
    heat: dict[str, np.ndarray] = {}
    h_sites = heat_sites(profile)
    if h_sites:
        heat_df = sheet(SHEET_HEAT)
        for sid in h_sites:
            heat[sid] = np.clip(
                _series_from_sheet(heat_df, sid, grid, SHEET_HEAT, report),
                0.0, None)

    series = InputSeries(
        grid=grid, da_price_pred=da, imb_price_pred=imb, gas_price=gas,
        afrr_cap_price_up=up, afrr_cap_price_dn=dn,
        pv_forecast=pv, consumption=cons, heat_demand=heat,
    )
    return series, report


def parse_actual_da(file, grid: TimeGrid, fx_czk_eur: float,
                    currency: str = "EUR") -> np.ndarray:
    """Skutečné ceny DA (run 3): xlsx/csv, poslední číselný sloupec je cena."""
    name = getattr(file, "name", str(file))
    if str(name).lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = [str(c).strip() for c in df.columns]
    num_cols = [c for c in df.columns
                if pd.to_numeric(df[c], errors="coerce").notna().any()
                and c.upper() != "MTU"]
    if not num_cols:
        raise InputError("V souboru se nepodařilo najít sloupec s cenou.")
    report: list[str] = []
    vals = _series_from_sheet(df, num_cols[-1], grid, "Skutečné DA", report)
    return _to_eur(vals, currency, fx_czk_eur)
