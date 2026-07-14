"""Šablony (profily) zdrojů — datový model a JSON persistence.

Profil = pojmenovaná, samostatně ukládaná konfigurace portfolia. Skládá se
z lokalit (Site); každá lokalita může nést libovolný počet assetů každého
typu (2× KGJ, 3× FVE, …) — assety jsou seznamy s vlastním asset_id.
Site.kind slouží jen jako UI šablona, model žádnou kombinaci nezakazuje.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = 1
PROFILES_DIR = Path(__file__).resolve().parent.parent / "data" / "profiles"

TDD_CLASSES = [f"TDD{i}" for i in range(1, 9)]


def slugify(name: str) -> str:
    """ASCII slug pro profile_id/site_id/asset_id (klíče sloupců ve workbooku)."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s or "item"


# ── Parametry assetů ─────────────────────────────────────────────────────────

@dataclass
class KGJParams:
    asset_id: str = "kgj1"
    name: str = "KGJ"
    k_th: float = 0.605            # jmenovitý tepelný výkon [MW_th]
    k_eff_th: float = 0.531        # tepelná účinnost [-]
    k_eff_el: float = 0.395        # elektrická účinnost [-]
    k_min: float = 0.5             # min. zatížení jako zlomek k_th [-]
    k_start_cost: float = 150.0    # [EUR/start]
    k_min_runtime_h: float = 4.0   # minimální doba běhu [h]
    k_service_cost: float = 14.0   # servisní náklad [EUR/h provozu]
    var_eff: bool = False          # linearizovaná účinnost dle zatížení
    eta_th_min: float | None = None
    eta_el_min: float | None = None
    gas_fix: bool = False
    gas_fix_price: float | None = None   # [EUR/MWh]
    ee_fix: bool = False                 # fixní výkupní cena EE (PPA/zelený bonus)
    ee_fix_price: float | None = None    # [EUR/MWh]
    afrr_capable: bool = True

    @property
    def k_el(self) -> float:
        """Odvozený jmenovitý elektrický výkon [MW_el]."""
        return self.k_th * (self.k_eff_el / self.k_eff_th)


@dataclass
class BoilerParams:
    asset_id: str = "kotel1"
    name: str = "Plynový kotel"
    b_max: float = 4.44            # [MW_th]
    boil_eff: float = 0.86
    gas_fix: bool = False
    gas_fix_price: float | None = None


@dataclass
class EKParams:
    asset_id: str = "ek1"
    name: str = "Elektrokotel"
    ek_max: float = 0.4            # [MW_th]
    ek_eff: float = 0.99
    ee_fix: bool = False
    ee_fix_price: float | None = None
    afrr_capable: bool = True


@dataclass
class TESParams:
    asset_id: str = "tes1"
    name: str = "Nádrž TES"
    tes_cap: float = 1.52          # [MWh_th]
    tes_loss_pct_h: float = 0.5    # ztráta [%/h]
    soc_start_frac: float = 0.5
    soc_end_min_frac: float = 0.0  # floor SoC na konci dne


@dataclass
class BESSParams:
    asset_id: str = "bess1"
    name: str = "Baterie BESS"
    bess_cap: float = 1.0          # [MWh]
    bess_p: float = 0.5            # [MW]
    bess_eff: float = 0.90         # účinnost nabíjení/vybíjení
    bess_cycle_cost: float = 5.0   # opotřebení [EUR/MWh toku]
    soc_start_frac: float = 0.2
    soc_end_min_frac: float = 0.2
    dist_buy_extra: bool = False   # extra distribuce na veškeré nabíjení
    dist_sell_extra: bool = False  # extra distribuce na veškeré vybíjení
    ee_fix: bool = False
    ee_fix_price: float | None = None
    afrr_capable: bool = True


@dataclass
class PVParams:
    asset_id: str = "fve1"
    name: str = "FVE"
    installed_mw: float = 1.0
    dist_sell: bool = False        # extra distribuce při prodeji z FVE
    allow_curtailment: bool = True


@dataclass
class HeatImportParams:
    asset_id: str = "imp1"
    name: str = "Import tepla"
    imp_max: float = 2.0           # [MW_th]
    imp_price: float = 150.0       # [EUR/MWh]


@dataclass
class ConsumptionSpec:
    mode: Literal["none", "tdd", "curve"] = "none"
    tdd_class: str | None = None       # 'TDD1'..'TDD8'
    annual_mwh: float | None = None    # roční spotřeba pro škálování TDD


# ── Lokalita a profil ────────────────────────────────────────────────────────

@dataclass
class Site:
    site_id: str
    name: str
    kind: Literal["heat", "fve_bess"] = "heat"
    # síť / distribuce
    dist_ee_buy: float = 12.0      # [EUR/MWh]
    dist_ee_sell: float = 0.0
    gas_dist: float = 10.0
    internal_ee_use: bool = True
    grid_export_limit_mw: float | None = None
    grid_import_limit_mw: float | None = None
    # teplo
    h_price: float = 95.0          # prodejní cena tepla [EUR/MWh]
    h_cover: float = 0.99          # minimální pokrytí poptávky
    shortfall_penalty: float = 500.0
    # assety (libovolný počet od každého typu)
    kgjs: list[KGJParams] = field(default_factory=list)
    boilers: list[BoilerParams] = field(default_factory=list)
    eks: list[EKParams] = field(default_factory=list)
    tes_units: list[TESParams] = field(default_factory=list)
    bess_units: list[BESSParams] = field(default_factory=list)
    pvs: list[PVParams] = field(default_factory=list)
    heat_imports: list[HeatImportParams] = field(default_factory=list)
    # spotřeba lokality (domácnosti / vlastní odběr)
    consumption: ConsumptionSpec = field(default_factory=ConsumptionSpec)
    # CO2 (informativní KPI; cena >0 vstupuje do objective)
    co2_price: float = 0.0
    co2_gas_factor: float = 0.202
    co2_grid_factor: float = 0.250

    @property
    def has_heat(self) -> bool:
        """Lokalita má tepelné hospodářství (vyžaduje list Teplo ve vstupech)."""
        return bool(self.kgjs or self.boilers or self.eks
                    or self.tes_units or self.heat_imports)

    def all_asset_ids(self) -> list[str]:
        ids = []
        for lst in (self.kgjs, self.boilers, self.eks, self.tes_units,
                    self.bess_units, self.pvs, self.heat_imports):
            ids += [a.asset_id for a in lst]
        return ids


@dataclass
class Profile:
    profile_id: str
    name: str
    description: str = ""
    sites: list[Site] = field(default_factory=list)
    afrr_activation_h: float = 1.0   # τ — kryté trvání aktivace pro SoC podmínky
    schema_version: int = SCHEMA_VERSION
    created_at: str = ""
    updated_at: str = ""

    # ── validace ────────────────────────────────────────────────────────────
    def validate(self) -> list[str]:
        """Vrátí seznam chyb (prázdný = OK)."""
        errs: list[str] = []
        if not self.sites:
            errs.append("Profil nemá žádnou lokalitu.")
        seen_sites: set[str] = set()
        for s in self.sites:
            if not re.fullmatch(r"[a-z0-9_]+", s.site_id or ""):
                errs.append(f"Neplatné site_id: {s.site_id!r} (jen a-z, 0-9, _).")
            if s.site_id in seen_sites:
                errs.append(f"Duplicitní site_id: {s.site_id!r}.")
            seen_sites.add(s.site_id)
            ids = s.all_asset_ids()
            if len(ids) != len(set(ids)):
                errs.append(f"Lokalita {s.site_id}: duplicitní asset_id.")
            if not ids and s.consumption.mode == "none":
                errs.append(f"Lokalita {s.site_id} je prázdná (žádný asset ani spotřeba).")
            if s.consumption.mode == "tdd":
                if s.consumption.tdd_class not in TDD_CLASSES:
                    errs.append(f"Lokalita {s.site_id}: neplatná TDD třída "
                                f"{s.consumption.tdd_class!r}.")
                if not s.consumption.annual_mwh or s.consumption.annual_mwh <= 0:
                    errs.append(f"Lokalita {s.site_id}: TDD vyžaduje roční spotřebu > 0.")
            for k in s.kgjs:
                if k.var_eff and (k.eta_th_min is None or k.eta_el_min is None):
                    errs.append(f"KGJ {s.site_id}/{k.asset_id}: var_eff vyžaduje "
                                f"eta_th_min a eta_el_min.")
                if not 0 < k.k_min <= 1:
                    errs.append(f"KGJ {s.site_id}/{k.asset_id}: k_min musí být v (0,1].")
        return errs


# ── (De)serializace ──────────────────────────────────────────────────────────

_ASSET_TYPES = {
    "kgjs": KGJParams, "boilers": BoilerParams, "eks": EKParams,
    "tes_units": TESParams, "bess_units": BESSParams, "pvs": PVParams,
    "heat_imports": HeatImportParams,
}


def profile_to_dict(p: Profile) -> dict:
    return asdict(p)


def _filter_kwargs(cls, d: dict) -> dict:
    """Ignoruj neznámé klíče (dopředná kompatibilita schémat)."""
    fields = {f for f in cls.__dataclass_fields__}
    return {k: v for k, v in d.items() if k in fields}


def profile_from_dict(d: dict) -> Profile:
    sites = []
    for sd in d.get("sites", []):
        kwargs = _filter_kwargs(Site, sd)
        for key, cls in _ASSET_TYPES.items():
            kwargs[key] = [cls(**_filter_kwargs(cls, a)) for a in sd.get(key, [])]
        kwargs["consumption"] = ConsumptionSpec(
            **_filter_kwargs(ConsumptionSpec, sd.get("consumption", {}) or {}))
        sites.append(Site(**kwargs))
    kwargs = _filter_kwargs(Profile, d)
    kwargs["sites"] = sites
    return Profile(**kwargs)


# ── Persistence ──────────────────────────────────────────────────────────────

def _path(profile_id: str, base_dir: Path | None = None) -> Path:
    return (base_dir or PROFILES_DIR) / f"{profile_id}.json"


def list_profiles(base_dir: Path | None = None) -> list[dict]:
    """Metadata všech profilů (id, name, description, updated_at, počty)."""
    base = base_dir or PROFILES_DIR
    out = []
    if not base.exists():
        return out
    for f in sorted(base.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out.append({
                "profile_id": d.get("profile_id", f.stem),
                "name": d.get("name", f.stem),
                "description": d.get("description", ""),
                "updated_at": d.get("updated_at", ""),
                "n_sites": len(d.get("sites", [])),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return out


def load_profile(profile_id: str, base_dir: Path | None = None) -> Profile:
    d = json.loads(_path(profile_id, base_dir).read_text(encoding="utf-8"))
    return profile_from_dict(d)


def save_profile(p: Profile, base_dir: Path | None = None) -> Path:
    """Atomický zápis (tmp + os.replace). Aktualizuje updated_at."""
    errs = p.validate()
    if errs:
        raise ValueError("Profil není validní:\n- " + "\n- ".join(errs))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not p.created_at:
        p.created_at = now
    p.updated_at = now
    path = _path(p.profile_id, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(profile_to_dict(p), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def duplicate_profile(profile_id: str, new_name: str,
                      base_dir: Path | None = None) -> Profile:
    p = load_profile(profile_id, base_dir)
    p.name = new_name
    base = slugify(new_name)
    new_id, i = base, 2
    while _path(new_id, base_dir).exists():
        new_id, i = f"{base}_{i}", i + 1
    p.profile_id = new_id
    p.created_at = ""
    save_profile(p, base_dir)
    return p


def delete_profile(profile_id: str, base_dir: Path | None = None) -> None:
    path = _path(profile_id, base_dir)
    if path.exists():
        path.unlink()
