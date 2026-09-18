"""Čisté výpočetní jádro opt_CHP — bez závislosti na Streamlitu.

Sem patří vše, co jde spustit a otestovat bez UI: kalendář svátků, definice
provozních profilů KGJ, linearizace účinnostní křivky a samotný MILP solver.
`app.py` tyhle funkce importuje; testy v `tests/` je importují přímo.
"""

import calendar
import datetime as _dt

import numpy as np
import pandas as pd
import pulp

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


# ════════════════════════════════════════════════════════════════════
# SCHEDULING PROFILES & SCENARIO MANAGEMENT
# ════════════════════════════════════════════════════════════════════

# SPECIAL profil — týdenní okna v rámci 168h týdne (idx = weekday*24 + hour)
# Měsíce 1,2,3,4,5,9,10,11,12: blok Po 06:00 → Pá 22:00 + So 06:00 → Ne 22:00 (152h/týden)
# Měsíce 6,7,8 (léto):         blok Po 06:00 → Čt 22:00                       (88h/týden)
SPECIAL_ON_OTHER  = set(range(6, 118)) | set(range(126, 166))
SPECIAL_ON_SUMMER = set(range(6, 94))
SPECIAL_SUMMER_MONTHS = {6, 7, 8}

# EXTPSUM profil — EXTPEAK s letní úpravou.
# Od 1. 6. do 30. 9. včetně nejde brát poledne (11:00–17:00), zato jdou
# navíc brát okraje dne: 04:00–06:00 a 22:00–24:00.
EXTPSUM_MONTHS = {6, 7, 8, 9}
EXTPSUM_BLOCKED = set(range(11, 17))          # 11:00-17:00
EXTPSUM_EXTRA = set(range(4, 6)) | set(range(22, 24))   # 04:00-06:00, 22:00-24:00


def create_profile_constraints(df, profile_type, custom_hours=None):
    """
    Vytvoří constrainty pro KGJ provoz dle profilu.
    Returns: list[int]  -1 = must OFF, 0 = free, 1 = must ON (baseload)

    Konvence (PXE/OTE + provozní úprava pro KGJ):
      PEAK    – po–pá (mimo CZ státní svátky), 8:00–20:00  (hodiny 8..19, 12 h)
      EXTPEAK – po–pá (mimo CZ státní svátky), 6:00–22:00  (hodiny 6..21, 16 h)
      EXTPSUM – jako EXTPEAK, ale VI–IX bez 11:00–17:00 a navíc s 04:00–06:00
                a 22:00–24:00 (14 h místo 16 h v letních měsících)
      OFFPEAK – doplněk peaku v rámci 24/7: víkendy a svátky celý den
                + po–pá hodiny 0..7 a 20..23
      SPECIAL – měsíční vzor s denním rytmem (CZ svátky se neuplatňují):
                · 1-5, 9-12: Po 06 → Pá 22 + So 06 → Ne 22 (152 h/týden)
                · 6-8 (léto): Po 06 → Čt 22                 (88 h/týden)
    """
    df_work = df.copy()
    dt = pd.to_datetime(df_work['datetime'])
    hours = dt.dt.hour.values
    bdays = dt.apply(is_business_day).values

    if profile_type == 'base':
        # BASE = baseload: KGJ jede 24/7, všechny sloty vynuceny ON
        constraints = [1] * len(df_work)

    elif profile_type == 'peak':
        constraints = [0 if (bd and 8 <= h < 20) else -1
                       for h, bd in zip(hours, bdays)]

    elif profile_type == 'extpeak':
        constraints = [0 if (bd and 6 <= h < 22) else -1
                       for h, bd in zip(hours, bdays)]

    elif profile_type == 'extpsum':
        months = dt.dt.month.values
        constraints = []
        for h, bd, m in zip(hours, bdays, months):
            if not bd:
                constraints.append(-1)
                continue
            if m in EXTPSUM_MONTHS:
                allowed = ((6 <= h < 22) and h not in EXTPSUM_BLOCKED) or h in EXTPSUM_EXTRA
            else:
                allowed = 6 <= h < 22
            constraints.append(0 if allowed else -1)

    elif profile_type == 'offpeak':
        constraints = [0 if ((not bd) or h < 8 or h >= 20) else -1
                       for h, bd in zip(hours, bdays)]

    elif profile_type == 'special':
        months = dt.dt.month.values
        weekdays = dt.dt.weekday.values
        week_idx = weekdays * 24 + hours
        constraints = [
            0 if wi in (SPECIAL_ON_SUMMER if m in SPECIAL_SUMMER_MONTHS else SPECIAL_ON_OTHER)
            else -1
            for m, wi in zip(months, week_idx)
        ]

    elif profile_type == 'custom' and custom_hours:
        constraints = [0 if h in custom_hours else -1 for h in hours]

    else:
        # 'free' nebo neurčeno = optimizer zcela volný
        constraints = [0] * len(df_work)

    return constraints


def apply_profile_constraints_to_model(model, on, constraints, T):
    """Aplikuj profile constrainty do PuLP modelu"""
    if constraints is None:
        return model
    for t in range(T):
        if constraints[t] == -1:
            model += on[t] == 0, f"profile_off_{t}"
        elif constraints[t] == 1:
            model += on[t] == 1, f"profile_on_{t}"
    return model


def calculate_smoothness_metrics(res):
    """
    Spočítej metriky hladkosti provozu KGJ
    """
    kgj_on = res['KGJ on'].values
    
    # Počet ON→OFF a OFF→ON přechodů
    transitions = np.sum(np.abs(np.diff(kgj_on)) > 0.5)
    
    # Délky kontinuálních běhů
    run_lengths = []
    current_run = 0
    for i in range(len(kgj_on)):
        if kgj_on[i] > 0.5:
            current_run += 1
        else:
            if current_run > 0:
                run_lengths.append(current_run)
            current_run = 0
    if current_run > 0:
        run_lengths.append(current_run)
    
    avg_run_length = np.mean(run_lengths) if run_lengths else 0
    min_run_length = np.min(run_lengths) if run_lengths else 0
    max_run_length = np.max(run_lengths) if run_lengths else 0
    
    # Stabilita skóre (0-100%, méně transakcí = vyšší skóre)
    stability_score = max(0, 100 * (1 - transitions / (len(kgj_on) / 2))) if len(kgj_on) > 0 else 0
    
    total_on = int(np.sum(kgj_on))
    utilization_pct = total_on / len(kgj_on) * 100 if len(kgj_on) > 0 else 0

    return {
        'transitions': int(transitions),
        'stability_score': stability_score,
        'avg_run_hours': avg_run_length,
        'min_run_hours': min_run_length,
        'max_run_hours': max_run_length,
        'total_on_hours': total_on,
        'utilization_pct': utilization_pct,
    }


# ────────────────────────────────────────────────
# POMOCNÉ FUNKCE PRO SOLVER
# ────────────────────────────────────────────────

def compute_linear_fuel_params(k_th, k_min, eta_th_rated, eta_th_min, eta_el_rated, eta_el_min):
    """
    Linearizace fuel/output funkce přes 2 provozní body: min zátěž a jmenovitý výkon.

    Vrátí (c0_th, c1_th, c0_el, c1_el) kde:
      gas_consumed[t] = c0_th * on[t] + c1_th * q_kgj[t]   [MWh_gas/h]
      ee_kgj[t]       = c0_el * on[t] + c1_el * q_kgj[t]   [MWh_el/h]

    Optimalizace zůstává lineární (LP) – žádné nové binary proměnné.
    """
    q_min = k_min * k_th
    q_max = k_th
    # Při k_min = 100 % (nebo k_th = 0) splynou oba body — pak není co linearizovat
    # a vracíme konstantní účinnost jmenovitého bodu.
    if abs(q_max - q_min) < 1e-9:
        if q_max <= 0:
            return 0.0, 0.0, 0.0, 0.0
        return 0.0, 1.0 / eta_th_rated, 0.0, eta_el_rated / eta_th_rated
    fuel_min = q_min / eta_th_min
    fuel_max = q_max / eta_th_rated
    c1_th = (fuel_max - fuel_min) / (q_max - q_min)
    c0_th = fuel_min - c1_th * q_min   # offset platí jen když on=1

    ee_min = q_min * (eta_el_min / eta_th_min)
    ee_max = q_max * (eta_el_rated / eta_th_rated)
    c1_el = (ee_max - ee_min) / (q_max - q_min)
    c0_el = ee_min - c1_el * q_min

    return c0_th, c1_th, c0_el, c1_el


def get_kgj_fix_price(p, profile_type):
    """Vrátí (is_active, price) — fixní výkupní cena KGJ pro daný profil.

    Pro profil 'custom' fix neexistuje (vždy spot cena z křivky).
    Pro FREE/BASE/PEAK/EXTPEAK/OFFPEAK/SPECIAL se vrátí (True, price) pokud je checkbox zapnutý.
    """
    if profile_type == 'custom':
        return False, None
    flag_key = f'kgj_ee_fix_{profile_type}'
    price_key = f'kgj_ee_fix_price_{profile_type}'
    if p.get(flag_key):
        return True, p.get(price_key)
    return False, None


# ────────────────────────────────────────────────
# ENHANCED SOLVER S PROFILE SUPPORT
# ────────────────────────────────────────────────

def run_optimization_with_profile(df, params, uses, profile_type='free', custom_hours=None,
                                   ee_delta=0.0, gas_delta=0.0, h_price_override=None, 
                                   time_limit=1200, gap_rel=0.01,
                                   max_starts_per_month=None, period_mask=None):
    """
    Enhanced solver s podporou KGJ scheduling profilů
    
    Parameters:
    - profile_type: 'free', 'base', 'peak', 'extpeak', 'custom'
    - custom_hours: list hodin (0-23) pokud profile_type=='custom'
    - max_starts_per_month: omezení startů na měsíc
    - period_mask: boolean array pro filtrování časového období
    """
    
    p        = params
    u        = uses
    T        = len(df)
    h_price  = h_price_override if h_price_override is not None else p['h_price']
    boil_eff = p.get('boil_eff', 0.95)
    ek_eff   = p.get('ek_eff',   0.98)

    # Koeficienty pro linearizovanou účinnostní křivku KGJ
    if p.get('kgj_var_eff') and u.get('kgj'):
        c0_th, c1_th, c0_el, c1_el = compute_linear_fuel_params(
            p['k_th'], p['k_min'],
            p['k_eff_th'], p.get('eta_th_min', p['k_eff_th']),
            p['k_eff_el'], p.get('eta_el_min', p['k_eff_el'])
        )
    else:
        c0_th, c1_th = 0.0, 1.0 / p.get('k_eff_th', 0.46)
        c0_el, c1_el = 0.0, p.get('k_eff_el', 0.40) / p.get('k_eff_th', 0.46)

    # Filtruj podle období
    if period_mask is not None:
        df = df[period_mask].reset_index(drop=True)
        T = len(df)
    
    # Vytvoř profile constrainty
    profile_constraints = create_profile_constraints(df, profile_type, custom_hours)

    # ── Rampy nájezdu / sjezdu KGJ ───────────────
    # Lineární rampa délky τ minut ⇒ hodina startu dodá průměrně P·(1 − τ/120),
    # hodina po vypnutí ještě P·(τ/120). τ je omezené na 0–60 min, aby se rampa
    # nikdy nerozlila do dalšího slotu (nad 60 min by vzorec neplatil).
    #
    # τ se chápe jako "ekvivalentní minuty rampy", ne jako doslovná strmost:
    # v hodinovém průměru nejde odlišit mrtvou dobu (u elektřiny synchronizace
    # generátoru) od pomalejší rampy — mrtvá doba d plus rampa r dá stejný
    # průměr jako lineární rampa délky 2d + r.
    #
    # Teplo se chová jinak než elektřina: vzniká hned při zážehu, ale nejdřív
    # ohřívá blok a výměník, a po odstavení ho dochlazení tlačí do sítě ještě
    # dlouho po tom, co se motor zastavil. Proto vlastní dvojice τ.
    # Plyn sleduje elektřinu — palivo jde do motoru a motor točí generátorem.
    def _alpha(key, fallback=0.0):
        raw = p.get(key)
        val = fallback if raw is None else float(raw)
        return min(max(val, 0.0), 60.0) / 120.0

    if p.get('kgj_ramp_on') and u.get('kgj'):
        a_up_el = _alpha('k_ramp_up_el_min')
        a_dn_el = _alpha('k_ramp_down_el_min')
        if p.get('kgj_ramp_th_split'):
            a_up_th = _alpha('k_ramp_up_th_min')
            a_dn_th = _alpha('k_ramp_down_th_min')
        else:
            # Bez rozlišení sleduje teplo elektřinu — chování jako před rozdělením.
            a_up_th, a_dn_th = a_up_el, a_dn_el
    else:
        a_up_el = a_dn_el = a_up_th = a_dn_th = 0.0
    ramp_on = bool(u.get('kgj')) and any(
        a > 0.0 for a in (a_up_el, a_dn_el, a_up_th, a_dn_th))

    # Strukturální prořezání: v hodině, kde profil start/stop vůbec nepřipouští,
    # se pomocné proměnné nezakládají. BASE (všude on==1) se tím smrskne na jeden
    # start v t=0, PEAK zhruba na čtvrtinu.
    if ramp_on:
        _c = profile_constraints
        can_start = [(_c[t] != -1) and (t == 0 or _c[t - 1] != 1) for t in range(T)]
        can_stop  = [(t >= 1) and (_c[t - 1] != -1) and (_c[t] != 1) for t in range(T)]
    else:
        can_start = can_stop = [False] * T

    model = pulp.LpProblem("KGJ_Dispatch_Profile", pulp.LpMaximize)

    # ── Proměnné ─────────────────────────────────
    if u['kgj']:
        q_kgj = pulp.LpVariable.dicts("q_KGJ",  range(T), 0, p['k_th'])
        on    = pulp.LpVariable.dicts("on",      range(T), 0, 1, "Binary")
        start = pulp.LpVariable.dicts("start",   range(T), 0, 1, "Binary")
    else:
        q_kgj = on = start = {t: 0 for t in range(T)}

    # Pomocné proměnné rampy. `stop` nemusí být binární — stávající constrainty na
    # `start` (start >= on[t]-on[t-1], <= on[t], <= 1-on[t-1]) ho pinují jednoznačně
    # přes identitu stop[t] = on[t-1] - on[t] + start[t], takže nepřibude ani jedna
    # binárka. `ru`/`rd` jsou McCormickem linearizované součiny binárka × spojitá:
    #   ru[t] = q_kgj[t]   * start[t]     (hladina, ze které se najíždí)
    #   rd[t] = q_kgj[t-1] * stop[t]      (hladina, ze které se sjíždí)
    if ramp_on:
        stop = {t: (pulp.LpVariable(f"stop_{t}", 0, 1) if can_stop[t] else 0)
                for t in range(T)}
        ru   = {t: (pulp.LpVariable(f"ramp_up_{t}", 0, p['k_th']) if can_start[t] else 0)
                for t in range(T)}
        rd   = {t: (pulp.LpVariable(f"ramp_dn_{t}", 0, p['k_th']) if can_stop[t] else 0)
                for t in range(T)}
    else:
        stop = ru = rd = {t: 0 for t in range(T)}

    if u['boil']:
        q_boil  = pulp.LpVariable.dicts("q_Boil",   range(T), 0, p['b_max'])
        on_boil = pulp.LpVariable.dicts("on_boil",  range(T), 0, 1, "Binary")
    else:
        q_boil  = {t: 0 for t in range(T)}
        on_boil = {t: 0 for t in range(T)}

    q_ek  = pulp.LpVariable.dicts("q_EK",   range(T), 0, p['ek_max'])  if u['ek']  else {t: 0 for t in range(T)}

    if u['ext_heat']:
        q_imp  = pulp.LpVariable.dicts("q_Imp",   range(T), 0, p['imp_max'])
        on_imp = pulp.LpVariable.dicts("on_imp",  range(T), 0, 1, "Binary")
    else:
        q_imp  = {t: 0 for t in range(T)}
        on_imp = {t: 0 for t in range(T)}

    if u['tes']:
        tes_soc = pulp.LpVariable.dicts("TES_SOC", range(T+1), 0, p['tes_cap'])
        tes_in  = pulp.LpVariable.dicts("TES_In",  range(T), 0)
        tes_out = pulp.LpVariable.dicts("TES_Out", range(T), 0)
        model  += tes_soc[0] == p['tes_cap'] * 0.5
    else:
        tes_soc = {t: 0 for t in range(T+1)}
        tes_in = tes_out = {t: 0 for t in range(T)}

    if u['bess']:
        bess_soc = pulp.LpVariable.dicts("BESS_SOC", range(T+1), 0, p['bess_cap'])
        bess_cha = pulp.LpVariable.dicts("BESS_Cha", range(T), 0, p['bess_p'])
        bess_dis = pulp.LpVariable.dicts("BESS_Dis", range(T), 0, p['bess_p'])
        model   += bess_soc[0] == p['bess_cap'] * 0.2
    else:
        bess_soc = {t: 0 for t in range(T+1)}
        bess_cha = bess_dis = {t: 0 for t in range(T)}

    ee_export      = pulp.LpVariable.dicts("ee_export",  range(T), 0)
    ee_import      = pulp.LpVariable.dicts("ee_import",  range(T), 0)
    heat_shortfall = pulp.LpVariable.dicts("shortfall",  range(T), 0)
    heat_dump      = pulp.LpVariable.dicts("heat_dump",  range(T), 0)  # přebytečné teplo zahozeno

    # Rozdělení toku EE do EK a do BESS nabíjení podle zdroje (lokální výroba vs. grid).
    # Lokální větve nepodléhají distribuci; grid větve ano. Když je internal_ee_use vypnut,
    # jsou lokální větve donuceny na 0 (vše musí fyzicky projít DS).
    if u['ek']:
        ee_ek_local = pulp.LpVariable.dicts("ee_ek_local", range(T), 0)
        ee_ek_grid  = pulp.LpVariable.dicts("ee_ek_grid",  range(T), 0)
    else:
        ee_ek_local = {t: 0 for t in range(T)}
        ee_ek_grid  = {t: 0 for t in range(T)}

    if u['bess']:
        ee_bess_local = pulp.LpVariable.dicts("ee_bess_local", range(T), 0)
        ee_bess_grid  = pulp.LpVariable.dicts("ee_bess_grid",  range(T), 0)
    else:
        ee_bess_local = {t: 0 for t in range(T)}
        ee_bess_grid  = {t: 0 for t in range(T)}

    # ── KGJ provozní omezení ─────────────────────
    if u['kgj']:
        for t in range(T):
            model += q_kgj[t] <= p['k_th'] * on[t]
            model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]
            
            # PROFILE CONSTRAINT: -1=must off, 0=free, 1=must on (baseload)
            if profile_constraints[t] == -1:
                model += on[t] == 0, f"profile_off_{t}"
            elif profile_constraints[t] == 1:
                model += on[t] == 1, f"profile_on_{t}"
        
        model += start[0] == on[0]
        for t in range(1, T):
            model += start[t] >= on[t] - on[t-1]
            model += start[t] <= on[t]
            model += start[t] <= 1 - on[t-1]
        
        # ── Rampy: definice stop + linearizace součinů ──
        if ramp_on:
            M = p['k_th']
            q_lo = p['k_min'] * p['k_th']
            for t in range(T):
                # stop[t] je exaktně určený; v t=0 je jednotka před horizontem vypnutá
                if can_stop[t]:
                    model += stop[t] == on[t - 1] - on[t] + start[t], f"stop_def_{t}"
                # ru[t] = q_kgj[t] * start[t]
                if can_start[t]:
                    model += ru[t] <= M * start[t],                f"ru_ub_bin_{t}"
                    model += ru[t] <= q_kgj[t],                    f"ru_ub_q_{t}"
                    model += ru[t] >= q_kgj[t] - M * (1 - start[t]), f"ru_lb_{t}"
                    model += ru[t] >= q_lo * start[t],             f"ru_cut_{t}"
                # rd[t] = q_kgj[t-1] * stop[t]
                if can_stop[t]:
                    model += rd[t] <= M * stop[t],                 f"rd_ub_bin_{t}"
                    model += rd[t] <= q_kgj[t - 1],                f"rd_ub_q_{t}"
                    model += rd[t] >= q_kgj[t - 1] - M * (1 - stop[t]), f"rd_lb_{t}"
                    model += rd[t] >= q_lo * stop[t],              f"rd_cut_{t}"

        min_rt = int(p['k_min_runtime'])
        for t in range(T):
            for dt in range(1, min_rt):
                if t + dt < T:
                    model += on[t+dt] >= start[t]
        
        # Roční limit hodin — pro BASE profil se ignoruje (KGJ jede vždy)
        if (p.get('kgj_hour_limit_on') and p.get('kgj_hour_limit')
                and profile_type != 'base'):
            model += pulp.lpSum(on[t] for t in range(T)) <= p['kgj_hour_limit']
        
        # NOVÉ: Limit startů za měsíc
        if max_starts_per_month is not None and u['kgj']:
            df_month = df.copy()
            df_month['month'] = pd.to_datetime(df_month['datetime']).dt.to_period('M')
            for month in df_month['month'].unique():
                month_indices = df_month[df_month['month'] == month].index.tolist()
                if len(month_indices) > 0:
                    model += pulp.lpSum(start[t] for t in month_indices) <= max_starts_per_month, f"starts_limit_{month}"

    # ── Kotel – on/off + roční limit ─────────────
    if u['boil']:
        for t in range(T):
            model += q_boil[t] <= p['b_max'] * on_boil[t]
        if p.get('boil_hour_limit_on') and p.get('boil_hour_limit'):
            model += pulp.lpSum(on_boil[t] for t in range(T)) <= p['boil_hour_limit']

    # ── Import tepla – on/off + roční limit ──────
    if u['ext_heat']:
        for t in range(T):
            model += q_imp[t] <= p['imp_max'] * on_imp[t]
        if p.get('imp_hour_limit_on') and p.get('imp_hour_limit'):
            model += pulp.lpSum(on_imp[t] for t in range(T)) <= p['imp_hour_limit']

    # Fixní výkupní cena EE z KGJ — vázaná na běžící profil
    kgj_fix_active, kgj_fix_price = get_kgj_fix_price(p, profile_type)

    def kgj_fix_here(t):
        """Platí pro hodinu t fixní výkupní cena profilu?

        Doběh po vypnutí spadne do hodiny, kterou profil vynucuje na on==0
        (PEAK končí ve 20:00, doběh běží 20:00–20:09). Taková MWh je mimo
        obchodní pásmo produktu, takže se vykupuje za spot, ne za PPA profilu.
        """
        return bool(u['kgj']) and kgj_fix_active and profile_constraints[t] != -1

    # ── Tvarování výstupu rampou ─────────────────
    # Jediné místo, kde se derating počítá — používá ho teplo, elektřina i plyn.
    def kgj_th(t):
        """Skutečně dodané teplo KGJ v hodině t [MW_th]."""
        if not u['kgj']:
            return 0
        if not ramp_on:
            return q_kgj[t]
        return q_kgj[t] - a_up_th * ru[t] + a_dn_th * rd[t]

    def kgj_lin(t, c0, c1, a_up, a_dn):
        """Afinní veličina KGJ (elektřina / plyn) po deratingu rampou.

        Využívá identit on[t]*start[t] == start[t] a on[t-1]*stop[t] == stop[t],
        takže i konstantní člen c0 zůstane lineární.

        `a_up` / `a_dn` se předávají, protože elektřina a teplo mají vlastní
        rampu; `ru` / `rd` na nich nezávisí, takže se sdílejí.
        """
        if not u['kgj']:
            return 0
        base = c0 * on[t] + c1 * q_kgj[t]
        if not ramp_on:
            return base
        return (base
                - a_up * (c0 * start[t] + c1 * ru[t])
                + a_dn * (c0 * stop[t]  + c1 * rd[t]))

    # ── Hlavní smyčka ─────────────────────────────
    obj = []
    for t in range(T):
        p_ee_m  = df['ee_price'].iloc[t]  + ee_delta
        p_gas_m = df['gas_price'].iloc[t] + gas_delta

        p_gas_kgj  = p.get('kgj_gas_fix_price',  p_gas_m) if (u['kgj']  and p.get('kgj_gas_fix'))  else p_gas_m
        p_gas_boil = p.get('boil_gas_fix_price', p_gas_m) if (u['boil'] and p.get('boil_gas_fix')) else p_gas_m
        p_ee_ek    = p.get('ek_ee_fix_price',    p_ee_m)  if (u['ek']   and p.get('ek_ee_fix'))   else p_ee_m

        h_dem = df['Poptávka po teple (MW)'].iloc[t]
        fve_p = float(df['FVE (MW)'].iloc[t]) if (u['fve'] and 'FVE (MW)' in df.columns) else 0.0

        if u['tes']:
            model += tes_soc[t+1] == tes_soc[t] * (1 - p['tes_loss']) + tes_in[t] - tes_out[t]
        if u['bess']:
            model += bess_soc[t+1] == bess_soc[t] + bess_cha[t]*p['bess_eff'] - bess_dis[t]/p['bess_eff']

        heat_delivered = kgj_th(t) + q_boil[t] + q_ek[t] + q_imp[t] + tes_out[t] - tes_in[t]
        model += heat_delivered + heat_shortfall[t] >= h_dem * p['h_cover']
        model += heat_delivered <= h_dem + heat_dump[t] + 1e-3

        ee_kgj_out = kgj_lin(t, c0_el, c1_el, a_up_el, a_dn_el)
        ee_ek_in   = q_ek[t] / ek_eff                            if u['ek']  else 0
        # Hlavní EE bilance (ponechána jako sanity check, je odvoditelná z níže uvedených split rovnic)
        model += ee_kgj_out + fve_p + ee_import[t] + bess_dis[t] == ee_ek_in + bess_cha[t] + ee_export[t]

        # Decomposition: rozdělení toku EE do EK / BESS na lokální a grid složku
        if u['ek']:
            model += ee_ek_local[t] + ee_ek_grid[t] == ee_ek_in
        if u['bess']:
            model += ee_bess_local[t] + ee_bess_grid[t] == bess_cha[t]
        # Grid import pokrývá výhradně grid-stranu lokálních spotřebičů
        model += ee_import[t] == ee_ek_grid[t] + ee_bess_grid[t]

        # Když je checkbox vypnut, žádné interní routování — vše musí přes grid
        if not p['internal_ee_use']:
            if u['ek']:
                model += ee_ek_local[t] == 0
            if u['bess']:
                model += ee_bess_local[t] == 0

        # Kontraktní cena EE pro BESS (paralela k p_ee_ek pro EK, aktivuje dosud nepoužitý parametr)
        p_ee_bess = p.get('bess_ee_fix_price', p_ee_m) if (u['bess'] and p.get('bess_ee_fix')) else p_ee_m

        # Distribuční sazby — vždy aplikovány na grid toky
        fve_dist_sell_cost  = p['dist_ee_sell'] if (u['fve'] and p.get('fve_dist_sell')) else 0.0
        bess_dist_buy_cost  = p['dist_ee_buy']  * bess_cha[t] if (u['bess'] and p.get('bess_dist_buy'))  else 0
        bess_dist_sell_cost = p['dist_ee_sell'] * bess_dis[t] if (u['bess'] and p.get('bess_dist_sell')) else 0

        # Green bonus: KGJ fixní výkupní cena per profil → bonus = (fix - spot) × ee_kgj_out
        # Když fix vypnutý / FREE / CUSTOM: bonus = 0 (žádný dopad).
        kgj_ee_bonus = (kgj_fix_price - p_ee_m) * ee_kgj_out if kgj_fix_here(t) else 0

        revenue = (h_price * (heat_delivered - heat_dump[t])
                   + (p_ee_m - p['dist_ee_sell'] - fve_dist_sell_cost) * ee_export[t]
                   + kgj_ee_bonus)
        co2_price = p.get('co2_price', 0.0)
        co2_cost = 0
        if co2_price > 0:
            co2_gas_factor  = p.get('co2_gas_factor',  0.202)
            co2_grid_factor = p.get('co2_grid_factor', 0.250)
            gas_kgj_mwh  = kgj_lin(t, c0_th, c1_th, a_up_el, a_dn_el)
            gas_boil_mwh = (q_boil[t] / boil_eff)               if u['boil'] else 0
            co2_cost = co2_price * (
                co2_gas_factor  * (gas_kgj_mwh + gas_boil_mwh) +
                co2_grid_factor * ee_import[t] -
                co2_grid_factor * ee_export[t]
            )
        # Náklady na EE pro EK / BESS — pouze grid složka platí tržní/kontraktní cenu + distribuci.
        # Lokální složka má jen opportunity cost (ušlý ee_export).
        ee_cost_ek_grid   = (p_ee_ek   + p['dist_ee_buy']) * ee_ek_grid[t]   if u['ek']   else 0
        ee_cost_bess_grid = (p_ee_bess + p['dist_ee_buy']) * ee_bess_grid[t] if u['bess'] else 0

        costs = (
            ((p_gas_kgj  + p['gas_dist']) * kgj_lin(t, c0_th, c1_th, a_up_el, a_dn_el)
             if u['kgj'] else 0) +
            ((p_gas_boil + p['gas_dist']) * (q_boil[t] / boil_eff)       if u['boil']     else 0) +
            ee_cost_ek_grid +
            ee_cost_bess_grid +
            (p['imp_price'] * q_imp[t]                                    if u['ext_heat'] else 0) +
            (p['k_start_cost'] * start[t]                                 if u['kgj']      else 0) +
            (p.get('k_service_cost', 0.0) * on[t]                            if u['kgj'] else 0) +
            (p['bess_cycle_cost'] * (bess_cha[t] + bess_dis[t])           if u['bess']     else 0) +
            bess_dist_buy_cost + bess_dist_sell_cost +
            p['shortfall_penalty'] * heat_shortfall[t] +
            co2_cost
        )
        obj.append(revenue - costs)

    model += pulp.lpSum(obj)
    # gap_rel rika CBC, jak blizko optimu staci dojit. Bez nej dokazuje
    # optimalitu, coz je u rocni ulohy s akumulaci exponencialne drahe -
    # najit dobre reseni je rychle, dokazat ze lepsi neexistuje uz ne.
    # 1 % je hluboko pod nejistotou FWD krivky, ze ktere se pocita.
    status = model.solve(pulp.PULP_CBC_CMD(
        msg=0, timeLimit=time_limit,
        gapRel=gap_rel if gap_rel else None))
    if status not in (1, 2):
        return None

    def vv(v, t):
        x = v[t]
        return float(x) if isinstance(x, (int, float)) else float(pulp.value(x) or 0)

    # Numerické protějšky kgj_th / kgj_lin nad vyřešeným modelem — stejný vzorec,
    # takže reportované hodnoty nemohou odbočit od objective.
    def kgj_parts_v(t, c0, c1, a_up, a_dn):
        """(setpoint, ztráta nájezdem, doběh) pro afinní veličinu KGJ.

        Jeden zdroj pro teplo i elektřinu, aby se obě sady sloupců nemohly
        rozejít. Platí setpoint − ztráta + doběh = skutečná hodnota.
        """
        if not u['kgj']:
            return 0.0, 0.0, 0.0
        setpoint = c0 * vv(on, t) + c1 * vv(q_kgj, t)
        if not ramp_on:
            return setpoint, 0.0, 0.0
        loss = a_up * (c0 * vv(start, t) + c1 * vv(ru, t))
        tail = a_dn * (c0 * vv(stop,  t) + c1 * vv(rd, t))
        return setpoint, loss, tail

    def kgj_lin_v(t, c0, c1, a_up, a_dn):
        setpoint, loss, tail = kgj_parts_v(t, c0, c1, a_up, a_dn)
        return setpoint - loss + tail

    # Teplo je afinní veličina s (c0, c1) = (0, 1) — stejný helper.
    th_parts = [kgj_parts_v(t, 0.0, 1.0, a_up_th, a_dn_th) for t in range(T)]
    el_parts = [kgj_parts_v(t, c0_el, c1_el, a_up_el, a_dn_el) for t in range(T)]

    res = pd.DataFrame({
        'Čas':                  df['datetime'],
        'Poptávka tepla [MW]':  df['Poptávka po teple (MW)'],
        'KGJ [MW_th]':          [sp - lo + ta for sp, lo, ta in th_parts],
        'KGJ setpoint [MW_th]': [sp for sp, _, _ in th_parts],
        'KGJ nájezd ztráta [MW_th]': [lo for _, lo, _ in th_parts],
        'KGJ doběh [MW_th]':    [ta for _, _, ta in th_parts],
        # Plyn sleduje motor, tedy elektrickou rampu.
        'Plyn KGJ [MWh]':       [kgj_lin_v(t, c0_th, c1_th, a_up_el, a_dn_el)
                                 for t in range(T)],
        'Kotel [MW_th]':        [vv(q_boil, t) for t in range(T)],
        'Elektrokotel [MW_th]': [vv(q_ek,   t) for t in range(T)],
        'Import tepla [MW_th]': [vv(q_imp,  t) for t in range(T)],
        'TES příjem [MW_th]':   [vv(tes_in,  t) for t in range(T)],
        'TES výdej [MW_th]':    [vv(tes_out, t) for t in range(T)],
        'TES SOC [MWh]':        [vv(tes_soc, t+1) for t in range(T)],
        'BESS nabíjení [MW]':   [vv(bess_cha, t) for t in range(T)],
        'BESS vybíjení [MW]':   [vv(bess_dis, t) for t in range(T)],
        'BESS SOC [MWh]':       [vv(bess_soc, t+1) for t in range(T)],
        'Shortfall [MW]':       [vv(heat_shortfall, t) for t in range(T)],
        'Zahozené teplo [MW]':  [vv(heat_dump, t) for t in range(T)],
        'EE export [MW]':       [vv(ee_export, t) for t in range(T)],
        'EE import [MW]':       [vv(ee_import, t) for t in range(T)],
        'EE z KGJ [MW]':        [sp - lo + ta for sp, lo, ta in el_parts],
        'EE z KGJ setpoint [MW]': [sp for sp, _, _ in el_parts],
        'EE z KGJ nájezd ztráta [MW]': [lo for _, lo, _ in el_parts],
        'EE z KGJ doběh [MW]':  [ta for _, _, ta in el_parts],
        'EE z FVE [MW]':        [float(df['FVE (MW)'].iloc[t]) if (u['fve'] and 'FVE (MW)' in df.columns) else 0.0 for t in range(T)],
        'EE do EK [MW]':        [vv(q_ek, t)/ek_eff if u['ek'] else 0.0 for t in range(T)],
        'EE do EK lokál [MW]':  [vv(ee_ek_local, t) for t in range(T)],
        'EE do EK grid [MW]':   [vv(ee_ek_grid,  t) for t in range(T)],
        'EE do BESS lokál [MW]':[vv(ee_bess_local, t) for t in range(T)],
        'EE do BESS grid [MW]': [vv(ee_bess_grid,  t) for t in range(T)],
        'Cena EE [€/MWh]':     (df['ee_price'] + ee_delta).values,
        'Cena výkupu EE z KGJ [€/MWh]': [
            kgj_fix_price if kgj_fix_here(t)
                          else float(df['ee_price'].iloc[t]) + ee_delta
            for t in range(T)
        ],
        'Cena plyn [€/MWh]':   (df['gas_price'] + gas_delta).values,
        'KGJ on':               [vv(on, t) for t in range(T)],
        'KGJ stop':             [vv(stop, t) for t in range(T)],
        'Kotel on':             [vv(on_boil, t) for t in range(T)],
        'Import tepla on':      [vv(on_imp, t) for t in range(T)],
    })
    
    res['TES netto [MW_th]'] = res['TES výdej [MW_th]'] - res['TES příjem [MW_th]']
    res['Dodáno tepla [MW]'] = (res['KGJ [MW_th]'] + res['Kotel [MW_th]'] +
                                res['Elektrokotel [MW_th]'] + res['Import tepla [MW_th]'] +
                                res['TES netto [MW_th]'])
    res['Měsíc']      = pd.to_datetime(res['Čas']).dt.month
    res['Hodina dne'] = pd.to_datetime(res['Čas']).dt.hour

    # ── Hodinové ekonomické toky ──────────────────
    rev_teplo_h, rev_ee_h = [], []
    c_gas_kgj_h, c_gas_boil_h = [], []
    c_ee_imp_h, c_ee_ek_h, c_imp_heat_h = [], [], []
    c_start_h, c_bess_h, c_penalty_h, c_service_h = [], [], [], []
    co2_kgj_h, co2_kotel_h, co2_grid_h = [], [], []
    co2_gas_f   = p.get('co2_gas_factor',  0.202)
    co2_grid_f  = p.get('co2_grid_factor', 0.250)

    for t in range(T):
        p_ee_m   = df['ee_price'].iloc[t]  + ee_delta
        p_gas_m  = df['gas_price'].iloc[t] + gas_delta
        p_gas_kj = p.get('kgj_gas_fix_price',  p_gas_m) if (u['kgj']  and p.get('kgj_gas_fix'))  else p_gas_m
        p_gas_bh = p.get('boil_gas_fix_price', p_gas_m) if (u['boil'] and p.get('boil_gas_fix')) else p_gas_m
        p_ee_ekh = p.get('ek_ee_fix_price',    p_ee_m)  if (u['ek']   and p.get('ek_ee_fix'))   else p_ee_m
        p_ee_bessh = p.get('bess_ee_fix_price', p_ee_m) if (u['bess'] and p.get('bess_ee_fix')) else p_ee_m

        fve_ds   = p['dist_ee_sell'] if (u['fve'] and p.get('fve_dist_sell')) else 0.0

        # Objective počítá h_price*(heat_delivered - heat_dump); zahozené teplo
        # se neprodá, takže se musí odečíst i tady, jinak se report rozejde s optimem.
        rt  = h_price * (res['Dodáno tepla [MW]'].iloc[t] - res['Zahozené teplo [MW]'].iloc[t])
        # Green bonus pro KGJ fix cenu (per profil) — promítne se do hodinového Rev EE
        kgj_bonus_h = (kgj_fix_price - p_ee_m) * res['EE z KGJ [MW]'].iloc[t] if kgj_fix_here(t) else 0
        re  = ((p_ee_m - p['dist_ee_sell'] - fve_ds) * res['EE export [MW]'].iloc[t]
              + kgj_bonus_h)
        cg1 = (p_gas_kj + p['gas_dist']) * res['Plyn KGJ [MWh]'].iloc[t] if u['kgj'] else 0
        cg2 = (p_gas_bh + p['gas_dist']) * (res['Kotel [MW_th]'].iloc[t] / boil_eff)      if u['boil'] else 0
        # Náklady EE — split podle cíle: grid→EK + grid→BESS (každý se svou kontraktní/tržní cenou + distribucí).
        # Lokální složka má jen opportunity cost (ušlý export), nevstupuje sem.
        ce1 = ((p_ee_ekh   + p['dist_ee_buy']) * res['EE do EK grid [MW]'].iloc[t]   if u['ek']   else 0) \
            + ((p_ee_bessh + p['dist_ee_buy']) * res['EE do BESS grid [MW]'].iloc[t] if u['bess'] else 0)
        ce2 = 0
        ci  = p['imp_price'] * res['Import tepla [MW_th]'].iloc[t] if u['ext_heat'] else 0
        cs  = p['k_start_cost'] * vv(start, t) if u['kgj'] else 0
        csv = p.get('k_service_cost', 0.0) * res['KGJ on'].iloc[t] if u['kgj'] else 0
        cb  = (p['bess_cycle_cost'] * (res['BESS nabíjení [MW]'].iloc[t] + res['BESS vybíjení [MW]'].iloc[t])
               if u['bess'] else 0)
        cp  = p['shortfall_penalty'] * res['Shortfall [MW]'].iloc[t]

        gas_kgj_mwh  = res['Plyn KGJ [MWh]'].iloc[t] if u['kgj'] else 0
        gas_boil_mwh = (res['Kotel [MW_th]'].iloc[t] / boil_eff) if u['boil'] else 0
        co2_kgj_h.append(co2_gas_f  * gas_kgj_mwh)
        co2_kotel_h.append(co2_gas_f * gas_boil_mwh)
        co2_grid_h.append(co2_grid_f * (res['EE import [MW]'].iloc[t] - res['EE export [MW]'].iloc[t]))

        rev_teplo_h.append(rt);  rev_ee_h.append(re)
        c_gas_kgj_h.append(cg1); c_gas_boil_h.append(cg2)
        c_ee_imp_h.append(ce1);  c_ee_ek_h.append(ce2)
        c_imp_heat_h.append(ci); c_start_h.append(cs)
        c_bess_h.append(cb);     c_penalty_h.append(cp)
        c_service_h.append(csv)

    res['Rev teplo [€]']      = rev_teplo_h
    res['Rev EE [€]']         = rev_ee_h
    res['Nákl plyn KGJ [€]']  = c_gas_kgj_h
    res['Nákl plyn kotel [€]']= c_gas_boil_h
    res['Nákl EE import [€]'] = c_ee_imp_h
    res['Nákl EE EK [€]']     = c_ee_ek_h
    res['Nákl imp tepla [€]'] = c_imp_heat_h
    res['Nákl starty [€]']    = c_start_h
    res['Nákl servis KGJ [€]']= c_service_h
    res['Nákl BESS [€]']      = c_bess_h
    res['Nákl penalizace [€]']= c_penalty_h
    res['Hodinový zisk [€]']  = [
        rev_teplo_h[t] + rev_ee_h[t]
        - c_gas_kgj_h[t] - c_gas_boil_h[t]
        - c_ee_imp_h[t] - c_ee_ek_h[t]
        - c_imp_heat_h[t] - c_start_h[t]
        - c_service_h[t]
        - c_bess_h[t] - c_penalty_h[t]
        for t in range(T)
    ]
    res['Kumulativní zisk [€]'] = res['Hodinový zisk [€]'].cumsum()

    # ── CO₂ emise ─────────────────────────────────
    res['CO₂ KGJ [tCO₂]']   = co2_kgj_h
    res['CO₂ Kotel [tCO₂]'] = co2_kotel_h
    res['CO₂ Síť [tCO₂]']   = co2_grid_h
    res['CO₂ Celkem [tCO₂]']= [co2_kgj_h[t] + co2_kotel_h[t] + co2_grid_h[t] for t in range(T)]

    total_co2 = sum(co2_kgj_h[t] + co2_kotel_h[t] + co2_grid_h[t] for t in range(T))

    return {'res': res, 'start': start, 'stop': stop, 'on': on,
            'on_boil': on_boil, 'on_imp': on_imp,
            'status': status, 'total_profit': res['Hodinový zisk [€]'].sum(),
            'lp_objective': float(pulp.value(model.objective) or 0.0),
            'total_co2': total_co2}
# ────────────────────────────────────────────────
# PROVOZNI PLAN - mesicni mrizka den x hodina
# ────────────────────────────────────────────────

MONTH_NAMES_FULL = {
    1: 'LEDEN', 2: 'ÚNOR', 3: 'BŘEZEN', 4: 'DUBEN', 5: 'KVĚTEN', 6: 'ČERVEN',
    7: 'ČERVENEC', 8: 'SRPEN', 9: 'ZÁŘÍ', 10: 'ŘÍJEN', 11: 'LISTOPAD',
    12: 'PROSINEC',
}

# Popisky radku. Misto cisel 1-24 rovnou intervaly, at neni pochyb o tom,
# jestli "hodina 1" znamena pulnoc nebo jednu rano.
HOUR_LABELS = [f'{h:02d}:00-{h + 1:02d}:00' for h in range(24)]


def build_month_grid(res, month):
    """Sestaví mřížku provozu KGJ pro jeden měsíc.

    Vrací (dny, mřížka), kde mřížka[h][i] je 'P' (provoz), 'X' (klid) nebo
    None pro hodinu, která v datech není. `dny` je seznam čísel dní
    kalendářního měsíce, takže mřížka pokrývá celý měsíc i při dílčí analýze.

    Čte se sloupec 'KGJ on', tedy **nasazení**, ne skutečný výkon: doběhová
    hodina po odstavení má on = 0, takže vyjde jako 'X', přestože v ní
    jednotka ještě dodává zbytkové teplo z bloku.
    """
    times = pd.to_datetime(res['Čas'])
    mask = times.dt.month == month
    if not mask.any():
        return [], []

    sub = res.loc[mask]
    sub_times = times.loc[mask]
    year = int(sub_times.dt.year.iloc[0])
    n_days = calendar.monthrange(year, month)[1]
    days = list(range(1, n_days + 1))

    # Pri prechodu na zimni cas jsou v datech dve hodiny se stejnym razitkem.
    # Bereme max, tedy 'P', pokud jednotka bezela aspon v jedne z nich.
    running = {}
    for day, hour, on in zip(sub_times.dt.day, sub_times.dt.hour,
                             sub['KGJ on'] > 0.5):
        key = (int(day), int(hour))
        running[key] = running.get(key, False) or bool(on)

    grid = [[None] * n_days for _ in range(24)]
    for (day, hour), on in running.items():
        grid[hour][day - 1] = 'P' if on else 'X'
    return days, grid


def month_grid_totals(grid):
    """(počty P po dnech, počty X po dnech) — pro kontrolu proti vzorcům."""
    if not grid:
        return [], []
    n_days = len(grid[0])
    p = [sum(1 for h in range(24) if grid[h][d] == 'P') for d in range(n_days)]
    x = [sum(1 for h in range(24) if grid[h][d] == 'X') for d in range(n_days)]
    return p, x
