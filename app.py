import io
import re
import pickle
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

CACHE_DIR  = Path(__file__).parent / 'cache'
CACHE_FILE = CACHE_DIR / 'last_run.pkl'
CACHE_KEYS = [
    'scenario_results', 'monthly_profile_results', 'annual_plan_result',
    'sensitivity_results', 'df_main', 'uses', 'fwd_data',
    'avg_ee_raw', 'avg_gas_raw', 'ee_new', 'gas_new',
]


def save_cache():
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {k: st.session_state.get(k) for k in CACHE_KEYS}
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.warning(f"Nepodařilo se uložit cache: {e}")


def load_cache():
    if not CACHE_FILE.exists():
        return False
    try:
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        for k, v in data.items():
            if v is not None:
                st.session_state[k] = v
        return True
    except Exception as e:
        st.warning(f"Nepodařilo se načíst cache: {e}")
        return False


def clear_cache():
    try:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
    except Exception:
        pass
    for k in CACHE_KEYS:
        st.session_state[k] = None
    st.session_state._cache_loaded = True

# ── Barvy profilů (konzistentní napříč všemi grafy) ──────────────────
PROFILE_COLORS = {
    'free':    '#2196F3',  # modrá
    'base':    '#4CAF50',  # zelená
    'peak':    '#FF9800',  # oranžová
    'extpeak': '#F44336',  # červená
    'offpeak': '#9C27B0',  # fialová
    'custom':  '#607D8B',  # šedá
}

# ── CSS – vizuální vylepšení ──────────────────────────────────────────
st.markdown("""
<style>
/* KPI karty */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e2a3a 0%, #243447 100%);
    border: 1px solid #2d4a6b;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
div[data-testid="metric-container"] label {
    color: #8ab4d4 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
}
/* Sekce nadpisy */
h3 { color: #e8f4fd; }
/* Sidebar */
section[data-testid="stSidebar"] { background: #0f1923; }
section[data-testid="stSidebar"] label { color: #c5d8ea !important; }
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 { color: #4fc3f7 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────
for key, default in [
    ('fwd_data', None), ('avg_ee_raw', 100.0), ('avg_gas_raw', 50.0),
    ('ee_new', 100.0), ('gas_new', 50.0), ('results', None), ('df_main', None),
    ('scenario_results', None), ('selected_profile', 'free'),
    ('monthly_profile_results', None), ('sensitivity_results', None),
    ('annual_plan_result', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state.get('_cache_loaded'):
    load_cache()
    st.session_state._cache_loaded = True

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

MONTH_NAMES = {1:'Led',2:'Úno',3:'Bře',4:'Dub',5:'Kvě',6:'Čvn',
               7:'Čvc',8:'Srp',9:'Zář',10:'Říj',11:'Lis',12:'Pro'}

# ════════════════════════════════════════════════════════════════════
# SCHEDULING PROFILES & SCENARIO MANAGEMENT
# ════════════════════════════════════════════════════════════════════

def create_profile_constraints(df, profile_type, custom_hours=None):
    """
    Vytvoří constrainty pro KGJ provoz dle profilu.
    Returns: list[int]  -1 = must OFF, 0 = free, 1 = must ON (baseload)
    """
    df_work = df.copy()
    df_work['hour'] = pd.to_datetime(df_work['datetime']).dt.hour

    if profile_type == 'base':
        # BASE = baseload: KGJ jede 24/7, všechny sloty vynuceny ON
        constraints = [1] * len(df_work)

    elif profile_type == 'peak':
        # Peak hodiny 9-21
        constraints = [0 if h in range(9, 22) else -1 for h in df_work['hour']]

    elif profile_type == 'extpeak':
        # Extended Peak 6-22
        constraints = [0 if h in range(6, 23) else -1 for h in df_work['hour']]

    elif profile_type == 'offpeak':
        # Off-peak: noční hodiny 0-8 a 22-23
        constraints = [0 if (h <= 8 or h >= 22) else -1 for h in df_work['hour']]

    elif profile_type == 'custom' and custom_hours:
        constraints = [0 if h in custom_hours else -1 for h in df_work['hour']]

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


def create_scenario_comparison_df(scenarios):
    """Vytvoř DataFrame s porovnáním scénářů"""
    data = []
    
    for profile_name, scenario in scenarios.items():
        if scenario['result'] is None:
            continue
        res = scenario['result']['res']
        smooth = scenario['smoothness']
        
        profit = scenario['result']['total_profit']
        shortfall = res['Shortfall [MW]'].sum() if 'Shortfall [MW]' in res.columns else 0
        co2_total = res['CO₂ Celkem [tCO₂]'].sum() if 'CO₂ Celkem [tCO₂]' in res.columns else None

        row = {
            'Profil': profile_name.upper(),
            'Zisk [€]': f"{profit:,.0f}",
            'Využití KGJ [%]': f"{smooth['utilization_pct']:.1f}",
            'Stabilita [%]': f"{smooth['stability_score']:.1f}",
            'Transitions': smooth['transitions'],
            'Avg Runtime [h]': f"{smooth['avg_run_hours']:.1f}",
            'Total Hours ON': smooth['total_on_hours'],
            'Shortfall [MWh]': f"{shortfall:.1f}",
        }
        if co2_total is not None:
            row['CO₂ [tCO₂]'] = f"{co2_total:,.1f}"
        data.append(row)
    
    return pd.DataFrame(data)


# ────────────────────────────────────────────────
# EXCEL EXPORTNÍ FUNKCE
# ────────────────────────────────────────────────

def create_detail_comparison_strip(scenarios, p):
    """Kompaktní porovnávací tabulka všech profilů – stejné metriky jako detail view."""
    rows = []
    for name, scenario in scenarios.items():
        if scenario['result'] is None:
            continue
        res    = scenario['result']['res']
        smooth = scenario['smoothness']
        profit = scenario['result']['total_profit']
        shortfall = res['Shortfall [MW]'].sum()
        target    = (res['Poptávka tepla [MW]'] * p['h_cover']).sum()
        coverage  = 100.0 * (1 - shortfall / target) if target > 0 else 100.0
        co2 = res['CO₂ Celkem [tCO₂]'].sum() if 'CO₂ Celkem [tCO₂]' in res.columns else None
        rows.append({
            'Profil':           name.upper(),
            'Zisk [€]':         round(profit, 0),
            'Pokrytí [%]':      round(coverage, 1),
            'CO₂ [tCO₂]':       round(co2, 1) if co2 is not None else None,
            'Přechodů':         smooth['transitions'],
            'Avg Runtime [h]':  round(smooth['avg_run_hours'], 1),
            'Stabilita [%]':    round(smooth['stability_score'], 1),
            'Hod KGJ':          int(res['KGJ on'].sum()),
        })
    return pd.DataFrame(rows)


def _wb_formats(workbook):
    hdr = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
    num = workbook.add_format({'num_format': '#,##0.00'})
    txt = workbook.add_format({'num_format': '@'})
    grn = workbook.add_format({'bold': True, 'bg_color': '#C6EFCE', 'border': 1})
    return hdr, num, txt, grn

def _safe_sheet(name: str) -> str:
    """Odstraní znaky neplatné v Excel názvech listů, zkrátí na 31 znaků."""
    return re.sub(r'[/\\*?:\[\]]', '-', name)[:31]

def _write_sheet(writer, df, sheet_name, hdr_fmt, num_fmt, txt_fmt):
    safe = _safe_sheet(sheet_name)
    df.to_excel(writer, index=False, sheet_name=safe)
    ws = writer.sheets[safe]
    for col_idx, col_name in enumerate(df.columns):
        is_num = pd.api.types.is_numeric_dtype(df.iloc[:, col_idx])
        ws.set_column(col_idx, col_idx, 18, num_fmt if is_num else txt_fmt)
        ws.write(0, col_idx, col_name, hdr_fmt)


def to_excel_scenarios(scenarios):
    """Excel export pro scenáristickou analýzu."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        workbook = writer.book
        hdr_fmt, num_fmt, txt_fmt, grn_fmt = _wb_formats(workbook)

        # List: Porovnání scénářů
        comp_df = create_scenario_comparison_df(scenarios)
        _write_sheet(writer, comp_df, 'Porovnání scénářů', hdr_fmt, num_fmt, txt_fmt)

        # List: Breakdown příjmů/nákladů
        bd_rows = []
        for profile, scenario in scenarios.items():
            if scenario['result'] is None:
                continue
            r = scenario['result']['res']
            bd_rows.append({
                'Profil':            profile.upper(),
                'Rev teplo [k€]':   r['Rev teplo [€]'].sum() / 1000,
                'Rev EE [k€]':      r['Rev EE [€]'].sum() / 1000,
                'Nákl plyn [k€]':  -(r['Nákl plyn KGJ [€]'].sum() + r['Nákl plyn kotel [€]'].sum()) / 1000,
                'Nákl EE [k€]':    -(r['Nákl EE import [€]'].sum() + r['Nákl EE EK [€]'].sum()) / 1000,
                'Nákl imp tepla [k€]': -(r['Nákl imp tepla [€]'].sum()) / 1000,
                'Nákl starty+BESS [k€]': -(r['Nákl starty [€]'].sum() + r['Nákl BESS [€]'].sum()) / 1000,
                'Nákl servis KGJ [k€]': -(r['Nákl servis KGJ [€]'].sum()) / 1000,
                'Zisk celkem [k€]': r['Hodinový zisk [€]'].sum() / 1000,
            })
        if bd_rows:
            _write_sheet(writer, pd.DataFrame(bd_rows), 'Breakdown nákladů', hdr_fmt, num_fmt, txt_fmt)

        # Listy s hodinovými daty pro každý profil
        skip_cols = {'Měsíc', 'Hodina dne', 'KGJ on', 'Kotel on', 'Import tepla on'}
        for profile, scenario in scenarios.items():
            if scenario['result'] is None:
                continue
            res_df = scenario['result']['res']
            df_exp = res_df[[c for c in res_df.columns if c not in skip_cols]].round(4)
            sheet = profile.upper()[:31]
            _write_sheet(writer, df_exp, sheet, hdr_fmt, num_fmt, txt_fmt)

    return buf.getvalue()


def to_excel_monthly(monthly_pr, month_names):
    """Excel export pro měsíční analýzu profilů."""
    buf = io.BytesIO()
    months_sorted  = sorted(monthly_pr.keys())
    all_profiles   = sorted({pr for m in monthly_pr.values() for pr in m.keys()})

    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        workbook = writer.book
        hdr_fmt, num_fmt, txt_fmt, grn_fmt = _wb_formats(workbook)

        # List: Nejlepší profil per měsíc
        best_rows = []
        for month in months_sorted:
            m_data = monthly_pr[month]
            if not m_data:
                continue
            best_pr = max(m_data, key=lambda pr: m_data[pr]['profit'])
            br = {
                'Měsíc':           month_names.get(month, month),
                'Nejlepší profil': best_pr.upper(),
                'Zisk [€]':        round(m_data[best_pr]['profit'], 0),
                'Zisk/hod [€]':    round(m_data[best_pr]['profit_per_h'], 2),
                'Využití KGJ [%]': round(m_data[best_pr]['smoothness']['utilization_pct'], 1),
                'Stabilita [%]':   round(m_data[best_pr]['smoothness']['stability_score'], 1),
            }
            co2_val = m_data[best_pr].get('total_co2')
            if co2_val is not None:
                br['CO₂ [tCO₂]'] = round(co2_val, 1)
            best_rows.append(br)
        if best_rows:
            _write_sheet(writer, pd.DataFrame(best_rows), 'Nejlepší profil / měsíc', hdr_fmt, num_fmt, txt_fmt)

        # List: Matice zisk [€] – měsíce × profily
        matrix_profit = []
        for month in months_sorted:
            row = {'Měsíc': month_names.get(month, month)}
            for pr in all_profiles:
                row[pr.upper()] = round(monthly_pr[month].get(pr, {}).get('profit', None) or 0, 0)
            matrix_profit.append(row)
        df_matrix = pd.DataFrame(matrix_profit)
        _write_sheet(writer, df_matrix, 'Matice zisk [€]', hdr_fmt, num_fmt, txt_fmt)

        # Podmíněné formátování – zelená = max v řádku
        ws = writer.sheets[_safe_sheet('Matice zisk [€]')]
        n_prof = len(all_profiles)
        if n_prof > 0:
            ws.conditional_format(1, 1, len(months_sorted), n_prof, {
                'type': 'cell', 'criteria': '>=',
                'value': 0, 'format': workbook.add_format({'bg_color': '#C6EFCE'})
            })

        # List: Matice zisk/hod [€/h]
        matrix_pph = []
        for month in months_sorted:
            row = {'Měsíc': month_names.get(month, month)}
            for pr in all_profiles:
                row[pr.upper()] = round(monthly_pr[month].get(pr, {}).get('profit_per_h', None) or 0, 2)
            matrix_pph.append(row)
        _write_sheet(writer, pd.DataFrame(matrix_pph), 'Matice zisk_hod [€/h]', hdr_fmt, num_fmt, txt_fmt)

    return buf.getvalue()


def to_excel_sensitivity(sa_df, profile_name, gas_range, ee_range, steps):
    """Excel export pro citlivostní analýzu."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        workbook = writer.book
        hdr_fmt, num_fmt, txt_fmt, _ = _wb_formats(workbook)

        # List: Data
        export_df = sa_df[['typ', 'delta', 'profit', 'delta_pct']].rename(columns={
            'typ': 'Parametr', 'delta': 'Δ cena [€/MWh]',
            'profit': 'Zisk [€]', 'delta_pct': 'Změna [%]'
        }).round(2)
        _write_sheet(writer, export_df, 'Citlivostní analýza', hdr_fmt, num_fmt, txt_fmt)

        # List: Parametry analýzy
        meta_df = pd.DataFrame([
            {'Parametr': 'Profil', 'Hodnota': profile_name.upper()},
            {'Parametr': 'Rozsah plyn [±€/MWh]', 'Hodnota': gas_range},
            {'Parametr': 'Rozsah EE [±€/MWh]', 'Hodnota': ee_range},
            {'Parametr': 'Počet kroků', 'Hodnota': steps},
        ])
        _write_sheet(writer, meta_df, 'Parametry analýzy', hdr_fmt, num_fmt, txt_fmt)

    return buf.getvalue()


# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    if CACHE_FILE.exists():
        try:
            _ts = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime).strftime('%d.%m. %H:%M')
            st.caption(f"💾 Cache: poslední běh {_ts}")
        except Exception:
            st.caption("💾 Cache: dostupná")
        if st.button("🗑️ Vyčistit cache výsledků", use_container_width=True):
            clear_cache()
            st.rerun()
    else:
        st.caption("💾 Cache: prázdná")

    st.divider()
    st.header("⚙️ Technologie na lokalitě")
    use_kgj      = st.checkbox("Kogenerace (KGJ)",    value=True)
    use_boil     = st.checkbox("Plynový kotel",        value=True)
    use_ek       = st.checkbox("Elektrokotel",         value=True)
    use_tes      = st.checkbox("Nádrž (TES)",          value=True)
    use_bess     = st.checkbox("Baterie (BESS)",       value=True)
    use_fve      = st.checkbox("Fotovoltaika (FVE)",   value=True)
    use_ext_heat = st.checkbox("Nákup tepla (Import)", value=True)

uses = dict(kgj=use_kgj, boil=use_boil, ek=use_ek, tes=use_tes,
            bess=use_bess, fve=use_fve, ext_heat=use_ext_heat)

with st.sidebar:
    st.divider()
    st.header("📈 Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])
    if fwd_file is not None:
        try:
            df_raw = pd.read_excel(fwd_file)
            df_raw.columns = [str(c).strip() for c in df_raw.columns]
            date_col = df_raw.columns[0]
            df_raw[date_col] = pd.to_datetime(df_raw[date_col], dayfirst=True)
            years    = sorted(df_raw[date_col].dt.year.unique())
            sel_year = st.selectbox("Rok pro analýzu", years)
            df_year  = df_raw[df_raw[date_col].dt.year == sel_year].copy()

            avg_ee  = float(df_year.iloc[:, 1].mean())
            avg_gas = float(df_year.iloc[:, 2].mean())
            st.session_state.avg_ee_raw  = avg_ee
            st.session_state.avg_gas_raw = avg_gas
            st.info(f"Průměr EE: **{avg_ee:.1f} €/MWh** | Plyn: **{avg_gas:.1f} €/MWh**")

            ee_new  = st.number_input("Cílová base cena EE [€/MWh]",   value=round(avg_ee,  1), step=1.0)
            gas_new = st.number_input("Cílová base cena Plyn [€/MWh]", value=round(avg_gas, 1), step=1.0)

            df_fwd = df_year.copy()
            df_fwd.columns = ['datetime', 'ee_original', 'gas_original']
            df_fwd['ee_price']  = df_fwd['ee_original']  + (ee_new  - avg_ee)
            df_fwd['gas_price'] = df_fwd['gas_original'] + (gas_new - avg_gas)
            st.session_state.fwd_data = df_fwd
            st.session_state.ee_new   = ee_new
            st.session_state.gas_new  = gas_new
            st.success("FWD načteno ✔")
        except Exception as e:
            st.error(f"Chyba při načítání FWD: {e}")

    # ── NOVÉ: KGJ Scheduling Profily ──
    st.divider()
    st.header("📅 Analýza Období & Profily KGJ")
    
    # Výběr Období
    st.subheader("1️⃣ Období Analýzy")
    analysis_mode = st.radio("Vyberte režim:", ["Celá data", "Vlastní rozsah"])
    
    period_start = None
    period_end = None
    if analysis_mode == "Vlastní rozsah" and st.session_state.fwd_data is not None:
        min_date = pd.to_datetime(st.session_state.fwd_data['datetime']).min().date()
        max_date = pd.to_datetime(st.session_state.fwd_data['datetime']).max().date()
        
        col_ps, col_pe = st.columns(2)
        with col_ps:
            period_start = st.date_input("Od", value=min_date, key="period_start")
        with col_pe:
            period_end = st.date_input("Do", value=max_date, key="period_end")
    
    # KGJ Scheduling Profily
    st.subheader("2️⃣ Scheduling Profily KGJ")
    
    profiles_to_run = st.multiselect(
        "Které profily testovat?",
        options=['free', 'base', 'peak', 'extpeak', 'offpeak', 'custom'],
        default=['free', 'base', 'peak', 'extpeak', 'offpeak'],
        help="Spusť optimalizaci pro vybrané profily a porovnej je"
    )
    st.caption("💡 BASE = KGJ vždy zapnuto 24/7 (ignoruje limit hodin provozu)")

    profile_definitions = {
        'free':    {'name': 'Volná Opt.',          'hours': None,                          'desc': 'Bez omezení'},
        'base':    {'name': 'Base (0-24h)',         'hours': list(range(24)),               'desc': 'Celý den'},
        'peak':    {'name': 'Peak (9-21h)',         'hours': list(range(9, 22)),            'desc': '12 hodin'},
        'extpeak': {'name': 'ExtPeak (6-22h)',      'hours': list(range(6, 23)),            'desc': '16 hodin'},
        'offpeak': {'name': 'Offpeak (0-8,22-23h)', 'hours': list(range(0, 9))+[22, 23],   'desc': '11 hodin'},
    }
    
    custom_hours = None
    if 'custom' in profiles_to_run:
        st.write("**Custom Profil** - Vyberte hodiny:")
        custom_hours = st.multiselect(
            "Povolené hodiny (0-23):",
            options=list(range(24)),
            default=list(range(6, 23)),
            key="custom_hours_selector"
        )
        profile_definitions['custom'] = {
            'name': f'Custom ({len(custom_hours)}h)',
            'hours': custom_hours,
            'desc': f"Custom"
        }
    
    # Provozní Omezení
    st.subheader("3️⃣ Omezení Provozování")
    
    use_month_start_limit = st.checkbox("Omezit max. startů za měsíc", value=False)
    max_starts_per_month = None
    if use_month_start_limit:
        max_starts_per_month = st.number_input("Max. startů/měsíc", value=5, min_value=1, max_value=30)
    

# ────────────────────────────────────────────────
# FWD GRAFY
# ────────────────────────────────────────────────
if st.session_state.fwd_data is not None:
    df_fwd = st.session_state.fwd_data
    with st.expander("📈 FWD křivka – originál vs. upravená", expanded=True):
        tab_ee, tab_gas, tab_dur = st.tabs(["Elektřina [€/MWh]", "Plyn [€/MWh]", "Křivky trvání"])
        with tab_ee:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_fwd['datetime'], y=df_fwd['ee_original'],
                name='EE – originál', line=dict(color='#95a5a6', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df_fwd['datetime'], y=df_fwd['ee_price'],
                name='EE – upravená', line=dict(color='#2ecc71', width=2)))
            fig.add_hline(y=st.session_state.avg_ee_raw, line_dash="dash", line_color="#95a5a6",
                annotation_text=f"Orig. průměr {st.session_state.avg_ee_raw:.1f}")
            fig.add_hline(y=st.session_state.ee_new, line_dash="dash", line_color="#27ae60",
                annotation_text=f"Nový průměr {st.session_state.ee_new:.1f}")
            fig.update_layout(height=340, hovermode='x unified', margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)
        with tab_gas:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_fwd['datetime'], y=df_fwd['gas_original'],
                name='Plyn – originál', line=dict(color='#95a5a6', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df_fwd['datetime'], y=df_fwd['gas_price'],
                name='Plyn – upravená', line=dict(color='#e67e22', width=2)))
            fig.add_hline(y=st.session_state.avg_gas_raw, line_dash="dash", line_color="#95a5a6",
                annotation_text=f"Orig. průměr {st.session_state.avg_gas_raw:.1f}")
            fig.add_hline(y=st.session_state.gas_new, line_dash="dash", line_color="#e67e22",
                annotation_text=f"Nový průměr {st.session_state.gas_new:.1f}")
            fig.update_layout(height=340, hovermode='x unified', margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)
        with tab_dur:
            ee_s  = df_fwd['ee_price'].sort_values(ascending=False).values
            gas_s = df_fwd['gas_price'].sort_values(ascending=False).values
            hrs   = list(range(1, len(ee_s)+1))
            fig = make_subplots(rows=1, cols=2,
                subplot_titles=("Křivka trvání – EE", "Křivka trvání – Plyn"))
            fig.add_trace(go.Scatter(x=hrs, y=ee_s, name='EE',
                line=dict(color='#2ecc71', width=2), fill='tozeroy',
                fillcolor='rgba(46,204,113,0.15)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hrs, y=gas_s, name='Plyn',
                line=dict(color='#e67e22', width=2), fill='tozeroy',
                fillcolor='rgba(230,126,34,0.15)'), row=1, col=2)
            fig.update_xaxes(title_text="Hodiny [h]")
            fig.update_yaxes(title_text="€/MWh")
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────
# PARAMETRY
# ────────────────────────────────────────────────
p = {}
t_gen, t_tech, t_co2 = st.tabs(["Obecné", "Technika", "Emise CO₂"])

with t_gen:
    col1, col2 = st.columns(2)
    with col1:
        p['dist_ee_buy']       = st.number_input("Distribuce nákup EE [€/MWh]",   value=33.0)
        p['dist_ee_sell']      = st.number_input("Distribuce prodej EE [€/MWh]",  value=2.0)
        p['gas_dist']          = st.number_input("Distribuce plyn [€/MWh]",        value=5.0)
    with col2:
        p['internal_ee_use']   = st.checkbox(
            "Interní spotřeba EE bez distribuce", value=True,
            help="ON (doporučeno): distribuce se platí jen na skutečně dovezenou EE ze sítě "
                 "(ee_import → EK). Lokální výroba (KGJ/FVE/BESS) spotřebovaná v EK je bez distribuce. "
                 "Distribuce na export se neplatí. "
                 "OFF (legacy): účtuje navíc distribuci na celou spotřebu EK.")
        p['h_price']           = st.number_input("Prodejní cena tepla [€/MWh]",   value=95.0)
        p['h_cover']           = st.slider("Minimální pokrytí poptávky tepla", 0.0, 1.0, 0.99, step=0.01)
        p['shortfall_penalty'] = st.number_input("Penalizace za nedodání tepla [€/MWh]", value=500.0,
            help="Doporučeno 3–5× cena tepla. Vyšší = silnější priorita pokrytí poptávky.")

with t_tech:
    # ── KGJ ──────────────────────────────────────
    if use_kgj:
        st.subheader("Kogenerace (KGJ)")
        c1, c2 = st.columns(2)
        with c1:
            p['k_th']          = st.number_input("Jmenovitý tepelný výkon [MW]",  value=0.605)
            p['k_eff_th']      = st.number_input("Tepelná účinnost η_th [-]",      value=0.531)
            p['k_eff_el']      = st.number_input("Elektrická účinnost η_el [-]",   value=0.395)
            p['k_min']         = st.slider("Min. zatížení [%]", 0, 100, 50) / 100
        with c2:
            p['k_start_cost']  = st.number_input("Náklady na start [€/start]",    value=150.0)
            p['k_min_runtime'] = st.number_input("Min. doba běhu [hod]",          value=4, min_value=1)
            p['k_service_cost']= st.number_input("Servisní náklad [€/h provozu]", value=14.0)
        k_el_derived = p['k_th'] * (p['k_eff_el'] / p['k_eff_th'])
        p['k_el'] = k_el_derived
        st.caption(f"ℹ️ Odvozený el. výkon: **{k_el_derived:.3f} MW** | "
                   f"Celková účinnost: **{p['k_eff_th']+p['k_eff_el']:.2f}**")
        # Roční limit hodin
        p['kgj_hour_limit_on'] = st.checkbox("Omezit max. počet provozních hodin KGJ / rok", value=False)
        if p['kgj_hour_limit_on']:
            p['kgj_hour_limit'] = st.number_input("Max. hodin provozu KGJ / rok", value=6000, min_value=1)
        p['kgj_gas_fix'] = st.checkbox("Fixní cena plynu pro KGJ", value=True)
        if p['kgj_gas_fix']:
            p['kgj_gas_fix_price'] = st.number_input("Fixní cena plynu – KGJ [€/MWh]",
                value=40.0)
        # Proměnná účinnost dle výkonu
        p['kgj_var_eff'] = st.checkbox("Proměnná účinnost dle výkonu", value=False,
            help="Linearizovaná 2-bodová křivka: účinnost při min. zátěži vs. jmenovitém výkonu")
        if p['kgj_var_eff']:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                p['eta_th_min'] = st.number_input("η_th při min. zátěži [-]",
                    value=round(p['k_eff_th'] * 0.9, 3), min_value=0.1, max_value=1.0, step=0.01,
                    help="Tepelná účinnost při minimálním zatížení KGJ")
            with col_v2:
                p['eta_el_min'] = st.number_input("η_el při min. zátěži [-]",
                    value=round(p['k_eff_el'] * 0.9, 3), min_value=0.05, max_value=1.0, step=0.01,
                    help="Elektrická účinnost při minimálním zatížení KGJ")
            st.caption(f"ℹ️ Při jmenovitém výkonu: η_th={p['k_eff_th']}, η_el={p['k_eff_el']}")

    # ── Kotel ─────────────────────────────────────
    if use_boil:
        st.subheader("Plynový kotel")
        p['b_max']    = st.number_input("Max. výkon [MW]",    value=4.44)
        p['boil_eff'] = st.number_input("Účinnost kotle [-]", value=0.86)
        p['boil_hour_limit_on'] = st.checkbox("Omezit max. počet provozních hodin kotle / rok", value=False)
        if p['boil_hour_limit_on']:
            p['boil_hour_limit'] = st.number_input("Max. hodin provozu kotle / rok", value=4000, min_value=1)
        p['boil_gas_fix'] = st.checkbox("Fixní cena plynu pro kotel", value=True)
        if p['boil_gas_fix']:
            p['boil_gas_fix_price'] = st.number_input("Fixní cena plynu – kotel [€/MWh]",
                value=40.0)

    # ── Elektrokotel ──────────────────────────────
    if use_ek:
        st.subheader("Elektrokotel")
        p['ek_max'] = st.number_input("Max. výkon [MW]",  value=0.4)
        p['ek_eff'] = st.number_input("Účinnost EK [-]",  value=0.99)
        p['ek_ee_fix'] = st.checkbox("Fixní cena EE pro elektrokotel")
        if p['ek_ee_fix']:
            p['ek_ee_fix_price'] = st.number_input("Fixní cena EE – EK [€/MWh]",
                value=float(st.session_state.avg_ee_raw))

    # ── TES ───────────────────────────────────────
    if use_tes:
        st.subheader("Nádrž TES")
        p['tes_cap']  = st.number_input("Kapacita [MWh]", value=1.52)
        p['tes_loss'] = st.number_input("Ztráta [%/h]",   value=0.5) / 100

    # ── BESS ──────────────────────────────────────
    if use_bess:
        st.subheader("Baterie BESS")
        c1, c2 = st.columns(2)
        with c1:
            p['bess_cap']        = st.number_input("Kapacita [MWh]",                 value=1.0)
            p['bess_p']          = st.number_input("Max. výkon [MW]",                 value=0.5)
            p['bess_eff']        = st.number_input("Účinnost nabíjení/vybíjení [-]",  value=0.90)
            p['bess_cycle_cost'] = st.number_input("Náklady na opotřebení [€/MWh]",   value=5.0)
        with c2:
            st.markdown("**Distribuce pro arbitráž**")
            p['bess_dist_buy']  = st.checkbox("Účtovat distribuci NÁKUP do BESS",  value=False)
            p['bess_dist_sell'] = st.checkbox("Účtovat distribuci PRODEJ z BESS",  value=False)
            st.caption("💡 Interní arbitráž distribuci neplatí při zapnuté volbě 'Ušetřit distribuci'.")
        p['bess_ee_fix'] = st.checkbox("Fixní cena EE pro BESS")
        if p['bess_ee_fix']:
            p['bess_ee_fix_price'] = st.number_input("Fixní cena EE – BESS [€/MWh]",
                value=float(st.session_state.avg_ee_raw))

    # ── FVE ───────────────────────────────────────
    if use_fve:
        st.subheader("Fotovoltaika FVE")
        p['fve_installed_p'] = st.number_input("Instalovaný výkon [MW]", value=1.0,
            help="Profil FVE v lokálních datech = capacity factor 0–1.")
        p['fve_dist_sell'] = st.checkbox("Účtovat distribuci PRODEJ z FVE do sítě", value=False)

    # ── Import tepla ──────────────────────────────
    if use_ext_heat:
        st.subheader("Nákup tepla (Import)")
        p['imp_max']   = st.number_input("Max. výkon [MW]",      value=2.0)
        p['imp_price'] = st.number_input("Cena importu [€/MWh]", value=150.0)
        p['imp_hour_limit_on'] = st.checkbox("Omezit max. počet hodin importu tepla / rok", value=False)
        if p['imp_hour_limit_on']:
            p['imp_hour_limit'] = st.number_input("Max. hodin importu tepla / rok", value=2000, min_value=1)

with t_co2:
    st.markdown("#### Emisní faktory a CO₂ cena")
    co2_col1, co2_col2 = st.columns(2)
    with co2_col1:
        p['co2_gas_factor']  = st.number_input(
            "Emisní faktor plynu [tCO₂/MWh]", value=0.202, step=0.001, format="%.3f",
            help="Typická hodnota zemní plyn: 0.202 tCO₂/MWh")
        p['co2_grid_factor'] = st.number_input(
            "Emisní faktor sítě [tCO₂/MWh]", value=0.250, step=0.001, format="%.3f",
            help="CZ grid mix 2023 cca 0.25 tCO₂/MWh (EEA)")
    with co2_col2:
        p['co2_price'] = st.number_input(
            "Cena CO₂ [€/tCO₂]", value=0.0, min_value=0.0, step=1.0,
            help="EU ETS cena. 0 = CO₂ jen jako KPI (bez vlivu na optimalizaci). "
                 "Kladná hodnota → CO₂ náklad vstupuje do objective funkce.")
    st.caption(
        "CO₂ bilance = emise z plynu (KGJ + kotel) + emise z nákupu EE − úspora za export EE. "
        "Zobrazuje se ve výsledcích jako informativní KPI."
    )

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
    fuel_min = q_min / eta_th_min
    fuel_max = q_max / eta_th_rated
    c1_th = (fuel_max - fuel_min) / (q_max - q_min)
    c0_th = fuel_min - c1_th * q_min   # offset platí jen když on=1

    ee_min = q_min * (eta_el_min / eta_th_min)
    ee_max = q_max * (eta_el_rated / eta_th_rated)
    c1_el = (ee_max - ee_min) / (q_max - q_min)
    c0_el = ee_min - c1_el * q_min

    return c0_th, c1_th, c0_el, c1_el


# ────────────────────────────────────────────────
# ENHANCED SOLVER S PROFILE SUPPORT
# ────────────────────────────────────────────────

def run_optimization_with_profile(df, params, uses, profile_type='free', custom_hours=None,
                                   ee_delta=0.0, gas_delta=0.0, h_price_override=None, 
                                   time_limit=300, max_starts_per_month=None, period_mask=None):
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

    model = pulp.LpProblem("KGJ_Dispatch_Profile", pulp.LpMaximize)

    # ── Proměnné ─────────────────────────────────
    if u['kgj']:
        q_kgj = pulp.LpVariable.dicts("q_KGJ",  range(T), 0, p['k_th'])
        on    = pulp.LpVariable.dicts("on",      range(T), 0, 1, "Binary")
        start = pulp.LpVariable.dicts("start",   range(T), 0, 1, "Binary")
    else:
        q_kgj = on = start = {t: 0 for t in range(T)}

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

        heat_delivered = q_kgj[t] + q_boil[t] + q_ek[t] + q_imp[t] + tes_out[t] - tes_in[t]
        model += heat_delivered + heat_shortfall[t] >= h_dem * p['h_cover']
        model += heat_delivered <= h_dem + heat_dump[t] + 1e-3

        ee_kgj_out = (c0_el * on[t] + c1_el * q_kgj[t]) if u['kgj'] else 0
        ee_ek_in   = q_ek[t] / ek_eff                            if u['ek']  else 0
        model += ee_kgj_out + fve_p + ee_import[t] + bess_dis[t] == ee_ek_in + bess_cha[t] + ee_export[t]

        # Distribuci na export NEÚČTUJEME (prodej do sítě).
        # Distribuci na import účtujeme vždy – ee_import[t] je grid EE,
        # která jde do EK (BESS má vlastní flag bess_dist_buy).
        dist_sell_net       = 0.0
        dist_buy_net        = p['dist_ee_buy']
        # Přirážka na celkovou spotřebu EK (legacy režim při internal_ee_use=False).
        # Při internal_ee_use=True (default) = 0, distribuci platíme jen
        # přes ee_import[t] na skutečně dovezené EE z gridu do EK.
        dist_ek             = 0.0 if p['internal_ee_use'] else p['dist_ee_buy']
        fve_dist_sell_cost  = p['dist_ee_sell'] if (u['fve'] and p.get('fve_dist_sell')) else 0.0
        bess_dist_buy_cost  = p['dist_ee_buy']  * bess_cha[t] if (u['bess'] and p.get('bess_dist_buy'))  else 0
        bess_dist_sell_cost = p['dist_ee_sell'] * bess_dis[t] if (u['bess'] and p.get('bess_dist_sell')) else 0

        revenue = (h_price * (heat_delivered - heat_dump[t])
                   + (p_ee_m - dist_sell_net - fve_dist_sell_cost) * ee_export[t])
        co2_price = p.get('co2_price', 0.0)
        co2_cost = 0
        if co2_price > 0:
            co2_gas_factor  = p.get('co2_gas_factor',  0.202)
            co2_grid_factor = p.get('co2_grid_factor', 0.250)
            gas_kgj_mwh  = (c0_th * on[t] + c1_th * q_kgj[t]) if u['kgj']  else 0
            gas_boil_mwh = (q_boil[t] / boil_eff)               if u['boil'] else 0
            co2_cost = co2_price * (
                co2_gas_factor  * (gas_kgj_mwh + gas_boil_mwh) +
                co2_grid_factor * ee_import[t] -
                co2_grid_factor * ee_export[t]
            )
        costs = (
            ((p_gas_kgj  + p['gas_dist']) * (c0_th * on[t] + c1_th * q_kgj[t]) if u['kgj'] else 0) +
            ((p_gas_boil + p['gas_dist']) * (q_boil[t] / boil_eff)       if u['boil']     else 0) +
            (p_ee_m + dist_buy_net) * ee_import[t] +
            ((p_ee_ek + dist_ek) * ee_ek_in                               if u['ek']       else 0) +
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
    status = model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    if status not in (1, 2):
        return None

    def vv(v, t):
        x = v[t]
        return float(x) if isinstance(x, (int, float)) else float(pulp.value(x) or 0)

    res = pd.DataFrame({
        'Čas':                  df['datetime'],
        'Poptávka tepla [MW]':  df['Poptávka po teple (MW)'],
        'KGJ [MW_th]':          [vv(q_kgj,  t) for t in range(T)],
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
        'EE export [MW]':       [vv(ee_export, t) for t in range(T)],
        'EE import [MW]':       [vv(ee_import, t) for t in range(T)],
        'EE z KGJ [MW]':        [(c0_el * vv(on, t) + c1_el * vv(q_kgj, t)) if u['kgj'] else 0.0 for t in range(T)],
        'EE z FVE [MW]':        [float(df['FVE (MW)'].iloc[t]) if (u['fve'] and 'FVE (MW)' in df.columns) else 0.0 for t in range(T)],
        'EE do EK [MW]':        [vv(q_ek, t)/ek_eff if u['ek'] else 0.0 for t in range(T)],
        'Cena EE [€/MWh]':     (df['ee_price'] + ee_delta).values,
        'Cena plyn [€/MWh]':   (df['gas_price'] + gas_delta).values,
        'KGJ on':               [vv(on, t) for t in range(T)],
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

        fve_ds   = p['dist_ee_sell'] if (u['fve'] and p.get('fve_dist_sell')) else 0.0
        dist_s   = 0.0
        dist_b   = p['dist_ee_buy']
        dist_ek  = 0.0 if p['internal_ee_use'] else p['dist_ee_buy']

        rt  = h_price * res['Dodáno tepla [MW]'].iloc[t]
        re  = (p_ee_m - dist_s - fve_ds) * res['EE export [MW]'].iloc[t]
        cg1 = (p_gas_kj + p['gas_dist']) * (c0_th * res['KGJ on'].iloc[t] + c1_th * res['KGJ [MW_th]'].iloc[t]) if u['kgj'] else 0
        cg2 = (p_gas_bh + p['gas_dist']) * (res['Kotel [MW_th]'].iloc[t] / boil_eff)      if u['boil'] else 0
        ce1 = (p_ee_m + dist_b) * res['EE import [MW]'].iloc[t]
        ce2 = (p_ee_ekh + dist_ek) * res['EE do EK [MW]'].iloc[t] if u['ek'] else 0
        ci  = p['imp_price'] * res['Import tepla [MW_th]'].iloc[t] if u['ext_heat'] else 0
        cs  = p['k_start_cost'] * vv(start, t) if u['kgj'] else 0
        csv = p.get('k_service_cost', 0.0) * res['KGJ on'].iloc[t] if u['kgj'] else 0
        cb  = (p['bess_cycle_cost'] * (res['BESS nabíjení [MW]'].iloc[t] + res['BESS vybíjení [MW]'].iloc[t])
               if u['bess'] else 0)
        cp  = p['shortfall_penalty'] * res['Shortfall [MW]'].iloc[t]

        gas_kgj_mwh  = (c0_th * res['KGJ on'].iloc[t] + c1_th * res['KGJ [MW_th]'].iloc[t]) if u['kgj'] else 0
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

    return {'res': res, 'start': start, 'on': on, 'on_boil': on_boil, 'on_imp': on_imp,
            'status': status, 'total_profit': res['Hodinový zisk [€]'].sum(),
            'total_co2': total_co2}


def run_scenario_analysis(df, params, uses, profiles_to_run, custom_hours=None, 
                          period_start=None, period_end=None, max_starts_per_month=None):
    """
    Spusť optimalizaci pro všechny vybrané profily a vrať porovnání
    """
    scenarios = {}
    
    # Vytvoř period mask pokud je zadáno
    period_mask = None
    if period_start is not None and period_end is not None:
        df_work = df.copy()
        df_work['date'] = pd.to_datetime(df_work['datetime']).dt.date
        period_mask = (df_work['date'] >= period_start) & (df_work['date'] <= period_end)
    
    progress_container = st.container()
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    for idx, profile in enumerate(profiles_to_run):
        status_text.write(f"⏳ Optimalizuji profil: **{profile.upper()}**...")

        profile_custom_hours = custom_hours if profile == 'custom' else None

        try:
            result = run_optimization_with_profile(
                df=df,
                params=params,
                uses=uses,
                profile_type=profile,
                custom_hours=profile_custom_hours,
                max_starts_per_month=max_starts_per_month,
                period_mask=period_mask,
                time_limit=60,
            )
        except Exception as exc:
            st.error(f"❌ Profil {profile.upper()} – výjimka: {exc}")
            result = None

        if result is not None:
            smoothness = calculate_smoothness_metrics(result['res'])
            scenarios[profile] = {
                'result': result,
                'smoothness': smoothness,
                'profile_name': profile.upper(),
            }
        else:
            st.warning(f"⚠️ Profil {profile.upper()} nenašel řešení – zkontroluj parametry nebo omezení hodin.")

        progress_bar.progress((idx + 1) / len(profiles_to_run))
    
    status_text.write("✅ Scenáře spočítány!")
    progress_bar.empty()
    
    return scenarios


def run_monthly_profile_analysis(df, params, uses, profiles_to_run,
                                  custom_hours=None, max_starts_per_month=None):
    """
    Pro každý měsíc v datech × každý profil spustí optimalizaci.
    Každý měsíc je nezávislý (TES/BESS startuje od 50 % kapacity).
    Vrátí: {month_int: {profile_str: {profit, profit_per_h, smoothness}}}
    """
    results = {}
    months = sorted(pd.to_datetime(df['datetime']).dt.month.unique())
    total_runs = len(months) * len(profiles_to_run)

    progress = st.progress(0)
    status = st.empty()
    run_idx = 0

    for month in months:
        mask = pd.to_datetime(df['datetime']).dt.month == month
        results[month] = {}
        for profile in profiles_to_run:
            status.write(f"⏳ Měsíc **{MONTH_NAMES.get(month, month)}**, profil **{profile.upper()}**...")
            try:
                r = run_optimization_with_profile(
                    df=df, params=params, uses=uses,
                    profile_type=profile,
                    custom_hours=custom_hours if profile == 'custom' else None,
                    max_starts_per_month=max_starts_per_month,
                    period_mask=mask,
                    time_limit=60,
                )
            except Exception as exc:
                st.warning(f"⚠️ {MONTH_NAMES.get(month, month)} / {profile.upper()}: výjimka: {exc}")
                r = None
            if r is not None:
                n_hours = int(mask.sum())
                results[month][profile] = {
                    'profit':       r['total_profit'],
                    'profit_per_h': r['total_profit'] / n_hours if n_hours > 0 else 0,
                    'smoothness':   calculate_smoothness_metrics(r['res']),
                    'total_co2':    r.get('total_co2', 0.0),
                }
            run_idx += 1
            progress.progress(run_idx / total_runs)

    progress.empty()
    status.empty()
    return results


def compute_quarterly_strategy(monthly_pr):
    """
    Seskupí měsíce do kvartálů Q1-Q4, najde nejlepší profil per kvartál.
    Vrátí list[dict]: quarter, best_profile, total_profit, avg_co2
    """
    quarters = {
        'Q1 (Led–Bře)': [1, 2, 3],
        'Q2 (Dub–Čvn)': [4, 5, 6],
        'Q3 (Čvc–Zář)': [7, 8, 9],
        'Q4 (Říj–Pro)': [10, 11, 12],
    }
    rows = []
    for q_label, months in quarters.items():
        totals, co2_lists = {}, {}
        for m in months:
            if m not in monthly_pr:
                continue
            for pr, data in monthly_pr[m].items():
                totals[pr] = totals.get(pr, 0.0) + data['profit']
                co2v = data.get('total_co2')
                if co2v is not None:
                    co2_lists.setdefault(pr, []).append(co2v)
        if not totals:
            continue
        best = max(totals, key=lambda p: totals[p])
        co2_avg = sum(co2_lists[best]) / len(co2_lists[best]) if co2_lists.get(best) else None
        rows.append({'quarter': q_label, 'best_profile': best,
                     'total_profit': totals[best], 'avg_co2': co2_avg})
    return rows


# ────────────────────────────────────────────────
# CITLIVOSTNÍ ANALÝZA
# ────────────────────────────────────────────────

def run_sensitivity_analysis(df, params, uses, profile_type, gas_range, ee_range, steps,
                              custom_hours=None):
    """
    Variuje gas_delta a ee_delta symetricky okolo základní varianty.
    Vrátí DataFrame: typ | delta | profit | delta_pct
    """
    base = run_optimization_with_profile(df, params, uses, profile_type,
                                          custom_hours=custom_hours)
    if base is None:
        return None
    base_profit = base['total_profit']
    if abs(base_profit) < 1e-6:
        base_profit = 1.0  # ochrana před dělením nulou

    rows = []
    gas_vals = np.linspace(-gas_range, gas_range, steps)
    ee_vals  = np.linspace(-ee_range,  ee_range,  steps)

    for gd in gas_vals:
        r = run_optimization_with_profile(df, params, uses, profile_type,
                                          gas_delta=gd, custom_hours=custom_hours)
        if r:
            rows.append({'typ': 'Cena plynu', 'delta': round(gd, 2),
                         'profit': r['total_profit'],
                         'delta_pct': (r['total_profit'] - base_profit) / abs(base_profit) * 100})

    for ed in ee_vals:
        r = run_optimization_with_profile(df, params, uses, profile_type,
                                          ee_delta=ed, custom_hours=custom_hours)
        if r:
            rows.append({'typ': 'Cena EE', 'delta': round(ed, 2),
                         'profit': r['total_profit'],
                         'delta_pct': (r['total_profit'] - base_profit) / abs(base_profit) * 100})

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────
# LOKÁLNÍ DATA + SPUŠTĚNÍ OPTIMALIZACE
# ────────────────────────────────────────────────
st.divider()
st.markdown("**Formát lokálních dat:** 1. sloupec = datetime | `Poptávka po teple (MW)` "
            "| `FVE (MW)` jako capacity factor **0–1**.")
loc_file = st.file_uploader("📂 Lokální data (poptávka tepla, FVE profil, ...)", type=["xlsx"])

if st.session_state.fwd_data is not None and loc_file is not None:
    df_loc = pd.read_excel(loc_file)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    df_loc.rename(columns={df_loc.columns[0]: 'datetime'}, inplace=True)
    df_loc['datetime'] = pd.to_datetime(df_loc['datetime'], dayfirst=True)
    df = pd.merge(st.session_state.fwd_data, df_loc, on='datetime', how='inner').fillna(0)
    if use_fve and 'fve_installed_p' in p and 'FVE (MW)' in df.columns:
        df['FVE (MW)'] = df['FVE (MW)'].clip(0, 1) * p['fve_installed_p']
    T = len(df)
    st.info(f"Načteno **{T}** hodin ({df['datetime'].min().date()} → {df['datetime'].max().date()})")

    uses = dict(kgj=use_kgj, boil=use_boil, ek=use_ek, tes=use_tes,
                bess=use_bess, fve=use_fve, ext_heat=use_ext_heat)

    # ════════════════════════════════════════════════
    # SPUŠTĚNÍ ANALÝZY – jedno tlačítko, vše najednou
    # ════════════════════════════════════════════════
    n_runs = len(profiles_to_run) * pd.to_datetime(df['datetime']).dt.month.nunique()
    st.markdown(
        f"Profily: **{', '.join(pr.upper() for pr in profiles_to_run)}** · "
        f"Celé období + měsíční analýza ({n_runs} kombinací)"
    )

    if st.button("🚀 Spustit analýzu všech profilů", type="primary", key="btn_run_all"):
        with st.spinner("⏳ Spouštím optimalizaci pro všechny profily …"):
            _ch = custom_hours if 'custom' in profiles_to_run else None
            _ms = max_starts_per_month if use_month_start_limit else None

            # 1) Porovnání profilů přes celé období
            scenarios = run_scenario_analysis(
                df=df, params=p, uses=uses,
                profiles_to_run=profiles_to_run,
                custom_hours=_ch,
                period_start=period_start, period_end=period_end,
                max_starts_per_month=_ms
            )
            if not scenarios:
                st.error("❌ Žádný profil nebyl úspěšně vypočten. Zkontroluj parametry.")
                st.stop()

            # 2) Měsíční analýza
            monthly_res = run_monthly_profile_analysis(
                df=df, params=p, uses=uses,
                profiles_to_run=profiles_to_run,
                custom_hours=_ch, max_starts_per_month=_ms
            )

        st.session_state.scenario_results = scenarios
        st.session_state.monthly_profile_results = monthly_res
        st.session_state.results = None
        st.session_state.annual_plan_result = None
        st.session_state.df_main = df.copy()
        st.session_state.uses = uses
        for _k in ('_xlsx_scen', '_xlsx_monthly', '_xlsx_sa'):
            st.session_state.pop(_k, None)
        save_cache()
        st.success("✅ Analýza dokončena!")

# ────────────────────────────────────────────────
# MONTHLY PROFILE ANALYSIS VIEW
# ────────────────────────────────────────────────
if st.session_state.monthly_profile_results is not None:
    monthly_pr = st.session_state.monthly_profile_results
    st.divider()
    st.subheader("🗓️ Měsíční Analýza Profilů")

    # Sestavit tabulku a heatmapu
    all_profiles = sorted({pr for m_data in monthly_pr.values() for pr in m_data.keys()})
    months_sorted = sorted(monthly_pr.keys())

    # Tabulka: nejlepší profil per měsíc
    best_rows = []
    for month in months_sorted:
        m_data = monthly_pr[month]
        if not m_data:
            continue
        best_pr = max(m_data, key=lambda pr: m_data[pr]['profit'])
        row_best = {
            'Měsíc':           MONTH_NAMES.get(month, month),
            'Nejlepší profil': best_pr.upper(),
            'Zisk/hod [€]':    f"{m_data[best_pr]['profit_per_h']:,.1f}",
            'Zisk celkem [€]': f"{m_data[best_pr]['profit']:,.0f}",
            'Stabilita [%]':   f"{m_data[best_pr]['smoothness']['stability_score']:.1f}",
        }
        co2_m = m_data[best_pr].get('total_co2')
        if co2_m is not None:
            row_best['CO₂ [tCO₂]'] = f"{co2_m:,.1f}"
        best_rows.append(row_best)

    if best_rows:
        df_best = pd.DataFrame(best_rows)
        st.dataframe(df_best, use_container_width=True, hide_index=True)
        total_opt = sum(
            monthly_pr[m][max(monthly_pr[m], key=lambda pr: monthly_pr[m][pr]['profit'])]['profit']
            for m in months_sorted if monthly_pr[m]
        )
        st.success(f"💡 Celkový potenciál při optimálním výběru profilu per měsíc: **{total_opt:,.0f} €**")

    # Heatmapa: profily × měsíce, hodnota = profit/hod
    heat_z, heat_x, heat_y = [], all_profiles, [MONTH_NAMES.get(m, m) for m in months_sorted]
    for pr in all_profiles:
        row = [monthly_pr[m].get(pr, {}).get('profit_per_h', None) for m in months_sorted]
        heat_z.append(row)

    fig_heat = go.Figure(go.Heatmap(
        z=heat_z, x=heat_y, y=[pr.upper() for pr in heat_x],
        colorscale='RdYlGn', zmid=0,
        colorbar=dict(title='€/hod'),
        hovertemplate='Měsíc: %{x}<br>Profil: %{y}<br>Zisk/hod: %{z:.1f} €<extra></extra>'
    ))
    fig_heat.update_layout(
        height=320, title="Zisk/hod [€] dle profilu a měsíce",
        xaxis_title="Měsíc", yaxis_title="Profil"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Sezónní strategie ──────────────────────────
    st.divider()
    st.subheader("🌍 Sezónní strategie (kvartální doporučení)")
    q_rows = compute_quarterly_strategy(monthly_pr)
    if q_rows:
        tbl = [{'Kvartál': r['quarter'],
                'Nejlepší profil': r['best_profile'].upper(),
                'Celkový zisk [€]': f"{r['total_profit']:,.0f}",
                'Průměr CO₂/měsíc [tCO₂]': f"{r['avg_co2']:,.1f}" if r['avg_co2'] is not None else '–'}
               for r in q_rows]
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
        narrative = " | ".join(
            f"{r['quarter'].split()[0]}: **{r['best_profile'].upper()}**" for r in q_rows)
        st.info(f"Doporučená strategie: {narrative}")

    # ── Download měsíční analýzy ──
    if st.button("📦 Připravit Excel měsíční analýzy ke stažení", key="prep_monthly"):
        with st.spinner("⏳ Generuji Excel …"):
            st.session_state['_xlsx_monthly'] = to_excel_monthly(monthly_pr, MONTH_NAMES)
    if st.session_state.get('_xlsx_monthly') is not None:
        st.download_button(
            label="📥 Stáhnout měsíční analýzu (Excel)",
            data=st.session_state['_xlsx_monthly'],
            file_name="kgj_mesicni_analyza.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_monthly"
        )

    # ── R2: Kombinovaný roční plán ──
    st.divider()
    st.subheader("📋 Kombinovaný roční plán (nejlepší profil / měsíc)")

    st.markdown("**Kvartální override profilu** (volitelné – ponech 'Auto' pro max zisk per měsíc):")
    _qcols = st.columns(4)
    _q_override = {}
    _quarter_months = {
        'Q1 (Led–Bře)': [1, 2, 3],
        'Q2 (Dub–Čvn)': [4, 5, 6],
        'Q3 (Čvc–Zář)': [7, 8, 9],
        'Q4 (Říj–Pro)': [10, 11, 12],
    }
    for _qi, (_qlabel, _qmonths) in enumerate(_quarter_months.items()):
        with _qcols[_qi]:
            _choice = st.selectbox(
                _qlabel,
                ['Auto (max zisk)'] + [_pr.upper() for _pr in profiles_to_run],
                key=f"q_override_{_qi}"
            )
            if _choice != 'Auto (max zisk)':
                for _m in _qmonths:
                    _q_override[_m] = _choice.lower()

    if st.button("📋 Sestavit roční plán", key="btn_annual_plan"):
        with st.spinner("⏳ Sestavuji roční plán …"):
            df_annual_src = st.session_state.df_main
            combined_frames = []
            months_plan = sorted(monthly_pr.keys())
            prog_annual = st.progress(0)
            for idx, month in enumerate(months_plan):
                m_data = monthly_pr[month]
                if not m_data:
                    continue
                _override = _q_override.get(month)
                if _override and _override in m_data:
                    best_pr = _override
                else:
                    best_pr = max(m_data, key=lambda pr: m_data[pr]['profit'])
                mask = pd.to_datetime(df_annual_src['datetime']).dt.month == month
                r = run_optimization_with_profile(
                    df=df_annual_src, params=p, uses=uses,
                    profile_type=best_pr,
                    custom_hours=custom_hours if best_pr == 'custom' else None,
                    period_mask=mask
                )
                if r is not None:
                    frame = r['res'].copy()
                    frame['_best_profile'] = best_pr.upper()
                    combined_frames.append(frame)
                prog_annual.progress((idx + 1) / len(months_plan))
            prog_annual.empty()
            if combined_frames:
                res_annual = pd.concat(combined_frames, ignore_index=True)
                st.session_state['annual_plan_result'] = res_annual
                save_cache()
                st.success(f"✅ Roční plán sestaven – {len(res_annual):,} hodin, zisk: {res_annual['Hodinový zisk [€]'].sum():,.0f} €")
            else:
                st.error("❌ Nepodařilo se sestavit roční plán.")

    if st.session_state.get('annual_plan_result') is not None:
        res_ap = st.session_state['annual_plan_result']
        total_ap = res_ap['Hodinový zisk [€]'].sum()
        st.info(f"Celkový zisk kombinovaného plánu: **{total_ap:,.0f} €**")

        # ── Barevný kalendář ročního plánu ───────────────────────────
        if '_best_profile' in res_ap.columns:
            st.markdown("#### 📅 Profil per měsíc")
            res_ap['_month'] = pd.to_datetime(res_ap['Čas']).dt.month
            month_summary = (
                res_ap.groupby('_month')
                .agg(profil=('_best_profile', 'first'),
                     zisk=('Hodinový zisk [€]', 'sum'),
                     hodiny_kgj=('KGJ on', 'sum'))
                .reset_index()
            )
            MONTH_LABELS = {1:'Led',2:'Úno',3:'Bře',4:'Dub',5:'Kvě',6:'Čvn',
                            7:'Čvc',8:'Srp',9:'Zář',10:'Říj',11:'Lis',12:'Pro'}
            cal_cols = st.columns(12)
            for _, mrow in month_summary.iterrows():
                m = int(mrow['_month'])
                pr = mrow['profil'].lower()
                color = PROFILE_COLORS.get(pr, '#888')
                zisk_k = mrow['zisk'] / 1000
                cal_cols[m - 1].markdown(
                    f"""<div style="background:{color};border-radius:10px;padding:10px 4px;
                    text-align:center;margin:2px;">
                    <div style="color:white;font-weight:700;font-size:0.85rem">{MONTH_LABELS[m]}</div>
                    <div style="color:white;font-size:0.7rem;opacity:0.9">{mrow['profil']}</div>
                    <div style="color:white;font-size:0.75rem;font-weight:600">{zisk_k:+.1f} k€</div>
                    </div>""",
                    unsafe_allow_html=True
                )

        # Graf – tepelné pokrytí
        st.markdown("#### 🔥 Pokrytí tepelné poptávky (kombinovaný plán)")
        fig_ap = go.Figure()
        for col, name, color in [
            ('KGJ [MW_th]',          'KGJ',         '#27ae60'),
            ('Kotel [MW_th]',        'Kotel',        '#3498db'),
            ('Elektrokotel [MW_th]', 'Elektrokotel', '#9b59b6'),
            ('Import tepla [MW_th]', 'Import tepla', '#e74c3c'),
        ]:
            if col in res_ap.columns:
                fig_ap.add_trace(go.Scatter(x=res_ap['Čas'], y=res_ap[col].clip(lower=0),
                    name=name, stackgroup='teplo', fillcolor=color, line_width=0))
        fig_ap.add_trace(go.Scatter(x=res_ap['Čas'], y=res_ap['Poptávka tepla [MW]'] * p['h_cover'],
            name='Cílová poptávka', mode='lines', line=dict(color='black', width=2, dash='dot')))
        fig_ap.update_layout(height=420, hovermode='x unified')
        st.plotly_chart(fig_ap, use_container_width=True)

        # Graf – kumulativní zisk
        st.markdown("#### 💰 Kumulativní zisk (kombinovaný plán)")
        fig_ap2 = go.Figure()
        fig_ap2.add_trace(go.Scatter(
            x=res_ap['Čas'], y=res_ap['Hodinový zisk [€]'].cumsum(),
            fill='tozeroy', fillcolor='rgba(39,174,96,0.2)',
            line_color='#27ae60', name='Kum. zisk'
        ))
        fig_ap2.update_layout(height=300, hovermode='x unified')
        st.plotly_chart(fig_ap2, use_container_width=True)

        # Download kombinovaného plánu
        skip_cols_ap = {'Měsíc', 'Hodina dne', 'KGJ on', 'Kotel on', 'Import tepla on'}
        df_ap_exp = res_ap[[c for c in res_ap.columns if c not in skip_cols_ap]].round(4)
        buf_ap = io.BytesIO()
        with pd.ExcelWriter(buf_ap, engine='xlsxwriter') as writer_ap:
            workbook_ap = writer_ap.book
            hdr_ap, num_ap, txt_ap, _ = _wb_formats(workbook_ap)
            _write_sheet(writer_ap, df_ap_exp, 'Kombinovaný plán', hdr_ap, num_ap, txt_ap)
        st.download_button(
            label="📥 Stáhnout kombinovaný roční plán (Excel)",
            data=buf_ap.getvalue(),
            file_name="kgj_rocni_plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_annual_plan"
        )


# ────────────────────────────────────────────────
# SCENARIO COMPARISON VIEW
# ────────────────────────────────────────────────
if st.session_state.scenario_results is not None:
    scenarios = st.session_state.scenario_results

    st.divider()
    st.subheader("📋 Porovnání Scénářů (Scenario Comparison)")

    # ── KPI karty – vítěz a přehled ──────────────────────────────────
    valid_scenarios = {k: v for k, v in scenarios.items() if v['result'] is not None}
    if valid_scenarios:
        best_profile = max(valid_scenarios, key=lambda k: valid_scenarios[k]['result']['total_profit'])
        worst_profile = min(valid_scenarios, key=lambda k: valid_scenarios[k]['result']['total_profit'])
        best_profit = valid_scenarios[best_profile]['result']['total_profit']
        worst_profit = valid_scenarios[worst_profile]['result']['total_profit']
        profit_spread = best_profit - worst_profit
        best_util = valid_scenarios[best_profile]['smoothness']['utilization_pct']
        best_stab = valid_scenarios[best_profile]['smoothness']['stability_score']

        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.metric("Nejlepší profil", best_profile.upper(),
                      f"zisk {best_profit:,.0f} €")
        with kpi_cols[1]:
            st.metric("Potenciální zisk navíc", f"{profit_spread:,.0f} €",
                      f"vs. {worst_profile.upper()}",
                      delta_color="normal")
        with kpi_cols[2]:
            st.metric("Využití KGJ (vítěz)", f"{best_util:.1f} %")
        with kpi_cols[3]:
            st.metric("Stabilita (vítěz)", f"{best_stab:.1f} %")
        st.markdown("")

    # Comparison Table
    comparison_df = create_scenario_comparison_df(scenarios)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Metric Comparison Charts
    col_chart_1, col_chart_2 = st.columns(2)
    
    with col_chart_1:
        st.markdown("**Zisk vs. Stabilita**")
        chart_data = []
        for profile, scenario in scenarios.items():
            if scenario['result'] is not None:
                profit = scenario['result']['total_profit']
                stability = scenario['smoothness']['stability_score']
                chart_data.append({
                    'Profil': profile.upper(),
                    'Zisk [k€]': profit / 1000,
                    'Stabilita [%]': stability
                })
        
        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            bar_colors = [PROFILE_COLORS.get(p.lower(), '#888') for p in df_chart['Profil']]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_chart['Profil'],
                y=df_chart['Zisk [k€]'],
                name='Zisk [k€]',
                marker_color=bar_colors,
                marker_line_color='rgba(255,255,255,0.3)',
                marker_line_width=1,
                yaxis='y1'
            ))
            fig.add_trace(go.Scatter(
                x=df_chart['Profil'],
                y=df_chart['Stabilita [%]'],
                name='Stabilita [%]',
                line=dict(color='#3498db', width=3),
                marker=dict(size=10),
                yaxis='y2'
            ))
            fig.update_layout(
                height=380,
                yaxis=dict(title="Zisk [k€]", side='left'),
                yaxis2=dict(title="Stabilita [%]", overlaying='y', side='right'),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col_chart_2:
        st.markdown("**Charakteristiky Provozu**")
        chart_data2 = []
        for profile, scenario in scenarios.items():
            if scenario['result'] is not None:
                smooth = scenario['smoothness']
                chart_data2.append({
                    'Profil': profile.upper(),
                    'Transitions': smooth['transitions'],
                    'Avg Runtime [h]': smooth['avg_run_hours']
                })
        
        if chart_data2:
            df_chart2 = pd.DataFrame(chart_data2)
            bar_colors2 = [PROFILE_COLORS.get(p.lower(), '#888') for p in df_chart2['Profil']]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=df_chart2['Profil'],
                y=df_chart2['Transitions'],
                name='Přechodů ON↔OFF',
                marker_color=bar_colors2,
                marker_line_color='rgba(255,255,255,0.3)',
                marker_line_width=1,
                opacity=0.85,
                yaxis='y1'
            ))
            fig2.add_trace(go.Scatter(
                x=df_chart2['Profil'],
                y=df_chart2['Avg Runtime [h]'],
                name='Avg Runtime [h]',
                line=dict(color='#f39c12', width=3),
                marker=dict(size=10),
                yaxis='y2'
            ))
            fig2.update_layout(
                height=380,
                yaxis=dict(title="Přechodů", side='left'),
                yaxis2=dict(title="Avg Runtime [h]", overlaying='y', side='right'),
                hovermode='x unified'
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Breakdown příjmů a nákladů ──
    st.markdown("**Breakdown příjmů a nákladů dle profilu**")
    breakdown_data = []
    for profile, scenario in scenarios.items():
        if scenario['result'] is None:
            continue
        r = scenario['result']['res']
        breakdown_data.append({
            'Profil':           profile.upper(),
            'Rev teplo':        r['Rev teplo [€]'].sum() / 1000,
            'Rev EE':           r['Rev EE [€]'].sum() / 1000,
            'Nákl plyn':       -(r['Nákl plyn KGJ [€]'].sum() + r['Nákl plyn kotel [€]'].sum()) / 1000,
            'Nákl EE':         -(r['Nákl EE import [€]'].sum() + r['Nákl EE EK [€]'].sum()) / 1000,
            'Nákl imp tepla':  -(r['Nákl imp tepla [€]'].sum()) / 1000,
            'Nákl starty/BESS':-(r['Nákl starty [€]'].sum() + r['Nákl BESS [€]'].sum()) / 1000,
            'Nákl servis KGJ':-(r['Nákl servis KGJ [€]'].sum()) / 1000,
        })

    if breakdown_data:
        df_bd = pd.DataFrame(breakdown_data)
        fig_bd = go.Figure()
        colors_pos = ['#27ae60', '#2ecc71']
        colors_neg = ['#e74c3c', '#c0392b', '#e67e22', '#95a5a6', '#7f8c8d']
        pos_cols = ['Rev teplo', 'Rev EE']
        neg_cols = ['Nákl plyn', 'Nákl EE', 'Nákl imp tepla', 'Nákl starty/BESS', 'Nákl servis KGJ']
        for col, color in zip(pos_cols, colors_pos):
            fig_bd.add_trace(go.Bar(name=col, x=df_bd['Profil'], y=df_bd[col],
                                    marker_color=color))
        for col, color in zip(neg_cols, colors_neg):
            fig_bd.add_trace(go.Bar(name=col, x=df_bd['Profil'], y=df_bd[col],
                                    marker_color=color))
        fig_bd.update_layout(
            barmode='relative', height=420,
            title="Příjmy (kladné) a náklady (záporné) dle profilu [k€]",
            yaxis_title="k€", hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_bd, use_container_width=True)

    # ── Waterfall – rozkad zisku per profil ──────────────────────────
    if breakdown_data:
        st.markdown("**Waterfall – rozkad čistého zisku dle profilu**")
        wf_cols = st.columns(len(breakdown_data))
        for col_idx, row in enumerate(breakdown_data):
            pr_key = row['Profil'].lower()
            pr_color = PROFILE_COLORS.get(pr_key, '#888')
            items = {
                'Teplo': row['Rev teplo'],
                'Elektřina': row['Rev EE'],
                'Plyn': row['Nákl plyn'],
                'EE nákup': row['Nákl EE'],
                'Imp. teplo': row['Nákl imp tepla'],
                'Starty/BESS': row['Nákl starty/BESS'],
                'Servis KGJ': row['Nákl servis KGJ'],
            }
            labels = [''] + list(items.keys()) + ['Zisk']
            values = [0.0] + list(items.values()) + [0.0]
            measures = ['absolute'] + ['relative'] * len(items) + ['total']
            fig_wf = go.Figure(go.Waterfall(
                orientation='v',
                measure=measures,
                x=labels,
                y=values,
                connector=dict(line=dict(color='rgba(255,255,255,0.2)', width=1)),
                increasing=dict(marker_color='#4CAF50'),
                decreasing=dict(marker_color='#F44336'),
                totals=dict(marker_color=pr_color),
                texttemplate='%{y:.1f}',
                textposition='outside',
            ))
            fig_wf.update_layout(
                height=340,
                title=dict(text=f"<b>{row['Profil']}</b>", font=dict(color=pr_color, size=14)),
                yaxis_title="k€",
                showlegend=False,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            with wf_cols[col_idx]:
                st.plotly_chart(fig_wf, use_container_width=True, key=f"wf_{row['Profil'].lower()}")

    # ── Detailní view – záložka per profil ──────────────────────────
    st.divider()
    st.subheader("🔍 Detailní analýza per profil")

    _tab_prs = [pr for pr in profiles_to_run if pr in scenarios and scenarios[pr]['result'] is not None]
    if _tab_prs:
        _dtabs = st.tabs([pr.upper() for pr in _tab_prs])
        for _dtab, pr in zip(_dtabs, _tab_prs):
            with _dtab:
                _sc     = scenarios[pr]
                result  = _sc['result']
                res     = result['res']
                smoothness = _sc['smoothness']

                col_sm_1, col_sm_2, col_sm_3, col_sm_4, col_sm_5, col_sm_6 = st.columns(6)
                with col_sm_1:
                    st.metric("Využití KGJ", f"{smoothness['utilization_pct']:.1f}%",
                              help="Podíl hodin, kdy KGJ běželo")
                with col_sm_2:
                    st.metric("Stabilita", f"{smoothness['stability_score']:.1f}%",
                              help="100% = velmi hladký provoz")
                with col_sm_3:
                    st.metric("Přechodů", f"{smoothness['transitions']}",
                              help="Počet start-stop cyklů")
                with col_sm_4:
                    st.metric("Avg Runtime", f"{smoothness['avg_run_hours']:.1f} h")
                with col_sm_5:
                    st.metric("Min Runtime", f"{smoothness['min_run_hours']:.0f} h")
                with col_sm_6:
                    st.metric("Max Runtime", f"{smoothness['max_run_hours']:.0f} h")

                total_profit    = result['total_profit']
                total_shortfall = res['Shortfall [MW]'].sum()
                target_heat     = (res['Poptávka tepla [MW]'] * p['h_cover']).sum()
                coverage        = 100*(1 - total_shortfall/target_heat) if target_heat > 0 else 100.0
                total_ee_gen    = res['EE z KGJ [MW]'].sum() + res['EE z FVE [MW]'].sum()
                kgj_hours       = int(res['KGJ on'].sum())
                rev_teplo_total = res['Rev teplo [€]'].sum()
                rev_ee_total    = res['Rev EE [€]'].sum()
                c_gas_total     = res['Nákl plyn KGJ [€]'].sum() + res['Nákl plyn kotel [€]'].sum()
                c_ee_total      = res['Nákl EE import [€]'].sum() + res['Nákl EE EK [€]'].sum()
                c_imp_total     = res['Nákl imp tepla [€]'].sum()
                c_other_total   = (res['Nákl starty [€]'].sum() + res['Nákl BESS [€]'].sum()
                                   + res['Nákl servis KGJ [€]'].sum())

                st.markdown("#### 📊 Klíčové Metriky")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Celkový zisk", f"{total_profit:,.0f} €")
                m2.metric("Shortfall", f"{total_shortfall:,.1f} MWh")
                m3.metric("Pokrytí poptávky", f"{coverage:.1f} %")
                m4.metric("Export EE", f"{res['EE export [MW]'].sum():,.1f} MWh")
                m5.metric("Výroba EE", f"{total_ee_gen:,.1f} MWh")
                m6.metric("Provozní hodiny KGJ", f"{kgj_hours:,} h")

                st.markdown("#### 💰 Rozpad Zisku")
                r1, r2, r3, r4, r5, r6 = st.columns(6)
                r1.metric("🔥 Příjmy teplo", f"{rev_teplo_total:,.0f} €")
                r2.metric("⚡ Příjmy EE", f"{rev_ee_total:,.0f} €")
                r3.metric("🔴 Nákl plyn", f"{c_gas_total:,.0f} €")
                r4.metric("🔴 Nákl EE", f"{c_ee_total:,.0f} €")
                r5.metric("🔴 Nákl import", f"{c_imp_total:,.0f} €")
                r6.metric("🔴 Ostatní", f"{c_other_total:,.0f} €")

                if 'CO₂ Celkem [tCO₂]' in res.columns:
                    st.markdown("#### 🌿 Emise CO₂")
                    co_1, co_2, co_3, co_4 = st.columns(4)
                    co_1.metric("CO₂ celkem", f"{res['CO₂ Celkem [tCO₂]'].sum():,.1f} t")
                    co_2.metric("CO₂ KGJ", f"{res['CO₂ KGJ [tCO₂]'].sum():,.1f} t")
                    co_3.metric("CO₂ Kotel", f"{res['CO₂ Kotel [tCO₂]'].sum():,.1f} t")
                    co_4.metric("CO₂ Síť (netto)", f"{res['CO₂ Síť [tCO₂]'].sum():,.1f} t")

                st.markdown("#### 🔥 Pokrytí Tepelné Poptávky")
                fig = go.Figure()
                for col, name, color in [
                    ('KGJ [MW_th]',          'KGJ',         '#27ae60'),
                    ('Kotel [MW_th]',        'Kotel',        '#3498db'),
                    ('Elektrokotel [MW_th]', 'Elektrokotel', '#9b59b6'),
                    ('Import tepla [MW_th]', 'Import tepla', '#e74c3c'),
                    ('TES netto [MW_th]',    'TES netto',    '#f39c12'),
                ]:
                    if col in res.columns:
                        fig.add_trace(go.Scatter(x=res['Čas'], y=res[col].clip(lower=0),
                            name=name, stackgroup='teplo', fillcolor=color, line_width=0))
                fig.add_trace(go.Scatter(x=res['Čas'], y=res['Shortfall [MW]'],
                    name='Nedodáno ⚠️', stackgroup='teplo', fillcolor='rgba(200,0,0,0.45)', line_width=0))
                fig.add_trace(go.Scatter(x=res['Čas'], y=res['Poptávka tepla [MW]']*p['h_cover'],
                    name='Cílová poptávka', mode='lines', line=dict(color='black', width=2, dash='dot')))
                fig.update_layout(height=450, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True, key=f"teplo_{pr}")

                st.markdown("#### ⚡ Bilance Elektřiny")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    row_heights=[0.5, 0.5], subplot_titles=("Zdroje EE [MW]", "Spotřeba / export EE [MW]"))
                for col, name, color in [
                    ('EE z KGJ [MW]',      'KGJ',       '#2ecc71'),
                    ('EE z FVE [MW]',      'FVE',        '#f1c40f'),
                    ('EE import [MW]',     'Import EE',  '#2980b9'),
                    ('BESS vybíjení [MW]', 'BESS výdej', '#8e44ad'),
                ]:
                    if col in res.columns:
                        fig.add_trace(go.Scatter(x=res['Čas'], y=res[col], name=name,
                            stackgroup='vyroba', fillcolor=color), row=1, col=1)
                for col, name, color in [
                    ('EE do EK [MW]',      'EK',            '#e74c3c'),
                    ('BESS nabíjení [MW]', 'BESS nabíjení', '#34495e'),
                    ('EE export [MW]',     'Export EE',     '#16a085'),
                ]:
                    if col in res.columns:
                        fig.add_trace(go.Scatter(x=res['Čas'], y=-res[col], name=name,
                            stackgroup='spotreba', fillcolor=color), row=2, col=1)
                fig.update_layout(height=600, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True, key=f"ee_{pr}")

                st.markdown("#### 💰 Kumulativní Zisk v Čase")
                _pr_color = PROFILE_COLORS.get(pr, '#27ae60')
                _pr_hex   = _pr_color.replace('#', '')
                _pr_r, _pr_g, _pr_b = int(_pr_hex[0:2],16), int(_pr_hex[2:4],16), int(_pr_hex[4:6],16)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res['Čas'], y=res['Kumulativní zisk [€]'],
                    fill='tozeroy', fillcolor=f'rgba({_pr_r},{_pr_g},{_pr_b},0.2)',
                    line_color=_pr_color, name='Kum. zisk'))
                fig.update_layout(height=350, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True, key=f"kum_{pr}")

    # ── Download scénářů ──
    st.divider()
    if st.button("📦 Připravit Excel scénářů ke stažení", key="prep_scenarios"):
        with st.spinner("⏳ Generuji Excel …"):
            st.session_state['_xlsx_scen'] = to_excel_scenarios(scenarios)
    if st.session_state.get('_xlsx_scen') is not None:
        st.download_button(
            label="📥 Stáhnout scénáře (Excel)",
            data=st.session_state['_xlsx_scen'],
            file_name="kgj_scenare.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_scenarios"
        )

# ─────────────────────────────────────────────
# CITLIVOSTNÍ ANALÝZA (standalone)
# ─────────────────────────────────────────────
if st.session_state.scenario_results is not None:
    _sa_df_main = st.session_state.df_main
    _sa_uses    = st.session_state.get('uses', uses)

    st.divider()
    st.subheader("📊 Citlivostní analýza")
    st.markdown("Zobrazí, jak se změní zisk při změně tržní ceny plynu nebo elektřiny.")

    col_sa1, col_sa2, col_sa3 = st.columns(3)
    with col_sa1:
        sa_gas_range = st.slider("Rozsah ceny plynu [±€/MWh]", 5, 50, 20, step=5,
                                 key="sa_gas_range")
    with col_sa2:
        sa_ee_range  = st.slider("Rozsah ceny EE [±€/MWh]", 5, 50, 20, step=5,
                                 key="sa_ee_range")
    with col_sa3:
        sa_steps = st.selectbox("Počet kroků:", [3, 5, 7, 9], index=1, key="sa_steps")

    available_profiles = list(st.session_state.scenario_results.keys())
    sa_profile = st.selectbox(
        "Profil pro analýzu:",
        options=available_profiles,
        format_func=lambda x: x.upper(),
        key="sa_profile"
    )

    if st.button("📊 Spustit citlivostní analýzu", key="btn_sensitivity"):
        with st.spinner("⏳ Probíhá citlivostní analýza …"):
            sa_df = run_sensitivity_analysis(
                df=_sa_df_main, params=p, uses=_sa_uses,
                profile_type=sa_profile,
                gas_range=sa_gas_range, ee_range=sa_ee_range, steps=sa_steps,
                custom_hours=custom_hours if sa_profile == 'custom' else None
            )
        if sa_df is None:
            st.error("❌ Citlivostní analýza selhala.")
        else:
            st.session_state.sensitivity_results = sa_df.to_dict('records')
            save_cache()
            st.success("✅ Hotovo!")

    if st.session_state.sensitivity_results:
        sa_df = pd.DataFrame(st.session_state.sensitivity_results)

        # Tornado chart – min/max per parametr
        tornado_rows = []
        for typ, grp in sa_df.groupby('typ'):
            tornado_rows.append({
                'Parametr': typ,
                'Min zisk [k€]': grp['profit'].min() / 1000,
                'Max zisk [k€]': grp['profit'].max() / 1000,
            })
        df_tornado = pd.DataFrame(tornado_rows)

        fig_t = go.Figure()
        for _, row in df_tornado.iterrows():
            fig_t.add_trace(go.Bar(
                y=[row['Parametr']],
                x=[row['Max zisk [k€]'] - row['Min zisk [k€]']],
                base=row['Min zisk [k€]'],
                orientation='h',
                name=row['Parametr'],
                text=f"{row['Min zisk [k€]']:.1f} → {row['Max zisk [k€]']:.1f} k€",
                textposition='inside'
            ))
        fig_t.update_layout(
            height=250, title="Tornádo chart – rozsah zisku dle cenové změny",
            xaxis_title="Zisk [k€]", showlegend=False, bargap=0.4
        )
        st.plotly_chart(fig_t, use_container_width=True)

        # Detailní tabulka
        st.dataframe(
            sa_df[['typ', 'delta', 'profit', 'delta_pct']].rename(columns={
                'typ': 'Parametr', 'delta': 'Δ cena [€/MWh]',
                'profit': 'Zisk [€]', 'delta_pct': 'Změna [%]'
            }).round(2),
            use_container_width=True, hide_index=True
        )

        if st.button("📦 Připravit Excel citlivostní analýzy ke stažení", key="prep_sensitivity"):
            with st.spinner("⏳ Generuji Excel …"):
                st.session_state['_xlsx_sa'] = to_excel_sensitivity(
                    sa_df=sa_df,
                    profile_name=sa_profile,
                    gas_range=sa_gas_range,
                    ee_range=sa_ee_range,
                    steps=sa_steps
                )
        if st.session_state.get('_xlsx_sa') is not None:
            st.download_button(
                label="📥 Stáhnout citlivostní analýzu (Excel)",
                data=st.session_state['_xlsx_sa'],
                file_name="kgj_citlivostni_analyza.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_sensitivity"
            )
