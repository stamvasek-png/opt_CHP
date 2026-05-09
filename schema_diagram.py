import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(24, 15))
ax.set_xlim(0, 24)
ax.set_ylim(0, 15)
ax.axis('off')
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# ── colour palette ────────────────────────────────────────────────────────
BG       = '#0d1117'
CARD     = '#161b22'
BORDER   = '#30363d'
C_GAS    = '#f97316'
C_GAS_D  = '#431407'
C_EL     = '#3b82f6'
C_EL_D   = '#172554'
C_HEAT   = '#ef4444'
C_HEAT_D = '#450a0a'
C_KGJ    = '#10b981'
C_KGJ_D  = '#022c22'
C_TES    = '#8b5cf6'
C_TES_D  = '#2e1065'
C_EK     = '#06b6d4'
C_EK_D   = '#083344'
C_BOIL   = '#f59e0b'
C_BOIL_D = '#451a03'
C_IMP    = '#6b7280'
C_IMP_D  = '#1f2937'
C_BUS    = '#1e40af'
C_BUS_D  = '#172554'
WHITE    = '#f0f6fc'
MUTED    = '#8b949e'
DIM      = '#484f58'

# ── helpers ───────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=1.8, alpha=1.0, r=0.3):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       alpha=alpha, zorder=3)
    ax.add_patch(p)

def t(x, y, s, sz=9, c=WHITE, bold=False, ha='center', va='center', z=5):
    ax.text(x, y, s, fontsize=sz, color=c, ha=ha, va=va,
            fontweight='bold' if bold else 'normal', zorder=z,
            fontfamily='DejaVu Sans')

def hr(x1, x2, y, c=BORDER):
    ax.plot([x1, x2], [y, y], color=c, lw=0.7, zorder=2)

def arr(x1, y1, x2, y2, c, lw=2.2, style='arc3,rad=0.0', lbl='', lc=None, shrA=5, shrB=5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=lw,
                                mutation_scale=14,
                                connectionstyle=style,
                                shrinkA=shrA, shrinkB=shrB), zorder=6)
    if lbl:
        mx, my = (x1+x2)/2, (y1+y2)/2
        lcolor = lc if lc else c
        ax.text(mx, my+0.18, lbl, fontsize=7.5, color=lcolor,
                ha='center', va='bottom', zorder=7,
                bbox=dict(facecolor=BG, edgecolor='none', pad=1.2, alpha=0.85))

def badge(x, y, txt, fc, ec, sz=7.5):
    ax.text(x, y, txt, fontsize=sz, color=WHITE, ha='center', va='center',
            zorder=8, fontweight='bold',
            bbox=dict(facecolor=fc, edgecolor=ec, pad=2.5,
                      boxstyle='round,pad=0.3'))

# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════
rbox(0.3, 14.2, 23.4, 0.65, '#0f1923', '#1d4ed8', lw=2.0)
t(12, 14.52, 'KGJ Strategy & Dispatch Optimizer  —  Schema zdrojov, toku energii a cen',
  sz=13, bold=True, c='#93c5fd')

# ══════════════════════════════════════════════════════════════════════════
# ROW 1  —  VSTUPNI PALIVA  (y 11.0 – 13.9)
# ══════════════════════════════════════════════════════════════════════════

# --- ZEMNI PLYN ---
rbox(0.3, 11.0, 5.3, 2.9, C_GAS_D, C_GAS, lw=2.2)
t(2.95, 13.5, 'ZEMNI PLYN', sz=11, bold=True, c=C_GAS)
hr(0.55, 5.35, 13.1, C_GAS)
t(2.95, 12.75, 'Spotova cena (vstup):  variabilni  [€/MWh]', sz=8.5, c=WHITE)
t(2.95, 12.35, 'Fixni cena KGJ:    40 €/MWh', sz=8.5, c='#fdba74')
t(2.95, 11.95, 'Fixni cena Kotel:  40 €/MWh', sz=8.5, c='#fdba74')
t(2.95, 11.55, 'Distribuce plynu:  +5 €/MWh', sz=8.5, c=MUTED)
t(2.95, 11.15, 'CO2 faktor:  0.202 tCO2/MWh', sz=8.0, c=DIM)

# --- ELEKTRINA ---
rbox(6.3, 11.0, 5.3, 2.9, C_EL_D, C_EL, lw=2.2)
t(8.95, 13.5, 'ELEKTRINA  (GRID)', sz=11, bold=True, c='#93c5fd')
hr(6.55, 11.35, 13.1, C_EL)
t(8.95, 12.75, 'Spotova cena (vstup):  variabilni  [€/MWh]', sz=8.5, c=WHITE)
t(8.95, 12.35, 'Distribuce NAKUP:   +33 €/MWh', sz=8.5, c='#7dd3fc')
t(8.95, 11.95, 'Distribuce PRODEJ:   -2 €/MWh', sz=8.5, c='#86efac')
t(8.95, 11.55, 'Interne spotrebovana EE:', sz=8.5, c=MUTED)
t(8.95, 11.15, '→ distribuce usetrena', sz=8.0, c='#86efac')

# --- IMPORT TEPLA ---
rbox(12.3, 11.0, 5.0, 2.9, C_IMP_D, C_IMP, lw=2.2)
t(14.8, 13.5, 'IMPORT TEPLA  (zalohovy)', sz=11, bold=True, c='#d1d5db')
hr(12.55, 17.05, 13.1, C_IMP)
t(14.8, 12.75, 'Cena importu:  150 €/MWh', sz=8.5, c=WHITE)
t(14.8, 12.35, 'Max. vykon:    2.0 MW_th', sz=8.5, c='#d1d5db')
t(14.8, 11.95, 'Limit:  2 000 h/rok (volitelne)', sz=8.5, c=MUTED)
t(14.8, 11.55, 'Binarni promenna (On/Off)', sz=8.5, c=MUTED)
t(14.8, 11.15, 'Zalohovy zdroj — vysoka cena', sz=8.0, c=DIM)

# ══════════════════════════════════════════════════════════════════════════
# ARROWS  row1 → row2
# ══════════════════════════════════════════════════════════════════════════
arr(1.7, 11.0, 1.7, 10.1, C_GAS, lbl='plyn')
arr(3.5, 11.0, 7.5, 10.1, C_GAS, lbl='plyn', style='arc3,rad=-0.15')
arr(8.7, 11.0, 13.2, 10.1, C_EL, lbl='EE', style='arc3,rad=-0.1')
arr(14.8, 11.0, 14.8, 10.1, C_IMP, lbl='teplo')

# ══════════════════════════════════════════════════════════════════════════
# ROW 2  —  KONVERZNI ZDROJE  (y 6.8 – 10.0)
# ══════════════════════════════════════════════════════════════════════════

# --- KGJ ---
rbox(0.3, 6.8, 5.4, 3.1, C_KGJ_D, C_KGJ, lw=2.4)
t(3.0, 9.55, 'KGJ  —  Kogenerace', sz=11.5, bold=True, c='#6ee7b7')
hr(0.55, 5.45, 9.18, '#065f46')

# left col
t(0.65, 8.90, 'Tepelny vykon:', sz=7.5, c=MUTED, ha='left')
t(0.65, 8.60, '0.605 MW_th  (jm.)', sz=9, c=WHITE, ha='left', bold=True)
t(0.65, 8.25, 'Min. zatizeni:  50%  =  0.303 MW_th', sz=8.0, c='#6ee7b7', ha='left')
t(0.65, 7.95, 'Min. doba behu:  4 hod', sz=8.0, c=MUTED, ha='left')
hr(0.55, 5.45, 7.7, '#065f46')
t(0.65, 7.47, 'Elektr. vykon (odvozeno):', sz=7.5, c=MUTED, ha='left')
t(0.65, 7.17, '~0.45 MW_el', sz=9, c='#93c5fd', ha='left', bold=True)
hr(0.55, 5.45, 6.97, '#065f46')
t(0.65, 6.84, 'eta_th = 0.531   eta_el = 0.395   eta_cel = 0.926', sz=8.0, c='#6ee7b7', ha='left')

# right col — economic
rbox(3.35, 7.0, 2.2, 1.9, '#022c22', '#065f46', lw=1.0, r=0.2)
t(4.45, 8.55, 'Naklady', sz=7.5, c=MUTED)
t(4.45, 8.25, 'Plyn: 45 €/MWh', sz=8.0, c='#fdba74')
t(4.45, 7.95, 'Servis: 14 €/h', sz=8.0, c='#fbbf24')
t(4.45, 7.65, 'Start: 150 €/st', sz=8.0, c='#fbbf24')
t(4.45, 7.30, 'Max: 6 000 h/rok', sz=7.5, c=MUTED)

# --- PLYNOVY KOTEL ---
rbox(6.3, 6.8, 5.4, 3.1, C_BOIL_D, C_BOIL, lw=2.4)
t(9.0, 9.55, 'PLYNOVY KOTEL', sz=11.5, bold=True, c='#fcd34d')
hr(6.55, 11.45, 9.18, '#78350f')
t(6.65, 8.90, 'Max. vykon:', sz=7.5, c=MUTED, ha='left')
t(6.65, 8.60, '4.44 MW_th', sz=10, c=WHITE, ha='left', bold=True)
t(6.65, 8.25, 'Min. vykon:  0  (cont.)', sz=8.0, c=MUTED, ha='left')
t(6.65, 7.95, 'Binarni: On/Off (on_boil)', sz=8.0, c='#fcd34d', ha='left')
hr(6.55, 11.45, 7.7, '#78350f')
t(6.65, 7.47, 'Ucinnost:', sz=7.5, c=MUTED, ha='left')
t(6.65, 7.17, 'eta = 0.86', sz=9, c='#fcd34d', ha='left', bold=True)
hr(6.55, 11.45, 6.97, '#78350f')
t(6.65, 6.84, 'Plyn: 45 €/MWh  |  Max: 4 000 h/rok (opt.)', sz=8.0, c='#fdba74', ha='left')

# --- ELEKTROKOTEL ---
rbox(12.3, 6.8, 5.0, 3.1, C_EK_D, C_EK, lw=2.4)
t(14.8, 9.55, 'ELEKTROKOTEL  (EK)', sz=11.5, bold=True, c='#67e8f9')
hr(12.55, 17.05, 9.18, '#164e63')
t(12.65, 8.90, 'Max. vykon:', sz=7.5, c=MUTED, ha='left')
t(12.65, 8.60, '0.40 MW_th', sz=10, c=WHITE, ha='left', bold=True)
t(12.65, 8.25, 'Elektr. vstup: 0.40/0.99 = 0.404 MW', sz=8.0, c='#93c5fd', ha='left')
t(12.65, 7.95, 'Kontinualni promenna (q_EK)', sz=8.0, c='#67e8f9', ha='left')
hr(12.55, 17.05, 7.7, '#164e63')
t(12.65, 7.47, 'Ucinnost:', sz=7.5, c=MUTED, ha='left')
t(12.65, 7.17, 'eta = 0.99', sz=9, c='#67e8f9', ha='left', bold=True)
hr(12.55, 17.05, 6.97, '#164e63')
t(12.65, 6.84, 'EE: spot + 33 €/MWh  |  nebo fixni cena (opt.)', sz=8.0, c='#7dd3fc', ha='left')

# ══════════════════════════════════════════════════════════════════════════
# ARROWS  row2 → bus
# ══════════════════════════════════════════════════════════════════════════
arr(2.2,  6.8,  4.8,  6.05, C_HEAT, lbl='teplo', shrA=4, shrB=4)
arr(9.0,  6.8,  8.0,  6.05, C_HEAT, lbl='teplo', shrA=4, shrB=4)
arr(14.8, 6.8, 12.5,  6.05, C_HEAT, lbl='teplo', shrA=4, shrB=4)
arr(14.8, 11.0, 13.5, 6.05, C_IMP, lbl='', style='arc3,rad=0.12', shrA=4, shrB=4)

# KGJ elektrina → grid (napravo)
arr(5.0, 8.0, 17.8, 2.6, C_EL, lw=1.8, lbl='EE export',
    style='arc3,rad=-0.18', shrA=4, shrB=4)

# ══════════════════════════════════════════════════════════════════════════
# TEPELNA SIT / BUS  (y 5.2 – 5.95)
# ══════════════════════════════════════════════════════════════════════════
rbox(0.3, 5.2, 17.0, 0.75, '#0c1832', C_BUS, lw=2.4)
t(8.8, 5.57,
  'TEPELNA SIT  /  DISPATCH BUS',
  sz=10.5, bold=True, c='#93c5fd')
t(8.8, 5.28,
  'q_KGJ + q_Kotel + q_EK + q_Import + TES_vydej - TES_prijem  =  Poptavka_tepla',
  sz=8.2, c='#60a5fa')

# ══════════════════════════════════════════════════════════════════════════
# ROW 3  —  TES + VYSTUPY  (y 0.4 – 4.95)
# ══════════════════════════════════════════════════════════════════════════

# --- TES ---
rbox(0.3, 0.4, 4.8, 4.65, C_TES_D, C_TES, lw=2.4)
t(2.7, 4.72, 'TES  —  Tepelny zasobnik', sz=11, bold=True, c='#c4b5fd')
hr(0.55, 4.85, 4.38, '#3b0764')
t(0.65, 4.12, 'Kapacita:', sz=7.5, c=MUTED, ha='left')
t(0.65, 3.82, '1.52 MWh', sz=10, c=WHITE, ha='left', bold=True)
t(0.65, 3.52, 'Init SOC:  50 %  =  0.76 MWh', sz=8.0, c='#a78bfa', ha='left')
hr(0.55, 4.85, 3.3, '#3b0764')
t(0.65, 3.05, 'Ztrata:', sz=7.5, c=MUTED, ha='left')
t(0.65, 2.75, '0.5 % / hodinu', sz=9, c='#c4b5fd', ha='left', bold=True)
hr(0.55, 4.85, 2.5, '#3b0764')
t(0.65, 2.22, 'SOC_{t+1} = SOC_t x (1 - 0.005)', sz=8.0, c=MUTED, ha='left')
t(0.65, 1.92, '         + TES_in_t  -  TES_out_t', sz=8.0, c=MUTED, ha='left')
hr(0.55, 4.85, 1.68, '#3b0764')
t(0.65, 1.45, 'TES_in / TES_out: kontinualni', sz=8.0, c='#a78bfa', ha='left')
t(0.65, 1.15, 'Bez primych palivovych nakladu', sz=8.0, c=MUTED, ha='left')
t(0.65, 0.82, 'Optimalizovano spolecne s LP', sz=8.0, c=DIM, ha='left')
t(0.65, 0.55, 'Zacatek = konec periody (cyklicke)', sz=8.0, c=DIM, ha='left')

# --- TEPELNA POPTAVKA ---
rbox(5.5, 0.4, 5.8, 4.65, C_HEAT_D, C_HEAT, lw=2.4)
t(8.4, 4.72, 'TEPELNA POPTAVKA', sz=11, bold=True, c='#fca5a5')
hr(5.75, 11.05, 4.38, '#7f1d1d')
t(5.85, 4.12, 'Prodejni cena tepla:', sz=7.5, c=MUTED, ha='left')
t(5.85, 3.82, '95 €/MWh', sz=11, c='#fca5a5', ha='left', bold=True)
hr(5.75, 11.05, 3.6, '#7f1d1d')
t(5.85, 3.32, 'Penalizace za nedodani:', sz=7.5, c=MUTED, ha='left')
t(5.85, 3.02, '500 €/MWh', sz=11, c='#f87171', ha='left', bold=True)
hr(5.75, 11.05, 2.78, '#7f1d1d')
t(5.85, 2.50, 'Prebytek tepla:', sz=7.5, c=MUTED, ha='left')
t(5.85, 2.20, 'heat_dump  (nulova hodnota)', sz=8.5, c=MUTED, ha='left')
hr(5.75, 11.05, 1.95, '#7f1d1d')
t(5.85, 1.68, 'Vstup: hodinovy CSV profil', sz=8.0, c=DIM, ha='left')
t(5.85, 1.38, 'Casovy krok: 1 hodina', sz=8.0, c=DIM, ha='left')
hr(5.75, 11.05, 1.12, '#7f1d1d')
t(5.85, 0.82, 'Cil LP: max. zisk (prijem - naklady)', sz=8.0, c='#fca5a5', ha='left')
t(5.85, 0.52, 'Bilance plnena pro kazdy casovy krok', sz=8.0, c=DIM, ha='left')

# --- ELEKTRO GRID (vystupy) ---
rbox(11.7, 0.4, 5.6, 4.65, C_EL_D, C_EL, lw=2.4)
t(14.5, 4.72, 'ELEKTRO  (grid)', sz=11, bold=True, c='#93c5fd')
hr(11.95, 17.05, 4.38, '#1e3a8a')
t(12.05, 4.12, 'EXPORT (z KGJ):', sz=7.5, c=MUTED, ha='left')
t(12.05, 3.82, 'spot  -  2 €/MWh  (po distribuci)', sz=9, c='#4ade80', ha='left', bold=True)
hr(11.95, 17.05, 3.6, '#1e3a8a')
t(12.05, 3.32, 'NAKUP (pro EK / BESS):', sz=7.5, c=MUTED, ha='left')
t(12.05, 3.02, 'spot  +  33 €/MWh  (po distribuci)', sz=9, c='#f87171', ha='left', bold=True)
hr(11.95, 17.05, 2.78, '#1e3a8a')
t(12.05, 2.50, 'KGJ generuje EE jako vedlejsi produkt', sz=8.0, c=DIM, ha='left')
t(12.05, 2.20, 'Elektr. bilance: kazdy casovy krok', sz=8.0, c=DIM, ha='left')
hr(11.95, 17.05, 1.95, '#1e3a8a')
t(12.05, 1.68, 'CO2 faktor site:  0.250 tCO2/MWh', sz=8.0, c=DIM, ha='left')
t(12.05, 1.38, 'CO2 faktor plynu: 0.202 tCO2/MWh', sz=8.0, c=DIM, ha='left')
hr(11.95, 17.05, 1.12, '#1e3a8a')
t(12.05, 0.82, 'BESS (volitelne): kap. 1.0 MWh', sz=8.0, c=DIM, ha='left')
t(12.05, 0.52, 'Max vykon 0.5 MW  |  eta = 0.90', sz=8.0, c=DIM, ha='left')

# ══════════════════════════════════════════════════════════════════════════
# ARROWS  bus → row3
# ══════════════════════════════════════════════════════════════════════════
# Bus → TES (charge)
arr(3.5, 5.2, 2.8, 5.05, C_TES, lbl='charge', shrA=4, shrB=4)
# TES → Bus (discharge)
arr(2.5, 5.05, 3.1, 5.2, '#a78bfa', lbl='discharge',
    style='arc3,rad=0.5', shrA=4, shrB=4)
# Bus → tepelna poptavka
arr(8.5, 5.2, 8.0, 5.05, C_HEAT, lbl='teplo', shrA=4, shrB=4)
# Grid → EK (EE nakup)
arr(13.5, 5.2, 14.0, 5.05, C_EL, lbl='EE', shrA=4, shrB=4)

# ══════════════════════════════════════════════════════════════════════════
# LEGEND  (right strip)
# ══════════════════════════════════════════════════════════════════════════
rbox(17.5, 0.4, 6.15, 13.45, '#0c1018', BORDER, lw=1.5)
t(20.57, 13.55, 'LEGENDA', sz=10, bold=True, c=MUTED)
hr(17.7, 23.4, 13.2, BORDER)

leg = [
    (C_GAS,   'Tok zemniho plynu'),
    (C_EL,    'Tok elektriny'),
    (C_HEAT,  'Tok tepla'),
    (C_TES,   'TES charge / discharge'),
    (C_IMP,   'Import tepla'),
]
for i, (c, txt) in enumerate(leg):
    yy = 12.7 - i * 0.65
    ax.annotate('', xy=(19.0, yy), xytext=(17.95, yy),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=2.5,
                                mutation_scale=14), zorder=7)
    t(19.2, yy, txt, sz=8.5, c=WHITE, ha='left', va='center')

hr(17.7, 23.4, 9.45, BORDER)
t(20.57, 9.2, 'VYCHOZI HODNOTY PARAMETRU', sz=9, bold=True, c=MUTED)
hr(17.7, 23.4, 8.9, BORDER)

params = [
    ('Prodejni cena tepla',   '95 €/MWh',    '#fca5a5'),
    ('Fixni cena plynu',      '40 €/MWh',    '#fdba74'),
    ('Distribuce plynu',      '+5 €/MWh',    MUTED),
    ('Distribuce EE nakup',   '+33 €/MWh',   '#93c5fd'),
    ('Distribuce EE prodej',  '-2 €/MWh',    '#86efac'),
    ('Import tepla',          '150 €/MWh',   '#d1d5db'),
    ('Penalizace nedodani',   '500 €/MWh',   '#f87171'),
    ('Servis KGJ',            '14 €/h',      '#6ee7b7'),
    ('Start KGJ',             '150 €/start', '#fbbf24'),
    ('TES kapacita',          '1.52 MWh',    '#c4b5fd'),
    ('TES ztrata',            '0.5 %/h',     '#a78bfa'),
    ('BESS kapacita',         '1.0 MWh',     '#60a5fa'),
    ('FVE instalovany vykon', '1.0 MW',      '#fde68a'),
    ('CO2 cena',              '0 €/tCO2',    DIM),
]
for i, (k, v, vc) in enumerate(params):
    yy = 8.45 - i * 0.58
    t(17.85, yy, k, sz=8.0, c=MUTED, ha='left', va='center')
    t(23.3,  yy, v, sz=8.5, c=vc,    ha='right', va='center', bold=True)
    if i < len(params)-1:
        ax.plot([17.7, 23.4], [yy-0.25, yy-0.25], color='#1c2330', lw=0.5, zorder=2)

# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════
ax.text(8.9, 0.18,
        'LP optimalizace  (PuLP / CBC)   |   Hodinove kroky   |   Binarni: KGJ on/off, Kotel on/off   |   Optimalizovane promenne: q_KGJ, q_Boil, q_EK, q_Imp, TES_in, TES_out',
        fontsize=7.2, color=DIM, ha='center', va='center', zorder=5)

plt.tight_layout(pad=0)
plt.savefig('/home/user/opt_CHP/schema_zdroje_ceny.png', dpi=160,
            bbox_inches='tight', facecolor=BG)
print('DONE')
