# opt_CHP — D+1 plánování provozu a nominace OTE

MILP nástroj (PuLP + CBC, Streamlit) pro denní plánování portfolia
energetických zdrojů na českém trhu: příprava nabídek aFRR do denní aukce
ČEPS, plán provozu pro D+1, nominace pozice na OTE a následný re-dispatch
proti ceně odchylky. Vše v 15minutové granularitě (MTU), včetně dnů
přechodu času (92/100 intervalů).

## Denní workflow (den D pro dodávkový den D+1)

| Čas | Krok | Co se děje |
|---|---|---|
| ~08:00 | **aFRR nabídky** | Z predikcí (DA cena, ceny aFRR± po 4h blocích, odchylka) model spočítá žebřík *opportunity cost* — minimální bidové ceny [€/MW/h] pro různé kapacity, s doporučením co nabídnout do denní aukce ČEPS. |
| ~10:00 | **DA plán + nominace** | Po zadání výsledků aukce (vysoutěžené MW + ceny) model sestaví plán provozu s držením rezervované kapacity a čistou pozici portfolia = **nominaci na OTE** (export CSV/XLSX). Nominace se zmrazí. |
| po 14:00 | **Re-dispatch** | OTE zveřejní skutečné ceny DA. Nominace je fixní (zúčtuje se skutečnou cenou), model přeplánuje provoz a odchylky od nominace ocení predikovanou zúčtovací cenou odchylky (+ volitelná riziková přirážka λ). |

## Profily zdrojů

Portfolio se konfiguruje jako **profil** (JSON v `data/profiles/`), který
obsahuje libovolný počet lokalit a v každé lokalitě libovolný počet zdrojů
(i stejného typu — např. 2× KGJ, 3× FVE + 3× BESS v různých lokacích).
Podporované zdroje: KGJ (kogenerace), plynový kotel, elektrokotel, nádrž
TES, baterie BESS, FVE, import tepla. Spotřeba lokality: žádná / TDD třída
× roční spotřeba (koeficienty OTE) / vlastní křivka.

Obchodní den si při založení ukládá **snapshot profilu** — pozdější úpravy
šablony neovlivní historii ani nominaci.

## Vstupy

Jeden workbook „Vstupy D+1“ (šablona ke stažení v aplikaci, sloupce podle
profilu): listy `DA_ceny` [EUR/MWh], `Odchylka` [CZK/MWh], `aFRR_ceny`
[CZK/MW/h po 4h blocích], `FVE_vyroba` [MW per FVE], `Spotreba` [MW],
`Teplo` [MW_th], `Plyn` [EUR/MWh]. Parser přijme 15min i hodinové řady
(hodinové rozpadne — FVE lineární interpolací, ostatní opakováním; hodí se
pro hodinové predikce počasí např. z AG2). CZK↔EUR převádí kurz zadaný
u dne. Skutečné ceny DA pro re-dispatch se nahrávají zvlášť (xlsx/csv).

## Spuštění

```bash
pip install -r requirements.txt
streamlit run app.py
```

Demo data (syntetický den + vyplněný workbook pro demo profil):

```bash
python -m scripts.make_demo_day 2026-07-15
```

## Testy

```bash
pip install pytest
python -m pytest tests/ -q
```

Testy pokrývají časovou osu (DST), profily, TDD, parser vstupů, MILP
(bilanční identity, analytický toy case BESS arbitráže), aFRR rezervace
(numerická re-verifikace omezení na řešení), celý denní workflow včetně
restartu aplikace a exporty.

## Struktura

```
app.py              vstupní bod (st.navigation)
core/               timegrid, profiles, tdd, inputs, model (MILP),
                    runs (3 běhy), trading_day (persistence), export
ui/                 stránky: Obchodní den, Šablony zdrojů, Historie, Nastavení
data/profiles/      uložené profily (demo.json commitnutý)
data/tdd/           TDD koeficienty per rok (parquet)
data/runs/          obchodní dny (day.json + parquet artefakty)
scripts/            make_demo_day.py — syntetická demo data
tests/              pytest
```

## Ruční QA checklist

1. `python -m scripts.make_demo_day` → vzniknou demo soubory v `data/`.
2. Nastavení → vygenerovat syntetické TDD pro rok dodávky.
3. Obchodní den → založit den s profilem `demo`, stáhnout šablonu,
   nahrát `data/Vstupy_D1_<datum>_demo.xlsx`.
4. Krok 1 → Spočítat nabídky → tabulka + ladder graf + XLSX export.
5. Krok 2 → zadat vysoutěženou kapacitu → Sestavit plán → grafy,
   nominace → Zmrazit → stáhnout CSV/XLSX.
6. Krok 3 → nahrát `data/Skutecne_DA_<datum>_demo.xlsx` → Přepočítat →
   graf odchylek + srovnání ekonomik.
7. Historie dní → den je vidět se zamčenou nominací a výsledky.
