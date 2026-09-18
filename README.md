# opt_CHP
optimalizace provozů s kgj

## Struktura

| Soubor | Obsah |
|---|---|
| `app.py` | Streamlit UI, scénáře, grafy a Excel exporty |
| `opt_core.py` | Výpočetní jádro bez Streamlitu — profily, linearizace účinnosti, MILP solver |
| `tests/` | Testy správnosti výpočtu (pytest) |
| `start_opt_chp.bat` | Launcher pro Windows — vyrobí venv a spustí UI |
| `windows/` | Skript pro aktualizaci z rozbaleného ZIPu |
| `.streamlit/config.toml` | Vazba serveru na localhost, bez telemetrie |

## Spuštění

```bash
pip install -r requirements.txt
streamlit run app.py
```

Na Windows použij `start_opt_chp.bat` — viz [níže](#spuštění-na-windows-i-bez-práv-správce).

## Testy

```bash
pip install -r requirements-dev.txt
pytest
```

## Spuštění na Windows (i bez práv správce)

Aplikace nepotřebuje instalátor ani práva správce — stačí Python s právem zapisovat
do vlastního profilu. Solver CBC se veze přímo v balíčku `pulp`, grafy má Streamlit
zabudované, takže za běhu není potřeba internet.

### První instalace

1. Na GitHubu **Code → Download ZIP**.
2. Rozbalit do pracovní složky a **přejmenovat na krátký název**:

   ```
   cd /d "%LOCALAPPDATA%\py"
   tar -xf "%USERPROFILE%\Downloads\opt_CHP-main.zip"
   move opt_CHP-main opt_chp
   ```

   Krátký název není kosmetika. Windows nezvládá cesty nad 260 znaků a nejdelší
   cesta uvnitř Streamlitu má 140 znaků. S výchozím názvem z GitHub ZIPu zbývá
   jen ~20 znaků rezervy, po přejmenování na `opt_chp` je to ~50.

3. Spustit `start_opt_chp.bat` (dvojklikem nebo z příkazové řádky).

   Při prvním spuštění si vyrobí `.venv` a doinstaluje balíčky z
   `requirements-lock.txt` — to trvá pár minut, průběh je v logu. Další starty
   jsou okamžité. Streamlit pak sám otevře `http://localhost:8501`.

`tar` i `robocopy` jsou ve Windows vestavěné, nic dalšího se neinstaluje.

### Kde co leží

| | Cesta |
|---|---|
| Kód | `%LOCALAPPDATA%\py\opt_chp` |
| Data a log | `%OPT_CHP_DATA_DIR%`, výchozí `%LOCALAPPDATA%\py\opt_chp_data` |

Cache se drží **mimo složku s kódem**, aby ji aktualizace nesmazala. Když
`OPT_CHP_DATA_DIR` nenastavíš, chová se aplikace jako dřív a píše do `./cache`.

### Aktualizace

`windows/update_opt_chp.bat` zkopíruj **o úroveň výš**, do `%LOCALAPPDATA%\py\`
— dávkový soubor se nesmí přepsat sám za běhu.

Pak stačí stáhnout nový ZIP na GitHubu (**Code → Download ZIP**) a na skript
**dvojkliknout**. Sám si najde nejnovější `opt_CHP-*.zip` ve složce Stažené
soubory, rozbalí ho a zeptá se, než začne přepisovat:

```
[opt_chp] ZIP: C:\Users\...\Downloads\opt_CHP-main.zip
[opt_chp] Cil: C:\Users\...\AppData\Local\py\opt_chp
Prepsat slozku s kodem? [A/N]
```

Potvrzuje se `A` nebo `Y` (na velikosti nezáleží, projde i `ano` / `yes`).
Cokoli jiného včetně prostého Enteru aktualizaci zruší a nic nepřepíše.

Kopíruje se přes `robocopy /MIR`, takže se v cíli smažou soubory, které v ZIPu
nejsou — proto ten dotaz. `.venv` a `__pycache__` zůstávají nedotčené a data
jsou stejně jinde (`OPT_CHP_DATA_DIR`). Pokud se změnil `requirements-lock.txt`,
skript zruší značku o instalaci a balíčky se při dalším startu doinstalují samy.

Když si ZIP rozbalíš sám, jde cesta předat jako parametr:
`update_opt_chp.bat "cesta\k\slozce"`.

Okno zůstane otevřené s výsledkem — a to i když něco selže.

### Za firemní proxy

Skripty proxy nikde nenastavují — berou ji z `HTTPS_PROXY` / `HTTP_PROXY`.
Pokud `pip` skončí na `ConnectionResetError`, nastav je (`setx` se projeví až
v **novém** okně) a spusť znovu. Ověřování certifikátů nikdy nevypínej.

### Poznámky

- Interpret se volá napřímo (`.venv\Scripts\python.exe`), ne přes `activate` —
  PowerShell aktivační skript často blokuje přes ExecutionPolicy.
- `.streamlit/config.toml` váže server na `localhost`. Ve výchozím stavu naslouchá
  Streamlit na všech rozhraních a Windows na to vyhodí dialog firewallu, jehož
  schválení chce práva správce.
- Verze v `requirements-lock.txt` jsou ověřené. `requirements.txt` je volnější,
  ale drží `pulp<4` — v PuLP 4.0 mizí API, které model používá.

## Provozní profily KGJ

| Profil | Okno |
|---|---|
| `free` | bez omezení |
| `base` | 24/7, KGJ vždy zapnuto |
| `peak` | Po–Pá 08:00–20:00 |
| `extpeak` | Po–Pá 06:00–22:00 |
| `extpsum` | jako `extpeak`, ale s letní úpravou (níže) |
| `offpeak` | doplněk peaku: víkendy a svátky celý den + Po–Pá 20:00–08:00 |
| `special` | měsíční vzor s denním rytmem |
| `custom` | ručně vybrané hodiny |

`peak`, `extpeak`, `extpsum` a `offpeak` respektují víkendy i české státní svátky.

### EXTPSUM

Základ je `extpeak`, tedy Po–Pá 06:00–22:00. **Od 1. 6. do 30. 9. včetně** se
okno mění:

| Čas | Zima (X–V) | Léto (VI–IX) |
|---|---|---|
| 04:00–06:00 | — | **lze** |
| 06:00–11:00 | lze | lze |
| 11:00–17:00 | lze | **nelze** |
| 17:00–22:00 | lze | lze |
| 22:00–24:00 | — | **lze** |
| **denně** | **16 h** | **14 h** |

Zákaz končí v 17:00, takže hodina 17:00–18:00 se už brát smí. Za rok 2026 to
dělá 3828 dostupných hodin proti 4000 u `extpeak`.

## Rychlost solveru

Roční úloha (8760 hodin) je pro CBC náročná a nejvíc na ní záleží, jestli je
zapnutá **akumulace**. Naměřeno na profilu FREE s limitem 10 minut:

| Technologie | Čas | Doběhlo samo? |
|---|---|---|
| jen KGJ + kotel | 141 s | ano |
| \+ elektrokotel + FVE | 237 s | ano |
| \+ nádrž (TES) | 644 s | ne, limit |
| všechno včetně baterie | 666 s | ne, limit |

Nádrž propojí stav nabití mezi všemi hodinami roku, takže se z úlohy stane
jeden provázaný problém — a to je ten zlom. Profil FREE je navíc nejtěžší,
protože nemá žádná profilová omezení a všech 8760 binárek zůstává volných.

### Tolerance od optima

V sidebaru je **„Tolerance od optima [%]"**, výchozí 1 %. Solver skončí, jakmile
ví, že je blíž než tahle mezera k optimu.

Rozdíl je zásadní: **najít dobré řešení je rychlé, dokázat že lepší neexistuje
může trvat řádově déle.** Poslední desetina procenta obvykle spotřebuje víc času
než prvních 99 %. U modelu, který stojí na odhadu FWD křivky, je přitom 1 %
hluboko pod nejistotou vstupů — dokazovat optimalitu takového zadání je spíš
formalita.

Naměřeno na téže roční úloze se všemi technologiemi, limit 15 minut:

| Nastavení | Čas | Dojelo na limit? | Zisk |
|---|---|---|---|
| bez tolerance | 946 s | ano | 636 374 € |
| 0,5 % | 398 s | ne | 636 374 € |
| **1 %** | **395 s** | ne | **636 374 €** |
| 1 % + `threads=6` | 400 s | ne | 636 374 € |

Zisk vyšel ve všech případech **identicky**. Tolerance nestála nic na kvalitě
řešení — solver ho našel dávno a zbylých 550 sekund jen dokazoval, že lepší
neexistuje. Na konkrétní hodnotě navíc moc nezáleží; rozhoduje, že tam nějaká
tolerance je.

Nastavení 0 znamená dokazovat optimalitu a u roční úlohy s akumulací může
běžet hodiny.

### Co nepomůže

Solver běží na **jednom jádře** a **GPU nepoužívá vůbec** — branch & bound je
sekvenční prohledávání stromu, které se na grafickou kartu nepřeloží. Rychlejší
procesor pomůže úměrně taktu jednoho jádra, ale exponenciální problém se
hardwarem neobejde: dvojnásobný výkon udělá z dvaceti hodin deset.

Ani víc jader nepomůže. V tabulce výše je vidět, že `threads=6` skončilo na
400 s proti 395 s bez něj — přiložený CBC 2.10.3 paralelně nepočítá.

## Provozní plán vybraného profilu

V sekci scénářů jde po analýze vybrat jeden profil a stáhnout k němu samostatný
sešit (`kgj_provozni_plan_<profil>.xlsx`):

| List | Obsah |
|---|---|
| `Přehled` | souhrn za období, tabulka po měsících, odkazy na měsíční listy |
| `<PROFIL>` | hodinový rozpad, stejný jako v exportu scénářů |
| `LEDEN` … | jeden list na měsíc s mřížkou provozu |
| `Parametry` | použité nastavení |

Měsíční list má dny ve sloupcích a hodiny v řádcích, `P` = provoz (zeleně),
`X` = klid (červeně). Řádky jsou popsané rovnou intervalem `00:00-01:00` až
`23:00-24:00`, aby nebylo nutné dohadovat, co znamená „hodina 1".

Počty pod tabulkou jsou **živé vzorce** (`COUNTIF` / `SUM`), takže ruční
přepsání buňky součty přepočítá. Žluté podmíněné formátování pro hodnotu `F`
zůstává připravené, i když se automaticky nikdy nezapíše.

**Doběhová hodina se počítá jako klid.** Mřížka se plní z nasazení
(`KGJ on`), ne ze skutečného výkonu — hodina po odstavení má `X`, přestože
v ní jednotka ještě dodává zbytkové teplo z bloku.

Při nekompletních datech zůstane buňka prázdná místo `X`: den mimo analyzované
období, nebo hodina, která kvůli přechodu na letní čas neexistuje. Zdvojená
hodina na konci října se sloučí do jedné buňky, `P` když jednotka běžela
aspoň v jedné z nich.

## Rampy nájezdu / sjezdu KGJ

Volitelně (záložka **Technika** → *Modelovat nájezd / sjezd KGJ*) model rozprostře
náběh a odstavení do hodinového průměru. Lineární rampa délky τ minut znamená, že
hodina startu dodá `P·(1 − τ/120)` a hodina po vypnutí ještě `P·(τ/120)`.

Pro jednotku 975 kW_el s nájezdem 12,7 min a sjezdem 8,9 min vyjde typický
tříhodinový běh jako `872 / 975 / 975 / 72` kW — poslední hodnota padá do hodiny,
kdy je jednotka už formálně vypnutá. Náklad na start pokrývá samostatný parametr
`Náklady na start [€/start]`.

Rampa je omezená na 0–60 min, aby se nikdy nerozlila do další hodiny.

### τ jsou ekvivalentní minuty, ne strmost

V hodinovém průměru **nejde odlišit mrtvou dobu od pomalejší rampy** — záleží jen
na tom, kolik minut plného výkonu celkem chybí. Mrtvá doba *d* plus rampa *r* dá
stejný průměr jako lineární rampa délky `2d + r`. Doba, než se generátor
synchronizuje na síť, se tím schová do τ a nepotřebuje vlastní parametr.

### Teplo se chová jinak než elektřina

| Fáze | Elektřina | Plyn | Teplo |
|---|---|---|---|
| Start | nic, dokud se generátor nesynchronizuje, pak najíždí | teče od první zážehy | vzniká hned, ale nejdřív ohřívá blok a výměník |
| Odstavení | řízené odlehčení, pak vypínač — končí rychle | končí s motorem | dochlazení 5–15 min, čerpadla dál tlačí zbytkové teplo do sítě |

Proto lze zaškrtnout *Tepelná rampa se liší od elektrické* a zadat pro teplo
vlastní dvojici τ. **Plyn sleduje vždy elektřinu**, protože palivo jde do motoru
a motor točí generátorem. Prakticky to znamená, že v doběhové hodině se dodá
teplo téměř bez plynu — je to energie uložená v hmotě motoru, ne spálené palivo.

Bez zaškrtnutí sleduje teplo elektřinu a model se chová jako před rozdělením.

**Pozor při zadávání:** náběh a sjezd tepla by měly vyjít zhruba stejně, protože
je to tatáž energie — nejdřív se uloží do hmoty motoru, pak se vrátí. Výrazně
delší tepelný sjezd než náběh znamená, že model dostává teplo zadarmo. Model to
nezakazuje, může to mít důvod, ale je dobré o tom vědět.

### Sloupce ve výsledcích

Pro teplo i elektřinu je k dispozici rozpad `setpoint − ztráta nájezdem + doběh`:

| Teplo | Elektřina |
|---|---|
| `KGJ setpoint [MW_th]` | `EE z KGJ setpoint [MW]` |
| `KGJ nájezd ztráta [MW_th]` | `EE z KGJ nájezd ztráta [MW]` |
| `KGJ doběh [MW_th]` | `EE z KGJ doběh [MW]` |
| `KGJ [MW_th]` | `EE z KGJ [MW]` |
