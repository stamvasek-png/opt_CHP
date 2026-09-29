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
| `season` | sezónní okno, jeden blok denně, 7 dní v týdnu (níže) |
| `seasonplus` | `season` s okny širšími o 1–2 h na každém konci |
| `p` | pásmo z nejlepších FWD hodin ceny EE (níže) |
| `prom26` | dodaná hodinová maska na rok 2026 (níže) |
| `s` | pevné pásmo nad FWD 2027 (3453 h), blok max. 16 h (níže) |
| `t` | totéž bez stropu délky bloku (3351 h, níže) |
| `u` | týdenní šablona po měsících, bloky přes více dní (7032 h, níže) |
| `v` | plán pro dispečink: blok max. 96 h, pauza min. 16 h, září stojí (4136 h, níže) |
| `offpeak` | doplněk peaku: víkendy a svátky celý den + Po–Pá 20:00–08:00 |
| `special` | měsíční vzor s denním rytmem |
| `custom` | ručně vybrané hodiny |

`peak`, `extpeak`, `extpsum` a `offpeak` respektují víkendy i české státní svátky.
`season`, `seasonplus`, `p`, `s` a `t` jedou 7 dní v týdnu — okno určuje jen
měsíc a hodina. `u` a `v` mají pro každý měsíc týdenní šablonu.

### Profily po celých měsících

Zaškrtávátko **Profily po celých měsících** (postranní panel, pod výběrem
profilů) mění, jak solver s profilem zachází: v každém kalendářním měsíci jede
KGJ buď ve **všech** hodinách okna profilu, nebo v žádné. Pojede tak i týden,
který je sám o sobě ztrátový, pokud se vyplatí měsíc jako celek — provoz je
po celý měsíc stejný.

Které měsíce, rozhoduje solver, a to ve dvou krocích. Nejdřív vybere
nejvýnosnější celé měsíce, které se vejdou do ročního limitu hodin — například
EXTPEAK do 3300 h. Co z limitu zbyde, smí pak dát do **jednoho** dalšího
měsíce z těch, které nevybral. Ten je neúplný a hodiny v něm se vybírají
volně v okně profilu (s min. dobou běhu a limitem startů jako obvykle).
Měsíc, který se vyplatí celý a do limitu se vejde, tak zůstane vždy celý.
Bez limitu hodin žádný neúplný měsíc nevzniká — jedou všechny měsíce, které
jako celek vydělávají.

- Výkon si model volí dál (min. zatížení až 100 %), stejně jako u BASE.
- V celých měsících se min. doba běhu a limit startů za měsíc neuplatní —
  hodiny určuje profil (PROM26 s 3h bloky tak jede i při min. době běhu 4 h).
- BASE se nemění, FREE znamená celé měsíce 24/7.
- Platí pro porovnání profilů, měsíční analýzu, roční plán i citlivostní analýzu.
- Výstupy a exporty zůstávají stejné; které měsíce solver zvolil, je vidět
  v přehledu hodin provozu jako u ostatních běhů.

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

### SEASON a SEASONPLUS

Jedno souvislé okno na den, **7 dní v týdnu** (svátky se neuplatňují — teplo se
topí každý den stejně). Okno se s ubývající poptávkou po teple zužuje a posouvá
do večera, takže v létě vypadne poledne i celá doba provozu FVE.

| Měsíc | `season` | h/den | `seasonplus` | h/den |
|---|---|---|---|---|
| I, II, XI, XII | 06:00–22:00 | 16 | 05:00–23:00 | 18 |
| III, X | 13:00–23:00 | 10 | 12:00–24:00 | 12 |
| IV | 15:00–23:00 | 8 | 13:00–24:00 | 11 |
| V–IX | 17:00–23:00 | 6 | 16:00–24:00 | 8 |
| **za rok 2026** | | **3698 h** | | **4458 h** |

Jeden blok denně znamená **jeden start denně**. Ranní letní špička se úmyslně
nebere: leží prakticky na bodu zvratu KGJ, takže druhý start za den se z ní
nezaplatí.

`seasonplus` má krajní hodiny, které se při holé forwardové ceně nevyplatí.
Smysl dávají, až výkupní cenu zvedne PPA nebo zelený bonus — pak je z čeho brát
a jde dojet na limit provozních hodin, aniž by se muselo do poledne.

### P

Pásmo poskládané z **nejlepších FWD hodin ceny EE**, 7 dní v týdnu, jedno okno
na měsíc.

| Měsíc | Okno | h/den | h/měsíc |
|---|---|---|---|
| I | 07:00–23:00 | 16 | 496 |
| II, III | 06:00–22:00 | 16 | 448 / 496 |
| IV, V | 18:00–24:00 | 6 | 180 / 186 |
| VI, VII, VIII | 17:00–24:00 | 7 | 210 / 217 / 217 |
| IX | 17:00–23:00 | 6 | 180 |
| X, XI, XII | 06:00–22:00 | 16 | 496 / 480 / 496 |
| **za rok 2026** | | | **4102 h** |

Okna nejsou odhad. Pro každý měsíc se prošla všechna souvislá okna a přes
měsíce se to složilo batohem (DP) na rozpočet ~4000 h se stropem 16 h na blok.
Účelová funkce je součet FWD ceny při pevné velikosti pásma, což je při pevném
počtu hodin totéž co jeho průměrná cena.

Na datech 2026 vychází pásmo na **91,5 €/MWh** proti **97,4 €/MWh**, které by
dal volný výběr 4000 nejdražších hodin — rozdíl je cena za to, že je pásmo
uvnitř měsíce konzistentní. Strop 16 h na blok stojí z toho 1,1 €/MWh.

Letní okna začínají v 17:00 resp. 18:00, takže poledne i doba provozu FVE
vypadnou samy, bez zvláštního pravidla.

### PROM26

Převzatá hodinová maska 0/1 na rok 2026 — **3420 h**. Beze zbytku se rozkládá
na okna po měsících a pět výjimečných dní, takže je v kódu v téhle podobě
(dá se přečíst), ne jako 8760 nul a jedniček. Jede se i o víkendech.

| Měsíc | Okno | h/den |
|---|---|---|
| I, II, XI, XII | 06:00–22:00 | 16 |
| III, X | 06:00–10:00 + 16:00–22:00 | 10 |
| IV | 06:00–09:00 + 17:00–22:00 | 8 |
| V | 06:00–09:00 + 19:00–22:00 | 6 |
| VI, VII, VIII | 06:00–09:00 | 3 |
| IX | 06:00–09:00 + 18:00–22:00 | 7 |

| Den | Okno |
|---|---|
| 1. 1. | klid |
| 1. 10. | 06:00–09:00 + 18:00–22:00 (ještě zářijové okno) |
| 24.–26. 12. | 13:00–24:00 |
| 31. 12. | 06:00–24:00 |

Výjimky platí jen pro rok 2026; v jiném roce se použijí samotná okna měsíců.

**Pozor na minimální dobu běhu.** V přechodných měsících jsou dva bloky denně
a několik z nich má jen 3 hodiny (v létě je to jediné okno). Model do bloku
kratšího, než je minimální doba běhu, KGJ nenastartuje vůbec — ani při
sebevyšší ceně. S výchozími 4 h tak zůstane nevyužito 642 hodin masky a
květen až srpen je celý bez provozu. Pro PROM26 nastav **min. dobu běhu 3 h**;
aplikace na krátké bloky upozorní před spuštěním.

### S a T

Pevné pásmo nad FWD křivkou 2027 (29. 9. 2026): jeden souvislý
blok denně, 7 dní v týdnu, v každém měsíci stejné okno. Metoda je stejná jako
u P — všechna souvislá okna po měsících a přes měsíce batoh s pevnou velikostí
pásma (~3300 h), tedy maximalizace jeho průměrné FWD ceny. Hodina smí do okna,
jen když poptávka po teple stačí na plný výkon KGJ každý den v měsíci — s
výjimkou září, které bylo doplněno dodatečně (níže).

| Měsíc | `s` (blok ≤ 16 h) | `t` (bez stropu) |
|---|---|---|
| I, II | 06:00–22:00 | 05:00–24:00 |
| III | 16:00–24:00 | 17:00–23:00 |
| IV | 18:00–24:00 | 18:00–23:00 |
| V | 18:00–24:00 | 19:00–23:00 |
| VI, VII | 18:00–24:00 | 18:00–24:00 |
| VIII | 18:00–24:00 | 18:00–23:00 |
| IX | 18:00–23:00 | 18:00–23:00 |
| X | 15:00–22:00 | 16:00–21:00 |
| XI | 06:00–22:00 | 06:00–23:00 |
| XII | 07:00–23:00 | 07:00–21:00 |
| **rok 2027** | **3453 h**, FWD 180,1 €/MWh | **3351 h**, FWD 182,9 €/MWh |

**Září 18–23** je doplněné nad rámec výběru: poptávka po teple je tam celý
měsíc 0,338 MW, na plný výkon KGJ (0,605 MW tepla) tedy nestačí, ale KGJ se
vejde na minimální zatížení 50 % (0,3025 MW). Za to přidá 150 h s průměrnou
FWD 198,4 €/MWh — zářijové večery patří k nejdražším hodinám roku.

**Únor u `t`** je upravený dodatečně: výběr dával 02:00–24:00, na přání má
stejné okno jako leden (05:00–24:00).

Letní okna začínají v 18:00 (resp. 19:00), takže solární propad kolem poledne
i jeho záporné ceny zůstávají mimo pásmo.

Rozdíl mezi nimi je jen v délce zimního bloku. `s` nikde nejede přes 16 h.
`t` je o 2,8 €/MWh dražší, ale v lednu a únoru běží 19 h denně.
Pro srovnání: bez září by volný výběr 3300 nejdražších hodin měl 191,4 €/MWh
proti 179,3 (`s`) a 182,2 €/MWh (`t`) — rozdíl je cena za to, že pásmo je
uvnitř měsíce konzistentní.

### U

Týdenní šablona po měsících nad FWD křivkou 2027 (29. 9. 2026). Na rozdíl od
profilů výše není vázaná na jedno okno denně — blok smí trvat přes několik dní.
Každý týden v měsíci je stejný, víkend se ale smí chovat jinak než pracovní den.

Rozpočet hodin není omezený. Šablony proto maximalizují **celkovou marži KGJ**
proti kotli: 0,45 MW × FWD + ušetřený plyn kotle za skutečně využitelné teplo −
plyn KGJ − servis 14 €/h − start 30 €. Bod zvratu je ~80 €/MWh, v září kvůli
malé poptávce po teple ~114 €/MWh. Hledal je MILP přes všech 12 × 168 hodin
týdne s těmito pravidly:

- blok nejvýš **96 h**, mezi bloky aspoň **8 h** pauza,
- blok aspoň **4 h** — do kratšího okna KGJ s výchozí min. dobou běhu nenastartuje,
- nejvýš **1 start** za kalendářní den.

| Měsíc | Týdenní šablona |
|---|---|
| I | So 07 → Po 22 · Út 06 → Pá 23 |
| II | So 15 → Po 22 · Út 06 → Pá 23 |
| III | Ne 16 → St 09 · St 17 → Ne 07 |
| IV | Ne 18 → St 10 · St 18 → Čt 10 · Čt 18 → Pá 09 · Pá 17 → So 09 · So 18 → Ne 08 |
| V | Ne 18 → Po 10 · Po 18 → Út 09 · Út 17 → Čt 10 · Čt 18 → Pá 10 · Pá 18 → So 09 · So 18 → Ne 05 |
| VI | Ne 18 → Út 09 · Út 17 → St 09 · St 17 → Čt 09 · Čt 17 → Pá 09 · Pá 17 → So 08 · So 18 → Ne 07 |
| VII | Ne 17 → Út 09 · Út 17 → So 09 · So 17 → Ne 09 |
| VIII | Ne 18 → Po 10 · Po 18 → Út 09 · Út 17 → St 09 · St 17 → Pá 10 · Pá 18 → So 09 · So 17 → Ne 08 |
| IX | Ne 18 → Po 09 · Po 17 → Po 22 · Út 06 → St 09 · St 17 → Čt 09 · Čt 17 → Pá 09 · Pá 17 → So 09 · So 17 → So 23 |
| X | Ne 15 → St 23 · Čt 07 → Ne 07 |
| XI | Ne 08 → St 22 · Čt 06 → Ne 00 |
| XII | Ne 08 → St 23 · Čt 07 → Ne 00 |

Konec bloku je výlučný (Po 22 = do 22:00). Od listopadu do března jede KGJ
prakticky nepřetržitě se dvěma pauzami týdně, od dubna do září v nočních
blocích od večerní do ranní špičky, takže vynechá solární poledne. V červenci
polední propad trvá jen ~7 h, méně než povinná pauza, a projet ho vychází lépe.

Za rok 2027 to je **7032 h**, průměrná FWD 152,7 €/MWh a **198 startů**. Při
cenách 2027 je KGJ zisková v 7346 hodinách roku, proto nejvýnosnější
konzistentní profil pokrývá tolik hodin. Se `s` a `t` (~3450 h, 365 startů)
se tedy nedá srovnávat cenou za MWh — `u` jede dvojnásobek hodin s poloviční
četností startů.

Pravidla 96/8/4 a 1 start denně jsou ověřená na kalendáři 2027 včetně
přechodů mezi měsíci a změny času. V jiném roce připadnou hranice měsíců na
jiné dny v týdnu a na přechodu mezi měsíci mohou být porušená.

### V

Plán pro dispečink nad FWD křivkou 2027 (29. 9. 2026): KGJ jede nejvýš
**96 h v kuse, pak aspoň 16 h stojí**. Cílem je plán, který se v každém
měsíci opakuje a v provozu se skoro nemění.

Denně opakovaný blok tak má nejvýš 8 h (24 − 16) — noční běh od večerní do
ranní špičky jako u `u` nejde. V zimě se do týdne vejdou dva bloky a dvě
pauzy, tedy nejvýš 136 h.

Šablony hledal stejný MILP jako u `u` (marže KGJ proti kotli, blok aspoň 4 h,
nejvýš 1 start za kalendářní den) s jedním pravidlem navíc: hodina (měsíc ×
den v týdnu × hodina) smí do okna, jen když KGJ v ní vydělává **aspoň ve 3 ze
4 dnů** měsíce. Bez něj by šablona držela i hodiny, které vycházejí jen
v průměru. V konkrétní dny s propadem ceny by pak model uvnitř okna zastavil
a plán by se v provozu rozpadal.

| Měsíc | Týdenní šablona |
|---|---|
| XI, XII, I, II | Ne 15 → St 21 · Čt 13 → So 23 |
| III | denně 17:00–24:00 |
| IV, VI, VII, VIII | denně 18:00–02:00 |
| V | denně 19:00–01:00 |
| IX | bez provozu |
| X | Po 06 → Čt 23 · Pá 15:00–23:00 · So 16:00–21:00 |

V zimě jedou dva bloky týdně (78 h a 58 h) s pauzami ze středy na čtvrtek
a ze soboty na neděli. Jedna šablona pro XI–II vychází stejně jako šablony
laděné po měsících. Od března do srpna jede KGJ každý den stejné večerní okno
a solární poledne vynechá; šablona po dnech v týdnu by přinesla jen
~3 tis. €/rok.

**Září je bez provozu.** Poptávka po teple je tam celý měsíc 0,338 MW, KGJ
by mařila 44 % svého tepla (~44 MWh za rok) kvůli 5,4 tis. € marže.

Za rok 2027 to je **4136 h**, průměrná FWD 168,9 €/MWh a **234 startů**.
Roční běh na dodaných datech 2027 (KGJ + kotel, start 30 €, min. doba běhu
4 h):

| Profil | Zisk | Plán | Odjeto | Neodjeto z plánu | Zastavení v okně |
|---|---|---|---|---|---|
| **V** | 636 336 € | 4136 h | 4101 h | **35 h (0,8 %)** | **4** |
| U | 702 087 € | 7032 h | 6722 h | 310 h (4,4 %) | 48 |

`v` vydělá méně než `u`, protože jede méně hodin, ale plán skoro celý odjede.
Tři ze čtyř zastavení jsou dřívější konec okna s následnou pauzou 18–19 h;
jen 28. 10. by model v okně zastavil na 6 h.

Pravidla 96/16/4 a 1 start denně jsou ověřená na kalendáři 2027 včetně
přechodů mezi měsíci a změny času. V jiném roce připadnou hranice měsíců na
jiné dny v týdnu a je třeba je ověřit znovu.

## Změna času v provozním plánu

Data jsou v místním čase včetně letního, takže poslední neděli v březnu chybí
hodina 02:00–03:00 a poslední neděli v říjnu je dvakrát. V měsíčních listech
provozního plánu jsou obě políčka označená:

| Změna | Políčko | Proč |
|---|---|---|
| jaro | `ZČ` na modré výplni | hodina neexistuje; do počtu P ani X se nezapočítá |
| podzim | P/X s tlustým modrým rámečkem | hodina proběhla dvakrát; P = KGJ běžela aspoň v jedné z nich |

Obě mají v buňce poznámku a pod tabulkou řádek legendy. Značí se jen tam, kde
změnu času data skutečně obsahují — obyčejná díra v datech ani data bez
letního času nic neoznačí.

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
