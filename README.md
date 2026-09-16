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

`windows/update_opt_chp.bat` zkopíruj **o úroveň výš**, mimo složku s kódem —
dávkový soubor se nesmí přepsat sám za běhu. Pak:

```
update_opt_chp.bat "cesta\k\nove\rozbalene\slozce"
```

Přepíše kód a nechá `.venv` být. Pokud se změnil `requirements-lock.txt`, zruší
značku o instalaci a balíčky se při dalším startu doinstalují samy.

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

## Rampy nájezdu / sjezdu KGJ

Volitelně (záložka **Technika** → *Modelovat nájezd / sjezd KGJ*) model rozprostře
náběh a odstavení do hodinového průměru. Lineární rampa délky τ minut znamená, že
hodina startu dodá `P·(1 − τ/120)` a hodina po vypnutí ještě `P·(τ/120)`.

Pro 975 kW jednotku s nájezdem 12,7 min a sjezdem 8,9 min vyjde typický tříhodinový
běh jako `872 / 975 / 975 / 72` kW — poslední hodnota padá do hodiny, kdy je jednotka
už formálně vypnutá. Teplo, elektřina i plyn se deratují stejným faktorem, takže se
během rampy nemění účinnost; náklad na start pokrývá samostatný parametr
`Náklady na start [€/start]`.

Rampa je omezená na 0–60 min, aby se nikdy nerozlila do další hodiny.
