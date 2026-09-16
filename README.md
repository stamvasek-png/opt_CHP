# opt_CHP
optimalizace provozů s kgj

## Struktura

| Soubor | Obsah |
|---|---|
| `app.py` | Streamlit UI, scénáře, grafy a Excel exporty |
| `opt_core.py` | Výpočetní jádro bez Streamlitu — profily, linearizace účinnosti, MILP solver |
| `tests/` | Testy správnosti výpočtu (pytest) |

## Spuštění

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Testy

```bash
pip install -r requirements-dev.txt
pytest
```

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
