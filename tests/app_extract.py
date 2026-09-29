"""Načtení exportních funkcí z app.py bez spuštění Streamlitu.

app.py je Streamlit skript, takže ho nejde importovat. Funkce se z něj
proto vytahují přes AST a spouštějí ve vlastním jmenném prostoru, kde je
k dispozici celé opt_core a velkými písmeny psané konstanty z app.py.

Jedno místo místo kopie v každém testu: přidání funkce nebo konstanty do
exportní cesty pak neznamená upravit seznam v několika souborech.
"""

import ast
import io
import re
import types
from pathlib import Path

import numpy as np
import pandas as pd
from xlsxwriter.utility import xl_col_to_name

import opt_core

APP = Path(opt_core.__file__).resolve().parent / 'app.py'

# Funkce, ze kterých se skládají Excel exporty
EXPORT_FUNCS = {
    '_wb_formats', '_safe_sheet', '_write_sheet', '_round_numeric',
    'build_parameters_df', '_write_month_sheet', '_write_hour_matrix',
    'create_scenario_comparison_df', 'to_excel_scenarios',
    'to_excel_operating_plan',
}


def _upper_names(target):
    """True pro `NAZEV = ...` i `A, B = ...`, kde jsou vsechna jmena velka."""
    if isinstance(target, ast.Name):
        return target.id.isupper()
    if isinstance(target, ast.Tuple):
        return all(_upper_names(e) for e in target.elts)
    return False


def load_app(funcs=EXPORT_FUNCS):
    tree = ast.parse(APP.read_text(encoding='utf-8'))
    mod = types.ModuleType('app_export')
    mod.__dict__.update({k: getattr(opt_core, k) for k in dir(opt_core)
                         if not k.startswith('__')})
    mod.__dict__.update({'io': io, 'pd': pd, 're': re, 'np': np,
                         'xl_col_to_name': xl_col_to_name})

    for node in tree.body:
        # Konstanty modulu (DST_GAP_MARK apod.). Ty, ktere sahaji na
        # Streamlit nebo prostredi, se tise preskoci - exporty je nepotrebuji.
        if (isinstance(node, ast.Assign)
                and all(_upper_names(t) for t in node.targets)):
            try:
                exec(compile(ast.Module([node], []), '<app>', 'exec'),
                     mod.__dict__)
            except Exception:
                pass
        elif isinstance(node, ast.FunctionDef) and node.name in funcs:
            exec(compile(ast.Module([node], []), '<app>', 'exec'), mod.__dict__)

    missing = set(funcs) - set(mod.__dict__)
    assert not missing, f'v app.py chybí {missing}'
    return mod
