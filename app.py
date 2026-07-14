"""D+1 Plánování & Nominace OTE — vstupní bod Streamlit aplikace.

Spuštění:  streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui.common import inject_css  # noqa: E402

st.set_page_config(page_title="D+1 Plánování & Nominace OTE",
                   page_icon="⚡", layout="wide")
inject_css()

pages = [
    st.Page("ui/page_workflow.py", title="Obchodní den", icon="📅",
            default=True),
    st.Page("ui/page_profiles.py", title="Šablony zdrojů", icon="🧩"),
    st.Page("ui/page_history.py", title="Historie dní", icon="🗂️"),
    st.Page("ui/page_settings.py", title="Nastavení", icon="⚙️"),
]

st.navigation(pages).run()
