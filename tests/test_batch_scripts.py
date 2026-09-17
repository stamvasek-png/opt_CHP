"""Statická kontrola dávkových souborů pro Windows.

Windows v CI ani ve vývojovém kontejneru není, takže se kontroluje aspoň to,
co jde ověřit ze souboru samotného. Každé pravidlo tu je proto, že odpovídající
chyba se už jednou stala.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BAT_FILES = [REPO / 'start_opt_chp.bat', REPO / 'windows' / 'update_opt_chp.bat']

# Promenne, jejichz rozvinuti muze obsahovat mezeru (AppData\Local, Downloads...)
PATH_VARS = ('LOCALAPPDATA', 'USERPROFILE', 'TEMP', 'OPT_CHP_HOME',
             'OPT_CHP_DATA_DIR', 'WORK', 'DL', 'SRC', 'ZIP', 'STAMP', 'OLDLOCK')


@pytest.fixture(params=BAT_FILES, ids=lambda p: p.name)
def bat(request):
    return request.param


def _text(path):
    return path.read_bytes().decode('ascii')


def test_file_is_not_truncated(bat):
    """Prázdný soubor projde všemi ostatními kontrolami — chytit ho zvlášť."""
    raw = bat.read_bytes()
    assert len(raw) > 100, f'podezřele malý soubor ({len(raw)} B)'
    assert raw.startswith(b'@echo off')


def test_is_pure_ascii(bat):
    """Diakritika se v dávce rozsype podle kódování konzole."""
    raw = bat.read_bytes()
    bad = [(i, b) for i, b in enumerate(raw) if b > 127]
    assert not bad, f'non-ASCII bajty na pozicích {[i for i, _ in bad[:5]]}'


def test_uses_crlf_line_endings(bat):
    raw = bat.read_bytes()
    assert not re.search(rb'(?<!\r)\n', raw), 'osamocený LF, dávka chce CRLF'


def test_lines_are_short(bat):
    """Dlouhé řádky se při vkládání do cmd sekají."""
    longs = [(i + 1, len(l)) for i, l in enumerate(_text(bat).split('\r\n'))
             if len(l) > 100]
    assert not longs, f'dlouhé řádky: {longs}'


def test_paths_with_variables_are_quoted(bat):
    """AppData\\Local obsahuje mezeru v některých lokalizacích i v profilu."""
    problems = []
    for i, line in enumerate(_text(bat).split('\r\n')):
        if line.strip().startswith(('echo', 'rem', 'set /p')):
            continue
        pattern = r'%(?:' + '|'.join(PATH_VARS) + r')%[^\s]*[\\/]'
        for m in re.finditer(pattern, line):
            if line[:m.start()].count('"') % 2 == 0:
                problems.append(f'ř.{i + 1}: {m.group(0)}')
    assert not problems, f'necitované cesty: {problems}'


def test_every_goto_has_a_label(bat):
    txt = _text(bat)
    labels = {l.strip()[1:].split()[0]
              for l in txt.split('\r\n') if re.match(r'^:\w', l.strip())}
    targets = {m.group(1) for m in re.finditer(r'(?:goto|call)\s+:?(\w+)', txt)}
    assert not targets - labels - {'eof'}, \
        f'skok bez labelu: {sorted(targets - labels - {"eof"})}'


def test_every_exit_path_pauses(bat):
    """Bez pause se okno při dvojkliku zavře dřív, než si to jde přečíst.

    Přesně tohle způsobilo, že update_opt_chp.bat jen probliknul a uživatel
    se nedozvěděl vůbec nic — ani že mu chybí argument.
    """
    lines = _text(bat).split('\r\n')
    problems = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not re.match(r'^(exit /b|goto :eof)\b', stripped):
            continue
        # Podprogram :cleanup se pres goto :eof legitimne vraci volajicimu.
        prev = [x.strip() for x in lines[:i] if re.match(r'^:\w', x.strip())]
        if prev and prev[-1] == ':cleanup':
            continue
        if 'pause' not in [x.strip() for x in lines[max(0, i - 6):i]]:
            problems.append(f'ř.{i + 1}: {stripped}')
    assert not problems, f'ukončení bez pause (okno zmizí): {problems}'
