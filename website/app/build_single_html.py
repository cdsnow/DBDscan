#!/usr/bin/env python3
"""
Build a single self-contained HTML file for GuestScan web app.

Inlines all scaffold data, mate PDBs, and pipeline.py into the HTML.
External CDN resources (Mol*, Pyodide) remain as remote links.

Usage:  python build_single_html.py
Output: guestscan.html (in the same directory)
"""

import os, json, re

APP_DIR = os.path.dirname(os.path.abspath(__file__))

def read(fname):
    with open(os.path.join(APP_DIR, fname)) as f:
        return f.read()

# ── 1. Read all data files ────────────────────────────────────────
DATA_FILES = {
    # key               → filename
    'scaffold_9YZJ_json': 'scaffold_9YZJ.json',
    'scaffold_9YZJ_pdb':  'scaffold_9YZJ.pdb',
    'mates_9YZJ_json':    'mates_9YZJ.json',
    'mate_9YZJ_bot_pdb':  'mate_9YZJ_bot.pdb',
    'mate_9YZJ_top_pdb':  'mate_9YZJ_top.pdb',
    'scaffold_9YZK_json': 'scaffold_9YZK.json',
    'scaffold_9YZK_pdb':  'scaffold_9YZK.pdb',
    'mates_9YZK_json':    'mates_9YZK.json',
    'mate_9YZK_bot_pdb':  'mate_9YZK_bot.pdb',
    'mate_9YZK_top_pdb':  'mate_9YZK_top.pdb',
    'pipeline_py':        'pipeline.py',
}

# ── 2. Build the embedded-data script block ───────────────────────
embedded_lines = [
    '<script>',
    '// ═══ Embedded data (for single-file deployment) ═══',
    'const _EMBEDDED = {};',
]

for key, fname in DATA_FILES.items():
    content = read(fname)
    if fname.endswith('.json'):
        # Inline JSON as a JS object literal (no escaping needed)
        embedded_lines.append(f'_EMBEDDED["{key}"] = {content.strip()};')
    else:
        # Escape backticks and ${ for JS template literal safety
        escaped = content.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
        embedded_lines.append(f'_EMBEDDED["{key}"] = `{escaped}`;')

embedded_lines.append('</script>')
embedded_block = '\n'.join(embedded_lines)

# ── 3. Read index.html and apply patches ──────────────────────────
html = read('index.html')

# 3a. Insert embedded data block right before the main app <script>
marker = '<!-- ═══════════════════════════════════════════════════════════════ -->\n<!--  JavaScript Layer'
html = html.replace(marker, embedded_block + '\n\n' + marker, 1)

# 3b. Replace fetchScaffoldData to use embedded data instead of fetch()
old_fetch_scaffold = """async function fetchScaffoldData(code) {
    const base = '.';
    const [scaffJson, scaffPdb, matesJson, botPdb, topPdb] = await Promise.all([
        fetch(`${base}/scaffold_${code}.json`).then(r => r.json()),
        fetch(`${base}/scaffold_${code}.pdb`).then(r => r.text()),
        fetch(`${base}/mates_${code}.json`).then(r => r.json()),
        fetch(`${base}/mate_${code}_bot.pdb`).then(r => r.text()),
        fetch(`${base}/mate_${code}_top.pdb`).then(r => r.text()),
    ]);
    scaffoldData[code] = {
        json: scaffJson,
        pdb: scaffPdb,
        matesJson: matesJson,
        botPdb: botPdb,
        topPdb: topPdb,
    };
}"""

new_fetch_scaffold = """async function fetchScaffoldData(code) {
    scaffoldData[code] = {
        json: _EMBEDDED['scaffold_' + code + '_json'],
        pdb: _EMBEDDED['scaffold_' + code + '_pdb'],
        matesJson: _EMBEDDED['mates_' + code + '_json'],
        botPdb: _EMBEDDED['mate_' + code + '_bot_pdb'],
        topPdb: _EMBEDDED['mate_' + code + '_top_pdb'],
    };
}"""

assert old_fetch_scaffold in html, "Could not find fetchScaffoldData function to replace"
html = html.replace(old_fetch_scaffold, new_fetch_scaffold)

# 3c. Replace pipeline.py fetch with embedded data
old_pipeline_fetch = "const pipelineCode = await fetch('./pipeline.py').then(r => r.text());"
new_pipeline_fetch = "const pipelineCode = _EMBEDDED['pipeline_py'];"

assert old_pipeline_fetch in html, "Could not find pipeline.py fetch to replace"
html = html.replace(old_pipeline_fetch, new_pipeline_fetch)

# ── 4. Write output ───────────────────────────────────────────────
out_path = os.path.join(APP_DIR, 'guestscan.html')
with open(out_path, 'w') as f:
    f.write(html)

size_kb = os.path.getsize(out_path) / 1024
print(f"Built {out_path}")
print(f"Size: {size_kb:.0f} KB ({size_kb/1024:.1f} MB)")
print(f"(will be ~{size_kb*0.18:.0f} KB gzipped over the wire)")
