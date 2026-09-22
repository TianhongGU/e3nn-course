#!/usr/bin/env python3
"""Inject translated markdown cells back into a notebook copy.

Usage: inject_md.py <original.ipynb> <translations.json> <output.ipynb>
translations.json maps cell-index -> translated markdown string.
Untranslated (missing) cells keep their original source.
"""
import json
import sys

nb_path, tr_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(nb_path) as f:
    nb = json.load(f)
with open(tr_path) as f:
    tr = json.load(f)

count = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and str(i) in tr:
        cell["source"] = tr[str(i)].splitlines(keepends=True)
        count += 1

with open(out_path, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(f"injected {count}/{len(tr)} translations -> {out_path}")
