#!/usr/bin/env python3
"""Build a bilingual notebook: Chinese first, original English collapsed in <details>.

Usage: make_bilingual.py <original.ipynb> <translations.json> <output.ipynb>
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
        zh = tr[str(i)].rstrip("\n")
        en = "".join(cell["source"]).rstrip("\n")
        cell["source"] = (
            zh + "\n\n<details><summary>English</summary>\n\n" + en + "\n\n</details>"
        ).splitlines(keepends=True)
        count += 1

with open(out_path, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(f"bilingual cells: {count}/{len(tr)} -> {out_path}")
