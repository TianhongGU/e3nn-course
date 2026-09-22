#!/usr/bin/env python3
"""Extract markdown cell sources from a notebook into a JSON file for translation."""
import json
import sys

nb_path, out_path = sys.argv[1], sys.argv[2]
with open(nb_path) as f:
    nb = json.load(f)

cells = {}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        cells[str(i)] = "".join(cell["source"])

with open(out_path, "w") as f:
    json.dump(cells, f, ensure_ascii=False, indent=1)

n_chars = sum(len(v) for v in cells.values())
print(f"{len(cells)} markdown cells, {n_chars} chars total")
