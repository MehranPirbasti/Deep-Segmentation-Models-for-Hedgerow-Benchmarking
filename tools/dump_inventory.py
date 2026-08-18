"""Refreshes the figure list used by analysis/audit.py from the .tex file."""
#!/usr/bin/env python3
"""
Refresh analysis/manuscript_figures.json from the LaTeX source.

The audit needs to know which figures the manuscript includes and which labels it
references. Shipping that as JSON keeps the audit runnable without the .tex file;
run this whenever the manuscript's figure set changes.

    python tools/dump_inventory.py path/to/paper.tex
"""
import json, re, sys

tex = sys.argv[1] if len(sys.argv) > 1 else "paper.tex"
s = open(tex, encoding="utf-8").read()
s = re.sub(r"(?<!\\)%.*", "", s)          # drop commented-out includes
inv = {
    "slots": re.findall(r"\\includegraphics\[[^\]]*\]\{Fig/([^}]*)\}", s),
    "labels": sorted(set(re.findall(r"\\label\{(fig:[^}]*)\}", s))),
    "refs": sorted(set(re.findall(r"\\ref\{(fig:[^}]*)\}", s))),
}
json.dump(inv, open("analysis/manuscript_figures.json", "w"), indent=1)
print(f"{len(inv['slots'])} image slots, {len(inv['labels'])} labels, "
      f"{len(inv['refs'])} references")
