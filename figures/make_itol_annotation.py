#!/usr/bin/env python3
"""
Generate iTOL annotation files so the two tree panels (Fig 4) are (a) drawn on the
SAME 107-tip taxon set as the headline RF comparison, and (b) clade-colored with the
SAME palette as the matplotlib panels, with a legend.

Outputs -> figures/paper_assets/itol/:
  reference_tree_pruned.nwk   copy of the matched 107-tip reference (platypus excluded,
                              synonyms reconciled) - UPLOAD THIS as the reference tree,
                              NOT the full MammalsPhylogeny.nwk.
  predicted_tree_UPGMA.nwk    copy of the predicted tree (already 107 tips).
  clade_colorstrip.txt        iTOL DATASET_COLORSTRIP (colored ring + legend) - upload
                              to BOTH trees so clade colors match Fig 5.
  clade_label_colors.txt      iTOL TREE_COLORS (colors each tip LABEL by clade).
  README_itol.md              step-by-step.

Clade colors come from pubstyle.CLADE_COLORS (the matplotlib palette); the two small
Laurasiatheria clades (Eulipotyphla n=1, Perissodactyla n=2) get two extra Tol-muted
hues. No data is altered - this only maps tip -> clade -> color.

Run: MPLCONFIGDIR=$PWD/.cache/matplotlib python3 figures/make_itol_annotation.py
"""
from __future__ import annotations
import os
import re
import shutil
import sys

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(FIG_DIR)
sys.path.insert(0, FIG_DIR)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import pubstyle  # noqa: E402
from elastic_net_regression_stratified_species_cv import build_clade_mapping  # noqa: E402

CV = os.path.join(REPO, "results", "classification_cv_tree_reconstruction")
REF_NWK = os.path.join(CV, "reference_tree_pruned.nwk")     # matched 107-tip reference
PRED_NWK = os.path.join(CV, "predicted_tree_UPGMA.nwk")
OUT = os.path.join(FIG_DIR, "paper_assets", "itol")

# clade -> hex. pubstyle.CLADE_COLORS covers the 7 enrichment clades + Marsupials/
# Monotremes; add the two small Laurasiatheria clades (Tol-muted olive/purple).
CLADE_HEX = dict(pubstyle.CLADE_COLORS)
CLADE_HEX["Laurasiatheria_Eulipotyphla"] = "#999933"   # olive
CLADE_HEX["Laurasiatheria_Perissodactyla"] = "#AA4499"  # purple
CLADE_HEX["Monotreme_outgroup"] = "#000000"


def tip_labels(nwk: str):
    return re.findall(r'[\(,]([^,\(\):]+):', open(nwk).read())


def main():
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(REF_NWK, os.path.join(OUT, "reference_tree_pruned.nwk"))
    shutil.copy(PRED_NWK, os.path.join(OUT, "predicted_tree_UPGMA.nwk"))

    mapping = build_clade_mapping()                       # {"Genus species": clade}
    tips = tip_labels(REF_NWK)
    pred_tips = set(tip_labels(PRED_NWK))
    assert set(tips) == pred_tips, "reference and predicted tip sets differ"

    rows = []                                             # (tip, clade, hex)
    unmapped = []
    for t in tips:
        sp = t.replace("_", " ")
        clade = mapping.get(sp)
        if clade is None or clade not in CLADE_HEX:
            unmapped.append(t)
            continue
        rows.append((t, clade, CLADE_HEX[clade]))
    if unmapped:
        raise SystemExit(f"{len(unmapped)} tips not mapped to a colored clade: {unmapped}")

    GREY = "#BBBBBB"   # neutral colour for the n<3 "Other" bucket in the 7-clade scheme
    SMALL_CLADES = {"Laurasiatheria_Eulipotyphla", "Laurasiatheria_Perissodactyla"}
    ORDER = ["Afrotheria", "Glires", "Primates", "Marsupials",
             "Laurasiatheria_Carnivora", "Laurasiatheria_Cetartiodactyla",
             "Laurasiatheria_Chiroptera", "Laurasiatheria_Eulipotyphla",
             "Laurasiatheria_Perissodactyla", "Monotreme_outgroup"]

    def _short(c):
        return c.replace("Laurasiatheria_", "Laur. ")

    def write_scheme(suffix, collapse):
        """Write a colorstrip + a label-color file. collapse=True folds the two n<3
        clades (Eulipotyphla, Perissodactyla) into a grey 'Other (n<3)' bucket so the
        legend shows the same 7 clades as the statistical figures (Fig 5)."""
        srows = [(t, ("Other (n<3)" if (collapse and c in SMALL_CLADES) else c))
                 for (t, c, _) in rows]

        def hexof(c):
            return GREY if c == "Other (n<3)" else CLADE_HEX[c]

        present = [c for c in ORDER if any(r[1] == c for r in srows)]
        if any(r[1] == "Other (n<3)" for r in srows):
            present = present + ["Other (n<3)"]
        n = len(present)
        cs = [
            "DATASET_COLORSTRIP", "SEPARATOR TAB", "DATASET_LABEL\tClade",
            "COLOR\t#000000", "STRIP_WIDTH\t30", "MARGIN\t2", "BORDER_WIDTH\t0",
            "SHOW_INTERNAL\t0", "LEGEND_TITLE\tClade",
            "LEGEND_SHAPES\t" + "\t".join(["1"] * n),
            "LEGEND_COLORS\t" + "\t".join(hexof(c) for c in present),
            "LEGEND_LABELS\t" + "\t".join(_short(c) for c in present),
            "DATA",
        ]
        cs += [f"{t}\t{hexof(c)}\t{_short(c)}" for (t, c) in srows]
        open(os.path.join(OUT, f"clade_colorstrip{suffix}.txt"), "w").write("\n".join(cs) + "\n")
        tc = ["TREE_COLORS", "SEPARATOR TAB", "DATA"]
        tc += [f"{t}\tlabel\t{hexof(c)}\tbold" for (t, c) in srows]
        open(os.path.join(OUT, f"clade_label_colors{suffix}.txt"), "w").write("\n".join(tc) + "\n")
        return present

    present9 = write_scheme("", collapse=False)             # 9-clade (every clade distinct)
    present7 = write_scheme("_7clade", collapse=True)        # 7-clade + grey Other(n<3)

    # ---- README ----
    hex_lines = "\n".join(f"- {_short(c)}: `{CLADE_HEX[c]}`" for c in present9)
    readme = f"""# iTOL package - matched taxa + color sync + scale

Both Fig-4 tree panels MUST be the **same 107-tip taxon set** (platypus excluded,
synonyms reconciled) and use the **same clade palette as Fig 5**.

## Upload
1. Reference panel: upload **`reference_tree_pruned.nwk`** (107 tips) - NOT the full
   `MammalsPhylogeny.nwk` (that one has 108 tips incl. platypus + unreconciled
   synonyms).
2. Predicted panel: upload **`predicted_tree_UPGMA.nwk`** (107 tips).
3. On EACH tree, drag in a colorstrip (colored ring + legend) and/or a label-color
   file. Use the SAME file on both trees.

## Two colour schemes (pick one, use it on BOTH trees)
- **9-clade** (`clade_colorstrip.txt` / `clade_label_colors.txt`): every clade its
  own colour, incl. Eulipotyphla (n=1) and Perissodactyla (n=2). Most informative;
  the two horses then show as their own colour where the predicted tree misplaces
  them (an honest illustration).
- **7-clade** (`clade_colorstrip_7clade.txt` / `clade_label_colors_7clade.txt`):
  the 7 clades tested in Fig 5 (n>=3) keep their colours; the 3 species in the two
  too-small-to-test clades (Galemys pyrenaicus; Equus caballus, Equus asinus) are a
  grey **"Other (n<3)"** bucket, so the tree legend matches the boxplots/enrichment.
  (NOT folded into Carnivora etc. - that would be taxonomically wrong.)

## Clade colors (match `figures/pubstyle.py` CLADE_COLORS); Other(n<3) = `{GREY}`
{hex_lines}

## Scale bar
Label the iTOL scale bar with units and note that the two panels are on different
distance quantities: reference branch lengths are **TimeTree divergence time (MY)**;
the predicted tree is UPGMA on **class-mapped patristic distance (MY)**. Either use a
shared, unit-labeled bar ("100 MY") or state both in the caption.

## Export
Export each as **PDF (vector)** at high resolution; save to
`figures/paper_assets/tree_reference_timetree.pdf` and
`figures/paper_assets/tree_predicted_domain.pdf`, then re-run
`figures/render_composites.py` to rebuild Fig 4.
"""
    open(os.path.join(OUT, "README_itol.md"), "w").write(readme)

    print(f"[itol] 9-clade scheme: {present9}")
    print(f"[itol] 7-clade scheme: {present7}  (Eulipotyphla+Perissodactyla -> grey 'Other (n<3)')")
    print(f"[itol] wrote -> {OUT}")
    print("[itol] files: reference_tree_pruned.nwk, predicted_tree_UPGMA.nwk, "
          "clade_colorstrip.txt + clade_label_colors.txt (9-clade), "
          "clade_colorstrip_7clade.txt + clade_label_colors_7clade.txt (7-clade), "
          "README_itol.md")


if __name__ == "__main__":
    main()
