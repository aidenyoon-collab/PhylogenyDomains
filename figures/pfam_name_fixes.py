"""
Canonical Pfam names for domains whose description is CORRUPTED in the source
data_raw/MammalDomainCount.tsv. That file dropped the leading character (and
truncated) ~2,128 of 23,656 domain descriptions (e.g. "PF18459:roprotein
convertase..." should be "Proprotein convertase..."). The accession (PFxxxxx) is
intact and is what every analysis keys on, so results are unaffected - but the
DISPLAYED descriptions on the figures were garbled.

These 11 are the corrupted domains that actually appear in the paper figures
(RF-importance top-20, ANOVA top-40 heatmap, ANOVA top-8 boxplots). Canonical
names verified from InterPro/Pfam (https://www.ebi.ac.uk/interpro/entry/pfam/<acc>).
Used only to relabel figure ticks/titles; no data is altered.
"""
PFAM_NAME_FIX = {
    "PF18459": "Proprotein convertase subtilisin-like/kexin type 9 C-terminal domain",
    "PF18463": "Proprotein convertase subtilisin-like/kexin type 9 C-terminal domain",
    "PF18464": "Proprotein convertase subtilisin-like/kexin type 9 C-terminal domain",
    "PF14565": "Interleukin 22 IL-10-related T-cell-derived-inducible factor",
    "PF06214": "Signaling lymphocytic activation molecule (SLAM) protein",
    "PF07831": "Pyrimidine nucleoside phosphorylase C-terminal domain",
    "PF20804": "Mastermind-like protein 2, transcriptional activation domain",
    "PF07458": "Sperm protein associated with nucleus, mapped to X chromosome",
    "PF16626": "Linking region between Kunitz_BPTI and I-set on papilin",
    "PF15189": "Meiosis-specific coiled-coil domain-containing protein MEIOC",
    "PF15621": "Proline-rich submaxillary gland androgen-regulated family",
}


def fix_domain_label(raw: str) -> str:
    """'PFxxxxx:description' -> 'PFxxxxx:<canonical>' for the known source-corrupted
    accessions; unchanged otherwise."""
    acc = raw.split(":", 1)[0].strip()
    if acc in PFAM_NAME_FIX:
        return f"{acc}:{PFAM_NAME_FIX[acc]}"
    return raw
