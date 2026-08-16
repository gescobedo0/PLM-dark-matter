# Per-residue phase — spec

The pooled-protein probe hit AUROC 0.83 (= M-Ionic's protein-level number) and
then plateaued, because **mean-pooling averages away the 3–4 residues that carry
the metal signal** (and dimension-wise max-pool is a non-selective "Frankenstein"
vector, which is why max was *worse*, not better). Metal coordination is a local,
per-residue property — so the analysis moves to the residue level, as originally
scoped in the deferred section of the handoff.

## How this differs from M-Ionic (Shenoy et al. 2024)

M-Ionic uses the **same model (ESM-2 650M, layer 33), same data (BioLiP), same
20% clustering, same PDB non-metal negatives**, and reports protein-level AUROC
0.83 — which our pooled pipeline independently reproduced. So we do **not** claim
a better predictor. Our contribution is:

1. **Mechanism / emergence** — M-Ionic used one model + one layer. We sweep model
   size (35M/150M/650M) and depth at the *residue* level: where does coordinating-
   vs-non-coordinating separation emerge? Early/small ⇒ local chemistry (the thesis).
2. **Dark-proteome application** — M-Ionic scores curated PDB proteins; we scan
   human Ribo-seq microproteins for candidate coordinating residues → a ranked,
   structure-checkable shortlist of candidate metalloproteins.
3. **Sharper control** — coordinating His/Cys vs *non-coordinating* His/Cys (same
   residue type): the residue-level analog of the composition baseline ("beyond
   knowing it's a His"). Doubles as the length-confound fix.
4. **M-Ionic as oracle, not rival** — run published M-Ionic (GitHub/Colab) on the
   microproteins; agreement with our probe = convergent evidence.

## Length / candidate-residue confound

- Unit of analysis = **residue**, not protein → protein length is not a per-residue
  covariate.
- Positives = coordinating residues; negatives = **non-coordinating residues of the
  same amino-acid type** (H,C,D,E,…). "More candidate residues" becomes the
  conditioning denominator, not a bias.
- Microprotein scan compares the **distribution of per-candidate-residue scores**
  vs length-matched controls (residue-level Mann–Whitney), never per-protein hit
  counts. Report a length-calibrated expected-false-positive rate too.

## Data prep — one fix required first

Our catalog stored coordinating residues in **PDB numbering** (BioLiP col 8, e.g.
`H94`). Per-residue work needs them aligned to the sequence → re-pull BioLiP
**col 9 (residues renumbered from 1)** for the metal subset and store as
0-based sequence indices. `neworfcatalog`/Swiss-Prot sequences need no mapping.
(BioLiP.txt.gz is cached / re-downloadable; a small reparse of 01's source.)

Scope: residue tensors only for the **metal subset + Study B** (per the handoff:
regenerate per-residue for the metal subset, not all of PDB). ~1,599 Study A +
~1,000 Study B proteins.

## Pipeline

| Script | Does |
|---|---|
| `05b_embed_residues.py` | per-token ESM-2 embeddings (no pooling) for the metal subset + Study B, chosen model/layers → HDF5 as a flat residue table `(protein_id, pos, aa, coord_ion, embedding)`; **fp16** to halve size. Reuses 05's forward pass with `include=per_tok`. |
| `11_residue_probe.py` | residue-level probe: coordinating vs non-coordinating, **GroupKFold on cluster30** (all residues of a protein/cluster stay in one fold). Metrics AUROC/MCC overall + per ion. Runs the **AA-matched control** (coordinating H/C/D/E vs non-coordinating same types) and an **identity baseline** (20-D residue one-hot — embedding must beat it). Repeats across model sizes + layers for the **emergence** curve. |
| `12_microprotein_residue_scan.py` | apply the trained probe to every candidate residue of each microprotein; per-protein best-site score + candidate table; **distribution comparison** microprotein vs length-matched control (Mann–Whitney + Cliff's δ); highlight overlays; ranked shortlist. |
| (optional) `13_mionic_oracle.py` | run/parse M-Ionic predictions on the microprotein FASTA; agreement vs our probe. |

## Storage

Flat residue table per (model, layer). ~384k Study A residues × 1280-D fp16 ≈
~1 GB/layer; store L6 + L33 (early-vs-late, the emergence contrast) ≈ ~2 GB.
Optionally restrict to candidate-AA residues (H,C,D,E,N,Q,S,T,Y,M,K + termini) to
cut ~half. Keyed like the pooled store; per-residue tensors stay regenerable, not
committed.

## Metrics & leakage

- Residue-level AUROC + MCC; per ion (Zn/Cu/Fe/Mn/Co/Ni).
- **Leakage-safe:** GroupKFold on `cluster30`; a protein's residues never split
  across folds.
- Headline comparisons: (a) embedding vs residue-identity baseline; (b) AA-matched
  coordinating-vs-not; (c) emergence across size/depth; (d) microprotein candidate
  residues vs length-matched control.

## Deferred / optional
Structure validation of the top microprotein candidates (AlphaFold → check His/Cys
spatial clustering into a plausible site) — strong orthogonal evidence for a
handful of candidates, ideal thesis figure.
