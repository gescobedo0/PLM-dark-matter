# Multi-label per-ion residue model — spec

Goal: give every candidate residue a **vector** of per-ion scores
`[Zn, Cu, Fe, Mn, Co, Ni]` (not a single binary call, not a percentile), then do
characterization analyses on those vectors. The model is *infrastructure*; the
deliverables are the similarity map, per-protein coherence, and microprotein
ion-profiles — not a benchmark predictor (that stays M-Ionic's job).

## Why multi-label (not multiclass)
Softmax forces one ion per residue and makes them compete — it cannot express
"Cu-ish and a bit Fe-ish". Independent **sigmoid heads per ion** (BCE) let a
residue score high on several ions, which is the real chemistry (metal
promiscuity; Cu/Fe/Ni coordinate similarly).

## Masked labels (resolves "don't punish Cu-being-Fe-ish" without soft labels)
Per ion head X: positives = residues coordinating X; negatives = **non-coordinating
residues only**; residues coordinating a *different* metal are **masked** (excluded
from head X's loss). So the Cu head is never trained to push Fe ligands down →
cross-ion similarity is preserved, not suppressed. The similarity then *emerges*
by scoring held-out other-metal ligands on each head.

## Data (scalable; low-data run is a floor)
- Labels: `catalog/residue_labels.parquet` — already per-ion over the FULL pool
  (23,902 seqs / 111,099 coordinating residues; Zn 72k, Mn 16k, Fe 12k, Cu 7k,
  Co 3k, Ni 455). Non-coordinating residues are negatives.
- `--n-clusters N` selects how many positive clusters to include (default = the
  current 800; up to 3,834). Bumping it means running `05b` residue embedding on
  the larger protein set — the only added cost. More data mainly helps Cu/Fe/Mn/Co;
  Ni is data-starved regardless.
- **Leakage-safe:** GroupKFold on `cluster30`; a protein's residues never split.
- Per-ion `pos_weight` for imbalance.

## Model
MLP: 1280 → LayerNorm → Linear(128) → LeakyReLU → Dropout → LayerNorm → **6 sigmoid
heads**. Masked BCE per head. The 128-D hidden layer is extractable (out-of-fold)
as a supervised representation for an optional atlas view.

## Outputs / analyses

1. **Honesty gate — per-ion held-out AUROC** (leakage-safe CV), plus the AA-matched
   version (coordinating-X H/C/D/E vs non-coordinating H/C/D/E). Tells us, ion by
   ion, which metals are even discriminable *before* trusting any profile.
2. **Metal-similarity map** — mean head-score of each true-ion's held-out ligands
   across all heads → a 6×6 matrix; off-diagonal = chemical similarity/promiscuity.
   Cluster ions by it (expect Cu/Fe/Ni close, Mn/Co, Zn its own).
3. **Per-residue score vectors** — the 6-D vector stored for all residues
   (knowns + Study B), the object every later analysis runs on.
4. **Per-protein coherence** — the real site signal: do a protein's top-scoring
   CHED residues **agree on an ion** (coherent site) or scatter (no site)?
   Coherence = concentration of the summed/aggregated score vector across the top
   residues. High coherence + high score on ion X = a specific-metal site hypothesis;
   noise-averages the weak per-residue signal.
5. **Microprotein ion-profiling** — per ORF: dominant ion(s), coherence, best
   residues, MetalNet2 flag → ranked shortlist of candidate metal-binding
   microproteins *with a predicted metal*.
6. **(optional) supervised hidden-layer atlas** — PCA of out-of-fold 128-D hidden
   activations coloured by ion. Labelled "supervised view — separability = the CV
   number in (1), not the picture" (same discipline as UMAP; never read structure
   off a projection optimized to show it).

## Honesty check to keep in view
The interesting transition metals (Cu/Fe/Mn/Co/Ni) are the weakest, most
overlapping per-residue signals — a clean 6-way split is not coming, and Ni is
tiny. Signal is extracted by **coherence (4) + MetalNet2 consensus**, not raw
per-residue scores. The per-ion AUROC in (1) is the honest arbiter of what's real.

## Scripts
- extend `05b_embed_residues.py` (or a small selector) to embed a chosen positive
  set beyond the 800 (for the data bump).
- new `15_ion_residue_model.py`: masked multi-label MLP + outputs (1)–(6).
