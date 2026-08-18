# metal_probe — ESM-2 metal-binding probe + microprotein geometry

Second PLM-dark-matter workstream (distinct from `sae_fidelity/`). Spec:
`~/Downloads/thesis_pipeline_handoff.md`. Thesis question: **do ESM-2 embeddings
carry metal-binding biophysics beyond memorized coevolution?** Metal coordination
(a few His/Cys/Asp/Glu in a geometry) is the conservation-independent probe.

- **Study A (rigorous core):** logistic-regression probe on pooled ESM-2 650M
  embeddings, metal-binder vs non-binder, cluster-aware `GroupKFold`, beating a
  20-D amino-acid-composition baseline (AUROC + MCC).
- **Study B (exploratory):** where human microproteins fall vs metal-binder
  clusters, with a length-matched control.

## Status

| Step | State |
|---|---|
| 0. Phase 0 counts → targets | ✅ done, signed off |
| 1. Catalog (provenance, dedup, cluster, split map) | ✅ done |
| 2. Embedding + probe + viz pipeline code | ✅ written, mock-validated (CPU) |
| 3. Full ESM-2 650M embeddings (Slurm/GPU) | ⬜ run on cluster |
| 4. Probe + composition baseline; PCA/UMAP | ⬜ (code ready) |
| 5. Size sweep (35M/150M/650M) emergence | ✅ done — 35M 0.805, 650M 0.827 (early L6 peak) |
| 6. Per-residue phase (spec + code) | ✅ built, mock-validated; run on cluster |

### Per-residue phase (`RESIDUE_PHASE_SPEC.md`)

The pooled probe plateaued at 0.83 (= M-Ionic's protein-level AUROC) because pooling
averages away the coordinating residues. The per-residue phase probes at the level
metal coordination lives. `catalog/residue_labels.parquet` (from `01b`, committed) has
the sequence-aligned coordinating residues.

```bash
PARTITION=gpu ENV_SETUP="source ~/miniconda3/bin/activate metalprobe" \
  MODEL=650M ./run_residue_pipeline.sh        # 05b embed residues -> 06b merge -> 11 probe
# emergence across sizes/depth (after 35M/150M residue embeds too):
python 11_residue_probe.py --emergence
```

`results/residue/residue_probe.md` — the make-or-break row is **`aa_matched` embedding
vs identity-baseline**: does the embedding distinguish a coordinating His/Cys from a
non-coordinating one (real site signal), or only detect "it's a His" (identity)?
**Result (650M L33): yes** — embed-mlp 0.933 / embed-logistic 0.897 vs identity 0.766.

Then the discovery step (needs the residue embeddings from above):
```bash
python 12_microprotein_residue_scan.py --model 650M     # scan microproteins
```
Trains the H/C/D/E coordinating scorer on Study A, scores every candidate residue
in the microproteins, and tests them per-residue and per-protein against the
**length-matched** `size_matched_short` control (never per-protein hit counts).
→ `results/microprotein_scan/` — `scan_stats.md`, ranked `candidates.csv` (with
highlighted ORFs + percentile vs control), and figures.

Site-atlas (which-metal characterization; needs residue embeddings):
```bash
python 14_site_atlas.py --model 650M --kmain 3
```
Represents each protein by the mean of its top-k coordinating-scored residues
(length-independent, site-focused). Runs a **pass-gate panel on knowns first**
(ion-type separation + anti-circularity random-CHED control + alignment-free
same-ion/different-cluster convergence), then projects microproteins and assigns
each a nearest ion centroid. → `results/site_atlas/` — `panel_stats.md`,
`microprotein_ion.csv` (ranked, nearest ion per ORF), and the atlas PCA.
Read Panel A + C first; the microprotein `nearest_ion` only means something if
ions separate and non-alignable same-ion proteins co-locate on the knowns.

Multi-label per-ion residue model (which-metal, supervised; needs residue embeddings):
```bash
python 15_ion_residue_model.py --model 650M
```
Independent sigmoid head per ion (a residue can be Cu-ish AND Fe-ish), masked BCE
(other-metal ligands masked per head), GroupKFold on cluster. Background-calibrated
(z above non-coordinating) so data-starved heads can't dominate. → `results/ion_residue/`:
`ion_model_report.md` (per-ion **`discriminate_auroc`** = the honest which-metal number;
`detect_auroc` is just coordination detection), `metal_similarity.png` (chemical
similarity from cross-scores), `protein_coherence.csv` (do a protein's top residues
agree on an ion = a site), `microprotein_profile.csv` (ranked strength×coherence,
dominant ion, MetalNet2 flagged), `residue_scores.parquet`, `hidden_atlas.png`.

## Locked decisions (user sign-off 2026-08-15)

- Positive ions: **Zn / Cu / Fe / Mn / Co / Ni** (Ni only where BioLiP-confirmed).
  Excluded Na/K/Cd/Hg (adventitious), Ca/Mg (structural), heme/Fe-S (cofactor).
- Homology threshold **30%** · microproteins from **`neworfcatalog.csv` v45** ·
  structures **X-ray, resolution < 3.0 Å**.
- **BioLiP replaces MetalPDB** as the metal-site annotation source (MetalPDB
  unreachable from the dev box; BioLiP also supplies coordinating residues +
  resolution + EC, and its biological-relevance curation removes His-tag Ni).

## Pipeline

Run order (all reproducible, local; no aligner needed — clustering uses RCSB's
precomputed 30% sequence-identity clusters):

```bash
python 00_phase0_counts.py      # RCSB per-ion structure/cluster counts
python 00b_merge_report.py      # + BioLiP cross-check -> phase0/PHASE0_REPORT.md
python 01_build_metal_positives.py   # BioLiP metal chains -> positives_pool
python 02_build_negatives.py         # RCSB non-metal Pfam chains -> negatives_pool
python 03_build_studyB.py            # local ORF + Swiss-Prot -> studyB_pool
python 04_assemble_catalog.py        # sample to targets -> catalog.{parquet,fasta}
```

### Embeddings + probe (Slurm/GPU)

**Recommended — one command, correctly ordered** (find your GPU partition with `sinfo`):

```bash
PARTITION=gpu ENV_SETUP="source ~/miniconda3/bin/activate metalprobe" ./run_pipeline.sh
```

This submits the embedding array job **and** a dependent merge+probe+plot job that
runs only after the embeddings finish. Watch with `squeue -u $USER`; results land
in `embeddings/esm2_650M.h5`, `results/probe_results.md`, `results/figures/`.
Repeat with `MODEL=150M` / `MODEL=35M` for the size sweep.

> **`sbatch` is asynchronous.** It queues the job and returns immediately — the
> GPU work happens later on a compute node. If you run the steps manually, you
> **must wait** for the embed job to finish before `06`/`07`/`08`. `run_pipeline.sh`
> enforces this with a Slurm dependency.
>
> **The model download is automatic.** ESM-2 weights (~2.5 GB for 650M) are pulled
> by fair-esm the first time `05` loads the model, into
> `~/.cache/torch/hub/checkpoints/` on the GPU node — there is no separate
> download command. The embed job's outputs are `embeddings/esm2_*_shard*.h5`.

Manual path (each step waits on the previous):

```bash
MODEL=650M sbatch slurm_embed.sh            # EDIT partition + env in the script first
# wait until done:  squeue -u $USER
python 06_merge_embeddings.py --model 650M  # shards -> embeddings/esm2_650M.h5
python 07_probe.py                          # -> results/probe_results.{csv,md}
python 08_visualize.py --model 650M --layer 24 --pooling mean
```

Config in `config.yaml` (models, candidate layers, pooling, max_len, CV). The
non-GPU path (pooling, HDF5 I/O, GroupKFold leakage-safety, probe, baseline,
07_probe CLI) is covered by `tests/` and runs on CPU:
`python tests/test_pooling.py && python tests/test_store.py && python tests/test_pipeline_mock.py`.

### Data sources

Data sources (reachability verified 2026-08-15): RCSB Search + Data/GraphQL APIs,
BioLiP (`download/BioLiP.txt.gz`), RCSB 30% cluster file, UniProt (metal-binding
keyword KW-0479), and in-repo `neworfcatalog.csv` + `data/swissprot_human.fasta`.
Large downloads cached under `data_cache/` (gitignored, regenerable).

## Catalog

`catalog/catalog.parquet` — 2,585 sequences, one row per sequence:

- **Study A** (probe): 800 metal + 799 non-metal = 1,599 seqs across **1,599
  distinct 30% clusters** (true 1-seq-per-cluster; 0 pos/neg cluster overlap).
- **Study B** (exploratory): 299 microproteins / 194 size-matched-short /
  493 general-human background.

Key provenance columns (guardrail #4): `source_db, pdb, chain, entity, uniprot,
organism, resolution, evidence, label, class, ions, coord_residues, ec, go,
orf_type, gene_name, length, cluster30, split_group, label_rule, seq`.

**`split_group`** = the leakage-safe `GroupKFold` key (= cluster30 for Study A;
null for Study B, which is not in the probe). `coord_residues` holds the
per-ion coordinating residues (ready for the deferred per-residue phase).
`protein_cluster_map.csv` is the standalone protein→cluster map.

Known confounds recorded for checking, not hidden: **taxonomic** (Study A spans
435/303 organisms, Study B is 100% human) and **length** (`length` on every row;
Study B control is length-histogram-matched to the microproteins).

## Notable Phase-0 / catalog findings

- BioLiP biological-relevance filter exposed adventitious metals: Ni 0.18
  sites/structure (His-tag Ni), Na 0.08, K 0.19, Cd 0.19, Hg 0.10 → excluded.
- Human reviewed proteome is nearly empty at 30–49 aa (13 + 29 proteins) while
  ~48% of microproteins live there → the "dark matter" gap, quantified; the
  size-matched control is capped at 194 by this real scarcity.
