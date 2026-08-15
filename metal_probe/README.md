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
| 5. Size sweep (35M/150M/650M) emergence | ⬜ (code ready) |

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

```bash
# 1) embed on the GPU cluster (array job; load model once, stream a shard)
MODEL=650M sbatch slurm_embed.sh            # then 150M, 35M for the size sweep
python 06_merge_embeddings.py --model 650M  # merge shards -> embeddings/esm2_650M.h5
# 2) probe + composition baseline over layers x pooling x models
python 07_probe.py                          # -> results/probe_results.{csv,md}
# 3) figures (PCA always; UMAP if umap-learn present)
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
