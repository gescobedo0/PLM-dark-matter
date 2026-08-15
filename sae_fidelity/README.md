# SAE fidelity diagnostic

Go / no-go on using Biohub's ESMC-6B sparse autoencoder
(`biohub/ESMC-6B-sae-layer60-k64-codebook16384`, 16,384 features) as **semantic
axes** for the microproteins in `../neworfcatalog.csv`. The SAE was trained on the
general protein distribution (~350 aa); our ORFs have **median length ~28 aa** and
are largely de-novo / non-conserved, so they may be out of distribution. If the SAE
cannot reconstruct them, its feature activations are noise dressed up as
interpretation. This package runs the diagnostic and reports a documented verdict.
It does **not** build the downstream feature space — that is gated on a "go".

Implements SAE_HANDOFF.md sections 1–4.

## Pipeline

| step | script | output |
|---|---|---|
| subsample + controls + markers | `01_prepare_subsample.py` | `data/subsample.parquet` |
| ESMC-6B layer-60 activations (GPU) | `02_extract_activations.py` | `data/activations/` (resumable shards) |
| SAE forward: FVU + max-pooled features | `03_sae_forward.py` | `data/sae_features/` |
| diagnostics (FVU / magnitude / entropy / markers) | `04_diagnostics.py` | `data/report/results.json`, `fvu_ecdf.png` |
| filled decision table + report | `05_report.py` | `data/report/report.md`, `decision.json` |

Three groups, plus positive controls, all flow through one model load:
- **microprotein** — ~5k ORFs, stratified by subclass (`orf_type` proxy) × length bin.
- **length_matched** — canonical N-terminal fragments truncated to the ORF length
  distribution; isolates the *length* confound from the *biology* one.
- **background** — random full-length human SwissProt; the SAE's home regime, the
  low-FVU anchor.
- **marker** — pre-registered positive controls (PF00096 zinc fingers, PF00037/PF00301
  Fe-S, TM peptides, disordered small proteins).

## Running

Local (no GPU) can run steps 01, 04-numeric, and the tests. Steps 02–03 need a GPU:
use `run_colab.ipynb` (prefer an L4/A100; **do not quantize** the 6B model — it
corrupts the reconstruction signal being measured).

```bash
pip install -r requirements.txt
python 01_prepare_subsample.py --smoke 100    # data prep (network: UniProt)
# 02, 03 on Colab GPU
python 04_diagnostics.py
python 05_report.py
```

## Verification order (do not skip)

1. **Background reconstructs** with low FVU. If not, the loader / layer-60 index /
   SAE weights are wrong — fix before interpreting anything.
2. **Zinc-finger set fires a metal feature.** Validates feature semantics end to end.
3. Only then trust the microprotein FVU and the decision.

`python tests/test_core.py` and `python tests/test_pipeline_mock.py` exercise the
math and the full 03→04→05 plumbing on CPU (no GPU, no network).

## Method notes / deviations

- **Max-pool, not mean-pool** (handoff rule). The precedent paper (arXiv 2606.12209)
  mean-pools for full-length enzymes; microprotein signals are local motifs, so
  mean-pooling both dilutes them and reintroduces the length confound.
- **FVU is our own diagnostic** (not in that paper). Baseline = background-group
  activation mean; raw MSE is reported alongside as a baseline-free cross-check.
- **Activation *count* is never used** as an OOD signal — a top-k=64 SAE fires
  exactly 64 features per residue by construction. We use magnitude + feature-usage
  entropy + description informativeness instead.

## Unconfirmed items to check on first GPU run (see plan)

- HF loader class for `biohub/ESMC-6B` (`AutoModel` vs the `esm` SDK) — `sae_lib/esmc.py`.
- The `hidden_states` index for "layer 60" (off-by-one) — validated by step-1 above.
- SAE checkpoint key names + FVU baseline convention — `sae_lib/sae.py` probes several.
- Feature-table column names — `sae_lib/features.py` auto-detects; pin if wrong.

## Open design choices (flagged, non-blocking)

- ORF min-length floor (default ≥10 aa; `config.yaml`).
- Length control = truncated canonical fragments (documented tradeoff).
- Conserved-vs-ORFan split uses `orf_type` as a proxy; MMseqs2 homology is a follow-on.
