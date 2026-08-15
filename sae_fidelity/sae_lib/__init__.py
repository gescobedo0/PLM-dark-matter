"""SAE fidelity diagnostic library for microprotein characterization.

Modules:
  config      - load config.yaml
  catalog     - parse neworfcatalog.csv, clean, label
  controls    - length-matched canonical + background sets
  esmc        - ESMC-6B layer-60 activation extraction (GPU)
  sae         - TopK SAE forward + FVU
  pooling     - per-residue -> per-protein (max; NOT mean)
  features    - SAE feature-description table + informativeness
  markers     - positive-control sequence sets
  diagnostics - FVU / magnitude / entropy / marker recovery
"""
