"""Interactive / point-level exploration of the catalog in embedding space.

CPU-only; consumes the precomputed embeddings HDF5 + catalog. Two lenses:
  - pca2d + make_figure : an interactive map (hover reveals protein identity).
  - neighbors           : full-D nearest neighbours of a chosen protein (the
                          rigorous per-point view; PCA is lossy and PC1~length).

Plotting deps (plotly) are imported lazily so load/pca2d/neighbors work without
plotly installed (and are unit-testable).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

from . import store

ROOT = Path(__file__).resolve().parents[1]
HOVER_COLS = ["class", "ions", "length", "ec", "organism", "gene_name", "orf_type"]


def load_embeddings(model="650M", layer=None, pooling="mean",
                    config_path=None, catalog_path=None):
    """-> (meta: DataFrame indexed by protein_id, E: (N,D) float64, ids: list).

    meta rows are aligned to E rows. layer defaults to the deepest stored layer.
    """
    cfg = yaml.safe_load(open(config_path or ROOT / "config.yaml"))
    cat = pd.read_parquet(catalog_path or ROOT / cfg["catalog"]).set_index("protein_id")
    h5 = ROOT / cfg["embed_dir"] / f"esm2_{model}.h5"
    if layer is None:
        layer = max(store.available(h5)[0])
    ids, arr = store.read(h5, layer, pooling)
    keep = [i for i in ids if i in cat.index]
    pos = {i: k for k, i in enumerate(ids)}
    E = arr[[pos[i] for i in keep]].astype(np.float64)
    meta = cat.loc[keep].copy()
    meta["_layer"] = layer
    meta["_pooling"] = pooling
    return meta, E, keep


def pca2d(E, seed=0):
    """Standardize + PCA to 2 comps. -> (coords (N,2), explained_var_pct (2,))."""
    Xs = StandardScaler().fit_transform(E)
    p = PCA(n_components=2, random_state=seed).fit(Xs)
    return p.transform(Xs), p.explained_variance_ratio_ * 100


def neighbors(query_id, ids, E, meta, k=15, restrict=None):
    """Full-D cosine nearest neighbours of query_id.

    restrict: None (any), "metal" (label==1), "non_metal", or a class name.
    -> DataFrame (nearest first) with distance + identity columns.
    """
    idx = {i: n for n, i in enumerate(ids)}
    if query_id not in idx:
        raise KeyError(f"{query_id} not in embeddings")
    En = normalize(E)
    q = En[idx[query_id]]
    sims = En @ q                                   # cosine similarity
    order = np.argsort(-sims)
    rows = []
    for j in order:
        pid = ids[j]
        if pid == query_id:
            continue
        m = meta.loc[pid]
        if restrict == "metal" and m.get("label") != 1:
            continue
        if restrict == "non_metal" and m.get("label") != 0:
            continue
        if restrict not in (None, "metal", "non_metal") and m.get("class") != restrict:
            continue
        rows.append({"protein_id": pid, "cos_dist": float(1 - sims[j]),
                     "class": m.get("class"), "ions": m.get("ions"),
                     "length": m.get("length"), "ec": m.get("ec"),
                     "organism": m.get("organism")})
        if len(rows) >= k:
            break
    return pd.DataFrame(rows)


def metal_ion_profile(query_id, ids, E, meta, k=15):
    """Of a query's k nearest metal binders, tally which ions -> a quick call on
    'if anything, which metal is this protein near?'."""
    nb = neighbors(query_id, ids, E, meta, k=k, restrict="metal")
    ions = {}
    for s in nb["ions"].dropna():
        for ion in str(s).split("+"):
            ions[ion] = ions.get(ion, 0) + 1
    return dict(sorted(ions.items(), key=lambda x: -x[1])), nb["cos_dist"].median()


def make_figure(coords, meta, color="class", title="PCA", highlight_col="highlight"):
    """Interactive Plotly scatter; hover reveals protein identity. Returns a fig."""
    import plotly.express as px
    import plotly.graph_objects as go
    df = meta.copy()
    df["PC1"], df["PC2"] = coords[:, 0], coords[:, 1]
    df["protein_id"] = df.index
    hover = [c for c in HOVER_COLS if c in df.columns]
    fig = px.scatter(df, x="PC1", y="PC2", color=color, hover_name="protein_id",
                     hover_data=hover, opacity=0.6, title=title,
                     color_discrete_sequence=px.colors.qualitative.Set2
                     if df[color].dtype == object else None)
    fig.update_traces(marker=dict(size=6))
    if highlight_col in df.columns and df[highlight_col].fillna(False).any():
        h = df[df[highlight_col].fillna(False)]
        fig.add_trace(go.Scatter(
            x=h["PC1"], y=h["PC2"], mode="markers", name="highlighted",
            marker=dict(symbol="star", size=16, color="black",
                        line=dict(width=1, color="white")),
            text=h["protein_id"], hoverinfo="text"))
    fig.update_layout(width=900, height=700, legend_title=color)
    return fig


def write_html(fig, path):
    """Self-contained interactive HTML (plotly.js embedded) — open anywhere."""
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    return path
