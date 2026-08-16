"""Study A linear probe + composition baseline, with cluster-aware CV.

The probe is logistic regression on pooled vectors; the baseline is the same
probe on 20-D amino-acid composition (the classical-motif control the embedding
must beat). Cross-validation is GroupKFold on cluster id (split_group) so
near-identical sequences never straddle a fold (guardrail #1). Metrics: AUROC
and MCC, the spec's reported pair. Probe accuracy is also the only fair way to
compare embedding spaces of different dimensionality across the size sweep.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}


def aa_composition(seqs):
    """(N, 20) amino-acid fraction vectors (non-standard residues ignored)."""
    X = np.zeros((len(seqs), 20), dtype=np.float32)
    for i, s in enumerate(seqs):
        for c in s:
            j = AA_IDX.get(c)
            if j is not None:
                X[i, j] += 1.0
        tot = X[i].sum()
        if tot:
            X[i] /= tot
    return X


def cluster_cv(X, y, groups, *, n_splits=5, C=1.0, max_iter=2000,
               standardize=True, seed=0):
    """GroupKFold logistic probe -> dict with mean/std AUROC & MCC and folds.

    n_splits is capped at the number of distinct groups. Folds missing a class
    in y_true are skipped for AUROC (still counted for MCC).
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    k = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=k)

    aurocs, mccs = [], []
    for tr, te in gkf.split(X, y, groups):
        steps = []
        if standardize:
            steps.append(StandardScaler())
        steps.append(LogisticRegression(C=C, max_iter=max_iter,
                                        class_weight="balanced"))
        clf = make_pipeline(*steps)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        if len(np.unique(y[te])) == 2:
            aurocs.append(roc_auc_score(y[te], proba))
        mccs.append(matthews_corrcoef(y[te], pred))
    return {
        "auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
        "auroc_std": float(np.std(aurocs)) if aurocs else float("nan"),
        "mcc_mean": float(np.mean(mccs)),
        "mcc_std": float(np.std(mccs)),
        "n_folds": k,
        "n_groups": int(n_groups),
    }


def mlp_cluster_cv(X, y, groups, *, n_splits=5, hidden=128, dropout=0.2,
                   epochs=30, lr=1e-3, seed=0):
    """Small MLP (LayerNorm->Linear->LeakyReLU->dropout->Linear), M-Ionic-flavoured,
    with the same GroupKFold + AUROC/MCC contract as cluster_cv. CPU-friendly."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    X = np.asarray(X, np.float32)
    y = np.asarray(y).astype(np.float32)
    groups = np.asarray(groups)
    k = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=k)
    aurocs, mccs = [], []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr = torch.tensor(sc.transform(X[tr])); Xte = torch.tensor(sc.transform(X[te]))
        ytr = torch.tensor(y[tr])
        pos_w = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())])
        net = nn.Sequential(nn.LayerNorm(X.shape[1]), nn.Linear(X.shape[1], hidden),
                            nn.LeakyReLU(), nn.Dropout(dropout),
                            nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(net(Xtr).squeeze(1), ytr)
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            proba = torch.sigmoid(net(Xte).squeeze(1)).numpy()
        pred = (proba >= 0.5).astype(int)
        if len(np.unique(y[te])) == 2:
            aurocs.append(roc_auc_score(y[te], proba))
        mccs.append(matthews_corrcoef(y[te].astype(int), pred))
    return {"auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
            "auroc_std": float(np.std(aurocs)) if aurocs else float("nan"),
            "mcc_mean": float(np.mean(mccs)), "mcc_std": float(np.std(mccs)),
            "n_folds": k, "n_groups": int(len(np.unique(groups)))}


def assert_no_group_leakage(groups, X, y, n_splits=5):
    """Verify GroupKFold never puts a group in both train and test."""
    groups = np.asarray(groups)
    k = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=k)
    for tr, te in gkf.split(X, y, groups):
        if set(groups[tr]) & set(groups[te]):
            return False
    return True
