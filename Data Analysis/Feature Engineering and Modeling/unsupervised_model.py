# unsupervised_model.py - unsupervised track (Plan B): clustering + anomaly scoring.
# pure-sklearn (no HDBSCAN/UMAP per the hybrid-dependency decision).
# reuses model_features.imputed_scaled_view + model_eval.make_splitter unchanged.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                              adjusted_mutual_info_score, roc_auc_score,
                              average_precision_score)

import model_features as mf
import model_eval as me

HERE = os.path.dirname(os.path.abspath(__file__))
K_RANGE = range(2, 9)  # ponytail: 2..8 covers typical cluster counts for this dataset size


# pca ----

def fit_pca(X_scaled, variance=0.95):
    """fit pca retaining `variance` fraction of explained variance; also 2d for plotting."""
    pca_full = PCA(n_components=variance, svd_solver="full", random_state=42).fit(X_scaled)
    pca_2d = PCA(n_components=2, random_state=42).fit(X_scaled)
    return pca_full, pca_2d


# clustering ----

def best_kmeans(X_pca, k_range=K_RANGE, seed=42):
    """sweep k by silhouette score; return (fitted model, labels, best_k, score_dict)."""
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_pca)
        scores[k] = silhouette_score(X_pca, labels)
    best_k = max(scores, key=scores.get)
    km_best = KMeans(n_clusters=best_k, random_state=seed, n_init=10).fit(X_pca)
    return km_best, km_best.labels_, best_k, scores


def best_gmm(X_pca, k_range=K_RANGE, seed=42):
    """sweep k by bic (lower = better); return (fitted model, labels, best_k, bic_dict)."""
    scores = {}
    for k in k_range:
        gm = GaussianMixture(n_components=k, random_state=seed, n_init=3, reg_covar=1e-3)
        gm.fit(X_pca)
        scores[k] = gm.bic(X_pca)
    best_k = min(scores, key=scores.get)
    gm_best = GaussianMixture(n_components=best_k, random_state=seed, n_init=3,
                               reg_covar=1e-3).fit(X_pca)
    return gm_best, gm_best.predict(X_pca), best_k, scores


def cluster_validation(labels, y, name):
    """ari + ami validation for a clustering vs the treatment labels."""
    y = np.asarray(y, int)
    return {
        "method": name,
        "n_clusters": int(len(np.unique(labels))),
        "ari": float(adjusted_rand_score(y, labels)),
        "ami": float(adjusted_mutual_info_score(y, labels)),
    }


# anomaly scoring ----

class _AnomalyProba:
    """wraps score_samples into predict_proba(X)[:, 1] = P(anomalous), ignores y on fit.
    useful for passing anomaly scorers into model_eval.evaluate_estimator if needed."""

    def __init__(self, scorer):
        self.scorer = scorer

    def fit(self, X, y=None):
        self.scorer.fit(X)
        return self

    def predict_proba(self, X):
        # score_samples: higher = more normal; flip so col 1 = anomalous.
        s = -np.asarray(self.scorer.score_samples(X))
        lo, hi = s.min(), s.max()
        s = (s - lo) / (hi - lo + 1e-12)
        return np.column_stack([1 - s, s])


def anomaly_auc_cv(scorer, X_scaled, y, groups, n_splits=5, seed=42):
    """grouped cv anomaly auc (fit on train, score on test, no label leak in fit).
    returns roc_auc_mean/std, pr_auc_mean/std, and a direction string:
    'tracks_treated' means anomalous -> treated (AUC >= 0.5);
    'tracks_untreated' means anomalous -> untreated (AUC < 0.5, expected when treated is majority)."""
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    splitter = me.make_splitter(n_splits=n_splits, seed=seed)
    aucs, praucs = [], []
    for tr, te in splitter.split(X_scaled, y, groups):
        fitted = clone(scorer).fit(X_scaled[tr])
        # flip sign: score_samples higher = more normal; we want higher = more anomalous.
        s = -fitted.score_samples(X_scaled[te])
        aucs.append(roc_auc_score(y[te], s))
        praucs.append(average_precision_score(y[te], s))
    direction = "tracks_treated" if float(np.mean(aucs)) >= 0.5 else "tracks_untreated"
    return {
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "pr_auc_mean": float(np.mean(praucs)),
        "pr_auc_std": float(np.std(praucs, ddof=1)) if len(praucs) > 1 else 0.0,
        "direction": direction,
    }


# figures ----

def _fig_pca_embedding(X_2d, y, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, label, color in [(0, "untreated", "steelblue"), (1, "treated", "tomato")]:
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=label, alpha=0.5, s=20)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA 2D - treated vs untreated")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "pca_embedding.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_cluster_treated_rate(labels_km, labels_gm, y, out_dir):
    overall = float(np.mean(y))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, labels, title in [(axes[0], labels_km, "KMeans"),
                               (axes[1], labels_gm, "GaussianMixture")]:
        clusters = np.unique(labels)
        rates = [float(np.mean(y[labels == c])) for c in clusters]
        ax.bar(clusters, rates, color="steelblue")
        ax.axhline(overall, color="red", linestyle="--", label=f"overall ({overall:.2f})")
        ax.set_xlabel("cluster")
        ax.set_ylabel("treated rate")
        ax.set_title(f"{title} - treated rate per cluster")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cluster_treated_rate.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_anomaly_scores(if_scores, lof_scores, y, out_dir):
    for name, s in [("isolation_forest", if_scores), ("lof", lof_scores)]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for cls, label, color in [(0, "untreated", "steelblue"), (1, "treated", "tomato")]:
            ax.hist(s[y == cls], bins=30, alpha=0.5, label=label, color=color, density=True)
        ax.set_xlabel("anomaly score (higher = more anomalous)")
        ax.set_ylabel("density")
        ax.set_title(f"{name} - anomaly score by treatment")
        ax.legend()
        fig.savefig(os.path.join(out_dir, f"anomaly_{name}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)


# main run ----

def run(out_dir=HERE, n_splits=5, seed=42, quick=False):
    """plan b: clustering + anomaly scoring, validated against labels.
    reads existing model_risk_scores.csv and model_metrics.csv and appends; idempotent."""
    df = mf.filter_modeling_rows(mf.load_descriptors())
    X, y, groups, meta = mf.build_xy(df)
    y_np = np.asarray(y, int)
    groups_np = np.asarray(groups)

    # imputed + scaled view for distance-based methods.
    X_scaled, _imp, _scl = mf.imputed_scaled_view(X)

    # pca: reduce to 95% variance for clustering; 2D for the embedding figure.
    pca_full, pca_2d = fit_pca(X_scaled)
    n_components = pca_full.n_components_
    X_pca = pca_full.transform(X_scaled)
    X_2d = pca_2d.transform(X_scaled)

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    _fig_pca_embedding(X_2d, y_np, fig_dir)

    # clustering: sweep k on the PCA-reduced space.
    k_range = range(2, 4) if quick else K_RANGE
    _, labels_km, best_k_km, _ = best_kmeans(X_pca, k_range=k_range, seed=seed)
    _, labels_gm, best_k_gm, _ = best_gmm(X_pca, k_range=k_range, seed=seed)
    _fig_cluster_treated_rate(labels_km, labels_gm, y_np, fig_dir)

    val_km = cluster_validation(labels_km, y_np, f"kmeans_k{best_k_km}")
    val_gm = cluster_validation(labels_gm, y_np, f"gmm_k{best_k_gm}")

    # anomaly scoring: grouped cv AUC, then refit on all data for exported scores.
    n_estimators = 50 if quick else 200
    n_neighbors = 10 if quick else 20
    n_splits_auc = 3 if quick else n_splits

    if_ = IsolationForest(n_estimators=n_estimators, contamination="auto",
                          random_state=seed, n_jobs=1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)

    if_auc = anomaly_auc_cv(if_, X_scaled, y_np, groups_np, n_splits=n_splits_auc, seed=seed)
    lof_auc = anomaly_auc_cv(lof, X_scaled, y_np, groups_np, n_splits=n_splits_auc, seed=seed)

    if_.fit(X_scaled)
    lof.fit(X_scaled)
    if_scores = -if_.score_samples(X_scaled)
    lof_scores = -lof.score_samples(X_scaled)
    # ponytail: simple average of both scorers; weighted combination if single-scorer output needed
    combined = (if_scores + lof_scores) / 2

    _fig_anomaly_scores(if_scores, lof_scores, y_np, fig_dir)

    # append new columns to model_risk_scores.csv - join on identity key, not positional.
    risk_path = os.path.join(out_dir, "model_risk_scores.csv")
    scores = pd.read_csv(risk_path)
    new_cols = pd.DataFrame({
        "PMT_SITE": meta["PMT_SITE"].values,
        "window": meta["window"].astype(int).values,
        "year": meta["year"].astype(int).values,
        "cluster_kmeans": labels_km,
        "cluster_gmm": labels_gm,
        "anomaly_if": if_scores,
        "anomaly_lof": lof_scores,
        "unsup_risk": combined,
    })
    # idempotent: drop any prior unsupervised columns before re-merging.
    for col in ["cluster_kmeans", "cluster_gmm", "anomaly_if", "anomaly_lof", "unsup_risk"]:
        if col in scores.columns:
            scores = scores.drop(columns=col)
    scores["window"] = scores["window"].astype(int)
    scores["year"] = scores["year"].astype(int)
    scores = scores.merge(new_cols, on=["PMT_SITE", "window", "year"], how="left")
    scores.to_csv(risk_path, index=False, float_format="%.4f")

    # append rows to model_metrics.csv (idempotent: remove prior unsupervised rows first).
    metrics_path = os.path.join(out_dir, "model_metrics.csv")
    existing = pd.read_csv(metrics_path)
    unsup_names = {"kmeans_clustering", "gmm_clustering", "isolation_forest", "lof"}
    existing = existing[~existing["model"].isin(unsup_names)]
    new_rows = [
        {"model": "kmeans_clustering", "ari": val_km["ari"], "ami": val_km["ami"],
         "n_clusters": val_km["n_clusters"]},
        {"model": "gmm_clustering", "ari": val_gm["ari"], "ami": val_gm["ami"],
         "n_clusters": val_gm["n_clusters"]},
        {"model": "isolation_forest", "roc_auc_mean": if_auc["roc_auc_mean"],
         "roc_auc_std": if_auc["roc_auc_std"], "pr_auc_mean": if_auc["pr_auc_mean"],
         "pr_auc_std": if_auc["pr_auc_std"], "direction": if_auc["direction"]},
        {"model": "lof", "roc_auc_mean": lof_auc["roc_auc_mean"],
         "roc_auc_std": lof_auc["roc_auc_std"], "pr_auc_mean": lof_auc["pr_auc_mean"],
         "pr_auc_std": lof_auc["pr_auc_std"], "direction": lof_auc["direction"]},
    ]
    updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    updated.to_csv(metrics_path, index=False, float_format="%.4f")

    print(f"PCA: {n_components} components retain >= 95% variance")
    print(f"KMeans best k={best_k_km} (silhouette) | GMM best k={best_k_gm} (BIC)")
    print(f"IF anomaly ROC-AUC {if_auc['roc_auc_mean']:.3f} ({if_auc['direction']})")
    print(f"LOF anomaly ROC-AUC {lof_auc['roc_auc_mean']:.3f} ({lof_auc['direction']})")

    return {"kmeans": val_km, "gmm": val_gm, "if_auc": if_auc, "lof_auc": lof_auc}


if __name__ == "__main__":
    run()
