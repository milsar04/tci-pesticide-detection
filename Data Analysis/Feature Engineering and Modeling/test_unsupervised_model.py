# test_unsupervised_model.py
import os
import sys
import shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unsupervised_model as um


def test_cluster_validation_returns_ari_ami():
    labels = np.array([0, 0, 1, 1, 0, 1])
    y      = np.array([0, 0, 1, 1, 0, 1])  # perfect match -> ARI = 1.0
    val = um.cluster_validation(labels, y, "test")
    assert "ari" in val and "ami" in val
    assert val["n_clusters"] == 2
    assert val["ari"] > 0.99


def test_anomaly_proba_no_nan_unit_range():
    from sklearn.ensemble import IsolationForest
    X = np.random.default_rng(0).normal(size=(30, 5))
    wrapper = um._AnomalyProba(IsolationForest(n_estimators=10, random_state=0))
    wrapper.fit(X)
    p = wrapper.predict_proba(X)
    assert p.shape == (30, 2)
    assert not np.isnan(p).any()
    assert (p >= 0).all() and (p <= 1).all()
    assert abs(p.sum(axis=1) - 1).max() < 1e-9  # rows sum to 1


def test_best_kmeans_picks_correct_k():
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal([0, 0], 0.2, (20, 2)),
                   rng.normal([5, 5], 0.2, (20, 2))])
    _, labels, best_k, scores = um.best_kmeans(X, k_range=range(2, 5), seed=42)
    assert best_k == 2
    assert len(labels) == 40
    assert scores[2] > scores[3]  # clean 2-cluster data -> k=2 has best silhouette


def test_integration_quick_run(tmp_path):
    """quick real-data run: checks new columns are appended and figures are written."""
    fem_dir = os.path.dirname(os.path.abspath(__file__))
    for f in ["model_risk_scores.csv", "model_metrics.csv"]:
        shutil.copy(os.path.join(fem_dir, f), tmp_path / f)
    res = um.run(out_dir=str(tmp_path), quick=True, n_splits=3)
    scores = pd.read_csv(tmp_path / "model_risk_scores.csv")
    assert "cluster_kmeans" in scores.columns
    assert "anomaly_if" in scores.columns
    assert "unsup_risk" in scores.columns
    assert len(scores) == 405
    metrics = pd.read_csv(tmp_path / "model_metrics.csv")
    assert "kmeans_clustering" in metrics["model"].values
    assert "isolation_forest" in metrics["model"].values
    assert os.path.exists(str(tmp_path / "figures" / "pca_embedding.png"))
    assert "kmeans" in res and "if_auc" in res
