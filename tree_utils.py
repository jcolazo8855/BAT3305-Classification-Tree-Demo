from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


DATASET_LABELS = {
    "Linear + noise": "Mostly linear separation with overlap; useful for observing when extra depth becomes unnecessary.",
    "Two moons": "Curved class structure that requires multiple axis-aligned splits.",
    "Concentric circles": "Nested nonlinear classes; trees approximate circles with step-like rectangular regions.",
    "Checkerboard": "Alternating regions that reward deeper partitioning and reveal overfitting risk.",
}


@dataclass(frozen=True)
class TreeResults:
    model: DecisionTreeClassifier
    metrics: Dict[str, float]
    confusion: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    tree_depth: int
    leaf_count: int


def make_dataset(name: str, n_samples: int, noise: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    if name == "Linear + noise":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=max(0.55, 1.65 - 1.5 * noise),
            flip_y=min(0.28, noise * 0.28),
            random_state=random_state,
        )
        X[:, 0] *= 1.35
    elif name == "Two moons":
        X, y = make_moons(n_samples=n_samples, noise=max(0.01, noise), random_state=random_state)
        X = X * np.array([2.0, 1.7])
    elif name == "Concentric circles":
        X, y = make_circles(
            n_samples=n_samples,
            noise=max(0.01, noise),
            factor=0.42,
            random_state=random_state,
        )
        X = X * 2.25
    elif name == "Checkerboard":
        rng = np.random.default_rng(random_state)
        X = rng.uniform(-2.4, 2.4, size=(n_samples, 2))
        cell_x = np.floor((X[:, 0] + 2.4) / 1.2).astype(int)
        cell_y = np.floor((X[:, 1] + 2.4) / 1.2).astype(int)
        y = ((cell_x + cell_y) % 2).astype(int)
        flip = rng.random(n_samples) < min(0.30, noise * 0.32)
        y = np.where(flip, 1 - y, y)
        X += rng.normal(0, noise * 0.10, size=X.shape)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def fit_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str,
    max_depth: int,
    min_samples_leaf: int,
    ccp_alpha: float,
    test_size: float,
    random_state: int,
) -> TreeResults:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        ccp_alpha=float(ccp_alpha),
        random_state=int(random_state),
    )
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {
        "Train accuracy": accuracy_score(y_train, train_pred),
        "Test accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred, zero_division=0),
        "Recall": recall_score(y_test, test_pred, zero_division=0),
        "F1": f1_score(y_test, test_pred, zero_division=0),
    }

    return TreeResults(
        model=model,
        metrics=metrics,
        confusion=confusion_matrix(y_test, test_pred, labels=[0, 1]),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tree_depth=model.get_depth(),
        leaf_count=model.get_n_leaves(),
    )


def decision_grid(X: np.ndarray, resolution: int = 240, padding: float = 0.8):
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, int(resolution)),
        np.linspace(y_min, y_max, int(resolution)),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def classification_path(model: DecisionTreeClassifier, x: np.ndarray) -> List[Dict]:
    x = np.asarray(x, dtype=float).reshape(1, -1)
    tree = model.tree_
    node_indicator = model.decision_path(x)
    leaf_id = int(model.apply(x)[0])
    nodes = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    steps: List[Dict] = []

    for node_id in nodes:
        if node_id == leaf_id:
            counts = tree.value[node_id][0]
            pred = int(np.argmax(counts))
            steps.append({
                "type": "leaf",
                "node": int(node_id),
                "class_counts": [int(round(v)) for v in counts],
                "prediction": pred,
            })
            continue

        feature = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])
        direction = "left" if x[0, feature] <= threshold else "right"
        steps.append({
            "type": "split",
            "node": int(node_id),
            "feature": feature,
            "feature_name": f"Feature {feature + 1}",
            "threshold": threshold,
            "direction": direction,
        })
    return steps


def pruning_curve(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str,
    min_samples_leaf: int,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )
    base = DecisionTreeClassifier(
        criterion=criterion,
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(random_state),
    )
    path = base.cost_complexity_pruning_path(X_train, y_train)
    alphas = np.unique(path.ccp_alphas)

    if len(alphas) > 18:
        idx = np.linspace(0, len(alphas) - 1, 18).astype(int)
        alphas = alphas[idx]

    rows = []
    for alpha in alphas:
        model = DecisionTreeClassifier(
            criterion=criterion,
            min_samples_leaf=int(min_samples_leaf),
            ccp_alpha=float(alpha),
            random_state=int(random_state),
        )
        model.fit(X_train, y_train)
        rows.append({
            "alpha": float(alpha),
            "leaves": int(model.get_n_leaves()),
            "depth": int(model.get_depth()),
            "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
            "test_accuracy": float(accuracy_score(y_test, model.predict(X_test))),
        })
    return pd.DataFrame(rows)
