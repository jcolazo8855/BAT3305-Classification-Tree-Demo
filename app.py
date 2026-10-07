from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.tree import plot_tree

from tree_utils import (
    DATASET_LABELS,
    classification_path,
    decision_grid,
    fit_and_evaluate,
    fit_random_forest,
    make_dataset,
    pruning_curve,
)

st.set_page_config(
    page_title="BAT 3305 - Colazo | Classification Tree Lab",
    page_icon="🌳",
    layout="wide",
)

st.markdown(
    """
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px;}
    .hero {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        padding: 1.15rem 1.35rem;
        margin-bottom: .8rem;
        background: linear-gradient(135deg, rgba(75,100,75,.10), rgba(90,90,90,.02));
    }
    .hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: -.03em;}
    .hero p {margin: .25rem 0 0 0; opacity: .78; font-size: 1.05rem;}
    .concept-card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        padding: .85rem 1rem;
        height: 100%;
    }
    .small-note {opacity: .73; font-size: .9rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        padding: .75rem .85rem;
        border-radius: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>BAT 3305 - Colazo</h1>
  <p><strong>Classification Trees:</strong> Splits, Depth, Impurity & Pruning</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Experiment controls")
    dataset_name = st.selectbox(
        "Dataset",
        list(DATASET_LABELS.keys()),
        index=0,
        help="Each dataset highlights a different strength or weakness of decision trees.",
    )
    st.caption(DATASET_LABELS[dataset_name])

    n_samples = st.slider("Observations", 80, 500, 220, 20)
    noise = st.slider("Noise", 0.00, 0.60, 0.16, 0.01)
    random_state = st.number_input("Random seed", 0, 9999, 42, 1)
    test_size = st.slider("Test-set share", 0.15, 0.40, 0.25, 0.05)

    st.divider()
    st.subheader("Tree settings")
    criterion_label = st.radio("Split criterion", ["Gini", "Entropy"], horizontal=True)
    criterion = criterion_label.lower()
    max_depth_choice = st.select_slider(
        "Maximum depth",
        options=[1, 2, 3, 4, 5, 6, 8, 10, 15],
        value=4,
        help="Deeper trees can represent more complex decision rules, but may overfit.",
    )
    min_samples_leaf = st.slider(
        "Minimum samples per leaf",
        1, 30, 4, 1,
        help="Larger leaves smooth the tree by preventing tiny terminal regions.",
    )
    ccp_alpha = st.select_slider(
        "Pruning strength (ccp_alpha)",
        options=[0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
        value=0.0,
        help="Higher values prune more aggressively using cost-complexity pruning.",
    )

    st.divider()
    st.caption("Teaching tip: keep the sample fixed and change one tree-control parameter at a time.")

X, y = make_dataset(dataset_name, n_samples, noise, int(random_state))
results = fit_and_evaluate(
    X=X,
    y=y,
    criterion=criterion,
    max_depth=max_depth_choice,
    min_samples_leaf=min_samples_leaf,
    ccp_alpha=ccp_alpha,
    test_size=test_size,
    random_state=int(random_state),
)


def draw_boundary():
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    xx, yy, grid = decision_grid(X, resolution=260)
    pred = results.model.predict(grid).reshape(xx.shape)
    proba = results.model.predict_proba(grid)[:, 1].reshape(xx.shape)

    ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5], alpha=0.12)
    ax.contour(xx, yy, proba, levels=[0.5], linewidths=2.0)

    ax.scatter(
        results.X_train[:, 0],
        results.X_train[:, 1],
        c=results.y_train,
        s=48,
        edgecolors="white",
        linewidths=0.55,
        label="Training observations",
    )
    ax.scatter(
        results.X_test[:, 0],
        results.X_test[:, 1],
        c=results.y_test,
        marker="X",
        s=80,
        linewidths=0.7,
        edgecolors="black",
        label="Test observations",
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Classification tree decision regions")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.12)
    fig.tight_layout()
    return fig


def draw_tree():
    fig, ax = plt.subplots(figsize=(15, 7.5))
    plot_tree(
        results.model,
        feature_names=["Feature 1", "Feature 2"],
        class_names=["Class 0", "Class 1"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        precision=2,
        ax=ax,
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def complexity_message():
    if results.tree_depth >= 8 and results.leaf_count >= 20:
        return "⚠️ This tree is highly segmented. Compare training and test accuracy carefully for signs of overfitting."
    if results.tree_depth <= 2:
        return "ℹ️ This is a very shallow tree. It is easy to explain, but may underfit nonlinear structure."
    if ccp_alpha >= 0.02:
        return "ℹ️ Pruning is fairly aggressive. Notice whether interpretability improves without sacrificing much test accuracy."
    return "✅ This is a useful middle-complexity setting for exploring how trees partition the feature space."


main_tab, tree_tab, forest_tab, split_tab, pruning_tab, path_tab, lab_tab = st.tabs(
    ["Decision Regions", "Tree Structure", "Random Forests", "How Splits Work", "Pruning", "Classify a Case", "Student Lab"]
)

with main_tab:
    left, right = st.columns([1.65, 0.85], gap="large")
    with left:
        st.pyplot(draw_boundary(), use_container_width=True)
        st.caption("The tree partitions the feature space into axis-aligned rectangular regions. X markers are held-out test observations.")
    with right:
        st.subheader("What to notice")
        st.info(complexity_message())
        st.markdown("**Current model**")
        st.write(f"Criterion: **{criterion_label}**")
        st.write(f"Maximum depth: **{max_depth_choice}**")
        st.write(f"Minimum leaf size: **{min_samples_leaf}**")
        st.write(f"ccp_alpha: **{ccp_alpha:g}**")
        st.write(f"Actual fitted depth: **{results.tree_depth}**")
        st.write(f"Terminal leaves: **{results.leaf_count}**")
        st.markdown("**Observe:**")
        st.markdown(
            "- Why are the regions rectangular?\n"
            "- Which observations cause additional splits?\n"
            "- Does increasing depth improve test performance?\n"
            "- Can pruning remove detail without hurting accuracy?"
        )

    st.subheader("Performance snapshot")
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Train accuracy", f"{results.metrics['Train accuracy']:.1%}")
    m2.metric("Test accuracy", f"{results.metrics['Test accuracy']:.1%}")
    m3.metric("Precision", f"{results.metrics['Precision']:.1%}")
    m4.metric("Recall", f"{results.metrics['Recall']:.1%}")
    m5.metric("F1", f"{results.metrics['F1']:.1%}")
    m6.metric("Kappa", f"{results.metrics['Kappa']:.3f}", help="Cohen's kappa adjusts observed agreement for agreement expected by chance.")
    m7.metric("Leaves", f"{results.leaf_count}")

with tree_tab:
    st.subheader("The fitted decision tree")
    st.pyplot(draw_tree(), use_container_width=True)
    st.caption("Each internal node asks one yes/no threshold question. A path from the root to a leaf is a complete classification rule.")

    fi = pd.DataFrame({
        "Feature": ["Feature 1", "Feature 2"],
        "Importance": results.model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    st.markdown("### Feature importance")
    st.bar_chart(fi.set_index("Feature"))
    st.caption("Tree importance measures how much each feature reduces impurity across the fitted tree. It is not a causal effect.")


with forest_tab:
    st.subheader("From one tree to a random forest")
    st.write(
        "A random forest grows many decision trees on bootstrap samples and averages their votes. "
        "It also randomizes the features considered at each split, reducing correlation among trees. "
        "The goal is to keep the flexibility of trees while reducing the instability and variance of a single tree."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rf_trees = st.slider("Number of trees", 10, 300, 100, 10, key="rf_trees")
    with c2:
        rf_depth_label = st.selectbox(
            "Maximum depth per tree",
            ["None", "3", "5", "8", "12"],
            index=0,
            key="rf_depth",
        )
        rf_depth = None if rf_depth_label == "None" else int(rf_depth_label)
    with c3:
        rf_leaf = st.slider("Minimum samples per leaf", 1, 20, 2, 1, key="rf_leaf")
    with c4:
        rf_features_label = st.radio(
            "Features tried at each split",
            ["√p (random subset)", "All features"],
            key="rf_features",
        )
        rf_features = "sqrt" if rf_features_label.startswith("√") else None

    forest = fit_random_forest(
        X=X,
        y=y,
        criterion=criterion,
        n_estimators=rf_trees,
        max_depth=rf_depth,
        min_samples_leaf=rf_leaf,
        max_features=rf_features,
        test_size=test_size,
        random_state=int(random_state),
    )

    left, right = st.columns([1.55, 0.95], gap="large")
    with left:
        fig, ax = plt.subplots(figsize=(9.2, 6.0))
        xx, yy, grid = decision_grid(X, resolution=240)
        rf_pred = forest.model.predict(grid).reshape(xx.shape)
        rf_proba = forest.model.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, rf_pred, levels=[-0.5, 0.5, 1.5], alpha=0.12)
        ax.contour(xx, yy, rf_proba, levels=[0.5], linewidths=2.0)
        ax.scatter(
            forest.X_train[:, 0],
            forest.X_train[:, 1],
            c=forest.y_train,
            s=46,
            edgecolors="white",
            linewidths=0.5,
            label="Training observations",
        )
        ax.scatter(
            forest.X_test[:, 0],
            forest.X_test[:, 1],
            c=forest.y_test,
            marker="X",
            s=80,
            linewidths=0.7,
            edgecolors="black",
            label="Test observations",
        )
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title(f"Random forest decision regions ({rf_trees} trees)")
        ax.legend(loc="best", frameon=True)
        ax.grid(alpha=0.12)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("The forest averages many rectangular tree partitions, which can create a smoother and more stable overall boundary.")

    with right:
        st.markdown("### Single tree vs. forest")
        comparison = pd.DataFrame(
            {
                "Metric": ["Test accuracy", "F1", "Kappa"],
                "Single tree": [
                    results.metrics["Test accuracy"],
                    results.metrics["F1"],
                    results.metrics["Kappa"],
                ],
                "Random forest": [
                    forest.metrics["Test accuracy"],
                    forest.metrics["F1"],
                    forest.metrics["Kappa"],
                ],
            }
        )
        display_comparison = comparison.copy()
        display_comparison["Single tree"] = display_comparison.apply(
            lambda r: f"{r['Single tree']:.3f}" if r["Metric"] == "Kappa" else f"{r['Single tree']:.1%}",
            axis=1,
        )
        display_comparison["Random forest"] = display_comparison.apply(
            lambda r: f"{r['Random forest']:.3f}" if r["Metric"] == "Kappa" else f"{r['Random forest']:.1%}",
            axis=1,
        )
        st.dataframe(display_comparison, hide_index=True, use_container_width=True)

        r1, r2 = st.columns(2)
        r1.metric("Forest test accuracy", f"{forest.metrics['Test accuracy']:.1%}")
        r2.metric("Forest kappa", f"{forest.metrics['Kappa']:.3f}")
        r3, r4 = st.columns(2)
        r3.metric("OOB accuracy", f"{forest.metrics['OOB accuracy']:.1%}", help="Out-of-bag observations were not used to fit the particular trees voting on them.")
        r4.metric("Trees", f"{rf_trees}")

        st.markdown("### Why a forest can improve on one tree")
        st.markdown(
            "- **Bootstrap sampling:** each tree sees a different resampled training set.\n"
            "- **Feature randomness:** trees are encouraged to make different mistakes.\n"
            "- **Voting:** averaging many noisy trees reduces variance.\n"
            "- **Tradeoff:** the forest is usually less interpretable than one tree."
        )

    st.markdown("### Classroom experiment")
    st.write(
        "Set the single tree to a fairly deep specification, then change the random seed several times. "
        "Compare how much the single-tree boundary and test score move versus the random forest. "
        "This demonstrates why forests are often more stable."
    )

with split_tab:
    st.subheader("How a classification-tree split works")
    st.write(
        "At each node, the algorithm searches candidate thresholds and selects the split that creates purer child nodes. "
        "A pure node contains mostly — or entirely — one class."
    )

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Gini index — course convention")
        st.latex(r"Gini = 1 - \sum_k p_k^2")
        st.write(
            "For BAT 3305, use binary Gini = p(1-p). It is 0 for a pure node and 0.25 for a 50/50 node. "
            "scikit-learn internally uses 2p(1-p) for binary Gini; this rescales impurity but chooses the same splits."
        )
    with b:
        st.markdown("#### Entropy")
        st.latex(r"Entropy = -\sum_k p_k \log_2(p_k)")
        st.write(
            "Each class contributes p × log2(p), and node entropy is the negative sum. Entropy is 0 for a pure node and reaches 1 for a 50/50 two-class node. "
            "Students can compare Gini and entropy while holding the sample fixed."
        )

    p = st.slider("Imagine a node with this proportion of Class 1", 0.0, 1.0, 0.5, 0.01)
    gini = p * (1 - p)
    entropy = 0.0
    for q in [p, 1-p]:
        if q > 0:
            entropy -= q * np.log2(q)
    c1, c2 = st.columns(2)
    c1.metric("Gini index", f"{gini:.3f}")
    c2.metric("Entropy", f"{entropy:.3f}")

    st.markdown("### Why the decision regions look like rectangles")
    st.write(
        "A standard CART classification tree tests one feature at a time, such as **Feature 1 ≤ 0.73**. "
        "Each threshold draws a vertical or horizontal cut in a two-dimensional plot. Repeating those cuts creates rectangular regions."
    )

with pruning_tab:
    st.subheader("Cost-complexity pruning")
    st.write(
        "Growing a very deep tree can fit noise. Cost-complexity pruning asks whether extra leaves improve fit enough to justify their added complexity. "
        "The parameter **ccp_alpha** controls that tradeoff."
    )
    curve = pruning_curve(
        X=X,
        y=y,
        criterion=criterion,
        min_samples_leaf=min_samples_leaf,
        test_size=test_size,
        random_state=int(random_state),
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["alpha"], curve["train_accuracy"], marker="o", label="Train accuracy")
    ax.plot(curve["alpha"], curve["test_accuracy"], marker="o", label="Test accuracy")
    ax.set_xscale("symlog", linthresh=0.0005)
    ax.set_xlabel("ccp_alpha")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.grid(alpha=.18)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.dataframe(
        curve[["alpha", "leaves", "depth", "train_accuracy", "test_accuracy"]].style.format({
            "alpha": "{:.5f}",
            "train_accuracy": "{:.1%}",
            "test_accuracy": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    best_row = curve.iloc[curve["test_accuracy"].argmax()]
    st.success(
        f"For this sample, the highest test accuracy along the pruning path occurs near ccp_alpha = {best_row['alpha']:.5f}, "
        f"with {int(best_row['leaves'])} leaves. This is a teaching illustration, not a substitute for cross-validation."
    )

with path_tab:
    st.subheader("Classify one new case")
    st.write("Move the two feature values and trace the exact yes/no rules the tree uses.")
    xmin, xmax = float(X[:,0].min()), float(X[:,0].max())
    ymin, ymax = float(X[:,1].min()), float(X[:,1].max())
    x1 = st.slider("Feature 1", xmin, xmax, float(np.median(X[:,0])), (xmax-xmin)/100)
    x2 = st.slider("Feature 2", ymin, ymax, float(np.median(X[:,1])), (ymax-ymin)/100)

    path = classification_path(results.model, np.array([x1, x2]))
    pred = int(results.model.predict([[x1, x2]])[0])
    prob = float(results.model.predict_proba([[x1, x2]])[0, 1])

    c1, c2 = st.columns(2)
    c1.metric("Predicted class", f"Class {pred}")
    c2.metric("Predicted P(Class 1)", f"{prob:.1%}")

    st.markdown("### Decision path")
    for step in path:
        if step["type"] == "split":
            symbol = "≤" if step["direction"] == "left" else ">"
            st.write(
                f"**Node {step['node']}** — {step['feature_name']} {symbol} {step['threshold']:.3f} "
                f"→ go **{step['direction']}**"
            )
        else:
            st.write(
                f"**Leaf {step['node']}** — class counts = {step['class_counts']}; predicted **Class {step['prediction']}**"
            )

with lab_tab:
    st.subheader("Guided classroom experiments")
    st.write("Use the sidebar controls and change **one variable at a time**.")

    experiments = pd.DataFrame(
        [
            ["1. Underfitting", "Two moons", "max depth = 1", "Why does one split fail to represent the class structure?"],
            ["2. Depth and complexity", "Two moons", "depth 1 → 2 → 4 → 8", "When does test accuracy stop improving even though training accuracy rises?"],
            ["3. Tiny leaves", "Checkerboard", "min leaf 1 vs 15", "How does minimum leaf size smooth the decision regions?"],
            ["4. Gini vs entropy", "Linear + noise", "switch criterion", "Do the resulting first splits or test metrics differ materially?"],
            ["5. Pruning", "Concentric circles", "increase ccp_alpha", "Can you simplify the tree while retaining similar test accuracy?"],
            ["6. Instability", "Two moons", "change random seed", "How stable are the exact splits when the sample changes slightly?"],
        ],
        columns=["Experiment", "Dataset", "Settings", "Question"],
    )
    st.dataframe(experiments, use_container_width=True, hide_index=True)

    st.markdown("### Exit questions")
    st.markdown(
        "1. Why does a classification tree produce **rectangular decision regions** in two dimensions?\n"
        "2. What does a **pure node** mean?\n"
        "3. How do **maximum depth**, **minimum leaf size**, and **ccp_alpha** each control complexity differently?\n"
        "4. Why can a tree have 100% training accuracy but worse test accuracy?\n"
        "5. What is the difference between reading the whole tree and tracing the classification path for one observation?\n"
        "6. Why might two samples from the same population produce somewhat different trees?"
    )

st.divider()
st.caption("BAT 3305 - Colazo • Interactive classification-tree demonstration • Synthetic data generated locally in the app")
