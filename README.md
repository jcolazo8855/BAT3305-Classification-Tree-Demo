# BAT 3305 - Colazo: Classification Tree Lab

Interactive Streamlit demo for teaching classification trees in BAT 3305.

## Learning goals

Students can explore:

- How a classification tree creates axis-aligned decision regions
- Gini impurity versus entropy
- Maximum depth and underfitting/overfitting
- Minimum samples per leaf as a smoothing control
- Cost-complexity pruning with `ccp_alpha`
- Train/test performance, Cohen's kappa, and confusion-matrix diagnostics
- Feature importance
- Exact root-to-leaf classification paths for new observations
- Tree instability across random samples
- A Random Forests tab comparing bagging/feature randomness against a single tree, including out-of-bag accuracy

## Included datasets

- Linear + noise
- Two moons
- Concentric circles
- Checkerboard

All data are generated locally in the app; no external files or API keys are required.

## Files

- `app.py` — Streamlit user interface
- `tree_utils.py` — data generation, fitting, pruning, and path helpers
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — Streamlit theme/server configuration

## Run locally

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Place these files in the repository root.
3. In Streamlit Community Cloud, choose **Create app**.
4. Select the repository and branch.
5. Set the main file path to `app.py`.
6. Deploy.

## Suggested classroom sequence

1. Start with **Two moons + depth 1** to show underfitting.
2. Increase depth gradually and compare train versus test accuracy.
3. Use **minimum leaf size** to smooth an overly fragmented tree.
4. Compare **Gini** and **Entropy** while holding the sample fixed.
5. Open the **Pruning** tab and identify a smaller tree with similar test accuracy.
6. Use **Classify a Case** to trace one observation from root to leaf.
7. Change the random seed to discuss tree instability.
