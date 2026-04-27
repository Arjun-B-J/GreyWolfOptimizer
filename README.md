# GreyWolfOptimizer

Implementations of the Grey Wolf Optimizer (GWO) and several variants in Python, applied primarily to **feature selection** on standard UCI machine learning datasets.

## About Grey Wolf Optimizer

Grey Wolf Optimizer is a population-based metaheuristic optimization algorithm inspired by the leadership hierarchy and hunting behavior of grey wolves in nature. Candidate solutions are organized into four social ranks — alpha, beta, delta, and omega — and the search is driven by the three best wolves (alpha/beta/delta) guiding the rest of the pack toward promising regions of the search space. A control parameter `a` decreases linearly from 2 to 0 across iterations to balance exploration and exploitation.

## What's in this repo

This repo collects multiple flavors of GWO that were built and experimented with:

- **Continuous GWO** — the classical single-objective variant (`GreyWolf/`).
- **Binary GWO (BGWO)** — binary-encoded variant that uses a sigmoid transfer function to map continuous wolf positions to `{0, 1}`, used here for feature selection.
- **Multi-Objective GWO (MOGWO)** and **Binary MOGWO (BMOGWO)** — multi-objective variants with an external archive of non-dominated solutions and grid-based leader selection (`MOGWOpy/`, `Model/Binary_MOGWO/`).

For the feature-selection experiments, each wolf's binary position vector picks a subset of features; the subset is fed to a Keras/TensorFlow ANN classifier (`Model/ANN_Classifier_v2.py`) and the resulting test error becomes the wolf's fitness. The multi-objective variants additionally consider the number of selected features as a second objective.

## Datasets

UCI-style classification datasets included under `Dataset/`: Breast Cancer (BreastEW), Heart, Hill-Valley, Ionosphere, KrVsKp, Lymphography, Madelon, Musk (v1), Sonar, SPECT, Waveform, Wine, Zoo. Per-dataset benchmark outputs are saved under `Results/` and `ExplorationResults/`.

## Tech stack

- Python 3
- NumPy, pandas, scikit-learn
- TensorFlow / Keras (ANN classifier used as the wrapper evaluator)
- Matplotlib, seaborn (plots)
- Jupyter notebooks for experiments

## Repository layout

```
GreyWolf/              Basic single-objective GWO
BinaryGreyWolf/        Binary GWO notebook
MOGWOpy/               Multi-objective GWO (Python port; MATLAB reference under matlab/)
Model/                 ANN classifier + GWO/MOGWO drivers used for feature selection
  Binary_MOGWO/        Binary MOGWO variants (v1, v2a..v2f)
  MOO-A/               NSGA-II baseline for comparison
Dataset/               UCI datasets used in experiments
Results/               Per-dataset run outputs
ExplorationResults/    Exploration analysis outputs (varied iteration counts)
Assets/                Plotting utilities and analysis notebook
```

## Running

1. Install dependencies:
   ```
   pip install numpy pandas scikit-learn tensorflow matplotlib seaborn jupyter
   ```
2. Open the notebooks (e.g. `GreyWolf/Basic_GWO.ipynb`, `BinaryGreyWolf/Binary_GreyWolf.ipynb`, `BMOGWO.ipynb`) in Jupyter, or run the Python drivers directly:
   ```
   cd MOGWOpy/BMOGWO
   python driver.py
   ```

## License

MIT — see `LICENSE`.
