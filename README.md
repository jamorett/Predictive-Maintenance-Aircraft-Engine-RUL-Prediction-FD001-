# ✈️ Predictive Maintenance: Aircraft Engine RUL Prediction (C-MAPSS)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Viz-Plotly-3F4F75?style=flat&logo=plotly)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

## 📖 Executive Summary

This repository hosts a robust machine learning pipeline designed to predict the **Remaining Useful Life (RUL)** of turbofan aircraft engines using the NASA C-MAPSS dataset (FD001).

The project moves beyond simple regression by implementing a **progressive analysis workflow**:
1.  **Dimensionality Reduction:** Compressing 21 sensor streams into 3 Principal Components (PCA) for visualization and noise reduction.
2.  **Piecewise Linear Modeling:** Optimizing RUL targets to reflect the physical reality of engine degradation (clipping).
3.  **Safety-Critical Classification:** A "Nuclear Option" Random Forest classifier that aggressively penalizes missed failures (Type II errors) using weighted loss functions.

## 🏗️ Technical Architecture

### 1. Preprocessing & PCA
* **Constant Removal:** Automatically detects and drops sensors with zero variance (static outputs).
* **Scaling:** `StandardScaler` (z-score) applied to normalize sensor magnitudes.
* **PCA Projection:** Projects the 21-dimensional sensor space into a 3D latent space ($PC_1, PC_2, PC_3$) to visualize the "degradation trajectory" of engines over time.

### 2. Regression Strategy (RUL Prediction)
We benchmark multiple regressors to predict the exact cycle of failure:
* **Baseline:** Linear Regression & KNN.
* **Random Forest:** Ensemble method to capture non-linear degradation patterns.
* **Optimization:** **Target Clipping**. We limit the maximum RUL to 125 cycles.
    * *Rationale:* Engines operate normally for a long period before degradation onset. predicting linear degradation from Day 1 introduces high error. Clipping the target reduces RMSE significantly.

### 3. The "Nuclear" Classification Option
Moving from *prediction* to *decision making*.
* **Zones:**
    * 🟢 **Green:** Safe (>75 cycles)
    * 🟡 **Yellow:** Planning Phase (30-75 cycles)
    * 🔴 **Red:** Urgent Maintenance (<30 cycles)
* **Cost-Sensitive Learning:** The model utilizes a custom class weight dictionary to heavily penalize missing a "Red Zone" engine.
    ```python
    weights = {0 (Red): 100, 1 (Yellow): 40, 2 (Green): 15}
    ```

## 📊 Results Summary

| Model Strategy | RMSE (Cycles) | R² Score | Note |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | ~37.0 | 0.40 | High bias, underfits complex data. |
| **Random Forest (Raw)** | ~28.0 | 0.55 | Good, but struggles with early cycles. |
| **RF + Clipped Target** | **< 20.0** | **0.75+** | **State of the Art performance.** |

## 🚀 Installation & Usage

### Prerequisites
* Python 3.8+
* NASA C-MAPSS Dataset (FD001) placed in the root directory.

### Dependencies
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn

```

### Running the Analysis

Execute the consolidated pipeline script:

```bash

python predictive_maintenance_pro.py

```






