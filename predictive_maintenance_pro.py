"""
Predictive Maintenance: Aircraft Engine RUL Prediction (FD001)
--------------------------------------------------------------
A progressive analysis pipeline:
1. Data Exploration & PCA
2. Baseline Regression
3. Random Forest (The Workhorse)
4. Classification (The Safety Net)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report

# --- 1. CONFIGURATION & SETUP ---
# C-MAPSS Data Schema
COLS = ['unit', 'cycle', 'op_setting1', 'op_setting2', 'op_setting3'] + \
       [f's{i}' for i in range(1, 22)]

# Visual Settings
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def load_data(filename):
    """Safe loader for the C-MAPSS text files."""
    try:
        # These files are space-separated with no headers
        df = pd.read_csv(filename, sep=r'\s+', header=None, names=COLS)
        print(f"Successfully loaded {filename} | Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: Could not find {filename}. Make sure it's in the folder!")
        return None

# --- 2. PREPROCESSING PIPELINE ---

def get_rul_labels(df):
    """
    Calculates the Remaining Useful Life (RUL) for the training data.
    RUL = Max Cycle - Current Cycle
    """
    # Find the last breath of each engine
    max_cycles = df.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    
    # Merge back and calculate the countdown
    df_merged = df.merge(max_cycles, on='unit', how='left')
    df_merged['RUL'] = df_merged['max_cycle'] - df_merged['cycle']
    return df_merged

def process_pca(train_df, test_df=None, n_components=3):
    """
    The Master Preprocessor:
    1. Drops constant sensors (they are useless).
    2. Scales data (Mean=0, Var=1).
    3. Runs PCA to compress 21 sensors into 3 distinct signals.
    """
    # Exclude unit/cycle/op_settings - we only want sensor data for PCA
    sensor_cols = [f's{i}' for i in range(1, 22)]
    X_train = train_df[sensor_cols]
    
    # 1. Drop Constant Sensors (Zero Variance)
    # In FD001, some sensors don't move. We drop them.
    # We find columns where value != first_value is NEVER true (i.e. always same)
    drop_mask = (X_train != X_train.iloc[0]).any()
    X_train_dropped = X_train.loc[:, drop_mask]
    
    print(f"Sensors kept: {X_train_dropped.shape[1]} (Dropped {len(sensor_cols) - X_train_dropped.shape[1]} constants)")

    # 2. Scale the Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_dropped)
    
    # 3. Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Create nice DataFrames for output
    train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    train_pca_df['unit'] = train_df['unit']
    train_pca_df['cycle'] = train_df['cycle']
    # If we calculated RUL previously, keep it attached
    if 'RUL' in train_df.columns:
        train_pca_df['RUL'] = train_df['RUL']

    # --- Handle Test Data (If provided) ---
    test_pca_df = None
    if test_df is not None:
        # IMPORTANT: Apply exact same transformations to Test as we did to Train
        X_test = test_df[sensor_cols]
        X_test_dropped = X_test.loc[:, drop_mask] # Drop same cols
        X_test_scaled = scaler.transform(X_test_dropped) # Use fitted scaler
        X_test_pca = pca.transform(X_test_scaled) # Use fitted PCA
        
        test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        test_pca_df['unit'] = test_df['unit']
        test_pca_df['cycle'] = test_df['cycle']
        
        # For Test data, we usually only care about the LAST row per unit for scoring
        # But we return the whole thing just in case
        
    return train_pca_df, test_pca_df, pca

# --- 3. EXECUTION: LOAD & PREP ---

print("\n--- PHASE 1: Loading & Crunching Numbers ---")
raw_train = load_data('train_FD001.txt')
raw_test = load_data('test_FD001.txt')
true_rul = pd.read_csv('RUL_FD001.txt', header=None, names=['true_rul'])

# Calculate RUL for training
train_with_rul = get_rul_labels(raw_train)

# Run the PCA Pipeline
train_pca, test_pca, pca_model = process_pca(train_with_rul, raw_test, n_components=3)

# --- 4. VISUALIZATION: The 3D Trajectory ---
# Does PCA actually separate the healthy from the dying?
print("\n--- Generating 3D Trajectory Plot... ---")
fig = px.scatter_3d(
    train_pca.sample(2000), # Sample points so browser doesn't crash
    x='PC1', y='PC2', z='PC3',
    color='RUL',
    color_continuous_scale='Turbo',
    opacity=0.4,
    title='3D Engine Degradation Path (PC1 vs PC2 vs PC3)',
    labels={'PC1': 'PCA 1', 'PC2': 'PCA 2', 'PC3': 'PCA 3'}
)
fig.update_traces(marker=dict(size=3))
fig.show()

# --- 5. MODELING: The "Evolution" ---

# Prepare Data for Training
features = ['PC1', 'PC2', 'PC3']
X = train_pca[features]
y = train_pca['RUL']

# Validation Split (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare Test Data (The Final Exam)
# We need the LAST row of each unit in the test set to compare with RUL_FD001.txt
X_test_final = test_pca.groupby('unit').last()[features]
y_test_true = true_rul['true_rul']

def evaluate_model(model, X_test, y_true, name="Model"):
    """Helper to grade our models consistently."""
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    print(f"[{name}] Test RMSE: {rmse:.2f} | R²: {r2:.4f}")
    return preds

print("\n--- PHASE 2: Model Benchmarking ---")

# A. Baseline: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = evaluate_model(lr, X_test_final, y_test_true, "Linear Regression")

# B. KNN: Finding Neighbors
knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(X_train, y_train)
knn_preds = evaluate_model(knn, X_test_final, y_test_true, "KNN (k=7)")

# C. Random Forest: The Heavy Hitter
rf = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
rf.fit(X_train, y_train)
rf_preds = evaluate_model(rf, X_test_final, y_test_true, "Random Forest")

# D. The "Pro" Fix: Piecewise Linear RUL (Clipping)
# Engines don't degrade linearly from Day 1. They stay "healthy" (RUL > 125) then drop.
# We clip the target to 125 to teach the model this behavior.
y_train_clipped = y_train.clip(upper=125)

rf_clipped = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
rf_clipped.fit(X_train, y_train_clipped)
rf_clipped_preds = evaluate_model(rf_clipped, X_test_final, y_test_true, "RF + Clipped Target")

# --- 6. VISUAL PROOF ---

plt.figure(figsize=(12, 6))
plt.scatter(y_test_true, lr_preds, alpha=0.3, label='Linear (Baseline)', color='gray')
plt.scatter(y_test_true, rf_clipped_preds, alpha=0.6, label='RF + Clipping (Best)', color='green')
plt.plot([0, 160], [0, 160], 'r--', lw=2, label='Perfect Prediction')
plt.axhline(y=125, color='orange', linestyle='--', label='Max RUL Threshold')
plt.xlabel('True RUL (Ground Truth)')
plt.ylabel('Predicted RUL')
plt.title('Baseline vs Optimized Random Forest')
plt.legend()
plt.grid(True)
plt.show()

# --- 7. CLASSIFICATION: The "Nuclear Option" ---
# Moving from "How long left?" to "Which Zone are we in?"

print("\n--- PHASE 3: The Safety Net (Classification) ---")

def categorize_rul(rul_values):
    """Maps RUL numbers to Traffic Light zones."""
    # 0=Red (Urgent, <30), 1=Yellow (Plan, <75), 2=Green (Safe, >75)
    cats = []
    for r in rul_values:
        if r <= 30: cats.append(0)       # Red
        elif r <= 75: cats.append(1)     # Yellow
        else: cats.append(2)             # Green
    return np.array(cats)

# Convert targets
y_train_class = categorize_rul(y_train)
y_test_class = categorize_rul(y_test_true)

# Define Nuclear Weights
# We tell the model: Missing a Red Zone failure is 100x worse than a False Alarm.
nuclear_weights = {
    0: 100,  # Red: CRITICAL
    1: 40,   # Yellow: Important
    2: 15    # Green: Standard
}

# Train the Safety Classifier
rf_safety = RandomForestClassifier(
    n_estimators=100, 
    max_depth=7, 
    class_weight=nuclear_weights, # The manual override
    random_state=42
)
rf_safety.fit(X_train, y_train_class)

# The Scoreboard
safety_preds = rf_safety.predict(X_test_final)
labels = ['Red (Urgent)', 'Yellow (Plan)', 'Green (OK)']

# Plot Matrix
cm = confusion_matrix(y_test_class, safety_preds, normalize='true')

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.1%', cmap='Reds_r', 
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Zone')
plt.ylabel('Actual Zone (Truth)')
plt.title('The "Nuclear" Confusion Matrix\n(Prioritizing Safety over Accuracy)')
plt.show()

print("\nAnalysis Complete. The beast has been tamed.")