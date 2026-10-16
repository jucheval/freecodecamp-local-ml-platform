"""
Generate synthetic datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.datasets import make_classification, make_regression


if __name__ == "__main__":
    print("Generating synthetic classification dataset...")
    features, target = make_regression(
        n_samples=240,
        n_features=8,
        n_informative=4,
        noise=10,
        random_state=1,
    )
    df = generate_transactions(n_samples=10000, fraud_ratio=0.02)

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Create data directory if it doesn't exist
    Path("data").mkdir(parents=True, exist_ok=True)

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(f"\nDataset generated successfully!")
    print(f"Training set: {len(train_df):,} transactions")
    print(f"Test set: {len(test_df):,} transactions")
    print(f"Overall fraud ratio: {df['is_fraud'].mean():.2%}")
    print(
        f"\nLegitimate transactions - Average amount: ${df[df['is_fraud'] == 0]['amount'].mean():.2f}"
    )
    print(
        f"Fraudulent transactions - Average amount: ${df[df['is_fraud'] == 1]['amount'].mean():.2f}"
    )
    print(f"\nMerchant category distribution (fraud):")
    print(df[df["is_fraud"] == 1]["merchant_category"].value_counts(normalize=True))

# TODO: look at the code

# --- 1. CORE GROUND TRUTH CONFIGURATION ---
N_FEATURES = 8
N_INFORMATIVE = 4
SEED = 1

# Generate local dataset
X_local, y_local, ground_truth_coef = make_regression(
    n_samples=240,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    noise=10.0,
    coef=True,
    random_state=SEED,
)
frame = pd.DataFrame(X_local, columns=[f"num_{i}" for i in range(X_local.shape[1])])
rng = np.random.RandomState(SEED)

frame["cat_a"] = pd.Series(
    rng.choice(["amber", "blue", "green", "NA"], size=len(y_local)), dtype=object
).replace({"NA": pd.NA})

missing = rng.rand(len(y_local), 3) < 0.08
for column_index in range(missing.shape[1]):
    frame.loc[missing[:, column_index], f"num_{column_index}"] = pd.NA

frame["group"] = rng.randint(0, 30, size=len(y_local))
frame["target"] = y_local

frame.to_csv("data/iid_regression.csv", index=False)

# --- 3. GENERATE EXTERIOR DATASET (Out-of-Distribution / Shifted) ---
# We use the same ground-truth coefficients but shift the feature distributions
# and simulate higher environmental noise/measurement errors.
X_ext_raw, _ = make_regression(
    n_samples=500,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    noise=0.0,  # Handle noise manually below
    random_state=SEED + 1,  # Different sample pool
)

# Simulate Covariate Shift: Shift the mean and scale of features
# For example, exterior sensors/locales have higher averages or wider variances
feature_shifts = np.array([1.5, -0.5, 2.0, 0.0, -1.0])
feature_scales = np.array([2.0, 1.0, 1.5, 0.5, 1.0])
X_exterior = (X_ext_raw * feature_scales) + feature_shifts

# Simulate Concept Drift / Environmental Noise: Higher target noise
exterior_noise = np.random.normal(loc=0.0, scale=25.0, size=X_exterior.shape[0])

# Compute targets using the exact same underlying ground truth linear equation
y_exterior = np.dot(X_exterior, ground_truth_coef) + BIAS + exterior_noise

# Convert to DataFrame
df_exterior = pd.DataFrame(
    X_exterior, columns=[f"feature_{i}" for i in range(N_FEATURES)]
)
df_exterior["target"] = y_exterior
df_exterior["dataset_type"] = "exterior"

# --- SUMMARY PRINT ---
print("Local Dataset Shape:", df_local.shape)
print("Exterior Dataset Shape:", df_exterior.shape)
print(f"True Coefficients: {ground_truth_coef}")


# TODO: look at the code

N_FEATURES = 8
N_INFORMATIVE = 4
CLASS_SEP = 1.2


def generate_base_classification_dataset(
    n_groups=20, n_repetitions=20, weights=None, seed=1
):
    """
    Generate a base classification dataset which is then transformed to create local and exterior datasets.
    """
    if weights is None:
        weights = [0.7, 0.3]  # Default class imbalance

    n_samples = n_groups * n_repetitions

    base, base_target = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        weights=weights,
        flip_y=0.0,  # Custom noise is injected later
        class_sep=CLASS_SEP,
        random_state=seed,
    )

    groups = np.tile(np.arange(n_groups), n_repetitions)
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    group_fingerprint = rng.normal(size=(n_groups, base.shape[1]))
    features = 0.4 * base + 4.0 * group_fingerprint[groups]
    group_rates = rng.beta(0.35, 0.35, size=n_groups)
    target = np.where(
        rng.rand(len(groups)) < 0.85,
        (rng.rand(len(groups)) < group_rates[groups]).astype(int),
        base_target,
    )

    frame = pd.DataFrame(
        features, columns=[f"num_{i}" for i in range(features.shape[1])]
    )

    sexes = [
        "male",
        "female",
    ]
    typo_mask = rng.rand(len(groups)) < 0.25
    typos = np.array(["m ale", "fema le", " male", "female "], dtype=object)
    diagnosis[typo_mask] = [
        f"{value}{typo}"
        for value, typo in zip(diagnosis[typo_mask], rng.choice(typos, typo_mask.sum()))
    ]
    frame["diagnosis"] = pd.Series(diagnosis, dtype=object)


# Generate local dataset
base, base_target = make_classification(
    n_samples=400,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    weights=[0.7, 0.3],  # Small class imbalance
    flip_y=0.01,  # Low label noise (1% misclassifications)
    class_sep=CLASS_SEP,  # Well-separated clusters
    random_state=SEED,
)
groups = np.tile(np.arange(30), 12)
rng.shuffle(groups)
group_fingerprint = rng.normal(size=(30, base.shape[1]))
features = 0.4 * base + 4.0 * group_fingerprint[groups]
group_rates = rng.beta(0.35, 0.35, size=30)
target = np.where(
    rng.rand(len(groups)) < 0.85,
    (rng.rand(len(groups)) < group_rates[groups]).astype(int),
    base_target,
)


frame.to_csv("data/iid_classification_local.csv", index=False)

# Convert to DataFrame
df_local = pd.DataFrame(X_local, columns=[f"feature_{i}" for i in range(N_FEATURES)])
df_local["target"] = y_local
df_local["dataset_type"] = "local"

# --- 3. GENERATE EXTERIOR DATASET (Out-of-Distribution / Shifted) ---
# We use the same configuration parameters but use a different random seed to pull
# from the same overall generation logic, then apply real-world shifts manually.
X_ext_raw, y_exterior = make_classification(
    n_samples=200,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    weights=[0.8, 0.2],  # Prior Probability Shift: Class 0 is now heavily dominant
    flip_y=0.0,  # We will inject custom noise below
    class_sep=CLASS_SEP,  # Keeps base cluster geometry consistent
    random_state=SEED + 1,
)

# A. Covariate Shift: Shift and scale the features (e.g., sensor drift in exterior environment)
feature_shifts = np.array([0.8, -0.4, 1.2, -0.2, 0.5])
feature_scales = np.array([1.5, 1.0, 2.0, 0.8, 1.2])
X_exterior = (X_ext_raw * feature_scales) + feature_shifts

# B. Label Noise Shift: Higher flip rate in the exterior world (e.g., poor annotation quality)
# Flip 10% of the labels randomly
noise_mask = np.random.RandomState(SEED).rand(len(y_exterior)) < 0.10
y_exterior[noise_mask] = 1 - y_exterior[noise_mask]

# Convert to DataFrame
df_exterior = pd.DataFrame(
    X_exterior, columns=[f"feature_{i}" for i in range(N_FEATURES)]
)
df_exterior["target"] = y_exterior
df_exterior["dataset_type"] = "exterior"

# --- SUMMARY PRINT ---
print("Local Dataset - Shape:", df_local.shape)
print("Local Class Balance:\n", df_local["target"].value_counts(normalize=True))
print("\nExterior Dataset - Shape:", df_exterior.shape)
print("Exterior Class Balance:\n", df_exterior["target"].value_counts(normalize=True))
