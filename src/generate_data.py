"""Generate synthetic classification datasets."""

import polars as pl
import numpy as np
from pathlib import Path

from sklearn.datasets import make_classification

N_FEATURES = 8
N_INFORMATIVE = 4
SEED = 1
CLASS_SEP = 1.2
BETA_CONCENTRATION = 1.8



def _rand_dates(rng, start_date, end_date, size) -> np.ndarray:
    dates = pl.date_range(
        pl.lit(start_date).str.to_date(format="%d/%m/%Y"),
        pl.lit(end_date).str.to_date(format="%d/%m/%Y"),
        interval="1d",
        eager=True,
    )
    return rng.choice(dates, size)


def generate_base_classification_dataset(
    n_groups=20, n_repetitions=20, weights=None, seed=1
) -> pl.DataFrame:
    """
    Generate a base classification dataset which is then transformed to create local and exterior datasets.
    """
    if weights is None:
        weights = [0.7, 0.3]  # Default class imbalance

    n_samples = n_groups * n_repetitions
    X_base, y_base = make_classification(
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

    X_group = rng.normal(size=(n_groups, N_FEATURES))
    X = 0.4 * X_base + 4.0 * X_group[groups] # numerical features are dominated by group contributions

    group_rates = rng.beta(BETA_CONCENTRATION * weights[1], BETA_CONCENTRATION * (1 - weights[1]), size=n_groups) # group-specific positive rates
    y = np.where(
        rng.rand(n_samples) < 0.85, # mixture rate
        rng.rand(n_samples) < group_rates[groups], # if group-specific
        y_base, # if group-independent
    )

    birth_dates = _rand_dates(rng, "01/01/1960", "31/12/1995", n_samples)
    exam_dates = _rand_dates(rng, "01/01/2000", "31/12/2020", n_samples)

    df = pl.DataFrame(
        X,
        schema=[f"num_{i}" for i in range(N_FEATURES)],
        orient="row",
    )
    df = df.with_columns(
        pl.Series("group", groups),
        pl.Series("target", y),
        pl.Series("sex", rng.choice(["male", "female"], size=n_samples)),
        pl.Series(
            "birth_date",
            birth_dates,
        ),
        pl.Series(
            "exam_date",
            exam_dates,
        ),
    )
    return df


def main():
    Path("data").mkdir(parents=True, exist_ok=True)

    # Generate local dataset
    df_local = generate_base_classification_dataset(20, 20, [0.7, 0.3], 1)
    df_local.write_csv("data/local_classification.csv")

    # Generate exterior dataset
    df_ext_raw = generate_base_classification_dataset(10, 20, [0.8, 0.2], 2)
    # Numerical features shift
    X_ext_raw = df_ext_raw.select([f"num_{i}" for i in range(N_FEATURES)]).to_numpy()
    rng = np.random.RandomState(SEED+2)
    X_shifts = rng.normal(0.0, 2.0, size=N_FEATURES)
    X_scales = rng.exponential(1.0, size=N_FEATURES)
    X_ext = (X_ext_raw * X_scales) + X_shifts
    # update df
    for i in range(N_FEATURES):
        df_ext_raw = df_ext_raw.with_columns(pl.Series(f"num_{i}", X_ext[:, i]))

    # Label noise
    y_ext = df_ext_raw["target"].to_numpy().copy()
    noise_mask = rng.rand(len(y_ext)) < 0.05
    y_ext[noise_mask] = 1 - y_ext[noise_mask]
    df_ext = df_ext_raw.with_columns(pl.Series("target", y_ext))

    df_ext.write_csv("data/exterior_classification.csv")

    print(f"\nDataset generated successfully!")
    print(f"Local dataset: {len(df_local):,} samples")
    print(f"Exterior dataset: {len(df_ext):,} samples")

if __name__ == "__main__":
    main()