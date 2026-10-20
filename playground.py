import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    import numpy as np
    from pathlib import Path

    from sklearn.datasets import make_regression, make_classification


@app.cell
def _():
    N_FEATURES = 8
    N_INFORMATIVE = 4
    SEED = 1
    CLASS_SEP = 1.2
    BETA_CONCENTRATION = 1.8
    return BETA_CONCENTRATION, CLASS_SEP, N_FEATURES, N_INFORMATIVE


@app.cell
def _(BETA_CONCENTRATION, CLASS_SEP, N_FEATURES, N_INFORMATIVE):
    def generate_base_classification_dataset(
        n_groups=20, n_repetitions=20, weights=None, seed=1
    ):
        """
        Generate a base classification dataset which is then transformed to create local and exterior datasets.
        """
        if weights is None:
            weights = [0.7, 0.3]  # Default class imbalance

        n_samples = n_groups * n_repetitions

        base_features, base_target = make_classification(
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

        group_features = rng.normal(size=(n_groups, base_features.shape[1]))
        features = 0.4 * base_features + 4.0 * group_features[groups] # features are dominated by group contributions
        df = pl.DataFrame(
                features,
                schema=[f"num_{i}" for i in range(features.shape[1])],
                orient="row",
            )
        df = df.with_columns(pl.Series("group", groups))

        group_rates = rng.beta(BETA_CONCENTRATION * weights[1], BETA_CONCENTRATION * (1 - weights[1]), size=n_groups) # group-specific positive rates
        target = np.where(
            rng.rand(n_samples) < 0.85, # mixture rate
            rng.rand(n_samples) < group_rates[groups], # if group-specific
            base_target, # if group-independent
        )
        df = df.with_columns(pl.Series("target", target))

        sex = rng.choice(["male", "female"], size=n_samples)
        df = df.with_columns(pl.Series("sex", sex))

        def _rand_dates(start_date, end_date, n):
            start_dt = pl.lit(start_date).str.to_date(format="%d/%m/%Y")
            end_dt = pl.lit(end_date).str.to_date(format="%d/%m/%Y")
            range_dt = pl.date_range(start_dt, end_dt, interval="1d", eager=True)

            return rng.choice(range_dt, n)

        birth_dates = _rand_dates("01/01/1960", "31/12/1995", n_samples)
        df = df.with_columns(pl.Series("birth_date", birth_dates))

        exam_dates = _rand_dates("01/01/2000", "31/12/2020", n_samples)
        df = df.with_columns(pl.Series("exam_date", exam_dates))

        return df

    return (generate_base_classification_dataset,)


@app.cell
def _(generate_base_classification_dataset):
    df = generate_base_classification_dataset()
    df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
