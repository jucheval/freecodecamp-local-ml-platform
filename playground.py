import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import pandas as pd
    import numpy as np
    from pathlib import Path

    from sklearn.datasets import make_regression, make_classification


@app.cell
def _():
    N_FEATURES = 8
    N_INFORMATIVE = 4
    SEED = 1
    CLASS_SEP = 1.2
    return (SEED,)


@app.cell
def _(SEED):
    rng = np.random.RandomState(SEED)
    base, base_target = make_classification(
        n_samples=360,
        n_features=8,
        n_informative=4,
        n_redundant=1,
        class_sep=1.0,
        flip_y=0.02,
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

    frame = pd.DataFrame(
        features, columns=[f"num{i}" for i in range(features.shape[1])]
    )
    categories = rng.choice(["amber", "blue", "green", None], size=len(groups))
    frame["cat_a"] = pd.Series(categories, dtype=object).replace({None: np.nan})
    missing = rng.rand(len(groups), 3) < 0.08
    for column_index in range(missing.shape[1]):
        frame.loc[missing[:, column_index], f"num{column_index}"] = np.nan

    roles = [
        "billing analyst",
        "field technician",
        "clinical assistant",
        "claims reviewer",
        "lab specialist",
        "care coordinator",
        "network auditor",
        "support engineer",
    ]
    levels = ["junior", "senior", "regional", "night shift", "contract"]
    role_ids = rng.randint(0, len(roles), size=len(groups))
    level_ids = rng.randint(0, len(levels), size=len(groups))
    diagnosis = np.array(
        [f"{roles[role]} {levels[level]}" for role, level in zip(role_ids, level_ids)],
        dtype=object,
    )
    typo_mask = rng.rand(len(groups)) < 0.25
    typos = np.array([" speclialist", " coordnator", " analyist"], dtype=object)
    diagnosis[typo_mask] = [
        f"{value}{typo}"
        for value, typo in zip(diagnosis[typo_mask], rng.choice(typos, typo_mask.sum()))
    ]
    frame["diagnosis"] = pd.Series(diagnosis, dtype=object)

    corrupt_dates = np.array(
        [
            "",
            "n/a",
            "31/02/2020",
            "03-04-2020",
            "Apr 3 2020",
            "2020/04/03",
            "03/04/20",
            "garbage",
            None,
            np.nan,
        ],
        dtype=object,
    )

    def dirty_dates(valid_values):
        values = np.array(rng.choice(valid_values, size=len(groups)), dtype=object)
        count = round(0.12 * len(groups))
        positions = rng.choice(len(groups), size=count, replace=False)
        values[positions] = rng.choice(corrupt_dates, size=count)
        return pd.Series(values, dtype=object)

    frame["birth_date"] = dirty_dates(
        ["1980-05-12", "03/04/2000", "1975-12-01", "14/07/1990", "2001-09-30"]
    )
    frame["exam_date"] = dirty_dates(
        ["2020-04-03", "03/04/2020", "2021-02-28", "15/01/2021"]
    )
    frame["group"] = groups
    frame["target"] = target
    return (frame,)


@app.cell
def _(frame):
    frame
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
