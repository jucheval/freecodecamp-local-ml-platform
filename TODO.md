# TODO list

The ML workflow I have in mind looks like this:

```text
[ DataOps / Joins ] ➔ [ Cleaning ] ➔ [ Vectorization ] ➔ [ Imputation ] ➔ [ Scaling ] ➔ [ Feature Selection ] ➔ [ Estimator ] ➔ [ Evaluation ] ➔ [ Logging ] ➔ [ Visualization ]
```

The objective of this repo if to cover all these steps. A somewhat preliminary step I want to cover too is exploratory data analysis (EDA) and visualization.

- [ ] update README according to `uv` usage
- [x] modify synthetic data generation
- [ ] include githooks: `pre-commit`, `pre-push`, `commit-msg`
- [ ] include some classification pipelines (use `skrub`'s `tabular_pipeline` as a simple example) with progressbars

    ```python
    callback = ProgressBar()
    estimator.set_callbacks(callback)
    estimator.fit(X)
    ```

  - [ ] Remove progressbars from unit tests via

    ```python
    monkeypatch.setattr(sklearn.callback.ProgressBar, "on_fit_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(sklearn.callback.ProgressBar, "on_fit_end", lambda *args, **kwargs: None)
    ```

- [ ] look at all possibilities of `mlflow.log`. For instance, custom metadata
- [ ] compare `MLflow` visualization with `marimo`.
- [ ] cleanup `requirements.txt`
- [ ] include data preprocessing with `skrub` (`Cleaner`, `deduplicate`) with progressbars. TableReport gives a quick overview of the dataset. We can see which columns are badly formatted or have lots of missing values.
- [ ] include EDA and visualization with `skrub` and `marimo` widgets
- [ ] include some regression pipelines
