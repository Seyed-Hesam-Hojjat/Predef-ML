# Contributing

Thanks for helping improve Tabular ML Workbench.

## Before opening an issue or pull request

- Do not include real patient data, personal information, secrets, or proprietary datasets in issues or test fixtures.
- For bug reports, include Python/package versions, task type, split strategy, a minimal synthetic example, the error message, and the intended behaviour.
- For scientific/methodological changes, explain the assumptions, leakage risks, validation strategy, relevant metrics, and how the change will be tested.
- Keep changes modular. Add tests for data-audit rules, preprocessing, split logic, metrics, exports, or UI logic as applicable.

## Development setup

```bash
python -m venv .venv
# Activate the environment for your OS, then:
python -m pip install --upgrade pip
pip install -r requirements.txt pytest
python -m pytest -q
streamlit run app.py
```

## Pull request checklist

- [ ] Tests pass locally.
- [ ] No real or sensitive data are included.
- [ ] New preprocessing is fit inside the cross-validation pipeline if it learns from observed data.
- [ ] The held-out test set is not used to choose features, tune hyperparameters, or select the model.
- [ ] Any changed metric is appropriate to the task and clearly documented.
- [ ] Documentation describes new assumptions and known limitations.
