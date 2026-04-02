from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import optuna
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def tune_hyperparameters(
    model,
    X,
    y,
    param_grid: Dict[str, Any],
    cv: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    verbose: int = 1,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Run GridSearchCV and return best estimator, params, and cv results."""
    search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.cv_results_


def save_tuned_params(params: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def load_tuned_params(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _create_model_from_class(
    model_class: str, params: Dict[str, Any], seed: int = 0
) -> Any:
    """Create a model instance from class name and parameters."""
    model_params = params.copy()
    
    if model_class == "RandomForestClassifier":
        model_params.setdefault("n_jobs", -1)
        model_params.setdefault("random_state", seed)
        model_params.setdefault("class_weight", "balanced")
        return RandomForestClassifier(**model_params)
    elif model_class == "SVC":
        model_params.setdefault("probability", True)
        model_params.setdefault("random_state", seed)
        model_params.setdefault("class_weight", "balanced")
        return SVC(**model_params)
    elif model_class == "XGBClassifier":
        model_params.setdefault("random_state", seed)
        model_params.setdefault("eval_metric", "logloss")
        return XGBClassifier(**model_params)
    elif model_class == "LGBMClassifier":
        model_params.setdefault("random_state", seed)
        model_params.setdefault("verbose", -1)
        return LGBMClassifier(**model_params)
    elif model_class == "AdaBoostClassifier":
        model_params.setdefault("random_state", seed)
        return AdaBoostClassifier(**model_params)
    elif model_class == "QuadraticDiscriminantAnalysis":
        return QuadraticDiscriminantAnalysis(**model_params)
    elif model_class == "LinearDiscriminantAnalysis":
        return LinearDiscriminantAnalysis(**model_params)
    else:
        raise ValueError(f"Unknown model class: {model_class}")


def tune_with_optuna(
    model_class: str,
    X,
    y,
    search_space: Dict[str, Tuple[str, Any]],
    n_trials: int = 50,
    cv: int = 5,
    scoring: str = "roc_auc",
    seed: int = 42,
    n_jobs: int = -1,
) -> Tuple[Dict[str, Any], optuna.Study]:
    """Run Optuna optimization and return best params and study.
    
    Args:
        model_class: Name of the model class (e.g., "RandomForestClassifier")
        X: Feature matrix
        y: Target vector
        search_space: Dictionary mapping parameter names to (type, ...) tuples
            Types: "int", "float", "float_log", "categorical", "int_or_none"
        n_trials: Number of Optuna trials
        cv: Number of cross-validation folds
        scoring: Scoring metric
        seed: Random seed
        n_jobs: Number of parallel jobs for cross-validation (-1 uses all cores)
        
    Returns:
        Tuple of (best_params, study)
    """
    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            spec_type = spec[0]
            
            if spec_type == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif spec_type == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif spec_type == "float_log":
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif spec_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec[1])
            elif spec_type == "int_or_none":
                # Suggest None with some probability, otherwise suggest int
                if trial.suggest_categorical(f"{name}_is_none", [True, False]):
                    params[name] = None
                else:
                    params[name] = trial.suggest_int(name, spec[1], spec[2])
            else:
                raise ValueError(f"Unknown search space type: {spec_type}")
        
        model = _create_model_from_class(model_class, params, seed)
        score = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs).mean()
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    
    # Filter out auxiliary parameters (e.g., "max_depth_is_none") that are not
    # valid model arguments
    best_params = {
        k: v for k, v in study.best_params.items() if not k.endswith("_is_none")
    }
    return best_params, study
