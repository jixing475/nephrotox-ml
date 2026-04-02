from __future__ import annotations

import argparse
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shap
from joblib import dump
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from . import config
from .data_loader import load_data_full, split_features_labels, FeatureEngineer
from .metrics import calculate_metrics, metrics_to_frame
from .tuning import (
    tune_hyperparameters,
    tune_with_optuna,
    save_tuned_params,
    load_tuned_params,
)
from .utils import (
    ensure_output_dir_with_confirm,
    save_dataframe,
    save_json,
    log_versions,
    update_json,
)


def run_single_seed(
    seed: int,
    df: pd.DataFrame,
    cfg: Dict,
    external_df: pd.DataFrame,
    tuned_params: Dict | None = None,
):
    # Split: 80/10/10 with exact fractions
    train_val_df, test_df = train_test_split(
        df, test_size=0.1, stratify=df["label"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=1.0 / 9.0,  # 10% of total from remaining 90%
        stratify=train_val_df["label"],
        random_state=seed,
    )

    # Separate IDs/features/labels
    train_ids, X_train, y_train = split_features_labels(train_df)
    val_ids, X_val, y_val = split_features_labels(val_df)
    test_ids, X_test, y_test = split_features_labels(test_df)

    # Feature engineering (fit on train only)
    fe = FeatureEngineer()
    X_train_proc = fe.fit_transform(X_train)
    X_val_proc = fe.transform(X_val)
    X_test_proc = fe.transform(X_test)

    # Model
    model = create_model(cfg, seed, tuned_params)
    model.fit(X_train_proc, y_train)

    # Predictions & metrics
    y_val_pred = model.predict(X_val_proc)
    y_val_proba = model.predict_proba(X_val_proc)[:, 1]
    y_test_pred = model.predict(X_test_proc)
    y_test_proba = model.predict_proba(X_test_proc)[:, 1]

    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

    # External test (use same fe transform)
    ext_ids, X_ext, y_ext = split_features_labels(external_df)
    X_ext_proc = fe.transform(X_ext)
    y_ext_pred = model.predict(X_ext_proc)
    y_ext_proba = model.predict_proba(X_ext_proc)[:, 1]
    ext_metrics = calculate_metrics(y_ext, y_ext_pred, y_ext_proba)

    # Feature importance
    fi_df = pd.DataFrame(columns=["feature", "importance"])
    if hasattr(model, "feature_importances_"):
        fi_vals = model.feature_importances_
        fi_df = pd.DataFrame(
            {
                "feature": X_train_proc.columns[: len(fi_vals)],
                "importance": fi_vals,
            }
        ).sort_values("importance", ascending=False)
    elif hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        fi_vals = np.abs(coef).mean(axis=0)
        fi_df = pd.DataFrame(
            {
                "feature": X_train_proc.columns[: len(fi_vals)],
                "importance": fi_vals,
            }
        ).sort_values("importance", ascending=False)

    warnings: List[dict] = []

    # SHAP (tree-based models only; skip unsupported/unstable cases)
    shap_df = pd.DataFrame(columns=["feature", "mean_abs_shap"])
    shap_supported_models = {"RandomForestClassifier", "XGBClassifier", "LGBMClassifier"}
    if cfg["model_class"] in shap_supported_models:
        shap_sample = X_ext_proc
        if len(shap_sample) > 200:
            shap_sample = shap_sample.sample(200, random_state=seed)

        shap_values_raw = None
        try:
            explainer = shap.TreeExplainer(
                model,
                feature_perturbation="interventional",
                model_output="raw",
                check_additivity=False,
            )
            shap_values_raw = explainer.shap_values(shap_sample)
        except Exception as e:
            # Continue training even if SHAP computation fails (e.g., additivity check issues)
            print(f"Skipping SHAP for {cfg['model_class']} due to error: {e}")
            warnings.append(
                {
                    "type": "shap_skipped",
                    "model_class": cfg.get("model_class"),
                    "seed": seed,
                    "train_file": str(cfg.get("train_file")),
                    "test_file": str(cfg.get("test_file")),
                    "reason": str(e),
                }
            )

        if shap_values_raw is not None:
            if isinstance(shap_values_raw, list):
                shap_values = shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0]
            elif hasattr(shap_values_raw, "ndim") and shap_values_raw.ndim == 3:
                shap_values = shap_values_raw[:, 1, :]
            else:
                shap_values = shap_values_raw

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            n_feats = min(len(shap_sample.columns), len(mean_abs_shap))
            shap_df = pd.DataFrame(
                {
                    "feature": shap_sample.columns[:n_feats],
                    "mean_abs_shap": mean_abs_shap[:n_feats],
                }
            ).sort_values("mean_abs_shap", ascending=False)

    # Predictions dataframe for external test
    preds_df = pd.DataFrame(
        {
            "seed": seed,
            "ID": ext_ids,
            "y_true": y_ext,
            "y_pred": y_ext_pred,
            "y_proba": y_ext_proba,
        }
    )

    # ROC data (external test)
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_ext, y_ext_proba)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

    # Confusion matrix (external test)
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_ext, y_ext_pred)
    cm_df = pd.DataFrame(cm, columns=["pred_0", "pred_1"], index=["true_0", "true_1"])

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "ext_metrics": ext_metrics,
        "preds_df": preds_df,
        "feature_importance": fi_df,
        "shap": shap_df,
        "roc": roc_df,
        "confusion": cm_df,
        "warnings": warnings,
    }


def train_final(
    seed: int,
    df_train: pd.DataFrame,
    cfg: Dict,
    external_df: pd.DataFrame,
    out_dir: pathlib.Path,
    config_key: str,
    tuned_params: Dict | None = None,
):
    """Train a single final model on all training data and save bundle."""
    # Split features/labels
    train_ids, X_train, y_train = split_features_labels(df_train)

    # Feature engineering (fit on all training data)
    fe = FeatureEngineer()
    X_train_proc = fe.fit_transform(X_train)

    # Model
    model = create_model(cfg, seed, tuned_params)
    model.fit(X_train_proc, y_train)

    # External evaluation
    ext_ids, X_ext, y_ext = split_features_labels(external_df)
    X_ext_proc = fe.transform(X_ext)
    y_ext_pred = model.predict(X_ext_proc)
    y_ext_proba = model.predict_proba(X_ext_proc)[:, 1]
    ext_metrics = calculate_metrics(y_ext, y_ext_pred, y_ext_proba)

    # Save bundle
    params = tuned_params.copy() if tuned_params else {}
    bundle = {
        "model": model,
        "fe": fe,
        "features": fe.remaining_features,
        "params": params,
        "seed": seed,
    }
    out_path = out_dir / f"{config_key.lower()}_final.joblib"
    dump(bundle, out_path)

    # Save metrics and predictions
    save_json({"external": ext_metrics}, out_dir / "final_metrics.json")
    preds_df = pd.DataFrame(
        {
            "ID": ext_ids,
            "y_true": y_ext,
            "y_pred": y_ext_pred,
            "y_proba": y_ext_proba,
        }
    )
    save_dataframe(preds_df, out_dir / "final_predictions.csv")

    return out_path, ext_metrics


def create_model(cfg: dict, seed: int, tuned_params: Dict | None = None):
    model_class = cfg["model_class"]
    params = tuned_params.copy() if tuned_params else {}
    if model_class == "RandomForestClassifier":
        params.setdefault("n_jobs", -1)
        params.setdefault("random_state", seed)
        params.setdefault("class_weight", "balanced")
        return RandomForestClassifier(**params)
    elif model_class == "SVC":
        params.setdefault("probability", True)
        params.setdefault("random_state", seed)
        params.setdefault("class_weight", "balanced")
        return SVC(**params)
    elif model_class == "XGBClassifier":
        params.setdefault("random_state", seed)
        params.setdefault("eval_metric", "logloss")
        return XGBClassifier(**params)
    elif model_class == "LGBMClassifier":
        params.setdefault("random_state", seed)
        params.setdefault("verbose", -1)
        return LGBMClassifier(**params)
    elif model_class == "AdaBoostClassifier":
        params.setdefault("random_state", seed)
        return AdaBoostClassifier(**params)
    elif model_class == "QuadraticDiscriminantAnalysis":
        return QuadraticDiscriminantAnalysis(**params)
    elif model_class == "LinearDiscriminantAnalysis":
        return LinearDiscriminantAnalysis(**params)
    else:
        raise ValueError(f"Unknown model class: {model_class}")


def run_tuning(
    cfg: dict,
    df_train: pd.DataFrame,
    out_dir: pathlib.Path,
    method: str = "grid",
    n_trials: int = 50,
):
    """Run hyperparameter tuning on full training data and persist best params.
    
    Args:
        cfg: Model configuration dictionary
        df_train: Training dataframe
        out_dir: Output directory
        method: Tuning method ("grid" or "optuna")
        n_trials: Number of Optuna trials (only used if method="optuna")
        
    Returns:
        Tuple of (best_model, best_params, results_dict)
    """
    _, X_train, y_train = split_features_labels(df_train)
    fe = FeatureEngineer()
    X_train_proc = fe.fit_transform(X_train)

    if method == "grid":
        param_grid = config.PARAM_GRIDS.get(cfg["model_class"])
        if not param_grid:
            raise ValueError(f"No PARAM_GRIDS entry for model class {cfg['model_class']}")

        base_model = create_model(cfg, seed=0, tuned_params=None)
        best_model, best_params, cv_results = tune_hyperparameters(
            base_model,
            X_train_proc,
            y_train,
            param_grid,
            cv=config.TUNING_CV_FOLDS,
            scoring="roc_auc",
        )

        save_tuned_params(best_params, out_dir / config.BEST_PARAMS_FILENAME)
        cv_results_df = pd.DataFrame(cv_results)
        save_dataframe(cv_results_df, out_dir / "tuning_cv_results.csv")
        return best_model, best_params, cv_results
    
    else:  # optuna
        search_space = config.OPTUNA_SEARCH_SPACES.get(cfg["model_class"])
        if not search_space:
            raise ValueError(
                f"No OPTUNA_SEARCH_SPACES entry for model class {cfg['model_class']}"
            )

        best_params, study = tune_with_optuna(
            cfg["model_class"],
            X_train_proc,
            y_train,
            search_space,
            n_trials=n_trials,
            cv=config.TUNING_CV_FOLDS,
            scoring="roc_auc",
            seed=0,
        )

        save_tuned_params(best_params, out_dir / config.BEST_PARAMS_FILENAME)
        
        # Save study trials as JSON
        trials_data = []
        for trial in study.trials:
            trials_data.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            })
        save_json(trials_data, out_dir / "optuna_study.json")
        
        # Save parameter importance if available
        try:
            importance = optuna.importance.get_param_importances(study)
            importance_df = pd.DataFrame(
                list(importance.items()), columns=["parameter", "importance"]
            ).sort_values("importance", ascending=False)
            save_dataframe(importance_df, out_dir / "optuna_importance.csv")
        except Exception:
            # Parameter importance requires at least 2 completed trials
            pass
        
        # Create a dummy model for compatibility
        best_model = create_model(cfg, seed=0, tuned_params=best_params)
        best_model.fit(X_train_proc, y_train)
        
        return best_model, best_params, {"study": study}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=config.DEFAULT_CONFIG_KEY, help="Config key from MODEL_CONFIGS")
    parser.add_argument(
        "--save-final",
        action="store_true",
        help="Train once on full train data and save <config>_final.joblib",
    )
    parser.add_argument("--final-seed", type=int, default=0, help="Seed used when --save-final is set")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning on full training data and save best params",
    )
    parser.add_argument(
        "--tuning-method",
        choices=["grid", "optuna"],
        default="grid",
        help="Hyperparameter tuning method: grid (GridSearchCV) or optuna (Bayesian optimization)",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (only used with --tuning-method optuna)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Number of random seeds to use (default: all 10 seeds)",
    )
    args = parser.parse_args()

    cfg = config.MODEL_CONFIGS.get(args.config)
    if cfg is None:
        raise ValueError(f"Config key {args.config} not found.")

    train_path = pathlib.Path(cfg["train_file"])
    test_path = pathlib.Path(cfg["test_file"])
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    df_train = load_data_full(train_path)
    df_external = load_data_full(test_path)

    # Ensure output directory exists (files will be overwritten)
    out_dir = ensure_output_dir_with_confirm(config.OUTPUT_DIR, args.config)
    tuned_params_path = out_dir / config.BEST_PARAMS_FILENAME

    # Tuning phase only
    if args.tune:
        _, tuned_params, cv_results = run_tuning(
            cfg, df_train, out_dir, method=args.tuning_method, n_trials=args.optuna_trials
        )
        print(f"Tuning complete ({args.tuning_method}). Best params saved to: {tuned_params_path}")
        print("Best params:", tuned_params)
        return

    # Load tuned params if available
    tuned_params = None
    if tuned_params_path.exists():
        tuned_params = load_tuned_params(tuned_params_path)
        print(f"Loaded tuned params from: {tuned_params_path}")

    # If only saving a single final model
    if args.save_final:
        model_path, ext_metrics = train_final(
            args.final_seed, df_train, cfg, df_external, out_dir, args.config, tuned_params
        )
        print(f"Saved final model to: {model_path}")
        print("External metrics:", ext_metrics)
        log_versions(out_dir / "experiment_info.json")
        paper_targets = {
            "val": {"AUC": 91.3, "ACC": 82.4, "SE": 78.9, "F1": 81.8, "Kappa": 64.7},
            "test": {"AUC": 91.7, "ACC": 83.1, "SE": 78.4, "F1": 82.4, "Kappa": 66.2},
        }
        save_json(paper_targets, out_dir / "paper_targets.json")
        return

    # Determine which seeds to use
    seeds_to_use = config.RANDOM_SEEDS
    if args.n_seeds is not None:
        seeds_to_use = config.RANDOM_SEEDS[:args.n_seeds]
        print(f"Using {len(seeds_to_use)} seed(s) for testing: {seeds_to_use}")

    # Containers for multi-seed experiment
    val_metrics_list: List[dict] = []
    test_metrics_list: List[dict] = []
    ext_metrics_list: List[dict] = []
    preds_all: List[pd.DataFrame] = []
    roc_all: List[pd.DataFrame] = []
    fi_all: List[pd.DataFrame] = []
    shap_all: List[pd.DataFrame] = []
    cm_all: List[pd.DataFrame] = []
    run_warnings: List[dict] = []

    for seed in tqdm(seeds_to_use, desc="Seeds"):
        results = run_single_seed(seed, df_train, cfg, df_external, tuned_params)
        val_metrics_list.append(results["val_metrics"])
        test_metrics_list.append(results["test_metrics"])
        ext_metrics_list.append(results["ext_metrics"])
        preds_all.append(results["preds_df"])
        roc_all.append(results["roc"].assign(seed=seed))
        fi_all.append(results["feature_importance"].assign(seed=seed))
        shap_all.append(results["shap"].assign(seed=seed))
        cm_all.append(results["confusion"].assign(seed=seed))
        run_warnings.extend(results.get("warnings", []))

    # Aggregate metrics
    val_df = metrics_to_frame(val_metrics_list, "val", seeds_to_use)
    test_df = metrics_to_frame(test_metrics_list, "test", seeds_to_use)
    ext_df = metrics_to_frame(ext_metrics_list, "external", seeds_to_use)

    # Mean/std summary
    def summarize(df, set_name):
        metric_cols = [c for c in df.columns if c not in ["seed", "set"]]
        summary = df[metric_cols].agg(["mean", "std"]).T.reset_index()
        summary.columns = ["metric", "mean", "std"]
        summary["set"] = set_name
        return summary

    summary_df = pd.concat([
        summarize(val_df, "val"),
        summarize(test_df, "test"),
        summarize(ext_df, "external"),
    ], ignore_index=True)

    # Paths
    per_fold_df = pd.concat([val_df, test_df, ext_df], ignore_index=True)
    save_dataframe(per_fold_df, out_dir / "cv_results_per_fold.csv")
    save_dataframe(summary_df, out_dir / "cv_summary.csv")

    # Predictions and curves
    preds_concat = pd.concat(preds_all, ignore_index=True)
    preds_concat["set"] = "external"
    save_dataframe(preds_concat, out_dir / "test_predictions.csv")

    save_dataframe(pd.concat(roc_all, ignore_index=True), out_dir / "roc_curve_data.csv")
    save_dataframe(pd.concat(fi_all, ignore_index=True), out_dir / "feature_importance.csv")
    save_dataframe(pd.concat(shap_all, ignore_index=True), out_dir / "shap_values.csv")
    # Confusion matrices per seed
    cm_concat = pd.concat(cm_all, keys=seeds_to_use, names=["seed", "true"])
    cm_concat.to_csv(out_dir / "confusion_matrix.csv")

    # Experiment info / versions
    log_versions(out_dir / "experiment_info.json")
    # Append warnings so missing SHAP is explicit in provenance
    shap_supported_models = {"RandomForestClassifier", "XGBClassifier", "LGBMClassifier"}
    if cfg.get("model_class") not in shap_supported_models:
        run_warnings.insert(
            0,
            {
                "type": "shap_disabled",
                "model_class": cfg.get("model_class"),
                "seed": None,
                "train_file": str(cfg.get("train_file")),
                "test_file": str(cfg.get("test_file")),
                "reason": "SHAP TreeExplainer is not enabled for this model class in this workflow.",
            },
        )
    if run_warnings:
        update_json(out_dir / "experiment_info.json", {"warnings": run_warnings})

    # Metrics comparison with paper (if desired)
    paper_targets = {
        "val": {"AUC": 91.3, "ACC": 82.4, "SE": 78.9, "F1": 81.8, "Kappa": 64.7},
        "test": {"AUC": 91.7, "ACC": 83.1, "SE": 78.4, "F1": 82.4, "Kappa": 66.2},
    }
    save_json(paper_targets, out_dir / "paper_targets.json")


if __name__ == "__main__":
    main()
