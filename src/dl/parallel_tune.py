"""Parallel hyperparameter tuning using multiple GPUs.

This script coordinates multiple Optuna workers, each running on a separate GPU,
sharing a SQLite database for the study state. Supports resume from checkpoint.

Usage:
    python -m src.parallel_tune --config GCN_Graph --n-trials 50 --n-gpus 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import optuna

from . import config
from .tuning import save_tuned_params, get_available_gpus
from .utils import ensure_output_dir_with_confirm, save_json


def create_study_storage(out_dir: Path, study_name: str) -> str:
    """Create SQLite storage URL for Optuna study.
    
    Args:
        out_dir: Output directory for the database file
        study_name: Name of the study
        
    Returns:
        SQLite storage URL
    """
    db_path = out_dir / f"{study_name}.db"
    return f"sqlite:///{db_path}"


def get_completed_trials(storage: str, study_name: str) -> int:
    """Get number of completed trials in existing study.
    
    Args:
        storage: Optuna storage URL
        study_name: Name of the study
        
    Returns:
        Number of completed trials, 0 if study doesn't exist
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        return len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    except KeyError:
        return 0


def run_parallel_tuning(
    cfg: dict,
    out_dir: Path,
    config_key: str,
    n_trials: int = 50,
    n_gpus: Optional[int] = None,
    nb_epoch: int = 200,
    early_stopping_patience: Optional[int] = None,
    early_stopping_metric: Optional[str] = None,
    use_early_stopping: bool = True,
    seed: int = 42,
    n_cpu_per_worker: Optional[int] = None,
) -> dict:
    """Run parallel hyperparameter tuning using multiple GPUs.
    
    Each GPU runs an independent Optuna worker process. All workers share
    a SQLite database for the study state, enabling distributed optimization
    and resume from checkpoint.
    
    Args:
        cfg: Model configuration dictionary
        out_dir: Output directory
        config_key: Configuration key (e.g., "DMPNN_Chemprop_Graph")
        n_trials: Total number of Optuna trials
        n_gpus: Number of GPUs to use (None = all available)
        nb_epoch: Number of training epochs per trial
        early_stopping_patience: Patience for early stopping
        early_stopping_metric: Metric for early stopping
        use_early_stopping: Whether to use early stopping
        seed: Random seed
        n_cpu_per_worker: Number of CPUs per worker for parallel featurization
        
    Returns:
        Best parameters found
    """
    import multiprocessing
    
    # Detect available GPUs
    available_gpus = get_available_gpus()
    if not available_gpus:
        raise RuntimeError("No GPUs available. Use single-GPU tuning instead.")
    
    if n_gpus is None:
        n_gpus = len(available_gpus)
    else:
        n_gpus = min(n_gpus, len(available_gpus))
    
    gpu_ids = available_gpus[:n_gpus]
    
    # Calculate CPUs per worker if not specified
    total_cpus = multiprocessing.cpu_count()
    if n_cpu_per_worker is None:
        n_cpu_per_worker = max(1, total_cpus // n_gpus)
    
    print(f"Using {n_gpus} GPUs: {gpu_ids}")
    print(f"CPUs per worker: {n_cpu_per_worker} (total: {total_cpus})")
    
    # Setup study storage with model-specific name
    backend = cfg.get("backend", "deepchem")
    algorithm = cfg.get("algorithm", "GCN")
    study_name = f"{algorithm.lower()}_{backend}_tuning"
    storage = create_study_storage(out_dir, study_name)
    
    # Check for existing progress
    completed = get_completed_trials(storage, study_name)
    if completed > 0:
        print(f"Resuming from checkpoint: {completed}/{n_trials} trials completed")
    
    remaining_trials = n_trials - completed
    if remaining_trials <= 0:
        print("All trials already completed. Loading results...")
        study = optuna.load_study(study_name=study_name, storage=storage)
        return study.best_params
    
    # Distribute trials across GPUs
    trials_per_gpu = remaining_trials // n_gpus
    extra_trials = remaining_trials % n_gpus
    
    # Build worker command template
    train_csv = cfg["train_file"]
    
    # Launch worker processes
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        # Assign extra trials to first few GPUs
        worker_trials = trials_per_gpu + (1 if i < extra_trials else 0)
        if worker_trials == 0:
            continue
        
        # Build command for this worker
        cmd = [
            sys.executable, "-m", "src.parallel_tune",
            "--worker",
            "--gpu-id", str(gpu_id),
            "--storage", storage,
            "--study-name", study_name,
            "--config-key", config_key,
            "--train-csv", str(train_csv),
            "--n-trials", str(worker_trials),
            "--nb-epoch", str(nb_epoch),
            "--seed", str(seed),
            "--n-cpu", str(n_cpu_per_worker),
        ]
        
        if early_stopping_patience is not None:
            cmd.extend(["--patience", str(early_stopping_patience)])
        if early_stopping_metric is not None:
            cmd.extend(["--metric", early_stopping_metric])
        if not use_early_stopping:
            cmd.append("--no-early-stop")
        
        # Set environment to use specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        print(f"Starting worker on GPU {gpu_id} with {worker_trials} trials")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append((gpu_id, proc))
    
    # Monitor processes and print output
    print(f"\n{'='*60}")
    print(f"Parallel tuning started: {remaining_trials} trials across {len(processes)} GPUs")
    print(f"{'='*60}\n")
    
    # Wait for all processes and collect output
    for gpu_id, proc in processes:
        # Stream output in real-time
        for line in proc.stdout:
            print(f"[GPU {gpu_id}] {line}", end="")
        proc.wait()
        
        if proc.returncode != 0:
            print(f"WARNING: Worker on GPU {gpu_id} exited with code {proc.returncode}")
    
    # Load final results
    study = optuna.load_study(study_name=study_name, storage=storage)
    best_params = study.best_params
    
    # Save results
    save_tuned_params(best_params, out_dir / config.BEST_PARAMS_FILENAME)
    
    # Save study trials as JSON
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            })
    save_json(trials_data, out_dir / "optuna_study.json")
    
    # Save parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        import pandas as pd
        importance_df = pd.DataFrame(
            list(importance.items()), columns=["parameter", "importance"]
        ).sort_values("importance", ascending=False)
        importance_df.to_csv(out_dir / "optuna_importance.csv", index=False)
    except Exception:
        pass
    
    print(f"\n{'='*60}")
    print(f"Parallel tuning complete!")
    print(f"Total completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Best ROC-AUC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*60}")
    
    return best_params


def run_worker_main(args):
    """Entry point for worker subprocess."""
    from .tuning import run_worker
    
    run_worker(
        gpu_id=args.gpu_id,
        storage=args.storage,
        study_name=args.study_name,
        config_key=args.config_key,
        train_csv=Path(args.train_csv),
        n_trials_per_worker=args.n_trials,
        nb_epoch=args.nb_epoch,
        early_stopping_patience=args.patience,
        early_stopping_metric=args.metric,
        use_early_stopping=not args.no_early_stop,
        seed=args.seed,
        n_cpu_per_worker=args.n_cpu,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Parallel Optuna tuning with multiple GPUs")
    
    # Worker mode arguments
    parser.add_argument("--worker", action="store_true", help="Run as worker subprocess")
    parser.add_argument("--gpu-id", type=int, help="GPU ID for worker")
    parser.add_argument("--storage", type=str, help="Optuna storage URL")
    parser.add_argument("--study-name", type=str, help="Optuna study name")
    parser.add_argument("--config-key", type=str, help="Configuration key (e.g., DMPNN_Chemprop_Graph)")
    parser.add_argument("--train-csv", type=str, help="Path to training CSV")
    
    # Main mode arguments
    parser.add_argument("--config", default=config.DEFAULT_CONFIG_KEY, help="Config key")
    parser.add_argument("--n-trials", type=int, default=config.OPTUNA_N_TRIALS, help="Total trials")
    parser.add_argument("--n-gpus", type=int, default=None, help="Number of GPUs (default: all)")
    parser.add_argument("--nb-epoch", type=int, default=config.TRAINING_PARAMS["nb_epoch"], help="Epochs per trial")
    
    # Shared arguments
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--metric", type=str, choices=["roc_auc", "loss"], default=None, help="Early stopping metric")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-cpu", type=int, default=8, help="CPUs per worker for parallel featurization")
    
    args = parser.parse_args()
    
    if args.worker:
        # Worker mode: run a single worker
        run_worker_main(args)
    else:
        # Main mode: coordinate multiple workers
        cfg = config.MODEL_CONFIGS.get(args.config)
        if cfg is None:
            raise ValueError(f"Config key {args.config} not found.")
        
        # Ensure output directory exists
        out_dir = ensure_output_dir_with_confirm(config.OUTPUT_DIR, args.config)
        
        # Run parallel tuning
        run_parallel_tuning(
            cfg=cfg,
            out_dir=out_dir,
            config_key=args.config,
            n_trials=args.n_trials,
            n_gpus=args.n_gpus,
            nb_epoch=args.nb_epoch,
            early_stopping_patience=args.patience,
            early_stopping_metric=args.metric,
            use_early_stopping=not args.no_early_stop,
            seed=args.seed,
            n_cpu_per_worker=args.n_cpu,
        )


if __name__ == "__main__":
    main()

