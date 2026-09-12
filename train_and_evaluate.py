# -*- coding: utf-8 -*-
"""
Comprehensive Training, Cross-Validation, and Benchmark Harness.
Evaluates model stability and performance across multiple folds and seeds,
benchmarks against alternative modeling paradigms, and computes rubric points.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features import FeaturePipeline
from src.model import HurdleRankEnsemble
from src.selection import select_response_portfolio
from src.metrics import (
    compute_ndcg_at_k,
    compute_spearman_correlation,
    compute_high_impact_recall,
    calculate_judging_rubric
)


def run_cv_evaluation(df: pd.DataFrame, seeds=(42, 100, 2024, 777, 999), n_splits=5):
    """
    Evaluates HurdleRankEnsemble across 25 splits (5 seeds x 5 folds)
    with strict anti-leakage guarantee on each fold.
    """
    all_ndcgs = []
    all_worst_per_seed = []
    all_spearmans = []
    all_recalls = []

    print(f"\n{'='*75}")
    print(f"RUNNING REPEATED 5-FOLD CROSS-VALIDATION ({len(seeds)} SEEDS, {len(seeds)*n_splits} TOTAL FOLDS)")
    print(f"{'='*75}")

    t0 = time.time()

    for s_idx, seed in enumerate(seeds):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        seed_ndcgs = []

        for fold, (trn_idx, val_idx) in enumerate(kf.split(df)):
            df_trn = df.iloc[trn_idx].copy()
            df_val = df.iloc[val_idx].copy()

            # Preprocessing fitted STRICTLY on training split
            pipeline = FeaturePipeline().fit(df_trn)
            X_trn = pipeline.transform(df_trn)
            X_val = pipeline.transform(df_val)

            # Model fitted strictly on training split
            model = HurdleRankEnsemble(random_state=seed)
            model.fit(X_trn, df_trn['area'].values)

            # Evaluation predictions
            val_scores = model.predict(X_val)

            # Metrics
            ndcg = compute_ndcg_at_k(df_val['area'].values, val_scores, k=25)
            sp = compute_spearman_correlation(df_val['area'].values, val_scores)

            # Portfolio selection
            sel_idx, _ = select_response_portfolio(df_val, val_scores, portfolio_size=25, max_per_cell=4)
            rec = compute_high_impact_recall(df_trn['area'].values, df_val['area'].values, sel_idx)

            all_ndcgs.append(ndcg)
            seed_ndcgs.append(ndcg)
            all_spearmans.append(sp)
            all_recalls.append(rec)

        all_worst_per_seed.append(np.min(seed_ndcgs))
        print(f"Seed {seed:4d} | Mean NDCG: {np.mean(seed_ndcgs):.4f} | Worst NDCG: {np.min(seed_ndcgs):.4f}")

    elapsed = time.time() - t0

    mean_ndcg = float(np.mean(all_ndcgs))
    worst_ndcg_overall = float(np.min(all_ndcgs))
    avg_worst_ndcg = float(np.mean(all_worst_per_seed))
    mean_spearman = float(np.mean(all_spearmans))
    mean_recall = float(np.mean(all_recalls))

    rubric = calculate_judging_rubric(
        mean_ndcg=mean_ndcg,
        worst_ndcg=worst_ndcg_overall,
        mean_spearman=mean_spearman,
        mean_recall=mean_recall,
        runtime_reproducibility=1.0
    )

    print(f"\n{'='*75}")
    print("FINAL 100-POINT JUDGING RUBRIC ESTIMATE (25-FOLD AVERAGE)")
    print(f"{'='*75}")
    print(f"1. Mean NDCG@25 (35 pts):               {mean_ndcg:.4f}  -->  {rubric['pts_ndcg_35']:.2f} / 35.0")
    print(f"2. High-Impact Recall@25 (25 pts):       {mean_recall:.4f}  -->  {rubric['pts_recall_25']:.2f} / 25.0")
    print(f"3. Worst-Split NDCG@25 (20 pts):        {worst_ndcg_overall:.4f}  -->  {rubric['pts_worst_ndcg_20']:.2f} / 20.0")
    print(f"   (Average Worst-Split per seed:       {avg_worst_ndcg:.4f})")
    print(f"4. Mean Spearman Correlation (10 pts):  {mean_spearman:.4f}  -->  {rubric['pts_spearman_10']:.2f} / 10.0")
    print(f"5. Runtime & Reproducibility (10 pts):   {elapsed:.2f}s  -->  {rubric['pts_reproducibility_10']:.2f} / 10.0")
    print(f"{'-'*75}")
    print(f"ESTIMATED TOTAL SCORE:                  {rubric['total_score_100']:.2f} / 100.0")
    print(f"{'='*75}\n")

    return rubric


def run_baseline_comparisons(df: pd.DataFrame, seed=42, n_splits=5):
    """
    Compares HurdleRankEnsemble against alternative modeling paradigms:
    - Random Baseline
    - Unregularized Random Forest
    - HistGradientBoosting (MSE)
    - Pure Ridge Regressor
    - Proposed Hurdle-Ensemble
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    baselines = {
        '1. Random Uniform Baseline': lambda trn_X, trn_y, val_X: np.random.RandomState(seed).rand(len(val_X)),
        '2. Random Forest Regressor': lambda trn_X, trn_y, val_X: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=seed).fit(trn_X, np.log1p(trn_y)).predict(val_X),
        '3. HistGradientBoosting (MSE)': lambda trn_X, trn_y, val_X: HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=seed).fit(trn_X, np.log1p(trn_y)).predict(val_X),
        '4. Pure Ridge Regressor': lambda trn_X, trn_y, val_X: Ridge(alpha=20.0, random_state=seed).fit(trn_X, np.log1p(trn_y)).predict(val_X),
        '5. Proposed Hurdle-Ensemble': None
    }

    results = {name: {'ndcg': [], 'worst_ndcg': [], 'spearman': [], 'recall': []} for name in baselines}

    for fold, (trn_idx, val_idx) in enumerate(kf.split(df)):
        df_trn = df.iloc[trn_idx].copy()
        df_val = df.iloc[val_idx].copy()

        pipeline = FeaturePipeline().fit(df_trn)
        X_trn = pipeline.transform(df_trn)
        X_val = pipeline.transform(df_val)

        for name, fn in baselines.items():
            if name == '5. Proposed Hurdle-Ensemble':
                m = HurdleRankEnsemble(random_state=seed)
                m.fit(X_trn, df_trn['area'].values)
                scores = m.predict(X_val)
            else:
                scores = fn(X_trn, df_trn['area'].values, X_val)

            ndcg = compute_ndcg_at_k(df_val['area'].values, scores, k=25)
            sp = compute_spearman_correlation(df_val['area'].values, scores)
            sel_idx, _ = select_response_portfolio(df_val, scores, portfolio_size=25, max_per_cell=4)
            rec = compute_high_impact_recall(df_trn['area'].values, df_val['area'].values, sel_idx)

            results[name]['ndcg'].append(ndcg)
            results[name]['spearman'].append(sp)
            results[name]['recall'].append(rec)

    print(f"\n{'='*82}")
    print(f"{'MODEL PARADIGM COMPARISON TABLE':^82}")
    print(f"{'='*82}")
    header = f"{'Model Architecture':<30} | {'Mean NDCG':>9} | {'Worst NDCG':>10} | {'Mean Sp':>7} | {'Mean Rec':>8} | {'Rubric Pts':>10}"
    print(header)
    print(f"{'-'*82}")

    for name in results:
        m_ndcg = float(np.mean(results[name]['ndcg']))
        w_ndcg = float(np.min(results[name]['ndcg']))
        m_sp = float(np.mean(results[name]['spearman']))
        m_rec = float(np.mean(results[name]['recall']))
        pts = 35.0 * m_ndcg + 25.0 * m_rec + 20.0 * w_ndcg + 10.0 * max(0.0, m_sp) + 10.0
        print(f"{name:<30} | {m_ndcg:9.4f} | {w_ndcg:10.4f} | {m_sp:7.4f} | {m_rec:8.4f} | {pts:10.2f}")
    print(f"{'='*82}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and benchmark wildfire response model.')
    parser.add_argument('--data-path', type=str, default=r'd:\hack idea\wildfire_response\data\forestfires.csv',
                        help='Path to forestfires.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns from {args.data_path}")

    # 1. Run baseline comparisons
    run_baseline_comparisons(df)

    # 2. Run multi-seed repeated cross-validation
    run_cv_evaluation(df)
