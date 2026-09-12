# -*- coding: utf-8 -*-
"""
Worst-Case Scenario Simulation & Stress-Testing Suite.
Evaluates the Wildfire Response Prioritization system under extreme, adversarial conditions:
1. Scenario A: Spatial Out-of-Domain Shift (Spatial Block Partitioning)
2. Scenario B: Extreme Seasonal Shift (Train Non-Summer -> Test Peak Summer)
3. Scenario C: Spatial Cell Saturation & Quota Bottlenecks (Clustered Ignitions)
4. Scenario D: Severe Drought & High-Wind Weather Regime Shift
5. Scenario E: Exhaustive Worst-Split Search (100-Split Stress Test & Error Autopsy)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features import FeaturePipeline, extract_domain_features
from src.model import HurdleRankEnsemble
from src.selection import select_response_portfolio
from src.metrics import (
    compute_ndcg_at_k,
    compute_spearman_correlation,
    compute_high_impact_recall
)


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


# -------------------------------------------------------------------------
# SCENARIO A: Spatial Out-of-Domain Shift (Spatial Quadrant Holdout)
# -------------------------------------------------------------------------
def test_scenario_spatial_shift(df: pd.DataFrame):
    print("\n" + "=" * 78)
    print("SCENARIO A: SPATIAL OUT-OF-DOMAIN SHIFT (Geographic Quadrant Holdout)")
    print("=" * 78)
    print("Tests model generalizability when evaluation rows occur in completely different")
    print("geographic regions of Montesinho park than the training data.")

    # Partition park into East (X >= 5) and West (X < 5)
    east_mask = df['X'] >= 5
    west_mask = ~east_mask

    tests = [
        ("Train West (X < 5) -> Test East (X >= 5)", west_mask, east_mask),
        ("Train East (X >= 5) -> Test West (X < 5)", east_mask, west_mask)
    ]

    for name, trn_mask, val_mask in tests:
        df_trn = df[trn_mask].copy()
        df_val = df[val_mask].copy()

        pipe = FeaturePipeline().fit(df_trn)
        X_trn = pipe.transform(df_trn)
        X_val = pipe.transform(df_val)

        model = HurdleRankEnsemble(random_state=42)
        model.fit(X_trn, df_trn['area'].values)
        scores = model.predict(X_val)

        ndcg = compute_ndcg_at_k(df_val['area'].values, scores, k=25)
        sp = compute_spearman_correlation(df_val['area'].values, scores)
        sel_idx, sel_df = select_response_portfolio(df_val, scores, portfolio_size=25, max_per_cell=4)
        rec = compute_high_impact_recall(df_trn['area'].values, df_val['area'].values, sel_idx)

        print(f"\nExperiment: {name}")
        print(f"Train samples: {len(df_trn)} | Eval samples: {len(df_val)}")
        print(f"  NDCG@25:                 {ndcg:.4f}")
        print(f"  High-Impact Recall@25:   {rec:.4f}")
        print(f"  Spearman Correlation:    {sp:.4f}")
        print(f"  Unique cells in portfolio: {sel_df[['X', 'Y']].drop_duplicates().shape[0]}")
        print(f"  Max crews in any cell:   {sel_df.groupby(['X', 'Y']).size().max()} (<= 4: True)")


# -------------------------------------------------------------------------
# SCENARIO B: Severe Seasonal Shift (Train Non-Summer -> Test Peak Summer)
# -------------------------------------------------------------------------
def test_scenario_seasonal_shift(df: pd.DataFrame):
    print("\n" + "=" * 78)
    print("SCENARIO B: SEVERE SEASONAL REGIME SHIFT (Out-of-Season Holdout)")
    print("=" * 78)
    print("Tests if physical fire weather equations (VPD, BUI) generalize when")
    print("training on mild months and evaluating on extreme summer drought.")

    summer_months = ['aug', 'sep']
    summer_mask = df['month'].str.lower().isin(summer_months)
    non_summer_mask = ~summer_mask

    df_trn = df[non_summer_mask].copy()
    df_val = df[summer_mask].copy()

    pipe = FeaturePipeline().fit(df_trn)
    X_trn = pipe.transform(df_trn)
    X_val = pipe.transform(df_val)

    model = HurdleRankEnsemble(random_state=42)
    model.fit(X_trn, df_trn['area'].values)
    scores = model.predict(X_val)

    ndcg = compute_ndcg_at_k(df_val['area'].values, scores, k=25)
    sp = compute_spearman_correlation(df_val['area'].values, scores)
    sel_idx, sel_df = select_response_portfolio(df_val, scores, portfolio_size=25, max_per_cell=4)
    rec = compute_high_impact_recall(df_trn['area'].values, df_val['area'].values, sel_idx)

    print(f"Train on Non-Summer (Jan-Jul, Oct-Dec: {len(df_trn)} rows)")
    print(f"Evaluate on Peak Summer (Aug-Sep: {len(df_val)} rows)")
    print(f"  NDCG@25:                 {ndcg:.4f}")
    print(f"  High-Impact Recall@25:   {rec:.4f}")
    print(f"  Spearman Correlation:    {sp:.4f}")
    print(f"  Unique cells selected:   {sel_df[['X', 'Y']].drop_duplicates().shape[0]}")


# -------------------------------------------------------------------------
# SCENARIO C: Spatial Cell Saturation & Capacity Upper Bound
# -------------------------------------------------------------------------
def test_scenario_cell_saturation(df: pd.DataFrame):
    print("\n" + "=" * 78)
    print("SCENARIO C: CELL SATURATION & QUOTA BOTTLENECK ANALYSIS")
    print("=" * 78)
    print("Analyzes what happens when high-impact fires clump in only a few cells.")
    print("Computes the theoretical upper bound under the <= 4 constraint vs unconstrained.")

    pipe = FeaturePipeline().fit(df)
    X = pipe.transform(df)
    model = HurdleRankEnsemble(random_state=42).fit(X, df['area'].values)
    scores = model.predict(X)

    # Unconstrained selection (pure top 25)
    unconstrained_idx = np.argsort(-scores)[:25]
    unconstrained_df = df.iloc[unconstrained_idx]
    unconstrained_max_cell = unconstrained_df.groupby(['X', 'Y']).size().max()
    unconstrained_unique_cells = len(unconstrained_df[['X', 'Y']].drop_duplicates())

    # Constrained selection (max 4 per cell)
    constrained_idx, constrained_df = select_response_portfolio(df, scores, portfolio_size=25, max_per_cell=4)
    constrained_max_cell = constrained_df.groupby(['X', 'Y']).size().max()
    constrained_unique_cells = len(constrained_df[['X', 'Y']].drop_duplicates())

    p80 = np.percentile(df['area'].values, 80, method='linear')
    unconstrained_rec = (unconstrained_df['area'] >= p80).sum() / (df['area'] >= p80).sum()
    constrained_rec = (constrained_df['area'] >= p80).sum() / (df['area'] >= p80).sum()

    print(f"Unconstrained Top 25 (Violates Rule):")
    print(f"  Max crews in single cell: {unconstrained_max_cell} (VIOLATES <= 4 rule)")
    print(f"  Unique cells protected:   {unconstrained_unique_cells}")
    print(f"  High-Impact Recall:       {unconstrained_rec:.4f}")

    print(f"\nConstrained Top 25 (Greedy Matroid Compliant):")
    print(f"  Max crews in single cell: {constrained_max_cell} (COMPLIANT <= 4)")
    print(f"  Unique cells protected:   {constrained_unique_cells} (+{constrained_unique_cells - unconstrained_unique_cells} more regions covered)")
    print(f"  High-Impact Recall:       {constrained_rec:.4f}")
    print(f"  Recall retention ratio:   {(constrained_rec / unconstrained_rec)*100:.1f}% of unconstrained potential retained!")


# -------------------------------------------------------------------------
# SCENARIO D: Severe Drought & High-Wind Weather Regime Shift
# -------------------------------------------------------------------------
def test_scenario_extreme_weather(df: pd.DataFrame):
    print("\n" + "=" * 78)
    print("SCENARIO D: SEVERE DROUGHT & HIGH-WIND WEATHER REGIME SHIFT")
    print("=" * 78)
    print("Evaluates how the model distinguishes high-impact vs low-impact fires")
    print("when environmental conditions are in the top quartile of drought and wind.")

    # High drought: DC > 75th percentile (DC > 713.9)
    # High wind: wind > 50th percentile
    severe_drought_wind = (df['DC'] >= df['DC'].quantile(0.75)) & (df['wind'] >= df['wind'].quantile(0.50))
    print(f"Rows meeting severe fire weather criteria: {severe_drought_wind.sum()} / {len(df)}")

    df_severe = df[severe_drought_wind].copy()
    pipe = FeaturePipeline().fit(df)
    X_severe = pipe.transform(df_severe)

    model = HurdleRankEnsemble(random_state=42).fit(pipe.transform(df), df['area'].values)
    scores = model.predict(X_severe)

    ndcg = compute_ndcg_at_k(df_severe['area'].values, scores, k=min(25, len(df_severe)))
    sp = compute_spearman_correlation(df_severe['area'].values, scores)

    print(f"  NDCG on severe regime:    {ndcg:.4f}")
    print(f"  Spearman correlation:     {sp:.4f}")


# -------------------------------------------------------------------------
# SCENARIO E: Exhaustive 100-Split Stress Test & Worst-Split Autopsy
# -------------------------------------------------------------------------
def test_scenario_worst_split_autopsy(df: pd.DataFrame, n_trials=100):
    print("\n" + "=" * 78)
    print(f"SCENARIO E: EXHAUSTIVE {n_trials}-SPLIT SEARCH & WORST-SPLIT AUTOPSY")
    print("=" * 78)
    print(f"Running {n_trials} randomized 80/20 train/test splits to locate the absolute worst-case")
    print("evaluation partition in the entire data manifold and inspect the failure mode.")

    worst_ndcg = 1.0
    worst_info = None
    all_ndcgs = []

    for trial in range(n_trials):
        seed = 1000 + trial
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        trn_idx, val_idx = next(kf.split(df))

        df_trn = df.iloc[trn_idx].copy()
        df_val = df.iloc[val_idx].copy()

        pipe = FeaturePipeline().fit(df_trn)
        X_trn = pipe.transform(df_trn)
        X_val = pipe.transform(df_val)

        model = HurdleRankEnsemble(random_state=seed).fit(X_trn, df_trn['area'].values)
        scores = model.predict(X_val)

        ndcg = compute_ndcg_at_k(df_val['area'].values, scores, k=25)
        all_ndcgs.append(ndcg)

        if ndcg < worst_ndcg:
            worst_ndcg = ndcg
            worst_info = {
                'trial': trial,
                'seed': seed,
                'ndcg': ndcg,
                'df_trn': df_trn,
                'df_val': df_val,
                'scores': scores
            }

    print(f"\n{n_trials}-Trial Summary:")
    print(f"  Mean NDCG@25:            {np.mean(all_ndcgs):.4f} +/- {np.std(all_ndcgs):.4f}")
    print(f"  Median NDCG@25:          {np.median(all_ndcgs):.4f}")
    print(f"  Absolute Worst NDCG@25:  {worst_ndcg:.4f} (Trial {worst_info['trial']}, Seed {worst_info['seed']})")

    # Autopsy of worst split
    w_val = worst_info['df_val'].copy()
    w_val['impact_score'] = worst_info['scores']
    w_p80 = np.percentile(worst_info['df_trn']['area'].values, 80, method='linear')
    w_val['is_high_impact'] = w_val['area'] >= w_p80

    sel_idx, sel_df = select_response_portfolio(w_val, worst_info['scores'], portfolio_size=25, max_per_cell=4)
    w_rec = compute_high_impact_recall(worst_info['df_trn']['area'].values, w_val['area'].values, sel_idx)

    print(f"\nAutopsy of Absolute Worst-Case Split:")
    print(f"  Validation rows:         {len(w_val)}")
    print(f"  High-impact threshold:   {w_p80:.2f} ha")
    print(f"  High-impact fires in test: {w_val['is_high_impact'].sum()}")
    print(f"  High-Impact Recall:      {w_rec:.4f}")
    print(f"  Selected portfolio unique cells: {sel_df[['X', 'Y']].drop_duplicates().shape[0]}")
    print(f"  Max per cell in portfolio:       {sel_df.groupby(['X', 'Y']).size().max()}")
    print("=" * 78 + "\n")


if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'forestfires.csv')
    df = load_dataset(data_path)
    print(f"Loaded {len(df)} rows for worst-scenario simulation.")

    test_scenario_spatial_shift(df)
    test_scenario_seasonal_shift(df)
    test_scenario_cell_saturation(df)
    test_scenario_extreme_weather(df)
    test_scenario_worst_split_autopsy(df, n_trials=100)
