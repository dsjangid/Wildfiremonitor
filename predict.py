# -*- coding: utf-8 -*-
"""
Production Inference and Response Portfolio Generator CLI.
Accepts arbitrary training and evaluation CSV paths.
Adheres strictly to the competition specifications:
- Training rows include target area
- Evaluation rows include only permitted input features (target area is not required or accessed)
- Preprocessing is fitted strictly on the provided training rows
- Zero target leakage, zero hard-coded lookups, zero identifiers
- Outputs predictions.csv (one impact score for every row)
- Outputs selected_portfolio.csv (exactly 25 rows, max 4 per (X, Y) cell)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features import FeaturePipeline, PERMITTED_FEATURES
from src.model import HurdleRankEnsemble
from src.selection import select_response_portfolio


def main():
    parser = argparse.ArgumentParser(description="Wildfire Response Prioritization Inference Engine.")
    parser.add_argument(
        "--train-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "forestfires.csv"),
        help="Path to training CSV file (must include permitted features and 'area')."
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "forestfires.csv"),
        help="Path to evaluation CSV file (must include permitted features; 'area' is ignored if present)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "outputs"),
        help="Directory where predictions.csv and selected_portfolio.csv will be saved."
    )
    parser.add_argument(
        "--portfolio-size",
        type=int,
        default=25,
        help="Exact number of responses to select (default: 25)."
    )
    parser.add_argument(
        "--max-per-cell",
        type=int,
        default=4,
        help="Maximum responses permitted from any single (X, Y) cell (default: 4)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    print(f"Loading training data from: {args.train_path}")
    df_train = pd.read_csv(args.train_path)
    if 'area' not in df_train.columns:
        raise ValueError("Training dataset must contain the 'area' target column.")

    print(f"Loading evaluation data from: {args.eval_path}")
    df_eval = pd.read_csv(args.eval_path)

    # Check permitted features
    for col in PERMITTED_FEATURES:
        if col not in df_train.columns:
            raise ValueError(f"Training dataset missing required feature: '{col}'")
        if col not in df_eval.columns:
            raise ValueError(f"Evaluation dataset missing required feature: '{col}'")

    print(f"Train rows: {len(df_train)} | Evaluation rows: {len(df_eval)}")

    # 2. Fit feature pipeline strictly on training data
    print("Fitting FeaturePipeline strictly on training partition...")
    pipeline = FeaturePipeline().fit(df_train)
    X_train_scaled = pipeline.transform(df_train)
    X_eval_scaled = pipeline.transform(df_eval)

    # 3. Fit model strictly on training data
    print("Fitting HurdleRankEnsemble strictly on training partition...")
    model = HurdleRankEnsemble(random_state=args.seed)
    model.fit(X_train_scaled, df_train['area'].values)

    # 4. Predict continuous impact scores on evaluation partition
    print("Generating continuous impact predictions for evaluation rows...")
    impact_scores = model.predict(X_eval_scaled)

    # 5. Constrained portfolio selection
    print(f"Selecting response portfolio ({args.portfolio_size} rows, max {args.max_per_cell} per cell)...")
    selected_indices, selected_df = select_response_portfolio(
        df_eval=df_eval,
        impact_scores=impact_scores,
        portfolio_size=args.portfolio_size,
        max_per_cell=args.max_per_cell
    )

    # 6. Save outputs
    # predictions.csv
    pred_df = df_eval[['X', 'Y']].copy()
    pred_df.insert(0, 'row_id', np.arange(len(df_eval)))
    pred_df['impact_score'] = impact_scores
    pred_df['predicted_rank'] = pd.Series(impact_scores).rank(ascending=False, method='min').astype(int)

    pred_out_path = os.path.join(args.output_dir, "predictions.csv")
    pred_df.to_csv(pred_out_path, index=False)
    print(f"Saved prediction scores to: {pred_out_path} ({len(pred_df)} rows)")

    # selected_portfolio.csv
    portfolio_out_path = os.path.join(args.output_dir, "selected_portfolio.csv")
    selected_df.to_csv(portfolio_out_path, index=False)
    print(f"Saved selected response portfolio to: {portfolio_out_path} ({len(selected_df)} rows)")

    # 7. Verification checks
    print("\n--- Output Verification Checks ---")
    print(f"Total rows in predictions.csv: {len(pred_df)} (matches eval set: {len(pred_df) == len(df_eval)})")
    print(f"Total rows in selected_portfolio.csv: {len(selected_df)} (exactly {args.portfolio_size}: {len(selected_df) == args.portfolio_size})")

    cell_counts = selected_df.groupby(['X', 'Y']).size()
    max_cell_count = cell_counts.max()
    print(f"Max observations from any (X, Y) cell: {max_cell_count} (<= {args.max_per_cell}: {max_cell_count <= args.max_per_cell})")
    print(f"Unique (X, Y) cells represented in portfolio: {len(cell_counts)}")
    print(f"Top 5 represented cells:\n{cell_counts.sort_values(ascending=False).head(5)}")
    print("----------------------------------\n")


if __name__ == '__main__':
    main()
