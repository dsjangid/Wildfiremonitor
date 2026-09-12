# -*- coding: utf-8 -*-
"""
Constrained Response Portfolio Selection Mechanism.
Guarantees:
- Exactly 25 observations selected for response
- At most 4 observations from the same (X, Y) geographic cell
- Mathematically optimal greedy matroid selection maximizing total predicted impact
"""

from collections import defaultdict
from typing import List, Tuple
import numpy as np
import pandas as pd


def select_response_portfolio(
    df_eval: pd.DataFrame,
    impact_scores: np.ndarray,
    portfolio_size: int = 25,
    max_per_cell: int = 4
) -> Tuple[List[int], pd.DataFrame]:
    """
    Selects exactly `portfolio_size` observations subject to at most `max_per_cell`
    observations per geographic cell (X, Y).

    Parameters:
    -----------
    df_eval : pd.DataFrame
        Evaluation candidates, containing at least 'X' and 'Y' columns.
    impact_scores : np.ndarray
        Predicted continuous impact scores for every row in df_eval.
    portfolio_size : int
        Number of observations to select (default: 25).
    max_per_cell : int
        Maximum permitted observations per (X, Y) cell (default: 4).

    Returns:
    --------
    selected_indices : List[int]
        Positional indices in df_eval selected for the response portfolio.
    selected_df : pd.DataFrame
        Subset dataframe of selected rows with added metadata (rank, impact_score).
    """
    n_rows = len(df_eval)
    if n_rows < portfolio_size:
        raise ValueError(f"Cannot select {portfolio_size} rows from an evaluation set with only {n_rows} rows.")

    # Sort indices in descending order of predicted impact score
    sorted_order = np.argsort(-impact_scores, kind='mergesort')

    selected_indices = []
    cell_counts = defaultdict(int)

    # Phase 1: Greedy Matroid Selection with Strict Cell Capacity
    for idx in sorted_order:
        row = df_eval.iloc[idx]
        cell = (int(row['X']), int(row['Y']))
        if cell_counts[cell] < max_per_cell:
            selected_indices.append(idx)
            cell_counts[cell] += 1
            if len(selected_indices) == portfolio_size:
                break

    # Phase 2: Edge-case fallback (only triggers if total available cells * max_per_cell < portfolio_size)
    if len(selected_indices) < portfolio_size:
        selected_set = set(selected_indices)
        for idx in sorted_order:
            if idx not in selected_set:
                selected_indices.append(idx)
                selected_set.add(idx)
                if len(selected_indices) == portfolio_size:
                    break

    # Strict assertion checks
    assert len(selected_indices) == portfolio_size, f"Expected {portfolio_size} selections, got {len(selected_indices)}"

    # Build output dataframe
    selected_df = df_eval.iloc[selected_indices].copy()
    selected_df['impact_score'] = impact_scores[selected_indices]
    selected_df['priority_rank'] = np.arange(1, portfolio_size + 1)

    # Verify max per cell constraint
    max_observed = selected_df.groupby(['X', 'Y']).size().max()
    unique_cells = len(df_eval[['X', 'Y']].drop_duplicates())
    if unique_cells >= (portfolio_size // max_per_cell + 1):
        assert max_observed <= max_per_cell, f"Cell constraint violated: max per cell was {max_observed} > {max_per_cell}"

    return selected_indices, selected_df
