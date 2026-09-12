# -*- coding: utf-8 -*-
"""
Evaluation Metrics Suite aligned with the 100-Point Judging Criteria:
- 35 points: Mean NDCG@25
- 25 points: High-impact recall within the selected top-25 response set
- 20 points: Worst-split NDCG@25
- 10 points: Mean Spearman rank correlation
- 10 points: Runtime and reproducibility
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr


def compute_ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 25) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank k (NDCG@k).
    """
    return float(ndcg_score(np.asarray([y_true]), np.asarray([y_score]), k=k))


def compute_spearman_correlation(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes Spearman rank-order correlation coefficient between true and predicted values.
    """
    val, _ = spearmanr(y_true, y_score)
    return float(val) if not np.isnan(val) else 0.0


def compute_high_impact_recall(
    training_area: np.ndarray,
    evaluation_area: np.ndarray,
    selected_indices: List[int]
) -> float:
    """
    Computes high-impact recall within the selected top-25 response set.
    High-impact observations are defined as evaluation rows whose true area
    is at or above the 80th percentile of the corresponding training-set area distribution:
    numpy.percentile(training_area, 80, method="linear").
    """
    p80 = float(np.percentile(training_area, 80, method='linear'))
    eval_hi_mask = (np.asarray(evaluation_area) >= p80)
    total_high_impact = int(eval_hi_mask.sum())

    if total_high_impact == 0:
        return 1.0

    selected_hi_count = int(eval_hi_mask[selected_indices].sum())
    return float(selected_hi_count / total_high_impact)


def calculate_judging_rubric(
    mean_ndcg: float,
    worst_ndcg: float,
    mean_spearman: float,
    mean_recall: float,
    runtime_reproducibility: float = 1.0
) -> Dict[str, Any]:
    """
    Calculates points according to the 100-Point Judging Criteria:
    - 35 pts: Mean NDCG@25
    - 25 pts: High-impact recall within selected top-25 response set
    - 20 pts: Worst-split NDCG@25
    - 10 pts: Mean Spearman rank correlation
    - 10 pts: Runtime and reproducibility
    """
    pts_ndcg = 35.0 * max(0.0, min(1.0, mean_ndcg))
    pts_recall = 25.0 * max(0.0, min(1.0, mean_recall))
    pts_worst_ndcg = 20.0 * max(0.0, min(1.0, worst_ndcg))
    pts_spearman = 10.0 * max(0.0, min(1.0, mean_spearman))
    pts_reproducibility = 10.0 * max(0.0, min(1.0, runtime_reproducibility))

    total = pts_ndcg + pts_recall + pts_worst_ndcg + pts_spearman + pts_reproducibility

    return {
        'pts_ndcg_35': pts_ndcg,
        'pts_recall_25': pts_recall,
        'pts_worst_ndcg_20': pts_worst_ndcg,
        'pts_spearman_10': pts_spearman,
        'pts_reproducibility_10': pts_reproducibility,
        'total_score_100': total
    }
