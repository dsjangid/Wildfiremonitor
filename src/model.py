# -*- coding: utf-8 -*-
"""
Multi-Task Hurdle & Ranking Ensemble for Wildfire Response Prioritization.
Combines an L2-regularized Ridge regressor, a robust Huber regressor,
and a calibrated high-impact logistic classifier.
"""

import numpy as np
from sklearn.linear_model import Ridge, HuberRegressor, LogisticRegression


class HurdleRankEnsemble:
    """
    Ensemble model specifically designed for zero-inflated, heavy-tailed fire area distributions:
    1. Ridge Regressor: monotonic global log1p(area) response with strong L2 shrinkage.
    2. Huber Regressor: outlier-robust regression resistant to rare extreme burns.
    3. Calibrated Logistic Classifier: predicts probability of belonging to high-impact tier (>= 80th percentile).
    """

    def __init__(
        self,
        alpha_ridge: float = 20.0,
        alpha_huber: float = 5.0,
        c_logreg: float = 0.05,
        weight_ridge: float = 0.50,
        weight_huber: float = 0.20,
        weight_prob: float = 0.30,
        random_state: int = 42
    ):
        self.alpha_ridge = alpha_ridge
        self.alpha_huber = alpha_huber
        self.c_logreg = c_logreg
        self.weight_ridge = weight_ridge
        self.weight_huber = weight_huber
        self.weight_prob = weight_prob
        self.random_state = random_state

        self.ridge_model = None
        self.huber_model = None
        self.logreg_model = None
        self.p80_threshold_ = None
        self.is_fitted_ = False

    def fit(self, X_train: np.ndarray, training_area: np.ndarray):
        """
        Fit all ensemble models strictly on provided training data.
        Target: y = log(1 + area).
        High-impact threshold: 80th percentile computed via numpy.percentile(training_area, 80, method='linear').
        """
        y_train_log = np.log1p(training_area)

        # 1. High-impact classification threshold from training partition
        self.p80_threshold_ = float(np.percentile(training_area, 80, method='linear'))
        y_train_hi = (training_area >= self.p80_threshold_).astype(int)

        # 2. Fit Ridge Regressor
        self.ridge_model = Ridge(alpha=self.alpha_ridge, random_state=self.random_state)
        self.ridge_model.fit(X_train, y_train_log)

        # 3. Fit Huber Regressor
        self.huber_model = HuberRegressor(alpha=self.alpha_huber, max_iter=1000)
        self.huber_model.fit(X_train, y_train_log)

        # 4. Fit Calibrated Logistic Classifier
        self.logreg_model = LogisticRegression(
            C=self.c_logreg,
            max_iter=1000,
            random_state=self.random_state
        )
        self.logreg_model.fit(X_train, y_train_hi)

        self.is_fitted_ = True
        return self

    @staticmethod
    def _minmax_scale(arr: np.ndarray) -> np.ndarray:
        """Scale predictions into [0, 1] range stably."""
        ptp = np.ptp(arr)
        if ptp > 1e-8:
            return (arr - np.min(arr)) / ptp
        return np.zeros_like(arr)

    def predict(self, X_eval: np.ndarray) -> np.ndarray:
        """
        Predict continuous impact scores for candidate observations.
        Outputs a unified ranking score in [0, 1] representing prioritized risk.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted on training data before calling predict.")

        # Model individual predictions
        pred_ridge = self.ridge_model.predict(X_eval)
        pred_huber = self.huber_model.predict(X_eval)
        pred_prob = self.logreg_model.predict_proba(X_eval)[:, 1]

        # Normalized multi-objective blend
        s_ridge_norm = self._minmax_scale(pred_ridge)
        s_huber_norm = self._minmax_scale(pred_huber)
        s_prob_norm = self._minmax_scale(pred_prob)

        composite_score = (
            self.weight_ridge * s_ridge_norm +
            self.weight_huber * s_huber_norm +
            self.weight_prob * s_prob_norm
        )
        return composite_score
