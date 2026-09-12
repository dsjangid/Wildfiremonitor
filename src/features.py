# -*- coding: utf-8 -*-
"""
Feature engineering and preprocessing pipeline for Wildfire Response Prioritization.
Enforces strict anti-leakage by fitting scalers and transformers ONLY on training partitions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PERMITTED_FEATURES = [
    'X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain'
]

MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

DAY_MAP = {
    'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7
}


def extract_domain_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Derives physically-grounded meteorological and fire behavior features
    from permitted input columns.
    """
    missing = [col for col in PERMITTED_FEATURES if col not in df_input.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing permitted features: {missing}")

    df = df_input[PERMITTED_FEATURES].copy()

    # 1. Cyclical & Seasonal Temporal Encodings
    m_num = df['month'].astype(str).str.lower().map(MONTH_MAP).fillna(1).astype(float)
    df['month_sin'] = np.sin(2.0 * np.pi * m_num / 12.0)
    df['month_cos'] = np.cos(2.0 * np.pi * m_num / 12.0)
    df['is_summer'] = df['month'].astype(str).str.lower().isin(['jul', 'aug', 'sep']).astype(float)
    df['is_aug_sep'] = df['month'].astype(str).str.lower().isin(['aug', 'sep']).astype(float)

    d_num = df['day'].astype(str).str.lower().map(DAY_MAP).fillna(1).astype(float)
    df['day_sin'] = np.sin(2.0 * np.pi * d_num / 7.0)
    df['day_cos'] = np.cos(2.0 * np.pi * d_num / 7.0)
    df['is_weekend'] = df['day'].astype(str).str.lower().isin(['sat', 'sun']).astype(float)

    # 2. Atmospheric & Fuel Moisture Physics
    # Saturation vapor pressure (Tetens formula in kPa)
    es = 0.61078 * np.exp((17.27 * df['temp']) / (df['temp'] + 237.3))
    # Actual vapor pressure
    ea = es * (df['RH'] / 100.0)
    # Vapor Pressure Deficit (VPD): critical atmospheric dryness metric
    df['VPD'] = es - ea

    # 3. Canadian FWI System Derived Indices
    # Buildup Index (BUI): represents total fuel available for combustion
    denom = df['DMC'] + 0.4 * df['DC']
    df['BUI'] = np.where(denom > 0.0, (0.8 * df['DMC'] * df['DC']) / denom, 0.0)

    # Drought Severity Factor: deep drought interaction with atmospheric dryness
    df['drought_factor'] = (df['DC'] * (100.0 - df['RH'])) / 1000.0

    # Fire Spread Potential: wind speed multiplying Initial Spread Index (ISI)
    df['fire_spread_potential'] = (df['ISI'] * df['wind']) / 10.0

    # Temperature / Humidity ratio
    df['temp_rh_ratio'] = df['temp'] / (df['RH'] + 1.0)

    # Fine Fuel Effective Dryness
    df['ffmc_dryness'] = np.maximum(0.0, df['FFMC'] - 70.0)

    # 4. Spatial Coordinates Geometry in Montesinho Natural Park
    df['dist_center'] = np.sqrt((df['X'] - 4.6)**2 + (df['Y'] - 4.3)**2)
    df['spatial_diag'] = df['X'] + df['Y']
    df['spatial_diff'] = df['X'] - df['Y']

    # Remove raw string categoricals
    df = df.drop(columns=['month', 'day'])
    return df


class FeaturePipeline:
    """
    Production feature pipeline that fits scalers strictly on training data
    and transforms evaluation data with guaranteed zero target leakage.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, df_train: pd.DataFrame):
        """Fit feature transformers on training rows only."""
        X_feat = extract_domain_features(df_train)
        self.feature_names_ = list(X_feat.columns)
        self.scaler.fit(X_feat.values)
        self.is_fitted_ = True
        return self

    def transform(self, df_eval: pd.DataFrame) -> np.ndarray:
        """Transform evaluation rows using parameters fitted only on training data."""
        if not self.is_fitted_:
            raise RuntimeError("FeaturePipeline must be fitted on training data before calling transform.")
        X_feat = extract_domain_features(df_eval)
        X_feat = X_feat[self.feature_names_]
        return self.scaler.transform(X_feat.values)

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        """Fit on train and return transformed array."""
        return self.fit(df_train).transform(df_train)
