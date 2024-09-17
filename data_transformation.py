import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from fancyimpute import IterativeImputer as MICE  # For MICE imputation
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Custom transformer for dropping unnecessary features


class DropFeature(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.features_to_drop, errors='ignore')

# Custom transformer for encoding categorical features


class EncodeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'Sex' in X.columns:
            X['Sex'] = X['Sex'].map({'m': 1, 'f': 2})
        return X

# Custom transformer for calculating and dropping the feature with the highest VIF


class DropHighestVIF(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_to_drop = None

    def fit(self, X, y=None):
        X_data = X.copy()
        vif_data = pd.DataFrame()
        vif_data['features'] = X_data.columns
        vif_data['vif_Factor'] = [variance_inflation_factor(
            X_data.values, i) for i in range(X_data.shape[1])]
        self.feature_to_drop = vif_data.loc[vif_data['vif_Factor'].idxmax(
        ), 'features']
        print(
            f"Dropping '{self.feature_to_drop}' with VIF: {vif_data['vif_Factor'].max()}")
        return self

    def transform(self, X):
        return X.drop(columns=[self.feature_to_drop], errors='ignore')

# Custom transformer for MICE imputation


class MICEImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        imputer = MICE()
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        return X_imputed

# Custom transformer to preserve DataFrame after scaling


class DataFrameStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns = None

    def fit(self, X, y=None):
        self.columns = X.columns
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.columns)

# Create the pipeline


def create_pipeline():
    pipeline = Pipeline(steps=[
        # Drop unnecessary features
        ('drop_feature', DropFeature(['Unnamed: 0'])),
        # Encode categorical features
        ('encode_features', EncodeFeatures()),
        # Impute missing values with MICE
        ('impute_missing', MICEImputer()),
        # Drop the feature with highest VIF
        ('drop_highest_vif', DropHighestVIF()),
        ('scale_data', DataFrameStandardScaler())        # Standardize the dataset
    ])
    return pipeline
