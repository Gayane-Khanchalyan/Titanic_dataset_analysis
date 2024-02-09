# Creating in house
import typing as t

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ComputeElapsedTime(BaseEstimator, TransformerMixin):

    def __init__(self, variables: t.List[str], referance_variable: str):
        if not isinstance(variables, list):
            raise ValueError('Variables should be a list.')
        self.variables = variables
        self.referance_variable = referance_variable

    def fit(self, X, y=None):

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for var in self.variables:
            X[var] = X[self.referance_variable] - X[var]
        return X


class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables: t.List[str], mappings: dict):
        if not isinstance(variables, list):
            raise ValueError('Variables should be a list.')
        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].map(self.mappings)
        return X
