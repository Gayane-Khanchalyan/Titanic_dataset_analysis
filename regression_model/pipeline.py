from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import (AddMissingIndicator, CategoricalImputer,
                                       MeanMedianImputer)
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler

from regression_model.config.core import config
from regression_model.processing.features import ComputeElapsedTime, Mapper

price_pipe = Pipeline([
    ('missing_imputation',
     CategoricalImputer(
         imputation_method='missing',
         variables=config.model_config.categorical_vars_with_na_missing)),
    ('frequent_imputation',
     CategoricalImputer(
         imputation_method='frequent',
         variables=config.model_config.categorical_vars_with_na_frequent)),

    ('missing_indicator',
     AddMissingIndicator(
         variables=config.model_config.numerical_vars_with_na)),
    ('mean_imputor',
     MeanMedianImputer(
         imputation_method='mean',
         variables=config.model_config.numerical_vars_with_na)),
    ('elapsed_time',
     ComputeElapsedTime(
         variables=config.model_config.temporal_vars,
         referance_variable=config.model_config.ref_var)),

    ('drop_features',
     DropFeatures(
         features_to_drop=config.model_config.ref_var)),
    ('log',
     LogTransformer(
         variables=config.model_config.numericals_log_vars)),
    ('binarizer',
     SklearnTransformerWrapper(
         transformer=Binarizer(threshold=0),
         variables=config.model_config.binarize_vars)),


    ('mapper_qual',
     Mapper(variables=config.model_config.qual_vars,
            mappings=config.model_config.qual_mappings)),

    ('mapper_exposure',
     Mapper(variables=config.model_config.exposure_vars,
            mappings=config.model_config.exposure_mappings)),

    ('mapper_finish',
     Mapper(variables=config.model_config.finish_vars,
            mappings=config.model_config.finish_mappings)),
    ('mapper_garage',
     Mapper(variables=config.model_config.garage_vars,
            mappings=config.model_config.garage_mappings)),

    ('rara_label_encoder',
     RareLabelEncoder(
         tol=0.01,
         n_categories=1, variables=config.model_config.categorical_vars)),

    ('categorical_encoder',
     OrdinalEncoder(
        encoding_method='ordered',
        variables=config.model_config.categorical_vars)),

    ('scaler', MinMaxScaler()),
    ('Lasso', Lasso(alpha=0.001, random_state=0)),
])

# price_pipe.fit(X_train, y_train)
