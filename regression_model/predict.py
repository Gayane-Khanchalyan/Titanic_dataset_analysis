import typing as t

import numpy as np
import pandas as pd

from regression_model import __version__ as _version
from regression_model.config.core import config
from regression_model.processing.data_manager import load_pipeline
from regression_model.processing.validation import validate_inputs
from regression_model.processing.data_manager import load_dataset


pipeline_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using sved pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {'predictions': None, 'version': None, 'errors': errors}
    if not errors:
        predictions = _price_pipe.predict(
            X=validated_data[config.model_config.features]
        )
        results = {
            'predictions': [np.exp(pred) for pred in predictions],
            'version': _version,
            'errors': errors
        }
    return results


if __name__ == '__main__':
    make_prediction(
        input_data=load_dataset(
            file_name=config.app_config.test_data_file))
