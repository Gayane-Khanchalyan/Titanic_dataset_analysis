from regression_model.config.core import config
from regression_model.processing.features import ComputeElapsedTime


def test_tepmoral_variable_transformer(sample_input_data):

    transformer = ComputeElapsedTime(
        variables=config.model_config.temporal_vars,
        referance_variable=config.model_config.ref_var)

    assert sample_input_data['YearRemodAdd'].iat[0] == 1961
    # when
    subject = transformer.fit_transform(sample_input_data)

    # then
    assert subject['YearRemodAdd'].iat[0] == 49
