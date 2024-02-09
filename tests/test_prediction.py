import math

from regression_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction = 113422
    expected_no_predictions = 1449

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get('predictions')
    assert isinstance(predictions, list)
    assert result.get('errors') is None
    assert isinstance(predictions[0], float)
    assert len(predictions) == expected_no_predictions
    assert math.isclose(expected_first_prediction, predictions[0], abs_tol=100)
