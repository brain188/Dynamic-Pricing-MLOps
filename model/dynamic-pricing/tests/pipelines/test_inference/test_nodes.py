"""Unit tests for ``dynamic_pricing.pipelines.inference.nodes``.

``load_production_model`` is intentionally not unit tested here with a
mock — a mocked MlflowClient would not verify the real alias-resolution
API is being called correctly. It's covered by the real-registry
integration test in ``test_integration.py`` instead, alongside the rest
of the pipeline.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from dynamic_pricing.pipelines.inference.nodes import (
    generate_predictions,
    validate_predictions,
)


@pytest.fixture
def base_parameters():
    return {
        "prediction_column_name": "predicted_fare",
        "sanity_bounds": {"min": 0, "max": 1000, "action": "clip"},
    }


def _make_mock_model(required_columns, predict_return):
    """Builds a minimal stand-in for an MLflow pyfunc model, exposing just
    the ``.metadata.get_input_schema()`` and ``.predict()`` surface that
    ``generate_predictions`` actually uses."""
    mock_schema = MagicMock()
    mock_schema.input_names.return_value = required_columns

    mock_model = MagicMock()
    mock_model.metadata.get_input_schema.return_value = mock_schema
    mock_model.predict.return_value = predict_return

    return mock_model


class TestGeneratePredictions:
    def test_selects_only_columns_the_model_signature_requires(self, base_parameters):
        encoded_data = pd.DataFrame(
            {"duration": [10, 20], "extra_feature": [1, 2], "another_col": [9, 9]}
        )
        model = _make_mock_model(required_columns=["duration"], predict_return=[100.0, 200.0])

        result = generate_predictions(encoded_data, model, base_parameters)

        called_with = model.predict.call_args[0][0]
        assert list(called_with.columns) == ["duration"]
        assert result["predicted_fare"].tolist() == [100.0, 200.0]

    def test_uses_all_columns_when_model_has_no_signature(self, base_parameters):
        encoded_data = pd.DataFrame({"duration": [10, 20], "extra_feature": [1, 2]})
        model = MagicMock()
        model.metadata.get_input_schema.return_value = None
        model.predict.return_value = [50.0, 60.0]

        result = generate_predictions(encoded_data, model, base_parameters)

        called_with = model.predict.call_args[0][0]
        assert list(called_with.columns) == ["duration", "extra_feature"]
        assert result["predicted_fare"].tolist() == [50.0, 60.0]

    def test_raises_when_required_column_is_missing(self, base_parameters):
        encoded_data = pd.DataFrame({"duration": [10, 20]})
        model = _make_mock_model(
            required_columns=["duration", "vehicle_type_premium"], predict_return=[1.0, 2.0]
        )

        with pytest.raises(ValueError, match="requires column"):
            generate_predictions(encoded_data, model, base_parameters)

    def test_preserves_original_columns_alongside_prediction(self, base_parameters):
        encoded_data = pd.DataFrame({"duration": [10, 20], "extra_feature": [1, 2]})
        model = _make_mock_model(required_columns=["duration"], predict_return=[100.0, 200.0])

        result = generate_predictions(encoded_data, model, base_parameters)

        assert "extra_feature" in result.columns
        assert "duration" in result.columns


class TestValidatePredictions:
    def test_no_violations_adds_false_flag_column(self, base_parameters):
        predictions = pd.DataFrame({"predicted_fare": [100.0, 200.0, 300.0]})
        result = validate_predictions(predictions, base_parameters)
        assert (result["is_out_of_bounds"] == False).all()  # noqa: E712
        assert result["predicted_fare"].tolist() == [100.0, 200.0, 300.0]

    def test_clip_action_clips_out_of_bounds_values(self, base_parameters):
        predictions = pd.DataFrame({"predicted_fare": [-50.0, 500.0, 2000.0]})
        result = validate_predictions(predictions, base_parameters)
        assert result["predicted_fare"].tolist() == [0.0, 500.0, 1000.0]
        assert (result["is_out_of_bounds"] == False).all()  # noqa: E712

    def test_flag_action_leaves_values_unchanged_but_flags_them(self, base_parameters):
        params = {**base_parameters, "sanity_bounds": {"min": 0, "max": 1000, "action": "flag"}}
        predictions = pd.DataFrame({"predicted_fare": [-50.0, 500.0, 2000.0]})
        result = validate_predictions(predictions, params)
        assert result["predicted_fare"].tolist() == [-50.0, 500.0, 2000.0]
        assert result["is_out_of_bounds"].tolist() == [True, False, True]

    def test_raises_on_unsupported_action(self, base_parameters):
        params = {**base_parameters, "sanity_bounds": {"min": 0, "max": 1000, "action": "bogus"}}
        predictions = pd.DataFrame({"predicted_fare": [-50.0]})
        with pytest.raises(ValueError, match="Unsupported sanity_bounds.action"):
            validate_predictions(predictions, params)
