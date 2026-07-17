from kedro.pipeline import Pipeline, node

from .nodes import (
    collect_trained_models,
    evaluate_models,
    log_and_register_best_model,
    split_data,
    train_baseline_model,
    train_catboost_model,
    train_lightgbm_model,
    train_random_forest,
    train_xgboost_model,
)


def create_training_pipeline(**kwargs) -> Pipeline:
    """Creates the ``training`` pipeline.

    """
    return Pipeline(
        [
            node(
                func    = split_data,
                inputs  = ["feature_table", "params:training"],
                outputs = ["X_train", "X_test", "y_train", "y_test"],
                name    = "split_data_node",
            ),
            node(
                func    = train_baseline_model,
                inputs  = ["X_train", "y_train", "params:training"],
                outputs = "baseline_model",
                name    = "train_baseline_model_node",
            ),
            node(
                func    = train_random_forest,
                inputs  = ["X_train", "y_train", "params:training"],
                outputs = "random_forest_model",
                name    = "train_random_forest_node",
            ),
            node(
                func    = train_lightgbm_model,
                inputs  = ["X_train", "y_train", "params:training"],
                outputs = "lightgbm_model",
                name    = "train_lightgbm_model_node",
            ),
            node(
                func    = train_xgboost_model,
                inputs  = ["X_train", "y_train", "params:training"],
                outputs = "xgboost_model",
                name    = "train_xgboost_model_node",
            ),
            node(
                func    = train_catboost_model,
                inputs  = ["X_train", "y_train", "params:training"],
                outputs = "catboost_model",
                name    = "train_catboost_model_node",
            ),
            node(
                func    = collect_trained_models,
                inputs  = [
                    "baseline_model",
                    "random_forest_model",
                    "lightgbm_model",
                    "xgboost_model",
                    "catboost_model",
                ],
                outputs = "trained_models",
                name    = "collect_trained_models_node",
            ),
            node(
                func    = evaluate_models,
                inputs  = ["trained_models", "X_test", "y_test", "params:training"],
                outputs = "model_comparison_table",
                name    = "evaluate_models_node",
            ),
            node(
                func    =log_and_register_best_model,
                inputs  = [
                    "trained_models",
                    "model_comparison_table",
                    "X_train",
                    "params:training",
                ],
                outputs = "training_summary",
                name    = "log_and_register_best_model_node",
            ),
        ]
    )
