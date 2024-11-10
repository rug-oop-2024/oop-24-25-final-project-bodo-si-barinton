import pickle
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric, get_metric
from autoop.core.ml.model import Model
from autoop.core.ml.model.classification import (
    SVM,
    BayesClassification,
    LogisticClassification,
)
from autoop.core.ml.model.regression import (
    DecisionTreeRegressor,
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.lasso import LassoRegression
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types

automl_system: AutoMLSystem = AutoMLSystem.get_instance()

MODEL_CLASSES: Dict[str, Dict[str, type[Model]]] = {
    "Regression": {
        "MultipleLinearRegression": MultipleLinearRegression,
        "Lasso": LassoRegression,
        "DecisionTreeRegressor": DecisionTreeRegressor,
    },
    "Classification": {
        "SVM": SVM,
        "BayesClassification": BayesClassification,
        "LogisticRegression": LogisticClassification,
    },
}


def load_model(model_type: str, model_name: str) -> Optional[Model]:
    """
    Load a model instance based on type and name.

    Args:
        model_type: Type of model (Regression or Classification)
        model_name: Name of the specific model class

    Returns:
        Model instance if found, None otherwise
    """
    model_class = MODEL_CLASSES[model_type].get(model_name)
    if model_class:
        return model_class()
    return None


def convert_metric_names_to_objects(metric_names: List[str]) -> List[Metric]:
    """
    Convert metric names to their corresponding metric objects.

    Args:
        metric_names: List of metric names to convert

    Returns:
        List of instantiated Metric objects
    """
    return [get_metric(name) for name in metric_names]


action: str = st.sidebar.selectbox(
    "Choose an action", ["View Saved Pipelines",
                         "Load Pipeline Summary and Predict"]
)


def view_saved_pipelines() -> None:
    """
    Display all saved pipelines and their details in the Streamlit interface.
    """
    st.title("📋 Existing Saved Pipelines")
    saved_pipelines: List[Artifact] = automl_system.registry.list(
        type="pipeline"
    )

    if not saved_pipelines:
        st.write("No saved pipelines found.")
        return

    st.write("## List of Saved Pipelines:")
    for pipeline in saved_pipelines:
        st.write(f"- Name: {pipeline.name}")
        st.write(f"  - Version: {pipeline.version}")
        st.write(f"  - Asset Path: {pipeline.asset_path}")
        st.write(f"  - Type: {pipeline.type}")
        st.write(
            f"  - Tags: "
            f"{', '.join(pipeline.tags) if pipeline.tags else 'None'}"
        )
        st.write(
            f"  - Metadata: "
            f"{pipeline.metadata if pipeline.metadata else 'None'}"
        )
        st.write("")


def load_and_show_pipeline_summary_and_predict() -> None:
    """
    Load a selected pipeline, display its summary.
    """
    st.title(
        "🔄 Load Pipeline Summary and Predict"
    )

    saved_pipelines: List[Artifact] = (
        automl_system.registry.list(type="pipeline")
    )
    
    if not saved_pipelines:
        st.write("No saved pipelines available.")
        return

    pipeline_options: Dict[str, Artifact] = {
        f"{p.name} (v{p.version})": p for p in saved_pipelines
    }
    selected_pipeline_name: str = st.selectbox(
        "Select a pipeline to view and use for predictions",
        list(pipeline_options.keys()),
    )

    if selected_pipeline_name:
        selected_pipeline: Artifact = pipeline_options[
            selected_pipeline_name
        ]

        try:
            pipeline_data: Dict[str, Any] = pickle.loads(
                selected_pipeline.data
            )

            st.write("## Pipeline Summary")
            st.write(f"Name: {selected_pipeline.name}")
            st.write(f"Version: {selected_pipeline.version}")
            st.write(f"Type: {selected_pipeline.type}")
            st.write(f"Asset Path: {selected_pipeline.asset_path}")
            st.write(
                f"Tags: {', '.join(selected_pipeline.tags) if selected_pipeline.tags else 'None'}"
            )
            st.write(
                f"Metadata: {selected_pipeline.metadata if selected_pipeline.metadata else 'None'}"
            )

            st.write("### Configuration Data:")
            st.write(pipeline_data)

            model_name: str = pipeline_data["model_type"]
            task_type: str = pipeline_data["task_type"]
            split_ratio: float = pipeline_data["split"]
            metric_names: List[str] = pipeline_data["metrics"]
            metrics: List[Metric] = convert_metric_names_to_objects(metric_names)

            model: Optional[Model] = load_model(task_type, model_name)

            if model:
                uploaded_file = st.file_uploader(
                    "Upload a CSV file for prediction", type="csv"
                )
                dataset_name: str = st.text_input("Dataset Name")
                asset_path: str = st.text_input("Asset Path (relative to storage)")

                if uploaded_file is not None:
                    input_data: pd.DataFrame = pd.read_csv(uploaded_file)

                    dataset1: Dataset = Dataset.from_dataframe(
                        input_data,
                        name=dataset_name,
                        asset_path=asset_path,
                        version="2.0.0",
                    )
                    features: List[Feature] = detect_feature_types(dataset1)
                    feature_names: List[str] = [feature.name for feature in features]
                    feature_types: Dict[str, str] = {  # noqa: F841
                        feature.name: feature.type for feature in features
                    }

                    input_features: List[str] = st.multiselect(
                        "Select input features", feature_names
                    )
                    target_feature: str = st.selectbox(
                        "Select target feature", feature_names
                    )

                    selected_features: List[Feature] = [
                        Feature(name=feature, type="numerical")
                        for feature in input_features
                    ]
                    target: Feature = Feature(
                        name=target_feature,
                        type="numerical"
                        if task_type == "Regression"
                        else "categorical",
                    )
                    pipeline: Pipeline = Pipeline(
                        dataset=dataset1,
                        model=model,
                        input_features=selected_features,
                        target_feature=target,
                        metrics=metrics,
                        split=split_ratio,
                    )

                    st.write("### Uploaded Data")
                    st.write(input_data)

                    if st.button("Run Pipeline"):
                        results: Dict[str, Any] = pipeline.execute()
                        st.write("Results:", results)

            else:
                st.error(
                    f"Model '{model_name}' for task '{task_type}' could not be loaded."
                )

        except Exception as e:
            st.error(f"Failed to load pipeline data: {e}")


if action == "View Saved Pipelines":
    view_saved_pipelines()
elif action == "Load Pipeline Summary and Predict":
    load_and_show_pipeline_summary_and_predict()
