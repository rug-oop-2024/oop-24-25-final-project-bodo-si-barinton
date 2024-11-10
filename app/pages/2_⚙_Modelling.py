import io
import pickle
from typing import Dict, List, Optional

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
    LassoRegression,
    MultipleLinearRegression,
)
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types

automl_system: AutoMLSystem = AutoMLSystem.get_instance()

if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = None

MODEL_CLASSES: Dict[str, Dict[str, Model]] = {
    "Regression": {
        "MultipleLinearRegression": MultipleLinearRegression(),
        "Lasso": LassoRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
    },
    "Classification": {
        "SVM": SVM(),
        "BayesClassification": BayesClassification(),
        "LogisticRegression": LogisticClassification(),
    },
}


def get_compatible_metrics(task_type: str) -> List[str]:
    """
    Get compatible metrics for a given task type.

    Args:
        task_type (str): The task type (Classification or Regression).

    Returns:
        List[str]: A list of compatible metrics.
    """
    if task_type == "Classification":
        return ["accuracy", "logloss", "micro", "macro"]
    elif task_type == "Regression":
        return [
            "mean_squared_error",
            "mean_absolute_error",
            "root_mean_squared_error"
        ]
    return []


st.set_page_config(page_title="Modeling", page_icon="📈")
st.title("⚙ Modeling")
st.write(
    "In this section, you can design a ml pipeline to train a model "
    "on a dataset."
)

action: str = st.sidebar.selectbox(
    "Choose an action", ["List Datasets", "Feature Selection"]
)


def list_datasets() -> None:
    """
    List all available datasets.
    """
    dataset_list: List[Artifact] = automl_system.registry.list(type="dataset")
    if not dataset_list:
        st.write("No datasets available.")
        return

    st.write("Available Datasets:")
    for dataset_artifact in dataset_list:
        st.write(f"- *Name:* {dataset_artifact.name}")
        st.write(f"  *Version:* {dataset_artifact.version}")
        st.write(f"  *Asset Path:* {dataset_artifact.asset_path}")


def feature_selection() -> None:
    """
    Perform feature selection for a selected dataset.
    """
    dataset_list: List[str] = [
        artifact.name
        for artifact in automl_system.registry.list(type="dataset")
    ]
    if not dataset_list:
        st.write("No datasets available.")
        return

    selected_dataset: str = st.selectbox(
        "Select a dataset for feature selection", dataset_list
    )
    if selected_dataset:
        dataset_artifact: Optional[Dataset] = next(
            (
                automl_system.registry.get(artifact.id)
                for artifact in automl_system.registry.list(type="dataset")
                if artifact.name == selected_dataset
            ),
            None,
        )

        if dataset_artifact:
            csv_data: str = dataset_artifact.data.decode()
            df: pd.DataFrame = pd.read_csv(io.StringIO(csv_data))
            st.write("Dataset Loaded:")
            st.write(df.head())

            dataset_instance: Dataset = Dataset.from_dataframe(
                df,
                name=selected_dataset,
                asset_path=dataset_artifact.asset_path
            )
            features: List[Feature] = detect_feature_types(dataset_instance)
            feature_names: List[str] = [feature.name for feature in features]
            feature_types: Dict[str, str] = {
                feature.name: feature.type for feature in features
            }

            input_features: List[str] = st.multiselect(
                "Select input features", feature_names
            )
            target_feature: str = st.selectbox(
                "Select target feature", feature_names
            )

            if input_features and target_feature:
                task_type: str = (
                    "Regression"
                    if feature_types[target_feature] == "numerical"
                    else "Classification"
                )
                available_models: Dict[str, Model] = MODEL_CLASSES[
                    task_type
                ]
                selected_model_name: str = st.selectbox(
                    "Select a model", list(available_models.keys())
                )
                selected_model_class: Model = available_models[
                    selected_model_name
                ]
                compatible_metrics: List[str] = get_compatible_metrics(
                    task_type
                )
                selected_metrics: List[str] = st.multiselect(
                    "Select metrics", compatible_metrics
                )
                metric_objects: List[Metric] = [
                    get_metric(metric) for metric in selected_metrics
                ]
                split_ratio: float = st.slider(
                    "Select train/test split ratio",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.8,
                    step=0.05,
                )

                if st.button("Run Pipeline"):
                    st.session_state.pipeline = Pipeline(
                        dataset=dataset_instance,
                        model=selected_model_class,
                        input_features=[
                            Feature(
                                name=feature,
                                type=feature_types[feature]
                            )
                            for feature in input_features
                        ],
                        target_feature=Feature(
                            name=target_feature,
                            type=feature_types[target_feature]
                        ),
                        metrics=metric_objects,
                        split=split_ratio,
                    )

                    st.session_state.pipeline._artifacts["config"] = {
                        "type": "pipeline_config",
                        "data": pickle.dumps(
                            {
                                "model_type": selected_model_name,
                                "split": split_ratio,
                                "metrics": selected_metrics,
                                "task_type": task_type,
                            }
                        ),
                    }

                    st.write("Pipeline executed successfully.")
                    st.write("### Pipeline Summary")
                    st.write(f"- *Model*: {selected_model_name}")
                    st.write(f"- *Task Type*: {task_type}")
                    st.write(
                        f"- *Selected Metrics*: "
                        f"{', '.join(selected_metrics)}"
                    )
                    st.write(f"- *Split Ratio*: {split_ratio}")
                    st.write(f"- *Input Features*: {input_features}")
                    st.write(f"- *Target Feature*: {target_feature}")

                    results = st.session_state.pipeline.execute()
                    st.write("Results:", results)

                pipeline_name: str = st.text_input(
                    "Enter a name for the pipeline", key="pipeline_name"
                )
                pipeline_version: str = st.text_input(
                    "Enter a version for the pipeline", key="pipeline_version"
                )

                if st.session_state.pipeline and \
                    st.button("Save Pipeline Artifacts"):
                    for (
                        artifact_key,
                        artifact_data,
                    ) in st.session_state.pipeline._artifacts.items():
                        updated_artifact: Artifact = Artifact(
                            name=f"{pipeline_name}_{artifact_key}",
                            version=pipeline_version,
                            asset_path="path/to/pipeline_config",
                            type="pipeline",
                            data=artifact_data["data"],
                            tags=["pipeline", "config"],
                            metadata={
                                "model_type": selected_model_name,
                                "task_type": task_type,
                            },
                        )

                        try:
                            automl_system.registry.register(updated_artifact)
                            st.success("Pipeline artifact saved successfully!")
                        except Exception as e:
                            st.error(f"Failed to save artifact: {e}")


if action == "List Datasets":
    list_datasets()
elif action == "Feature Selection":
    feature_selection()
