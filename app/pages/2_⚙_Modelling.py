import streamlit as st
import pandas as pd
import io
from app.core.system import AutoMLSystem  
from autoop.core.ml.feature import Feature
from typing import List
import os

automl_system = AutoMLSystem.get_instance()


st.set_page_config(page_title="Modeling", page_icon="📈")

st.title("⚙ Modeling")
st.write("In this section, you can design a machine learning pipeline to train a model on a dataset.")


action = st.sidebar.selectbox(
    "Choose an action", ["List Datasets", "Feature Selection"]
)


def list_datasets():
    dataset_list = automl_system.registry.list(type="dataset")

    if not dataset_list:
        st.write("No datasets available.")
        return

    st.write("Available Datasets:")
    for dataset_artifact in dataset_list:
        st.write(f"- **Name:** {dataset_artifact.name}")
        st.write(f"  **Version:** {dataset_artifact.version}")
        st.write(f"  **Asset Path:** {dataset_artifact.asset_path}")
        st.write("")


def detect_feature_types(df: pd.DataFrame) -> List[Feature]:
    """Detect feature types (categorical or numerical) in a DataFrame."""
    features: List[Feature] = []
    for column in df.columns:
        feature_type = "numerical" if pd.api.types.is_numeric_dtype(df[column]) else "categorical"
        feature = Feature(name=column, type=feature_type)
        feature.set_data(df[column].values)
        features.append(feature)
    return features

import os
from typing import List

def list_models(task_type: str) -> List[str]:
    """List models based on the task type from 'autoop/core/ml/model' subdirectories."""
    # Base directory for models
    base_model_folder = "autoop/core/ml/model"
    
    # Determine the subfolder based on task type
    subfolder = "regression" if task_type.lower() == "regression" else "classification"
    model_folder = os.path.join(base_model_folder, subfolder)

    available_models = []

    if not os.path.isdir(model_folder):
        st.error(f"The model folder '{model_folder}' does not exist.")
        return available_models

    # List all files in the model folder for the specified task type, excluding `__init__.py` and other non-model files
    for model_file in os.listdir(model_folder):
        model_name, ext = os.path.splitext(model_file)
        if ext in [".py", ".pkl"] and model_name != "__init__":  # Add or modify extensions as needed
            available_models.append(model_name)

    if not available_models:
        st.warning(f"No models found for {task_type}.")
    
    return available_models


def feature_selection():
    dataset_list = [artifact.name for artifact in automl_system.registry.list(type="dataset")]

    if not dataset_list:
        st.write("No datasets available.")
        return

    # Step 1: Select Dataset
    selected_dataset = st.selectbox("Select a dataset for feature selection", dataset_list)

    if selected_dataset:
        dataset_artifact = next(
            (artifact for artifact in automl_system.registry.list(type="dataset") if artifact.name == selected_dataset),
            None
        )

        if dataset_artifact:
            # Decode the data and load into DataFrame
            csv_data = dataset_artifact.data.decode()
            df = pd.read_csv(io.StringIO(csv_data))
            
            st.write("Dataset Loaded:")
            st.write(df.head())

            # Step 2: Detect Feature Types
            features = detect_feature_types(df)
            feature_names = [feature.name for feature in features]
            feature_types = {feature.name: feature.type for feature in features}

            # Step 3: Select Features
            input_features = st.multiselect("Select input features", feature_names)
            target_feature = st.selectbox("Select target feature", feature_names)

            if input_features and target_feature:
                st.write(f"Selected input features: {input_features}")
                st.write(f"Selected target feature: {target_feature} (Type: {feature_types[target_feature]})")

                # Step 4: Detect Task Type
                task_type = "Regression" if feature_types[target_feature] == "numerical" else "Classification"
                st.write(f"Detected task type: {task_type}")

                # Step 5: Prompt User to Select Model Based on Task Type
                available_models = list_models(task_type)
                if available_models:
                    selected_model = st.selectbox("Select a model", available_models)
                    st.write(f"Selected model: {selected_model}")


if action == "List Datasets":
    list_datasets()
elif action == "Feature Selection":
    feature_selection()
