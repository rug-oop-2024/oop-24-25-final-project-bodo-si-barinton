import streamlit as st
import pandas as pd
import io
import os
from app.core.system import AutoMLSystem  
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from typing import List

# Initialize AutoMLSystem singleton instance
automl_system = AutoMLSystem.get_instance()

# Set page configuration
st.set_page_config(page_title="Modeling", page_icon="📈")

# Title and description
st.title("⚙ Modeling")
st.write("In this section, you can design a machine learning pipeline to train a model on a dataset.")

# Sidebar for choosing actions
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

def list_models(task_type: str) -> List[str]:
    """List models based on the task type from 'autoop/core/ml/model' subdirectories."""
    base_model_folder = "autoop/core/ml/model"
    subfolder = "regression" if task_type.lower() == "regression" else "classification"
    model_folder = os.path.join(base_model_folder, subfolder)

    available_models = []

    if not os.path.isdir(model_folder):
        st.error(f"The model folder '{model_folder}' does not exist.")
        return available_models

    for model_file in os.listdir(model_folder):
        model_name, ext = os.path.splitext(model_file)
        if ext in [".py", ".pkl"] and model_name != "__init__":
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
            # Decode data and load into DataFrame
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

                    # Step 6: Select Dataset Split
                    split_ratio = st.slider("Select train/test split ratio", min_value=0.1, max_value=0.9, value=0.8, step=0.05)

                    # Placeholder for model and metrics
                    model = None  # Replace with actual model initialization
                    metrics = []  # Add code to initialize metrics as needed

                    # Step 7: Initialize and Run Pipeline
                    if st.button("Run Pipeline"):
                        # Initialize the Pipeline with selected configurations
                        pipeline = Pipeline(
                            dataset=dataset_artifact,
                            model=model,
                            input_features=[Feature(name=feature, type=feature_types[feature]) for feature in input_features],
                            target_feature=Feature(name=target_feature, type=feature_types[target_feature]),
                            metrics=metrics,
                            split=split_ratio
                            )
                            # Execute the pipeline and display results
                        results = pipeline.execute()
                        st.write("Pipeline executed successfully.")
                        st.write("Results:", results)
# Routing actions
if action == "List Datasets":
    list_datasets()
elif action == "Feature Selection":
    feature_selection()
