import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem  # Access AutoMLSystem for managing artifacts
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
import io
from typing import List
from autoop.core.ml.feature import Feature

# Initialize AutoMLSystem singleton instance
automl_system = AutoMLSystem.get_instance()

# Title for the Streamlit page
st.title("Dataset Manager")

# Sidebar for dataset management actions
action = st.sidebar.selectbox(
    "Choose an action", ["Upload Dataset", "View Dataset", "Delete Dataset", "List Datasets", "Feature Selection"]
)

# Function to handle dataset upload and save
def upload_dataset():
    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    dataset_name = st.text_input("Dataset Name")
    asset_path = st.text_input("Asset Path (relative to storage)")

    if uploaded_file is not None and dataset_name and asset_path:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())

        encoded_data = df.to_csv(index=False).encode()

        dataset_artifact = Dataset(
            name=dataset_name,
            asset_path=asset_path,
            version="1.0.0",
            data=encoded_data
        )

        automl_system.registry.register(dataset_artifact)
        
        st.success(f"Dataset '{dataset_name}' uploaded and saved successfully as an artifact.")

# Function to view details of a specific dataset
def view_dataset():
    dataset_list = [artifact.name for artifact in automl_system.registry.list(type="dataset")]

    if not dataset_list:
        st.write("No datasets available.")
        return

    selected_dataset = st.selectbox("Select a dataset to view", dataset_list)

    if selected_dataset:
        dataset_artifact = next(
            (artifact for artifact in automl_system.registry.list(type="dataset") if artifact.name == selected_dataset),
            None
        )

        if dataset_artifact:
            st.write("Dataset Details:")
            st.write(f"Name: {dataset_artifact.name}")
            st.write(f"Version: {dataset_artifact.version}")
            st.write("Dataset Content:")
            
            csv_data = dataset_artifact.data.decode()
            df = pd.read_csv(io.StringIO(csv_data))
            st.write(df)

# Function to delete a dataset
def delete_dataset():
    dataset_list = [artifact.name for artifact in automl_system.registry.list(type="dataset")]

    if not dataset_list:
        st.write("No datasets available.")
        return

    selected_dataset = st.selectbox("Select a dataset to delete", dataset_list)

    if st.button("Delete Dataset"):
        dataset_artifact = next(
            (artifact for artifact in automl_system.registry.list(type="dataset") if artifact.name == selected_dataset),
            None
        )
        
        if dataset_artifact:
            automl_system.registry.delete(dataset_artifact.id)
            st.success(f"Dataset '{selected_dataset}' deleted successfully.")

# Function to list all datasets with details
def list_datasets():
    dataset_list = automl_system.registry.list(type="dataset")

    if not dataset_list:
        st.write("No datasets available.")
        return

    st.write("Available Datasets:")
    for dataset_artifact in dataset_list:
        st.write(f"- Name: {dataset_artifact.name}")
        st.write(f"  Version: {dataset_artifact.version}")
        st.write(f"  Asset Path: {dataset_artifact.asset_path}")
        st.write("")

def detect_feature_types(df: pd.DataFrame) -> List[Feature]:
    """Detect feature types (categorical or numerical) in a DataFrame.

    Args:
        df: DataFrame containing the dataset.

    Returns:
        List[Feature]: List of Feature objects with detected types.
    """
    features: List[Feature] = []

    if df.empty:
        raise ValueError("The provided dataset is empty.")

    for column in df.columns:
        feature_type = "numerical" if pd.api.types.is_numeric_dtype(df[column]) else "categorical"
        feature = Feature(name=column, type=feature_type)
        feature.set_data(df[column].values)
        features.append(feature)
    
    return features

# Function to select features and detect task type
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
            # Decode the data from bytes and load it into a DataFrame
            csv_data = dataset_artifact.data.decode()
            df = pd.read_csv(io.StringIO(csv_data))
            
            st.write("Dataset Loaded:")
            st.write(df.head())

            # Step 2: Detect Feature Types using the DataFrame
            features = detect_feature_types(df)  # Pass DataFrame directly
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

# Routing actions based on the selected option
if action == "Upload Dataset":
    upload_dataset()
elif action == "View Dataset":
    view_dataset()
elif action == "Delete Dataset":
    delete_dataset()
elif action == "List Datasets":
    list_datasets()
elif action == "Feature Selection":
    feature_selection()
