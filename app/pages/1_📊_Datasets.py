import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem  # Access AutoMLSystem for managing artifacts
from autoop.core.ml.dataset import Dataset
import io

# Initialize AutoMLSystem singleton instance
automl_system = AutoMLSystem.get_instance()

# Title for the Streamlit page
st.title("Dataset Manager")

# Sidebar for dataset management actions
action = st.sidebar.selectbox(
    "Choose an action", ["Upload Dataset", "View Dataset", "Delete Dataset", "List Datasets"]
)

# Function to handle dataset upload and save
def upload_dataset():
    # File uploader for CSV files
    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    dataset_name = st.text_input("Dataset Name")
    asset_path = st.text_input("Asset Path (relative to storage)")

    if uploaded_file is not None and dataset_name and asset_path:
        # Load CSV data into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())

        # Convert DataFrame to CSV bytes to store as an artifact
        encoded_data = df.to_csv(index=False).encode()

        # Create a Dataset artifact and register it using AutoMLSystem
        dataset_artifact = Dataset(
            name=dataset_name,
            asset_path=asset_path,
            version="1.0.0",
            data=encoded_data,
            type="dataset"
        )

        # Register the dataset artifact
        automl_system.registry.register(dataset_artifact)
        
        st.success(f"Dataset '{dataset_name}' uploaded and saved successfully as an artifact.")

# Function to view details of a specific dataset
def view_dataset():
    dataset_list = [artifact.name for artifact in automl_system.registry.list(type="dataset")]

    if not dataset_list:
        st.write("No datasets available.")
        return

    # Dropdown to select a dataset by name
    selected_dataset = st.selectbox("Select a dataset to view", dataset_list)

    if selected_dataset:
        # Retrieve the selected dataset artifact
        dataset_artifact = next(
            (artifact for artifact in automl_system.registry.list(type="dataset") if artifact.name == selected_dataset),
            None
        )

        if dataset_artifact:
            st.write("Dataset Details:")
            st.write(f"Name: {dataset_artifact.name}")
            st.write(f"Version: {dataset_artifact.version}")
            st.write("Dataset Content:")
            
            # Convert CSV bytes back to DataFrame for display
            csv_data = dataset_artifact.data.decode()
            df = pd.read_csv(io.StringIO(csv_data))
            st.write(df)

# Function to delete a dataset
def delete_dataset():
    dataset_list = [artifact.name for artifact in automl_system.registry.list(type="dataset")]

    if not dataset_list:
        st.write("No datasets available.")
        return

    # Dropdown to select dataset for deletion
    selected_dataset = st.selectbox("Select a dataset to delete", dataset_list)

    if st.button("Delete Dataset"):
        dataset_artifact = next(
            (artifact for artifact in automl_system.registry.list(type="dataset") if artifact.name == selected_dataset),
            None
        )
        
        if dataset_artifact:
            # Delete the dataset artifact
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

# Routing actions based on the selected option
if action == "Upload Dataset":
    upload_dataset()
elif action == "View Dataset":
    view_dataset()
elif action == "Delete Dataset":
    delete_dataset()
elif action == "List Datasets":
    list_datasets()