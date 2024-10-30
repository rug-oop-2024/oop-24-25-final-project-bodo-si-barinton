import streamlit as st
import pandas as pd
import os

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


# Get the AutoML system instance
automl = AutoMLSystem.get_instance()

# List datasets from the registry
datasets = automl.registry.list(type="dataset")

# Title of the page
st.title("Dataset Management")

# Sidebar options
st.sidebar.header("Actions")
action = st.sidebar.selectbox("Select an action", ["View Datasets", "Add Dataset", "Edit Dataset", "Delete Dataset"])

# Function to display datasets in a table
def display_datasets(datasets):
    data = {
        "Name": [dataset.name for dataset in datasets],
        "Version": [dataset.version for dataset in datasets],
        "Path": [dataset.asset_path for dataset in datasets],
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

# Display datasets
if action == "View Datasets":
    st.subheader("Available Datasets")
    display_datasets(datasets)
    
    # Option to view details
    selected_dataset_name = st.selectbox("Select Dataset to View Details", [d.name for d in datasets])
    selected_dataset = next((d for d in datasets if d.name == selected_dataset_name), None)
    
    if selected_dataset:
        st.write("### Dataset Details")
        try:
            data = selected_dataset.read()
            st.dataframe(data)
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

elif action == "Add Dataset":
    st.subheader("Add a New Dataset")
    name = st.text_input("Dataset Name")
    version = st.text_input("Version", "1.0.0")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        file_name = os.path.basename(uploaded_file.name)
        asset_path = st.text_input("Asset Path", value=f"/assets/{file_name}")
        data = pd.read_csv(uploaded_file)
        
        st.write("Preview of Uploaded Data")
        st.dataframe(data.head())
        
        if st.button("Add Dataset"):
            encoded_data = data.to_csv(index=False).encode()
            new_dataset = Dataset(name=name, asset_path=asset_path, data=encoded_data, version=version)
            automl.registry.register(new_dataset)  # Use register instead of add
            st.success(f"Dataset '{name}' added successfully!")
    else:
        asset_path = st.text_input("Asset Path", value="")

elif action == "Edit Dataset":
    st.subheader("Edit an Existing Dataset")
    selected_dataset_name = st.selectbox("Select Dataset to Edit", [d.name for d in datasets])
    
    selected_dataset = next((d for d in datasets if d.name == selected_dataset_name), None)
    if selected_dataset:
        st.write("### Edit Dataset Details")
        new_version = st.text_input("New Version", value=selected_dataset.version)
        
        # Option to replace data
        uploaded_file = st.file_uploader("Upload New CSV", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of New Data")
            st.dataframe(data.head())
        
        if st.button("Save Changes"):
            if uploaded_file:
                selected_dataset.save(data)  # Save new data
            selected_dataset.version = new_version
            automl.registry.update(selected_dataset)
            st.success(f"Dataset '{selected_dataset_name}' updated successfully!")

elif action == "Delete Dataset":
    st.subheader("Delete a Dataset")
    selected_dataset_name = st.selectbox("Select Dataset to Delete", [d.name for d in datasets])
    
    if st.button("Delete"):
        automl.registry.delete(selected_dataset_name)
        st.success(f"Dataset '{selected_dataset_name}' deleted successfully!")
