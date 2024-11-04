import os
import csv
from io import StringIO
import streamlit as st
import json
import pandas as pd
from autoop.core.database import Database
from autoop.core.storage import LocalStorage
from autoop.core.ml.dataset import Dataset
from app.core.system import AutoMLSystem
# Initialize the storage and database
assets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../assets"))


storage = LocalStorage(base_path=assets_path)
database = Database(storage=storage)

st.title("Dataset Manager")

# Sidebar menu for actions
action = st.sidebar.selectbox("Choose an action", ["Add Dataset", "View Dataset", "Delete Dataset", "List Datasets", "Register Dataset"])



def csv_to_json(csv_file):
    # Read the CSV file into a list of dictionaries
    csv_reader = csv.DictReader(csv_file)
    data = [row for row in csv_reader]
    return data

# Function to add a new dataset
def add_dataset():
    st.subheader("Add a New Dataset")

    collection = st.text_input("Collection Name")
    dataset_id = st.text_input("Dataset ID")
    
    # Manual entry
    entry_data = st.text_area("Dataset JSON (in dictionary format)")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a dataset file", type=["json", "csv"])

    # Process uploaded file, if available
    file_data = None
    if uploaded_file is not None:
        if uploaded_file.type == "application/json":
            try:
                # Read JSON file
                file_data = json.load(uploaded_file)
                st.write("JSON data loaded successfully:", file_data)
            except Exception as e:
                st.error(f"Failed to read JSON file: {e}")
        elif uploaded_file.type == "text/csv":
            try:
                # Convert CSV to JSON format
                csv_file = StringIO(uploaded_file.getvalue().decode("utf-8"))
                file_data = csv_to_json(csv_file)
                st.write("CSV data converted to JSON format successfully:", file_data)
            except Exception as e:
                st.error(f"Failed to convert CSV file to JSON format: {e}")
    
    # Save dataset either from manual entry or file upload
    if st.button("Save Dataset"):
        try:
            # Determine the final entry data
            if file_data is not None:
                entry = {"data": file_data} if isinstance(file_data, list) else file_data
            else:
                entry = eval(entry_data)  # Convert text to dictionary (use cautiously)

            # Save data in the database
            saved_entry = database.set(collection, dataset_id, entry)
            st.success(f"Dataset saved successfully: {saved_entry}")
        except Exception as e:
            st.error(f"Failed to save dataset: {e}")

def view_dataset():
    st.subheader("View Dataset")

    collection = st.text_input("Collection Name")
    dataset_id = st.text_input("Dataset ID")

    if st.button("Fetch Dataset"):
        data = database.get(collection, dataset_id)
        if data:
            st.write(data)
        else:
            st.warning("Dataset not found.")

def delete_dataset():
    st.subheader("Delete Dataset")

    collection = st.text_input("Collection Name")
    dataset_id = st.text_input("Dataset ID")

    if st.button("Delete Dataset"):
        database.delete(collection, dataset_id)
        st.success(f"Dataset with ID '{dataset_id}' from '{collection}' deleted.")

def list_datasets():
    st.subheader("List All Datasets in a Collection")

    collection = st.text_input("Collection Name")

    if st.button("List Datasets"):
        datasets = database.list(collection)
        if datasets:
            for dataset_id, data in datasets:
                st.write(f"ID: {dataset_id} - Data: {data}")
        else:
            st.warning("No datasets found in this collection.")

def upload_and_register_dataset():
    st.subheader("Upload and Register Dataset")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", df)

        dataset_name = st.text_input("Dataset Name")
        asset_path = f"./assets/datasets/{dataset_name.lower().replace(' ', '_')}.csv"

        if st.button("Register Dataset"):
            try:
                dataset = Dataset.from_dataframe(data=df, name=dataset_name, asset_path=asset_path)

                automl_system = AutoMLSystem.get_instance()

                automl_system.registry.register(dataset)
                st.success("Dataset successfully registered in the Artifact Registry!")
            except Exception as e:
                st.error(f"An error occurred while registering the dataset: {e}")

if action == "Add Dataset":
    add_dataset()
elif action == "View Dataset":
    view_dataset()
elif action == "Delete Dataset":
    delete_dataset()
elif action == "List Datasets":
    list_datasets()
elif action == "Register Dataset":
    upload_and_register_dataset()
