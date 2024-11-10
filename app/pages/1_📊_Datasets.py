import io
from typing import Optional

import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset, List

automl_system: AutoMLSystem = AutoMLSystem.get_instance()

st.title("Dataset Manager")

action: str = st.sidebar.selectbox(
    "Choose an action", ["Upload Dataset", "View Dataset", "Delete Dataset"]
)


def upload_dataset() -> None:
    """
    Upload a dataset and save it as an artifact.
    """
    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    dataset_name = st.text_input("Dataset Name")
    asset_path = st.text_input("Asset Path (relative to storage)")

    if uploaded_file is not None and dataset_name and asset_path:
        df: pd.DataFrame = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())

        encoded_data: bytes = df.to_csv(index=False).encode()

        dataset_artifact: Dataset = Dataset(
            name=dataset_name,
            asset_path=asset_path,
            version="1.0.0",
            data=encoded_data
        )

        automl_system.registry.register(dataset_artifact)

        st.success(
            f"Dataset '{dataset_name}' uploaded and saved successfully."
        )


def view_dataset() -> None:
    """
    View details and content of a selected dataset.
    """
    dataset_list: List[str] = [
        artifact.name 
        for artifact in automl_system.registry.list(type="dataset")
    ]

    if not dataset_list:
        st.write("No datasets available.")
        return

    selected_dataset: str = st.selectbox(
        "Select a dataset to view", dataset_list
    )

    if selected_dataset:
        dataset_artifact: Optional[Dataset] = next(
            (
                artifact
                for artifact in automl_system.registry.list(type="dataset")
                if artifact.name == selected_dataset
            ),
            None,
        )

        if dataset_artifact:
            st.write("Dataset Details:")
            st.write(f"Name: {dataset_artifact.name}")
            st.write(f"Version: {dataset_artifact.version}")
            st.write("Dataset Content:")

            csv_data: str = dataset_artifact.data.decode()
            df: pd.DataFrame = pd.read_csv(io.StringIO(csv_data))
            st.write(df)


def delete_dataset() -> None:
    """
    Delete a selected dataset from the registry.
    """
    dataset_list: List[str] = [
        artifact.name 
        for artifact in automl_system.registry.list(type="dataset")
    ]

    if not dataset_list:
        st.write("No datasets available.")
        return

    selected_dataset: str = st.selectbox(
        "Select a dataset to delete", dataset_list
    )

    if st.button("Delete Dataset"):
        dataset_artifact: Optional[Dataset] = next(
            (
                artifact
                for artifact in automl_system.registry.list(type="dataset")
                if artifact.name == selected_dataset
            ),
            None,
        )

        if dataset_artifact:
            automl_system.registry.delete(dataset_artifact.id)
            st.success(f"Dataset '{selected_dataset}' deleted successfully.")


if action == "Upload Dataset":
    upload_dataset()
elif action == "View Dataset":
    view_dataset()
elif action == "Delete Dataset":
    delete_dataset()
