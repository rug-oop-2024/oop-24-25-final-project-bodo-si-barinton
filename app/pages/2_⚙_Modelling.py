import streamlit as st
import pandas as pd
import io
from app.core.system import AutoMLSystem  
from autoop.core.ml.feature import Feature
from typing import List


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


def feature_selection():
    dataset_list = [artifact.name for artifact in automl_system.registry.list(type="dataset")]

    if not dataset_list:
        st.write("No datasets available.")
        return

    
    selected_dataset = st.selectbox("Select a dataset for feature selection", dataset_list)

    if selected_dataset:
        dataset_artifact = next(
            (artifact for artifact in automl_system.registry.list(type="dataset") if artifact.name == selected_dataset),
            None
        )

        if dataset_artifact:
            csv_data = dataset_artifact.data.decode()
            df = pd.read_csv(io.StringIO(csv_data))
            
            st.write("Dataset Loaded:")
            st.write(df.head())

            features = detect_feature_types(df)
            feature_names = [feature.name for feature in features]
            feature_types = {feature.name: feature.type for feature in features}

            input_features = st.multiselect("Select input features", feature_names)
            target_feature = st.selectbox("Select target feature", feature_names)

            if input_features and target_feature:
                st.write(f"Selected input features: {input_features}")
                st.write(f"Selected target feature: {target_feature} (Type: {feature_types[target_feature]})")

                task_type = "Regression" if feature_types[target_feature] == "numerical" else "Classification"
                st.write(f"Detected task type: {task_type}")


if action == "List Datasets":
    list_datasets()
elif action == "Feature Selection":
    feature_selection()
