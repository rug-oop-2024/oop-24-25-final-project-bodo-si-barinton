import streamlit as st
import pandas as pd
import io
from app.core.system import AutoMLSystem
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.metric import METRICS, get_metric
from typing import List
from autoop.core.ml.model.classification import SVM, BayesClassification, LogisticRegression
from autoop.core.ml.model.regression import Lasso, MultipleLinearRegression, DecisionTreeRegressor
from autoop.core.ml.model import Model
from autoop.core.ml.dataset import Dataset

automl_system = AutoMLSystem.get_instance()

MODEL_CLASSES = {
    "Regression": {
        "MultipleLinearRegression": MultipleLinearRegression,
        "Lasso": Lasso,
        "DecisionTreeRegressor": DecisionTreeRegressor
    },
    "Classification": {
        "SVM": SVM,
        "BayesClassification": BayesClassification,
        "LogisticRegression": LogisticRegression
    }
}


def get_compatible_metrics(task_type: str) -> List[str]:
    """Return a list of compatible metrics based on the task type."""
    if task_type == "Classification":
        return ["accuracy", "logloss", "micro", "macro"]
    elif task_type == "Regression":
        return ["mean_squared_error", "mean_absolute_error", "root_mean_squared_error"]
    else:
        return []


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

            dataset_instance = Dataset.from_dataframe(df, name=selected_dataset, asset_path=dataset_artifact.asset_path)

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

                
                available_models = MODEL_CLASSES[task_type]
                selected_model_name = st.selectbox("Select a model", list(available_models.keys()))
                selected_model_class = available_models[selected_model_name]
                model_chosen = selected_model_class()

                
                compatible_metrics = get_compatible_metrics(task_type)
                selected_metrics = st.multiselect("Select metrics", compatible_metrics)
                metric_objects = [get_metric(metric) for metric in selected_metrics]

                
                split_ratio = st.slider("Select train/test split ratio", min_value=0.1, max_value=0.9, value=0.8, step=0.05)

                
                if st.button("Run Pipeline"):
                    pipeline = Pipeline(



                        dataset= dataset_instance,
                        model=model_chosen,
                        input_features=[Feature(name=feature, type=feature_types[feature]) for feature in input_features],
                        target_feature=Feature(name=target_feature, type=feature_types[target_feature]),
                        metrics=metric_objects,
                        split=split_ratio
                    )
                    
                    
                    st.write("### Pipeline Summary")
                    st.write(f"- **Model**: {selected_model_name}")
                    st.write(f"- **Task Type**: {task_type}")
                    st.write(f"- **Selected Metrics**: {', '.join(selected_metrics)}")
                    st.write(f"- **Split Ratio**: {split_ratio}")
                    st.write(f"- **Input Features**: {input_features}")
                    st.write(f"- **Target Feature**: {target_feature}")

                    
                    results = pipeline.execute()
                    st.write("Pipeline executed successfully.")
                    st.write("Results:", results)

                    
                    pipeline_name = st.text_input("Enter a name for this pipeline:")
                    pipeline_version = st.text_input("Enter a version for this pipeline:")
                    
                    if st.button("Save Pipeline"):
                        if pipeline_name and pipeline_version:
                            pipeline.save(name=pipeline_name, version=pipeline_version)
                            automl_system.registry.add(pipeline)
                            st.write(f"Pipeline '{pipeline_name}' (version {pipeline_version}) saved and registered successfully.")
                        else:
                            st.warning("Please enter both a name and a version for the pipeline.")




if action == "List Datasets":
    list_datasets()
elif action == "Feature Selection":
    feature_selection()
