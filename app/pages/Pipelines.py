import streamlit as st
import pandas as pd
import pickle
from app.core.system import AutoMLSystem
from autoop.core.ml.model.classification import BayesClassification, LogisticClassification, SVM
from autoop.core.ml.model.regression import DecisionTreeRegressor, MultipleLinearRegression
from autoop.core.ml.model.regression.lasso import LassoRegression
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.metric import get_metric

automl_system = AutoMLSystem.get_instance()

MODEL_CLASSES = {
    "Regression": {
        "MultipleLinearRegression": MultipleLinearRegression,
        "Lasso": LassoRegression,
        "DecisionTreeRegressor": DecisionTreeRegressor
    },
    "Classification": {
        "SVM": SVM,
        "BayesClassification": BayesClassification,
        "LogisticRegression": LogisticClassification
    }
}

def load_model(model_type, model_name):
    """Load model based on type and name."""
    model_class = MODEL_CLASSES[model_type].get(model_name)
    if model_class:
        return model_class()  
    return None

def convert_metric_names_to_objects(metric_names):
    """Convert a list of metric names to metric objects."""
    return [get_metric(name) for name in metric_names]  

action = st.sidebar.selectbox(
    "Choose an action", ["View Saved Pipelines", "Load Pipeline Summary and Predict"]
)

def view_saved_pipelines():
    st.title("📋 Existing Saved Pipelines")
    saved_pipelines = automl_system.registry.list(type="pipeline")

    if not saved_pipelines:
        st.write("No saved pipelines found.")
        return

    st.write("## List of Saved Pipelines:")
    for pipeline in saved_pipelines:
        st.write(f"- Name: {pipeline.name}")
        st.write(f"  - Version: {pipeline.version}")
        st.write(f"  - Asset Path: {pipeline.asset_path}")
        st.write(f"  - Type: {pipeline.type}")
        st.write(f"  - Tags: {', '.join(pipeline.tags) if pipeline.tags else 'None'}")
        st.write(f"  - Metadata: {pipeline.metadata if pipeline.metadata else 'None'}")
        st.write("")  

def load_and_show_pipeline_summary_and_predict():
    st.title("🔄 Load Pipeline Summary and Predict")

    saved_pipelines = automl_system.registry.list(type="pipeline")

    if not saved_pipelines:
        st.write("No saved pipelines available.")
        return

    pipeline_options = {f"{p.name} (v{p.version})": p for p in saved_pipelines}
    selected_pipeline_name = st.selectbox("Select a pipeline to view and use for predictions", list(pipeline_options.keys()))

    if selected_pipeline_name:
        selected_pipeline = pipeline_options[selected_pipeline_name]

        try:
            pipeline_data = pickle.loads(selected_pipeline.data)
            
            st.write("## Pipeline Summary")
            st.write(f"Name: {selected_pipeline.name}")
            st.write(f"Version: {selected_pipeline.version}")
            st.write(f"Type: {selected_pipeline.type}")
            st.write(f"Asset Path: {selected_pipeline.asset_path}")
            st.write(f"Tags: {', '.join(selected_pipeline.tags) if selected_pipeline.tags else 'None'}")
            st.write(f"Metadata: {selected_pipeline.metadata if selected_pipeline.metadata else 'None'}")
            
            st.write("### Configuration Data:")
            st.write(pipeline_data)

            model_name = pipeline_data["model_type"]
            task_type = pipeline_data["task_type"]
            split_ratio = pipeline_data["split"]
            metric_names = pipeline_data["metrics"]
            metrics = convert_metric_names_to_objects(metric_names)

            # Load the model
            model = load_model(task_type, model_name)

            if model:
                uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")
                dataset_name = st.text_input("Dataset Name")
                asset_path = st.text_input("Asset Path (relative to storage)")

                if uploaded_file is not None:
                    input_data = pd.read_csv(uploaded_file)
                    

                    dataset1 = Dataset.from_dataframe(input_data, name = dataset_name, asset_path=asset_path, version = "2.0.0")
                    features = detect_feature_types(dataset1)
                    feature_names = [feature.name for feature in features]
                    feature_types = {feature.name: feature.type for feature in features}

                    input_features = st.multiselect("Select input features", feature_names)
                    target_feature = st.selectbox("Select target feature", feature_names)


                    features = [Feature(name=feature, type="numerical") for feature in input_features]
                    target = Feature(name=target_feature, type="numerical" if task_type == "Regression" else "categorical")
                    pipeline = Pipeline(
                    dataset=dataset1, 
                    model=model,
                    input_features=features,
                    target_feature=target,
                    metrics=metrics,
                    split=split_ratio
                    )

                    
                    st.write("### Uploaded Data")
                    st.write(input_data)

                    if st.button("Run Pipeline"):
                         results = pipeline.execute()
                         st.write("Results :", results)
                    

                    
            else:
                st.error(f"Model '{model_name}' for task '{task_type}' could not be loaded.")

        except Exception as e:
            st.error(f"Failed to load pipeline data: {e}")

if action == "View Saved Pipelines":
    view_saved_pipelines()
elif action == "Load Pipeline Summary and Predict":
    load_and_show_pipeline_summary_and_predict()