import streamlit as st
import pandas as pd
import pickle
from app.core.system import AutoMLSystem
from autoop.core.ml.model.classification import BayesClassification, LogisticClassification, SVM
from autoop.core.ml.model.regression import DecisionTreeRegressor, MultipleLinearRegression
from autoop.core.ml.model.regression.lasso import LassoRegression
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline

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
        st.write(f"- **Name**: {pipeline.name}")
        st.write(f"  - **Version**: {pipeline.version}")
        st.write(f"  - **Asset Path**: {pipeline.asset_path}")
        st.write(f"  - **Type**: {pipeline.type}")
        st.write(f"  - **Tags**: {', '.join(pipeline.tags) if pipeline.tags else 'None'}")
        st.write(f"  - **Metadata**: {pipeline.metadata if pipeline.metadata else 'None'}")
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
            st.write(f"**Name**: {selected_pipeline.name}")
            st.write(f"**Version**: {selected_pipeline.version}")
            st.write(f"**Type**: {selected_pipeline.type}")
            st.write(f"**Asset Path**: {selected_pipeline.asset_path}")
            st.write(f"**Tags**: {', '.join(selected_pipeline.tags) if selected_pipeline.tags else 'None'}")
            st.write(f"**Metadata**: {selected_pipeline.metadata if selected_pipeline.metadata else 'None'}")
            
            st.write("### Configuration Data:")
            st.write(pipeline_data)

            model_name = pipeline_data["model_type"]
            task_type = pipeline_data["task_type"]
            input_features = pipeline_data["input_features"]
            target_feature = pipeline_data["target_feature"]
            split_ratio = pipeline_data["split"]
            metrics = pipeline_data["metrics"]

            model = load_model(task_type, model_name)

            if model:
                uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")

                if uploaded_file is not None:
                    input_data = pd.read_csv(uploaded_file)

                    missing_features = [feat for feat in input_features if feat not in input_data.columns]
                    if missing_features:
                        st.error(f"The following required features are missing from the uploaded data: {', '.join(missing_features)}")
                        return

                    st.write("### Uploaded Data")
                    st.write(input_data)

                    X = input_data[input_features]

                    if hasattr(model, "predict"):
                        predictions = model.predict(X)
                        st.write("### Predictions")
                        st.write(pd.DataFrame(predictions, columns=["Predictions"]))
                    else:
                        st.error("The selected model does not support predictions.")

            else:
                st.error(f"Model '{model_name}' for task '{task_type}' could not be loaded.")

        except Exception as e:
            st.error(f"Failed to load pipeline data: {e}")

if action == "View Saved Pipelines":
    view_saved_pipelines()
elif action == "Load Pipeline Summary and Predict":
    load_and_show_pipeline_summary_and_predict()

