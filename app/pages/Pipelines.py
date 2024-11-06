import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem


automl_system = AutoMLSystem.get_instance()

st.set_page_config(page_title="Pipeline Deployment", page_icon="🚀")


st.title("🚀 Pipeline Deployment")
st.write("Load and deploy a saved pipeline to make predictions on new data.")

def pipeline_deployment():
    saved_pipelines = automl_system.registry.list(type="pipeline")
    pipeline_names = [p.name for p in saved_pipelines]
    
    if pipeline_names:
        selected_pipeline = st.selectbox("Select a pipeline to load", pipeline_names)
        
        if selected_pipeline:
            pipeline = next((p for p in saved_pipelines if p.name == selected_pipeline), None)
            
            if pipeline:
                
                st.write("### Pipeline Summary")
                st.write(pipeline.summary())
                
                
                uploaded_file = st.file_uploader("Upload a CSV file for predictions")
                
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    st.write("Data Loaded for Prediction:")
                    st.write(df.head())
                    
                    predictions = pipeline.predict(df)
                    
                    st.write("### Predictions")
                    st.write(predictions)
                    
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=predictions.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
    else:
        st.write("No saved pipelines available.")

pipeline_deployment()
