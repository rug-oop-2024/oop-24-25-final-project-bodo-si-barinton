import streamlit as st
from app.core.system import AutoMLSystem
import pickle

automl_system = AutoMLSystem.get_instance()
action = st.sidebar.selectbox(
    "Choose an action", ["View Saved Pipelines", "Load Pipeline Summary"]
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
        st.write("")

def load_and_show_pipeline_summary():
    st.title("🔄 Load and Show Pipeline Summary")

    saved_pipelines = automl_system.registry.list(type="pipeline")

    if not saved_pipelines:
        st.write("No saved pipelines available.")
        return

    pipeline_options = {f"{p.name} (v{p.version})": p for p in saved_pipelines}
    selected_pipeline_name = st.selectbox("Select a pipeline to view", list(pipeline_options.keys()))

    if selected_pipeline_name:
        selected_pipeline = pipeline_options[selected_pipeline_name]

        try:
            pipeline_data = pickle.loads(selected_pipeline.data)
            st.write("## Pipeline Summary")
            st.write(f"**Name**: {selected_pipeline.name}")
            st.write(f"**Version**: {selected_pipeline.version}")
            st.write(f"**Type**: {selected_pipeline.type}")
            st.write(f"**Asset Path**: {selected_pipeline.asset_path}")
            st.write("### Configuration Data:")
            st.write(pipeline_data)  

        except Exception as e:
            st.error(f"Failed to load pipeline data: {e}")


if action == "View Saved Pipelines":
    view_saved_pipelines()
elif action == "Load Pipeline Summary":
    load_and_show_pipeline_summary()