
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio

# Set default plotly theme
pio.templates.default = "plotly_white"

st.set_page_config(page_title="MedQA Dashboard", layout="wide")
st.title(" MedQA Data Dashboard")

OFFLINE_DIR = "/mnt/object/data/dataset-split"
RETRAIN_DIR = "/mnt/object/data/production/retraining_data_transformed"

tab1, tab2 = st.tabs([" Offline MedQuAD Data", " Retraining Data"])

# --- TAB 1: OFFLINE DATA ---
with tab1:
    st.header(" Offline MedQuAD Data ")

    meta_path = os.path.join(OFFLINE_DIR, "metadata.json")
    paths = {
        "Training": os.path.join(OFFLINE_DIR, "training", "training.json"),
        "Validation": os.path.join(OFFLINE_DIR, "validation", "validation.json"),
        "Evaluation": os.path.join(OFFLINE_DIR, "evaluation", "testing.json"),
    }

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
        st.subheader("Metadata Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Source:** `{metadata.get('source', '-')}`")
            st.markdown(f"**Initial Records:** `{metadata.get('initial_records', '-')}`")
            st.markdown(f"**Final Records:** `{metadata.get('final_records', '-')}`")

        with col2:
            split = metadata.get("split_counts", {})
            st.markdown("**Split Counts:**")
            st.markdown(f"- Training: `{split.get('training', '-')}`")
            st.markdown(f"- Validation: `{split.get('validation', '-')}`")
            st.markdown(f"- Testing: `{split.get('testing', '-')}`")

        dropped = metadata.get("dropped", {})
        if dropped:
            st.markdown("**Dropped Records:**")
            drop_df = pd.DataFrame(list(dropped.items()), columns=["Type", "Count"])
            st.dataframe(drop_df, use_container_width=True)

            st.markdown("### Dropped Record Summary")
            fig = px.bar(
                drop_df,
                y="Type",
                x="Count",
                orientation="h",
                title="Drop Statistics by Type",
                color="Type",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)

        # st.markdown("###  Dataset Split Counts")
        # st.write(metadata.get("split_counts", {}))

        split_tabs = st.tabs(list(paths.keys()))
        for label, path in zip(paths.keys(), paths.values()):
            with split_tabs[list(paths.keys()).index(label)]:
                if not os.path.exists(path):
                    st.error(f"{label} set not found.")
                    continue

                df = pd.read_json(path, lines=True)
                st.markdown(f"###  {label} Set Explorer")

                qtype_options = df["question_type"].unique()
                selected_types = st.multiselect(f"Filter by question_type", qtype_options, key=f"{label}_qtype")
                keyword = st.text_input(f"Keyword in question", key=f"{label}_keyword")

                filtered_df = df.copy()
                if selected_types:
                    filtered_df = filtered_df[filtered_df["question_type"].isin(selected_types)]
                if keyword:
                    filtered_df = filtered_df[filtered_df["question"].str.contains(keyword, case=False, na=False)]

                st.write(f"Showing {len(filtered_df)} / {len(df)} records")
                st.dataframe(filtered_df.sample(min(10, len(filtered_df))), use_container_width=True)

                csv = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(" Download CSV", csv, file_name=f"{label.lower()}_filtered.csv")

                st.markdown(f"###  {label} - Question Type Pie Chart")
                fig_pie = px.pie(
                    df,
                    names="question_type",
                    title=f"{label} - Question Type Distribution",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown(f"###  {label} - Question Length Box & Histogram")
                df["question_length"] = df["question"].apply(lambda x: len(x.split()))
                fig_box = px.box(df, y="question_length", color_discrete_sequence=["#636EFA"])
                fig_hist = px.histogram(df, x="question_length", nbins=30, color_discrete_sequence=["#EF553B"])
                st.plotly_chart(fig_box, use_container_width=True)
                st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.error("metadata.json not found in dataset-split directory.")

# --- TAB 2: RETRAINING DATA ---
with tab2:
    st.header(" Retraining Data")

    versions = sorted(
        [v for v in os.listdir(RETRAIN_DIR) if v.startswith("v") and os.path.isdir(os.path.join(RETRAIN_DIR, v))],
        key=lambda x: int(x[1:])
    )

    if not versions:
        st.warning("No retraining versions found.")
    else:
        selected_version = st.selectbox("Select Version", versions)
        version_path = os.path.join(RETRAIN_DIR, selected_version)
        meta_path = os.path.join(version_path, "metadata.json")
        data_path = os.path.join(version_path, "retraining_data.json")

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

            st.subheader(" Metadata")
            st.markdown(f"- **Version**: {meta.get('version')}")
            st.markdown(f"- **Timestamp**: {meta.get('timestamp')}")
            st.markdown(f"- **Records**: {meta.get('record_count')}")

            if meta.get("archived_files"):
                st.markdown(f"** Archived Files:**")
                st.code("\n".join(meta["archived_files"]), language="bash")

            if meta.get("dropped"):
                dropped_df = pd.DataFrame(list(meta["dropped"].items()), columns=["Type", "Count"])
                st.markdown("### Dropped Record Summary")
                fig = px.bar(
                    dropped_df,
                    y="Type", x="Count", orientation="h",
                    title="Drop Statistics",
                    color="Type",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Metadata not found.")

        if os.path.exists(data_path):
            df = pd.read_json(data_path, lines=True)

            st.markdown("###  Question Type Distribution")
            if "question_type" in df.columns:
                type_counts = df["question_type"].value_counts()
                fig2 = px.pie(
                    df, names="question_type",
                    title="Question Type Distribution",
                    color_discrete_sequence=px.colors.sequential.Teal
                )
                st.plotly_chart(fig2, use_container_width=True)

                expected_types = {"symptoms", "complications", "dietary", "genetic changes"}
                present_types = set(df["question_type"].str.lower().unique())
                missing = expected_types - present_types
                if missing:
                    st.warning(f" Missing expected question types: {', '.join(missing)}")

            st.markdown("###  Question Length (Box + Histogram)")
            df["question_length"] = df["question"].apply(lambda x: len(x.split()))
            fig_box = px.box(df, y="question_length", color_discrete_sequence=["#00CC96"])
            fig_hist = px.histogram(df, x="question_length", nbins=30, color_discrete_sequence=["#AB63FA"])
            st.plotly_chart(fig_box, use_container_width=True)
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("###  Record Explorer")
            qtypes = df["question_type"].dropna().unique().tolist() if "question_type" in df.columns else []
            selected_qtypes = st.multiselect("Filter by question_type", qtypes, key="retraining_qtype")
            keyword = st.text_input("Keyword in question", key="retraining_keyword")

            filtered_df = df.copy()
            if selected_qtypes:
                filtered_df = filtered_df[filtered_df["question_type"].isin(selected_qtypes)]
            if keyword:
                filtered_df = filtered_df[filtered_df["question"].str.contains(keyword, case=False, na=False)]

            st.write(f"Showing {len(filtered_df)} / {len(df)} records")
            st.dataframe(filtered_df.sample(min(10, len(filtered_df))), use_container_width=True)

            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(" Download CSV", csv, file_name="retraining_filtered.csv")
        else:
            st.error("retraining_data.json not found.")

    st.markdown("###  Record Counts Over Time")
    history = []
    for v in versions:
        meta_fp = os.path.join(RETRAIN_DIR, v, "metadata.json")
        if os.path.exists(meta_fp):
            with open(meta_fp) as f:
                meta = json.load(f)
            history.append({
                "version": meta.get("version"),
                "records": meta.get("record_count", 0),
                "timestamp": meta.get("timestamp")
            })

    if history:
        df_hist = pd.DataFrame(history)
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
        fig_line = px.line(
            df_hist.sort_values("timestamp"),
            x="timestamp", y="records", markers=True,
            title="Retraining Record Volume Over Time"
        )
        st.plotly_chart(fig_line, use_container_width=True)
