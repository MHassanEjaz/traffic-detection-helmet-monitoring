
import os
import tempfile
from pathlib import Path

import streamlit as st
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]

MODEL_FILES = {
    "vehicle": "yolo26n.pt",
    "original": "helmet.pt",
    "finetuned": "helmet_finetuned.pt",
    "ridesafe": "helmet_ridesafe_best.pt",
}

HELMET_OPTIONS = {
    "Original Helmet Model": "original",
    "Fine-tuned Helmet Model": "finetuned",
    "RideSafe Helmet Model": "ridesafe",
}


@st.cache_resource(show_spinner=False)
def download_models(repo_id):
    paths = {}

    for key, filename in MODEL_FILES.items():
        paths[key] = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )

    return paths


def configure_models(paths):
    os.environ["MODEL_PATH"] = paths["vehicle"]
    os.environ["ORIGINAL_HELMET_MODEL"] = paths["original"]
    os.environ["FINETUNED_HELMET_MODEL"] = paths["finetuned"]
    os.environ["RIDESAFE_HELMET_MODEL"] = paths["ridesafe"]
    os.environ["TRACKER_PATH"] = str(
        ROOT / "trackers" / "bytetrack_custom.yaml"
    )


st.set_page_config(
    page_title="Traffic Detection & Helmet Monitoring",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Traffic Detection & Helmet Monitoring")
st.write(
    "AI-powered vehicle detection, traffic counting "
    "and motorcycle helmet monitoring."
)

st.info(
    "Cloud processing uses CPU inference. Processing speed "
    "depends on video size and available server resources."
)

repo_id = os.getenv("HF_REPO_ID", "")

if not repo_id and "HF_REPO_ID" in st.secrets:
    repo_id = st.secrets["HF_REPO_ID"]

if not repo_id:
    st.error("Model repository is not configured.")
    st.stop()


helmet_selection = st.selectbox(
    "Select Helmet Detection Model",
    list(HELMET_OPTIONS.keys()),
)

uploaded_file = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov", "mkv", "webm"],
)

if st.button(
    "Process Video",
    type="primary",
    use_container_width=True,
    disabled=uploaded_file is None,
):

    try:
        with st.spinner("Downloading AI models..."):
            model_paths = download_models(repo_id)
            configure_models(model_paths)

        # Import after model paths are configured.
        from app.core.video_processor import process_video

        progress_bar = st.progress(
            0,
            text="Preparing video..."
        )

        status_text = st.empty()

        def update_progress(progress, message):
            progress_bar.progress(
                min(max(progress, 0.0), 1.0),
                text=message,
            )
            status_text.write(message)

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_dir = Path(temp_dir)

            input_path = temp_dir / uploaded_file.name
            output_path = temp_dir / "processed_traffic_video.mp4"

            input_path.write_bytes(uploaded_file.getbuffer())

            selected_model = HELMET_OPTIONS[helmet_selection]
            helmet_model_path = model_paths[selected_model]

            counts, processed_frames = process_video(
                input_path=input_path,
                output_path=output_path,
                helmet_model_path=helmet_model_path,
                progress_callback=update_progress,
            )

            video_bytes = output_path.read_bytes()

        st.session_state["cloud_result"] = {
            "video": video_bytes,
            "counts": counts,
            "frames": processed_frames,
            "model": helmet_selection,
        }

        progress_bar.progress(1.0, text="Processing completed!")
        status_text.success("Video processing completed successfully.")

    except Exception as error:
        st.error(f"Processing failed: {error}")


if "cloud_result" in st.session_state:

    result = st.session_state["cloud_result"]

    st.divider()
    st.header("Processed Video")

    st.video(result["video"])

    st.download_button(
        "Download Processed Video",
        data=result["video"],
        file_name="processed_traffic_video.mp4",
        mime="video/mp4",
        use_container_width=True,
    )

    st.subheader("Traffic Counting Results")

    st.write(f"Processed Frames: {result['frames']}")
    st.write(f"Helmet Model: {result['model']}")

    rows = []

    for class_name, directions in result["counts"].items():
        rows.append({
            "Class": class_name,
            "Up": directions["Up"],
            "Down": directions["Down"],
            "Total": directions["Up"] + directions["Down"],
        })

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )
