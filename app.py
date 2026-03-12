import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd

# Import our custom modules
from preprocess import (
    grayscale_and_standardize,
    apply_clahe,
    extract_color_indices,
    extract_edges_and_texture,
    segment_leaf,
    extract_features,
)
from model_inference import (
    DiseaseClassifier,
    YOLOv8DiseaseClassifier,
    YOLOv8ObjectDetector,
)
from database import PhenotypeDatabase

# Page config
st.set_page_config(
    page_title="Automated Leaf Detection & Phenotyping", page_icon="🌿", layout="wide"
)

# Custom CSS for UI
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 30px;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .report-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin-top: 20px;
        color: #333333;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    # Placeholders for actual paths
    disease_model = "disease_model.pth"
    yolov8_model = r"runs\classify\runs\classify\leaf_disease_model2\weights\best.pt"
    class_names = "class_names.txt"

    resnet_classifier = DiseaseClassifier(disease_model, class_names_path=class_names)
    yolo_classifier = YOLOv8DiseaseClassifier(yolov8_model)
    yolo_detector = YOLOv8ObjectDetector("yolov8n.pt")  # Auto-downloads base model
    db = PhenotypeDatabase("phenotyping_results.db")

    return resnet_classifier, yolo_classifier, yolo_detector, db


def main():
    st.markdown(
        "<h1 class='main-header'>Automated Leaf Detection and Phenotyping</h1>",
        unsafe_allow_html=True,
    )

    resnet_classifier, yolo_classifier, yolo_detector, db = load_models()

    # Sidebar
    st.sidebar.title("Input Options")
    st.sidebar.markdown(
        "Upload a leaf image or take a picture using your camera to begin the phenotyping pipeline."
    )

    model_option = st.sidebar.selectbox(
        "Select Classification Model:",
        ("ResNet-18 (Default)", "YOLOv8-cls (Fast & Modern)"),
    )

    st.sidebar.divider()

    upload_option = st.sidebar.radio("Choose image source:", ("Upload Image", "Camera"))

    image_file = None
    if upload_option == "Upload Image":
        image_file = st.sidebar.file_uploader(
            "Choose a plant leaf image...", type=["jpg", "jpeg", "png", "bmp"]
        )
    else:
        image_file = st.sidebar.camera_input("Take a picture")

    if image_file is not None:
        try:
            # Read image
            image = Image.open(image_file).convert("RGB")
            img_np = np.array(image)

            # 1. Original Image
            st.subheader("1. Original Image")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption="Input Image", use_container_width=True)

            st.divider()

            # 2. Image Preprocessing Pipeline
            st.subheader("2. Image Preprocessing & Feature Extraction")

            with st.spinner("Processing image..."):
                # Run preprocessing
                gray, std_img = grayscale_and_standardize(img_np)
                clahe = apply_clahe((std_img * 255).astype(np.uint8))
                exg, exr, exg_vis, exr_vis = extract_color_indices(img_np)
                edges, lbp, lbp_vis = extract_edges_and_texture(gray)

                # Show results in columns
                p_col1, p_col2, p_col3, p_col4 = st.columns(4)

                with p_col1:
                    st.image(
                        clahe,
                        caption="CLAHE (Enhanced Contrast)",
                        use_container_width=True,
                        channels="GRAY",
                    )
                with p_col2:
                    st.image(
                        exg_vis,
                        caption="Excess Green (ExG)",
                        use_container_width=True,
                        channels="GRAY",
                    )
                with p_col3:
                    st.image(
                        edges,
                        caption="Canny Edges",
                        use_container_width=True,
                        channels="GRAY",
                    )
                with p_col4:
                    st.image(
                        lbp_vis,
                        caption="LBP (Texture)",
                        use_container_width=True,
                        channels="GRAY",
                    )

            st.divider()

            # 3. Segmentation and Detection
            st.subheader("3. Segmentation (Powered by PlantCV)")

            with st.spinner("Segmenting and running Object Detection..."):
                mask = segment_leaf(img_np)
                segmented_leaf = cv2.bitwise_and(img_np, img_np, mask=mask)

                # Run YOLOv8 Object Detection on the segmented leaf
                detected_leaf_img = yolo_detector.detect_and_draw(segmented_leaf)

                det_col1, det_col2 = st.columns(2)
                with det_col1:
                    st.image(
                        segmented_leaf,
                        caption="Segmented Leaf",
                        use_container_width=True,
                    )
                with det_col2:
                    st.image(
                        detected_leaf_img,
                        caption="YOLOv8 Detected Bounding Boxes",
                        use_container_width=True,
                    )

            st.divider()

            # 4. Phenotyping Report & Disease Classification
            st.subheader("4. Phenotyping Report & Disease Classification")

            with st.spinner("Extracting traits and classifying disease..."):
                traits = extract_features(img_np, mask)

                # Pass PIL Image directly to avoid NumPy-to-PIL conversion overhead
                if model_option == "YOLOv8-cls (Fast & Modern)":
                    disease_class, confidence, probs = yolo_classifier.classify(image)
                else:
                    disease_class, confidence, probs = resnet_classifier.classify(
                        image
                    )

                # Save to database (Assuming 1 leaf per frame for now)
                db.save_report(disease_class, confidence, 1, traits)

                r_col1, r_col2 = st.columns([1, 1])

                with r_col1:
                    st.markdown("### Extracted Traits")
                    if "error" in traits:
                        st.warning(f"Feature Extraction Error: {traits['error']}")
                    else:
                        # Display traits beautifully
                        df_traits = pd.DataFrame(
                            [
                                {
                                    "Trait": k.replace("_", " ").title(),
                                    "Value": (
                                        f"{v:.4f}" if isinstance(v, float) else str(v)
                                    ),
                                }
                                for k, v in traits.items()
                            ]
                        )
                        st.dataframe(
                            df_traits, hide_index=True, use_container_width=True
                        )

                with r_col2:
                    st.markdown("### Classification Result")

                    # Highlight based on result
                    if "Healthy" in disease_class:
                        color = "green"
                    elif "Model not loaded" in disease_class:
                        color = "gray"
                    else:
                        color = "red"

                    st.markdown(
                        f"""
                    <div class="report-card" style="border-left-color: {color};">
                        <h2>{disease_class}</h2>
                        <p><strong>Confidence:</strong> {confidence*100:.2f}%</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    st.success("Results saved to backend database successfully.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

    else:
        # Initial state before upload
        st.info("👈 Please select an input option from the sidebar to start.")

        # Display latest database entries
        st.subheader("Recent Database Records")
        try:
            # Re-use the already loaded db instance globally and use optimized backend query
            records = db.get_recent_reports(limit=5)
            if records:
                # Traits JSON is now excluded in the DB query to reduce memory usage
                df_records = pd.DataFrame(
                    records,
                    columns=[
                        "ID",
                        "Timestamp",
                        "Disease Class",
                        "Confidence",
                        "Leaves Detected",
                    ],
                )
                # Remove redundant head(5) truncation as the database limits it for us
                st.dataframe(df_records, hide_index=True)
            else:
                st.write("No records yet.")
        except Exception as e:
            st.write("Database not initialized yet.")


if __name__ == "__main__":
    main()
