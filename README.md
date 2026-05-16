# Automated Leaf Detection and Phenotyping

A complete end-to-end Python application for robust plant phenotyping and disease classification, featuring a modular computer-vision pipeline and an interactive web dashboard built with Streamlit.

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: Interactive web dashboard framework.
- **Ultralytics YOLOv8**: State-of-the-art fast models for both disease classification (`YOLOv8-cls`) and bounding-box object detection built on the PyTorch backend.
- **PyTorch**: Deep learning backend for the ResNet-18 disease classification model.
- **PlantCV & OpenCV**: Advanced computer vision libraries for image enhancement, morphological operations, and precise leaf segmentation via color space conversion.
- **SQLite**: Lightweight database for persistent storage of phenotyping reports.
- **Scikit-Image, NumPy, Pandas, Matplotlib**: Mathematical, array-level, and tabular data manipulation.

## Features

- **Advanced Preprocessing Pipeline**: Implements operations such as Grayscale Conversion, Standardization, and Contrast Limited Adaptive Histogram Equalization (CLAHE).
- **Color & Texture Indices**: Extracts and visualizes Excess Green (ExG), Excess Red (ExR), Canny edge contours, and Local Binary Patterns (LBP).
- **Dynamic Leaf Segmentation**: Utilizes **PlantCV**'s LAB color space conversion and dark-object Otsu thresholding to accurately separate the primary leaf from any noisy background.
- **Phenotypic Trait Extraction**: Automatically calculates critical morphological features (Area, Perimeter, Eccentricity, Solidity, Extent, Aspect Ratio) and color distribution profiles (RGB Mean and Standard Deviation) for analysis.
- **Deep Learning Classification**: Integrates both a **PyTorch ResNet-18** model and a **YOLOv8-cls** model capable of accurately diagnosing over 40 distinct plant health classes (including Apple Scab, Corn Leaf Blight, Peach Healthy, etc). 
- **YOLOv8 Object Detection**: Automatically draws tightly-fit bounding boxes around detected leaves for visual clarity.
- **Interactive UI**: A Streamlit frontend displays side-by-side processing outputs, a traits dataframe, and disease detection confidence metrics.
- **Data Persistence**: Records every phenotyping session and its corresponding geometric/color traits into an SQLite database (`phenotyping_results.db`).

---

## File Structure

- `app.py`: The main Streamlit dashboard application routing the front-end components.
- `preprocess.py`: Contains **PlantCV** and **scikit-image** logic for transforming the images and extracting geometric phenotypes.
- `model_inference.py`: Wraps the PyTorch ResNet, YOLOv8 Object Detection, and YOLOv8 Classification architectures into classes capable of translating image tensors into bounding boxes and disease labels.
- `dataset.py`: Defines a custom recursive PyTorch `Dataset` loader intended to dynamically parse a hierarchical directory structure of `Split/Species/Disease/image.jpg`.
- `train.py`: A complete hardware-accelerated (CUDA) training script loop designed to process the dataloaders and train the **ResNet** network.
- `prepare_yolo_dataset.py`: Flattens the hierarchical dataset directory structure via hard-links into a format compatible with YOLOv8.
- `train_yolo.py`: A complete training script designed to load the flattened dataset and train the **YOLOv8** classification backbone.
- `predict_yolo.py`: A standalone test script for passing images through the trained YOLOv8 classification model.
- `database.py`: A lightweight SQLite wrapper handling the SQL creation and insertion queries to permanently save the trait reports.

---

## How to Run the Application

### 1. Environment Setup
It is highly recommended to use a Python virtual environment.
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
Install the required packages specified in the text file. 
```bash
pip install -r requirements.txt
```
*(Note: The base `yolov8n.pt` object detection weights will be downloaded from the Ultralytics servers automatically when the application is launched).*

*(Note: If you have an NVIDIA GPU, you may want to install the CUDA-enabled version of PyTorch manually from the [PyTorch website](https://pytorch.org/) to accelerate model training).*

### 3. Provide Model Weights
Because deep learning models are large, the trained structural weights files are not included in the repository. Provide your trained model named `disease_model.pth` (for ResNet) in the root of the project directory.

Alternatively, if you possess the raw hierarchical `image data/` folder, you can run the training generation scripts yourself:

**To train the ResNet Model:**
```bash
python train.py
```
This will automatically parse your data, train the ResNet network across epochs, and drop the `disease_model.pth` and `class_names.txt` outputs into your root directory!

**To train the YOLOv8 Classification Model:**
```bash
python prepare_yolo_dataset.py
python train_yolo.py
```
This prepares the optimized folder structure and trains YOLO on the classes. 

### 4. Launch the Dashboard
With the dependencies mapped and the model weights present, you can start the local web server:
```bash
streamlit run app.py
```

Upload a leaf picture from your filesystem or capture one with your webcam using the sidebar. The application will compute the traits, render the sequential mask layers via PlantCV, run the deep neural network bounding box and disease classifiers, and log the output into the database!
