# Automated Leaf Detection and Phenotyping

A complete end-to-end Python application for robust plant phenotyping and disease classification, featuring a modular computer-vision pipeline and an interactive web dashboard built with Streamlit.

## Features

- **Advanced Preprocessing Pipeline**: Implements operations such as Grayscale Conversion, Standardization, and Contrast Limited Adaptive Histogram Equalization (CLAHE).
- **Color & Texture Indices**: Extracts and visualizes Excess Green (ExG), Excess Red (ExR), Canny edge contours, and Local Binary Patterns (LBP).
- **Dynamic Leaf Segmentation**: Utilizes a combination of color thresholds and morphological mapping to accurately separate the primary leaf from any noisy background.
- **Phenotypic Trait Extraction**: Automatically calculates critical morphological features (Area, Perimeter, Eccentricity, Solidity, Extent, Aspect Ratio) and color distribution profiles (RGB Mean and Standard Deviation) for analysis.
- **Deep Learning Classification**: Integrates a PyTorch ResNet-18 model capable of accurately diagnosing over 40 distinct plant health classes (including Apple Scab, Corn Leaf Blight, Peach Healthy, etc). 
- **Interactive UI**: A Streamlit frontend displays side-by-side processing outputs, a traits dataframe, and disease detection confidence metrics.
- **Data Persistence**: Records every phenotyping session and its corresponding geometric/color traits into an SQLite database (`phenotyping_results.db`).

---

## File Structure

- `app.py`: The main Streamlit dashboard application routing the front-end components.
- `preprocess.py`: Contains the raw OpenCV and `scikit-image` logic for transforming the images and extracting geometric phenotypes.
- `model_inference.py`: Wraps the PyTorch architecture into a class capable of translating image tensors into disease labels based on the weights.
- `dataset.py`: Defines a custom recursive PyTorch `Dataset` loader intended to dynamically parse a hierarchical directory structure of `Split/Species/Disease/image.jpg`.
- `train.py`: A complete hardware-accelerated (CUDA) training script loop designed to process the dataloaders and train the ResNet network, saving the weights and loss history.
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
*(Note: If you have an NVIDIA GPU, you may want to install the CUDA-enabled version of PyTorch manually from the [PyTorch website](https://pytorch.org/) to accelerate model training).*

Link for the dataset-  https://www.kaggle.com/datasets/asheniranga/leaf-disease-dataset-combination?select=image+data

### 3. Provide Model Weights
Because deep learning models are large, the trained `.pth` structural weights file is not included in the repository. Provide your trained model named `disease_model.pth` in the root of the project directory.

Alternatively, if you possess the raw hierarchical `image data/` folder, you can run the training generation script yourself:
```bash
python train.py
```
This will automatically parse your data, train the ResNet network across epochs, and drop the `disease_model.pth` and `class_names.txt` outputs into your root directory!

### 4. Launch the Dashboard
With the dependencies mapped and the model weights present, you can start the local web server:
```bash
streamlit run app.py
```

Upload a leaf picture from your filesystem or capture one with your webcam using the sidebar. The application will compute the traits, render the sequential mask layers, run the deep neural network disease classifier, and log the output into the database!
