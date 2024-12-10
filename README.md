# SCOPE

**Scientific Chart and Output Parsing Engine (SCOPE)**

Welcome to **SCOPE**, a framework and research initiative designed to detect, parse, and extract data from scientific charts and graphs within research publications. SCOPE (based on DeepRule) integrates multiple open-source components (such as CornerNet and MS COCO APIs) and leverages deep learning, computer vision, and OCR techniques. Its main goal is to serve as a foundation for a larger-scale system (REALM) that can automatically extract structured data from figures in scientific documents, identify key insights, and highlight “negative” or “insignificant” results.

This repository provides instructions for setting up the environment, installing dependencies, compiling necessary components, and running both training and inference pipelines. It also covers how to access and use the CHARTEX dataset, incorporate OCR, and extend the system’s functionalities.

---

## Project Overview

**DeepRule** (part of the SCOPE initiative) focuses on extracting data from charts and graphs within scientific PDFs. By leveraging deep learning and OCR, DeepRule converts visual information into structured, machine-readable data. Currently, it supports bar, line, and pie charts, with plans to expand to additional figure types and more complex scientific visualizations.

---

## Getting Started

### Prerequisites

- **Operating System:**  
  Linux-based system (e.g., Ubuntu 20.04+) recommended.

- **Hardware Requirements:**  
  - **GPU:** NVIDIA GPU with CUDA support  
  - **RAM:** Minimum 16 GB recommended  
  - **Disk Space:** At least 10 GB for datasets and dependencies

- **Python Version:**  
  Python 3.10.9 recommended.

- **Additional Tools:**  
  Poppler and `pdf2image` for converting PDF pages to images.

### Environment Setup

1. **Anaconda Installation:**  
   Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage Python environments.

2. **Conda Environment Creation:**  
   Use the provided `DeepRule.txt` file:
   ```bash
   conda create --name DeepRule --file DeepRule.txt
   source activate DeepRule
   
If this doesn't work due to outdated dependencies, try `requirements-2023.txt` or you may have to manually adjust packages

### GPU and CUDA Requirements

- **Ensure CUDA is installed:**
    ```bash
    nvcc --version
- **Verify GPU status:**
   ```bash
    nnvidia-smi

---

## Compiling Required Components

DeepRule relies on external modules like CornerNet and NMS (from Faster R-CNN and Soft-NMS).

### Compiling Corner Pooling Layers
Refer to the [CornerNet repository](https://github.com/princeton-vl/CornerNet):
   ```bash
    cd <CornerNet dir>/models/py_utils/_cpools/
    python setup.py build_ext --inplace 
   ```
If warnings or errors occur, ensure the required compilers (`gcc`, `g++`) and CUDA are installed.

### Compiling NMS
Compile NMS code (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/cpu_nms.pyx) and [Soft-NMS](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx):
   ```bash
    cd <CornerNet dir>/external
    make
   ```
If errors occur, ensure `make` and development tools are installed properly.

### Installing MS COCO APIs
   ```bash
   pip install pycocotools
   ```
Should just require a simple `pip` install, but if this is causing trouble, install using this method
   ```bash
   cd <CornerNet dir>/data
git clone git@github.com:cocodataset/cocoapi.git coco
cd <CornerNet dir>/data/coco/PythonAPI
make
   ```

---

## Datasets

### Downloading CHARTEX Data

Download the [CHARTEX](https://huggingface.co/datasets/asbljy/DeepRuleDataset/tree/main) dataset from Hugging Face.
Unzip the files into your data directory (e.g., `/workspace/data`).

### Data Description
Pie Data Example:
   ```json
   {"image_id": 74999, "category_id": 0, "bbox": [135.0, 60.0, 132.0, 60.0, 134.0, 130.0],   "area": 105.0263, "id": 433872}
   ```
- `bbox = [center_x, center_y, edge_1_x, edge_1_y, edge_2_x, edge_2_y]`

Line Data Example:
   ```json
 {"image_id": 120596, "category_id": 0, "bbox": [137.0, 131.0, 174.0, 113.0, 210.0, 80.0, 247.0, 85.0], "area": 0, "id": 288282}
   ```
- `bbox` represents data points `[d_1_x, d_1_y, …, d_n_x, d_n_y]`

Bar and Cls Data:
- Bounding boxes represent bars or classified components (titles, legends, etc.).

---

## OCR API

Originially, OCR functionalities used AZURE OCR service. If unavailable (or you wish to not pay), use `pytesseract`:

1. Install `pytesseract` and Tesseract OCR.
2. Rewrite the `ocr_result(image_path)` function to return:
- `word_info["text"]` for the recognized word
- `word_info["boundingBox"] = [top_left_x, top_left_y, bottom_left_x, bottom_left_y]`

---

## Environment Updates

Again, if you cannot install the original environment try `requirements-2023.txt` or manually adjust dependencies to match your Python/CUDA setup.

---

## Training and Evaluation

### Configuration Files and Models

- Place configuration JSON files (`<model>.json`) in `config/`.
- Place corresponding model files (`<model>.py`) in `models/`.

Examples:
- Bar Charts: `CornerNetPureBar`
- Pie Charts: `CornerNetPurePie`
- Line Charts: `CornerNetLine`
- Line Classification: `CornerNetLineClsReal`
- Classification: `CornerNetCls`

### Running Training
   ```bash
   python train_chart.py --cfg_file <model> --data_dir <data_path>

   # Example:
   python train_chart.py --cfg_file CornerNetBar --data_dir /home/data/bardata(1031)
   ```

### Server Mode
Run as a web server for on-deman inference:
   ```bash
   python manage.py runserver 8800
   ```
Access `http://localhost:8800` in a browser.

### Batch Testing
For batch testing (pre-assign chart type):
   ```bash
   python test_pipe_type_cloud.py --image_path <image_path> --save_path <save_path> --type <type>

    # Example:
    python test_pipe_type_cloud.py --image_path /data/bar_test --save_path save --type Bar
   ```

---

## Integration with OpenCV

OpenCV is used extensively within SCOPE for:

- Image preprocessing (resizing, cropping, normalization).
- Figure detection within PDFs converted to images.
- Drawing bounding boxes and annotations on extracted figures.
This ensures that the input data is properly formatted before being fed into deep learning models. Adjusting OpenCV parameters (e.g., thresholding, contour detection) can improve figure detection accuracy.

If needed, modify the OpenCV-based preprocessing functions in the provided scripts to handle different image qualities, resolutions, or chart types.

Further instructions are found in the comments of the jupyter notebook file.
