# Deep learning for atrial fibrillation diagnosis and phenotyping from the electrocardiogram time series

---
##  **IMPORTANT: Repository Branches** 

This repository maintains **two separate branches** for different use cases:

### **`main` Branch (Open Source)**
- **Purpose**: Open-source code for research, development, and training
- **Contents**: Full source code, training scripts, and model architecture
- **Model Format**: Trained models saved as `.pkl` files

### **`release` Branch (Deployment)**
- **Purpose**: Production-ready deployment with TensorFlow SavedModel
- **Contents**: Pre-trained exported SavedModel ready for inference
- **Model Format**: TensorFlow SavedModel format

**To switch branches:**
```bash
# For open-source development and training
git checkout main

# For production deployment
git checkout release
```

---

This README outlines the steps to set up the environment, install dependencies, and provide acess to both ArNet2 for AF detection ([LINK](https://github.com/aim-lab/Shany_Repo/tree/main/plug-and-play)) and phenotyping AF ([LINK]()), both developed as part of this work.

ArNet2 is a deep learning algorithm developed to detect AF events from windows/long-term beat-to-beat interval data. AF is the most common arrhythmia, linked to a significantly higher risk of stroke and increased mortality. Early and accurate detection of AF is crucial for timely intervention, risk assessment, and personalized treatment strategies.

For more details and to cite this work, please refer to our paper:

```
Biton, Shany, et al. "Generalizable and robust deep learning algorithm for atrial fibrillation diagnosis across geography, ages and sexes." NPJ Digital Medicine 6.1 (2023): 44.
```

## 1. Project Overview
This repository contains the implementation of ArNet2, a DL model for AF
detection using RR intervals from ECG recordings (Holter or 12-lead). 
It includes preprocessing scripts, model training pipelines, and inference tools.

## 2. Repository Structure 
```
Shany_Repo/
│
├── ArNet2/          # Core implementation of the ArNet2 architecture and model dependencies
├── parsing/         # Unified source code for database parsing used in this project
├── plug-and-play/   # Implementation and scripts for running ArNet2 in plug-and-play mode
├── utils/           # Utility functions and shared dependencies used across the project
│
├── README.md        # Project overview and usage documentation (this file)
├── pyproject.toml   # Poetry environment and dependency configuration
└── LICENSE          # License information (CC BY-NC 4.0)
```

## 3. Installation
Follow these steps to set up your environment and install dependencies.

### 3.1 Create Conda Environment + Poetry

1. Navigate to the project directory where the environment.yml file is located.

2. Create the Conda environment:
```
conda create -n arnet2 python=3.10 -y
```
3. Activate the environment:
```
conda activate arnet2
```

### 3.2 Install Additional Dependencies Using Poetry
1. Install Poetry (if not already installed):
```
pip install poetry
```
2. Navigate to the project directory where the `pyproject.toml` file is located.
3. Install dependencies:
```
poetry install
```

### 3.3 Verify TensorFlow and CUDA Compatibility

Ensure that your system meets the following requirements for TensorFlow 2.12 with GPU support:

Python Version: 3.8, 3.9, or 3.10

- TensorFlow Version: 2.12.*
- CUDA Version: 12.*

To verify the installation:
```angular2html
import tensorflow as tf
print(tf.__version__)  # Should print 2.12.*
print(tf.config.list_physical_devices('GPU'))  # Should list available GPUs
```

## 4. Setting `PYTHONPATH`

To ensure that Python can access `ArNet2` directory during runtime, you need to set the `PYTHONPATH` environment variable.

Run the following command in your terminal to set the `PYTHONPATH` variable with the necessary directory:

```angular2html
export PYTHONPATH="/root/folder/of/repo/:/root/folder/of/repo/ArNet2/src/models:$PYTHONPATH"
```

Note: Replace `/root/folder/of/repo` with the actual path to the repo directory

## 5. Usage

- **Run ArNet2 for AF detection**: Follow [readme](https://github.com/aim-lab/Shany_Repo/blob/main/plug-and-play/README.MD) under plug-and-play.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). For more details check the [LICENSE](https://github.com/aim-lab/Shany_Repo/blob/main/LICENCE.md).
