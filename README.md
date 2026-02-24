# AFtoolkit: Deep Learning for Atrial Fibrillation Detection and Phenotyping

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.12](https://img.shields.io/badge/tensorflow-2.12-orange.svg)](https://www.tensorflow.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

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

A comprehensive deep learning framework for **atrial fibrillation (AF) detection** and **circadian phenotype classification** from ECG-derived RR interval time series.

## Features

- **AF Detection**: Deep neural network models for detecting AF from beat-to-beat (RR) intervals or (RR-based) or raw ECG. 
- **AF Phenotyping**: Data-driven clustering to identify circadian AF subtypes with prognostic significance
- **Plug-and-Play Models**: Pre-trained models ready for inference on Holter or 12-lead ECG data
- **Generalizable**: Validated across diverse populations, geographies, ages, and sexes

## Modules

| Module | Description                                                             |
|--------|-------------------------------------------------------------------------|
| [**detection/**](./detection/README.MD) | AF detection from RR intervals or raw ECG data                          |
| [**phenotyping/**](./phenotyping/README.MD) | Hierarchical clustering for AF chronophenotype discovery and prediction |

---

## Background

### AF Detection

This module provides deep learning algorithms for detecting **atrial fibrillation events** from short-window and long-term ECG data extracted from ECG recordings.

- **ArNet2**: A generalizable and robust deep learning model based on **beat-to-beat interval (RR interval)** validated across geography, ages, and sexes (Biton et al., NPJ Digital Medicine 2023)

- **ArNetECG**: A deep learning model for AF detection from **raw single-lead ECG** (30 s windows @ 200 Hz), leveraging both morphological and rhythm information; validated on large-scale ECG datasets (Ben-Moshe et al., IEEE 2024)

AF is the most common cardiac arrhythmia, affecting over 40 million people worldwide and linked to a **5-fold increased risk of stroke** and elevated mortality. Early and accurate AF detection is crucial for:
- Timely clinical intervention
- Stroke risk assessment (CHA₂DS₂-VASc scoring)
- Personalized anticoagulation therapy

### AF Phenotyping

AF Phenotyping is a data-driven framework for identifying **circadian subtypes of paroxysmal AF** based on temporal burden patterns from 24-hour Holter recordings. Using hierarchical clustering of hourly AF burden profiles, this approach reveals **five distinct chronophenotypes**:

| Chronophenotype | Peak AF Time | Clinical Relevance |
|-----------------|--------------|-------------------|
| **Nocturnal-to-Morning** | 00:00–08:00 | Vagally-mediated AF pattern |
| **Evening-to-Early Morning** | 18:00–04:00 | Extended nocturnal burden |
| **Daytime** | 08:00–18:00 | Adrenergically-mediated AF |
| **Persistent AF** | All hours | High overall burden |
| **Non-AF** | None | Minimal/no AF detected |

These findings suggest that **circadian manifestation patterns carry prognostic relevance** for AF management and risk stratification.

---

## Citations

If you use this code, please cite our papers:

### AF Detection 
#### (ArNet2)
```bibtex
@article{biton2023generalizable,
  title={Generalizable and robust deep learning algorithm for atrial fibrillation diagnosis across geography, ages and sexes},
  author={Biton, Shany and others},
  journal={NPJ Digital Medicine},
  volume={6},
  number={1},
  pages={44},
  year={2023},
  publisher={Nature Publishing Group}
}
```

#### (ArNetECG)
```bibtex
@article{10538381,
  author={Ben-Moshe, Noam and Tsutsui, Kenta and Brimer, Shany Biton and Zvuloni, Eran and Sörnmo, Leif and Behar, Joachim A.},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={RawECGNet: Deep Learning Generalization for Atrial Fibrillation Detection From the Raw ECG}, 
  year={2024},
  volume={28},
  number={9},
  pages={5180-5188},
  keywords={Electrocardiography;Recording;Detectors;Rhythm;Training;Deep learning;Data models;Atrial fibrillation;atrial flutter;deep learning;electrocardiogram},
  doi={10.1109/JBHI.2024.3404877}}
```

### AF Phenotyping
```bibtex
@article{brimer2025temporal,
  title={Temporal Phenotyping of Paroxysmal Atrial Fibrillation Reveals Prognostic Circadian Subtypes},
  author={Brimer, Shany and others},
  journal={Machine Learning for Health},
  year={2025}
}
```

---

## Repository Structure

```
aftoolkit/
│
├── detection/       # AF detection pipeline with pre-trained models
├── phenotyping/     # AF chronophenotype clustering and prediction
├── utils/           # Shared utility functions
│
├── README.md        # This file
├── pyproject.toml   # Poetry dependency configuration
└── LICENSE          # CC BY-NC 4.0 License
```

---

## Installation

### Step 1: Create Conda Environment

```bash
conda create -n aftoolkit python=3.10 -y
conda activate aftoolkit
```

### Step 2: Install Dependencies with Poetry

```bash
pip install poetry
poetry install
```

### Step 3: Verify GPU Support (Optional)

For TensorFlow 2.12 with GPU acceleration, ensure CUDA 12.x is installed:

```python
import tensorflow as tf
print(tf.__version__)  # Should print 2.12.*
print(tf.config.list_physical_devices('GPU'))  # Lists available GPUs
```

---

## Setting PYTHONPATH

### For Detection:
```bash
# Linux/Mac
export PYTHONPATH="/path/to/repo/:$PYTHONPATH"

# Windows (PowerShell)
$env:PYTHONPATH = "C:\path\to\repo\;$env:PYTHONPATH"
```

### For Phenotyping:
```bash
# Linux/Mac
export PYTHONPATH="/path/to/repo/phenotyping:$PYTHONPATH"

# Windows (PowerShell)
$env:PYTHONPATH = "C:\path\to\repo\phenotyping;$env:PYTHONPATH"
```

---

## Usage

| Task | Documentation |
|------|---------------|
| **AF Detection** | [detection/README.MD](./detection/README.MD) |
| **Chronophenotype Discovery** | [phenotyping/README.MD](./phenotyping/README.MD) |

---

## Keywords

`atrial fibrillation` · `AF detection` · `arrhythmia detection` · `ECG analysis` · `electrocardiogram` · `RR intervals` · `heart rate variability` · `HRV` · `Holter monitoring` · `deep learning` · `neural network` · `TensorFlow` · `cardiac arrhythmia` · `AF phenotyping` · `circadian rhythm` · `chronophenotype` · `clustering` · `machine learning` · `digital health` · `cardiology AI`

---

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free for academic and non-commercial use.

See [LICENSE](./LICENCE.md) for details.
