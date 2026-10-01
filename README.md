# H4H Hackathon

A Python-based health intelligence project focused on early risk detection using two complementary machine learning models:

- a lifestyle/clinical risk model for cardiovascular risk prediction
- an ECG-based arrhythmia detection model for abnormal heart rhythm classification

The repository contains the training scripts, processed datasets, and exported TensorFlow Lite model files used for experimentation and mobile-friendly inference.

Note: 
- For the demonstration app, I have only created A UI to display the ECG-based arrhythmia model working
- The app itself is just a showcase of what could be possible, since it uses an unused set of data from MIT-BIH Arrhythmia Database instead of collecting new data
## Overview

This project combines two approaches to heart health monitoring:

1. Lifestyle and medical feature model
   - Trains on patient-level health indicators such as age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol use, and activity.
   - Produces a binary risk prediction model for cardiovascular risk.
   - Exports a TensorFlow Lite model: `samsung_lifestyle_model.tflite`

2. ECG arrhythmia detection model
   - Reads ECG signal records from the MIT-BIH Arrhythmia Database.
   - Extracts heartbeat windows around annotated beats.
   - Trains a 1D Convolutional Neural Network to distinguish normal vs abnormal beats.
   - Exports a TensorFlow Lite model: `ecg_cnn_model.tflite`

## Repository structure

```text
H4H-Hackathon/
├── README.md
├── train_heart_model.py              # Lifestyle risk model training script
├── ecg_CNN.py                        # ECG CNN training script
├── cardio_base.csv                   # Raw lifestyle/clinical dataset
├── cardiac_failure_processed.csv     # Processed training dataset
├── samsung_lifestyle_model.tflite   # Exported lifestyle model
├── ecg_cnn_model.tflite             # Exported ECG model
├── .gitignore
├── .gitattributes
└── LICENSE (if added later)
```

## Key scripts

### `train_heart_model.py`

This script:

- loads `cardiac_failure_processed.csv`
- drops irrelevant columns such as ID and row index
- standardizes numerical input features
- trains a Keras neural network for binary classification
- evaluates performance using precision, recall, F1-score, and confusion matrix
- saves the final model as `samsung_lifestyle_model.tflite`

It also prints scaling values (`MEANS` and `SCALES`) intended for Android/mobile app integration.

### `ecg_CNN.py`

This script:

- loads annotated ECG records from a local MIT-BIH dataset path
- extracts heartbeats around annotation indices
- builds a 1D CNN for classifying normal vs abnormal beats
- trains the model and evaluates accuracy
- saves the result as `ecg_cnn_model.tflite`

## Data

### Lifestyle data

The repository includes:

- `cardio_base.csv`
- `cardiac_failure_processed.csv`

These files are used for training the risk prediction model. The processed dataset appears to contain patient-level features with a target column named `cardio`.

### ECG data

The ECG model expects a local copy of the MIT-BIH Arrhythmia Database. The script includes a dataset path configuration and references records such as:

- `100`, `106`, `109`, `118`, `119`, `124`, `200`, `202`, `210`, `213`, `214`, `219`

You must ensure the dataset is present locally before running `ecg_CNN.py`.

## Setup

### Prerequisites

- Python 3.9+
- pip
- TensorFlow
- NumPy
- pandas
- scikit-learn
- wfdb

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install tensorflow pandas numpy scikit-learn wfdb
```

## Usage

### Train the lifestyle risk model

```bash
python train_heart_model.py
```

This generates:

- `samsung_lifestyle_model.tflite`
- training metrics printed in the terminal

### Train the ECG model

Update the `DATA_PATH` variable in `ecg_CNN.py` to the location of your local MIT-BIH dataset, then run:

```bash
python ecg_CNN.py
```

This generates:

- `ecg_cnn_model.tflite`

## Notes

- The project is experimental and intended for learning, prototyping, and hackathon-style model exploration.
- The scripts are configured for local execution and export to TensorFlow Lite for deployment in lightweight environments such as mobile apps.
- The ECG model depends on a local external dataset, so the repository alone will not be enough to fully reproduce that pipeline without the MIT-BIH database files.

## Potential future enhancements

- add a `requirements.txt` file
- add a `notebooks/` folder for analysis and experimentation
- document the exact feature list used by the lifestyle model
- add model evaluation plots and confusion matrices saved to disk
- include deployment instructions for Android or embedded inference
- add a proper license

## License

No license is currently specified in this repository. If you intend to share or distribute the project publicly, consider adding an open-source license such as MIT or Apache 2.0.

## Contributing

This repository is suitable for hackathon or prototype-driven development. Contributions are welcome if you want to improve:

- model accuracy
- data preprocessing
- documentation
- deployment support
- code quality and reproducibility
