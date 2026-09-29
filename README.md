# Schizophrenia Detection with EEGNet and Attention

A deep-learning project for binary schizophrenia classification from EEG recordings using an **EEGNet-based architecture with a temporal attention mechanism**.

## Project Overview

The original implementation was developed in Google Colab and works with EEG recordings stored in EDF format. The workflow is:

```text
EDF EEG recordings
        ↓
Load with MNE
        ↓
0.5–45 Hz band-pass filtering
        ↓
2-second non-overlapping windows
        ↓
Per-window StandardScaler normalization
        ↓
Flattened feature matrix + labels
        ↓
Train/Test split (80/20, stratified)
        ↓
EEGNet backbone
        ↓
Temporal attention
        ↓
Softmax classification
        ↓
Accuracy / AUC / Precision / Recall / F1 / Sensitivity / Specificity / MCC / Cohen's Kappa
```

## Model

The model in the supplied notebook contains:

- Temporal convolution
- Depthwise spatial convolution
- Batch normalization
- ELU activation
- Average pooling
- Dropout
- Second convolution block
- Attention scores generated with `Dense(1, activation='tanh')`
- Softmax attention across the temporal dimension
- Weighted context vector
- Final softmax classifier

## Preprocessing

The supplied notebook:

1. Loads EDF files with MNE.
2. Reads each file's actual sampling frequency.
3. Applies a Butterworth band-pass filter from **0.5 to 45 Hz**.
4. Creates **2-second non-overlapping windows** using the recording's actual sampling frequency.
5. Applies `StandardScaler` independently to each window.
6. Flattens each window and stores the result in `X_final_data.csv`.
7. Stores labels in `y_final_labels.csv`.

Control recordings are assigned label `0`; patient recordings are assigned label `1`.

## Training Configuration

The supplied notebook uses:

- Optimizer: Adam
- Learning rate: `1e-3`
- Epochs: `50` maximum
- Batch size: `32`
- Validation split: `0.2`
- Early stopping patience: `15`
- Best-model checkpoint monitored on `val_accuracy`
- Train/test split: `80/20`
- `random_state=42`
- Stratified splitting

## Evaluation

The notebook calculates:

- Accuracy
- ROC AUC
- Precision
- Recall
- F1 Score
- Sensitivity
- Specificity
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Confusion matrix
- Classification report

It also generates accuracy, loss, ROC and confusion-matrix visualizations.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│   └── schizophrenia_eeg_eegnet_attention.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
└── results/
    ├── README.md
    ├── figures/
    └── metrics/
```

## Dataset Setup

The raw EDF recordings are not included in this repository. Create:

```text
data/
├── controls/
└── patients/
```

and place the corresponding EDF recordings in each folder.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the processed CSV files:

```bash
python src/preprocessing.py
```

Train the model:

```bash
python src/train.py
```

Evaluation utilities are available in `src/evaluate.py`.

## Important Note

The notebook's preprocessing creates windows using each EDF file's actual sampling frequency, while the later reshape/training section uses a fixed `SAMPLING_FREQUENCY = 250`. This repository preserves that behavior from the supplied notebook; verify that the recordings used for training are compatible with this assumption before relying on the modular scripts for a fresh run.

## Notebook

The original notebook is retained under `notebooks/` so the repository preserves the original experimental workflow and results.
