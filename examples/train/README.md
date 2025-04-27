
# LC-NICER Training Guide

This directory contains the training pipeline for the Lung Cancer Neo-adjuvant Immuno-Chemotherapy Response Predictor (LC-NICER).

## Overview

The training process involves multiple steps including feature extraction, feature selection, and model training. The main script `lc_nicer_train.py` orchestrates the entire training workflow.

## Prerequisites

- Python 3.9.18 environment with required dependencies (see main README.md)
- Properly formatted training data in the required directory structure
- CUDA-compatible GPU for deep learning feature extraction

## Training Pipeline

### 1. Feature Extraction

The system extracts three types of features:
- **Radiomics Features**: Conventional quantitative features from tumor regions
- **Deep Learning Features**: High-dimensional features extracted using the MAE3D foundation model
- **Habitat Imaging Features**: Novel features capturing tumor heterogeneity across sub-regions

### 2. Feature Selection and Processing

Features undergo a multi-stage selection process:
- Variance threshold filtering (removing low-variance features)
- Correlation-based selection (removing highly correlated features)
- Univariate statistical testing (Mann-Whitney U test)
- LASSO feature selection (identifying most predictive features)

### 3. Model Training

The system trains multiple prediction models:
- **$\text{LC-NICER}_\alpha$**: Pre-treatment model using only baseline features
- **$\text{LC-NICER}_\delta$**: Delta model incorporating longitudinal changes between pre/post treatment

## Running Training

```python
python lc_nicer_train.py
```

This single command will:
1. Generate habitat imaging features
2. Extract radiomics, deep learning, and habitat imaging features for both training and validation sets
3. Prepare the data structure for model training
4. Perform feature selection across all feature types
5. Merge selected features from different sources
6. Train and validate the LC-NICER models
7. Save trained models and normalization parameters for inference

## Output Structure

Trained models and feature selection records in `pkl/`

## Advanced Customization

Training parameters can be modified in the YAML configuration files:
- `tasks/rad_dl_hi_feat_{train|val}.yaml`: Feature extraction settings
- `tasks/hi_train.yaml`: Habitat imaging specific settings
- `tasks/dl_train_PCA.yaml`: Deep learning PCA settings
- `tasks/LC-NICER_train.yaml`: Main model training settings
