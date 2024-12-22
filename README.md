# Potato Leaf Disease Detection using Dense Convolutional Neural Networks (D-CNNs)

This repository contains the implementation of a Dense Convolutional Neural Network (D-CNN) for detecting potato leaf diseases. The project was developed as part of a **Computer Vision class**, where the goal was to pick a research paper and modify its methodology to improve the outcomes. 

## Main Paper
The project is based on the following paper:
- **Title**: Erlin, Indra Fuadi, Ramalia Noratama Putri, Dewi Nasien, Gusrianty, and Dwi Oktarina. *"Deep Learning Approaches for Potato Leaf Disease Detection: Evaluating the Efficacy of Convolutional Neural Network Architectures."* Revue d'Intelligence Artificielle, Vol. 38, No. 2, April 2024, pp. 717-727. DOI: [10.18280/ria.380236](https://doi.org/10.18280/ria.380236).

### Modifications Made
In this project, the standard Convolutional Neural Network (CNN) architectures discussed in the main paper were modified by introducing a Dense Convolutional Neural Network (D-CNN). This modification enhances:
- **Feature Propagation**: By introducing dense connections between layers, the model reuses features and improves overall learning efficiency.
- **Mitigation of Vanishing Gradient Problem**: Dense connections facilitate gradient flow during training, enabling deeper networks.

These enhancements significantly improved the model's ability to detect potato leaf diseases compared to the results in the main paper.

---

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup Instructions](#setup-instructions)
- [Results](#results)
- [References](#references)

---

## Introduction
Potato plants are vulnerable to diseases like early blight and late blight, which significantly impact agricultural productivity. This project proposes an automated solution for early and accurate disease detection using a Dense Convolutional Neural Network, aiming to improve crop management and food security.

## Key Features
- **Dense Blocks**: Improve feature reuse and gradient flow.
- **Transition Layers**: Reduce feature maps and dimensionality to prevent excessive model width.
- **Data Augmentation**: Enhances robustness to variations in leaf appearances.
- **High Accuracy**: Achieved 99% test accuracy.

## Dataset
The dataset is sourced from the [PlantVillage dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1) and contains 3,000 annotated images of potato leaves across three categories:
- **Potato___healthy**
- **Potato___Early_blight**
- **Potato___Late_blight**

### Data Split:
- Training Set: 70%
- Validation Set: 20%
- Test Set: 10%

## Model Architecture
The Dense Convolutional Neural Network (D-CNN) consists of:
1. **Input Layer**: Accepts images resized to `224x224x3`.
2. **Dense Blocks**: Connected layers improve feature propagation.
3. **Transition Layers**: Reduce spatial dimensions and feature maps.
4. **Global Average Pooling**: Reduces parameters.
5. **Output Layer**: Softmax activation for multi-class classification.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/potato-leaf-disease-detection-dcnn.git
   cd potato-leaf-disease-detection-dcnn
   ```

2. **Install Dependencies**:
   Install the required libraries using:
   ```bash
   pip install -r data/requirements.txt
   ```

3. **Prepare the Dataset**:
   - Inside the `data` folder, open `data.txt` and use the provided Google Drive link to download the dataset.
   - The dataset includes three folders: `train`, `val`, and `test`.
   - Once downloaded, update the dataset paths in the `model.ipynb` notebook to reflect the location where you saved these folders.


4. **Update Configurations**:
   Modify parameters in `config.yaml` such as:
   - `input_shape`
   - `num_classes`
   - `dropout_rate`
   - `batch_size`
   - `epochs`
   - `learning_rate`

5. **Run the Model**:
   - Open `model.ipynb` in Jupyter Notebook or any compatible environment.
   - Follow the steps in the notebook to:
     - Load the dataset.
     - Train the Dense Convolutional Neural Network (D-CNN).
     - Evaluate the model's performance on the test set.

5. **Monitor Training**:
   - Training and validation accuracy/loss will be displayed **inside the notebook** during execution of the training cell.

  
6. **Evaluate the Model**:
   - To generate classification reports and confusion matrices, locate the specific evaluation cell in `model.ipynb` and run it after training.

8. **Modify Hyperparameters and Re-run**:
   - Experiment with different settings in `config.yaml`.

## Results
- **Training Accuracy**: 98.47%
- **Validation Accuracy**: 99.00%
- **Test Accuracy**: 99.00%

### Classification Metrics:
| Class               | Precision | Recall | F1-Score |
|---------------------|-----------|--------|----------|
| Potato___healthy    | 1.00      | 0.98   | 0.99     |
| Potato___Early_blight | 1.00    | 0.98   | 0.99     |
| Potato___Late_blight  | 0.96    | 1.00   | 0.98     |

### Confusion Matrix:
Minimal misclassifications were observed, mostly between **early blight** and **late blight**.

## References
1. Erlin, Indra Fuadi, et al. "Deep Learning Approaches for Potato Leaf Disease Detection: Evaluating the Efficacy of Convolutional Neural Network Architectures." Revue d'Intelligence Artificielle, Vol. 38, No. 2, April 2024, pp. 717-727. DOI: [10.18280/ria.380236](https://doi.org/10.18280/ria.380236).

2. Hughes, D.P., & Salathé, M. "An open access repository of images on plant health to enable the development of mobile disease diagnostics." arXiv preprint arXiv:1511.08060. Available at: [PlantVillage dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1).

