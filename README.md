# Classification with Cross Validation GUI

A Python desktop application that provides an intuitive graphical interface for performing machine learning classification tasks with cross-validation. The application supports multiple datasets and classification algorithms, making it easy to experiment with different classification approaches and visualize the results.

## Features

- **User-friendly GUI** for dataset and classifier selection
- **Built-in Datasets**: 
  - Iris
  - Breast Cancer
  - Wine
- **Classification Algorithms**:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Gaussian mixture model (GMM)
- **Automated Parameter Optimization** using k-fold cross-validation
- **Rich Visualizations**:
  - Parameter tuning plots
  - Confusion matrix heatmaps
  - Accuracy metrics
<img width="1512" alt="Breast Cancer- GMM" src="https://github.com/user-attachments/assets/614baeac-f237-47fd-b9e5-dea715fff963" />

## Usage Guide

1. **Select Dataset**: Choose from one of the built-in datasets using the radio buttons
2. **Choose Classifier**: Select either KNN or SVM classification algorithm
3. **Run Analysis**: Click "Run Classification" to start the process
4. **View Results**: 
   - Parameter optimization plot shows cross-validation results
   - Confusion matrix visualizes classification performance
   - Accuracy score displays overall performance metric

## How It Works

The application follows a streamlined classification workflow:

1. **Data Preparation**:
   - Automatic 80/20 train-test split
   - Feature scaling where appropriate

2. **Parameter Optimization**:
   - Implements 5-fold cross-validation
   - Automatically selects optimal parameters:
     - K value for KNN
     - C parameter for SVM
     - BIC score for GMM

3. **Model Training**:
   - Trains final model using best parameters
   - Validates on held-out test set

4. **Results Visualization**:
   - Real-time plotting of results
   - Interactive confusion matrix display

## Dependencies

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Tkinter
- Seaborn

