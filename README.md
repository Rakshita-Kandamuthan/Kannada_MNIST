# Kannada MNIST

This repository contains Python code for performing classification on the Kannada MNIST dataset using Principal Component Analysis (PCA) for dimensionality reduction and various machine learning models. The code aims to analyze the performance of different classifiers with varying sizes of PCA components and visualize the Receiver Operating Characteristic (ROC) curves and ROC-AUC scores.

## Prerequisites
Before running the code, ensure that you have the following installed:

   - Python 3.x
   - **Required Python libraries**: scikit-learn, numpy, matplotlib


   You can install the required libraries using pip:
   ```
   !pip install scikit-learn 
   !pip install numpy 
   !pip install matplotlib
   ```

## Dataset

The code uses the Kannada MNIST dataset, which consists of handwritten digits in the Kannada script. The dataset is divided into training and testing sets, along with their corresponding labels. The dataset is loaded using the NumPy **np.load** function from the provided NPZ files.

## Principal Component Analysis (PCA)

PCA is performed to reduce the dimensionality of the dataset. The code applies PCA with different numbers of components, and for each size, it trains and evaluates multiple classifiers.

## Classification Models

The code trains and evaluates the following classifiers on the reduced feature set obtained from PCA:

   - Decision Tree Classifier
   - Random Forest Classifier
   - Naive Bayes Classifier (Gaussian Naive Bayes)
   - K-Nearest Neighbors (KNN) Classifier
   - Support Vector Machine (SVM) Classifier 


## Running the Code

1. Ensure that you have the required Python libraries installed as mentioned in the Prerequisites section.

2. Download the Kannada MNIST dataset and place the corresponding NPZ files in the appropriate directories. Modify the paths in the code to point to the correct locations of the training and testing data.

3. Run the code in a Python environment.

The code will perform PCA with different component sizes, train each classifier on the reduced feature set, and evaluate their performance. For each classifier and PCA component size, it will display the following:

   - Classification Report (including precision, recall, F1-score, and   support for each class)
   - Confusion Matrix
   - ROC-AUC Score 
   - ROC Curve

### Note

The performance of the classifiers may vary based on the PCA component size and the dataset. Experiment with different sizes and classifiers to achieve the best results.

If you use this code or find it helpful, consider giving credits and references to this repository.

Feel free to modify and adapt the code as per your specific requirements and datasets.

Happy coding!



