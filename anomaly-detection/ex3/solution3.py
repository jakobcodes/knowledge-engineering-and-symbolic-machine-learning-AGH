from sklearn import svm
from utils import binary2neg_boolean
import numpy as np


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    """Detect anomalies using One-Class SVM
    
    :param train_data: Training data with only nominal samples (no anomalies)
    :param test_data: Test data containing both nominal and anomaly samples
    :return: Binary labels for test_data samples (0 = nominal, 1 = anomaly)
    """
    # Initialize One-Class SVM with tuned parameters:
    # - nu=0.01: Expect about 1% outliers in training data
    # - kernel='rbf': Radial Basis Function kernel for non-linear boundary
    # - gamma=0.1: Larger gamma means closer fit to training data
    ocsvm = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=0.1)
    
    # Train the model on nominal data
    ocsvm.fit(train_data)
    
    # Get predictions for test data
    # OneClassSVM returns:
    # 1 for inliers (nominal)
    # -1 for outliers (anomalies)
    predictions = ocsvm.predict(test_data)
    
    # Convert predictions to required format:
    # -1 -> 1 (anomaly)
    # 1 -> 0 (nominal)
    predictions = binary2neg_boolean(predictions)
    
    return predictions
