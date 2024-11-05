import numpy as np
from sklearn.covariance import MinCovDet

def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:

    # Fit robust covariance estimator
    robust_cov = MinCovDet(random_state=0)
    robust_cov.fit(train_data)
    
    # Calculate Mahalanobis distances for training data
    train_mahal_dist = robust_cov.mahalanobis(train_data)
    
    # Set threshold as the 97.5th percentile of training distances
    # This is a common choice that allows for some outliers in training data
    threshold = np.percentile(train_mahal_dist, 97.5)
    
    # Calculate Mahalanobis distances for test data
    test_mahal_dist = robust_cov.mahalanobis(test_data)
    
    # Make predictions (1 for anomalies, 0 for normal)
    predictions = (test_mahal_dist > threshold).astype(int)
    
    return predictions.tolist()
