import numpy as np
from scipy import stats

def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:

    # 1. Assume normal distribution (Gaussian) for training data
    
    # 2. Estimate distribution parameters from training data
    mu = np.mean(train_data)
    sigma = np.std(train_data)
    
    # 3. Determine threshold using probability density function
    # Use 99.7th percentile (3-sigma rule) for the normal distribution
    threshold = stats.norm.ppf(0.997, mu, sigma)
    
    # 4. Calculate detection results
    # If probability of sample being from normal distribution is very low,
    # classify it as anomaly
    predictions = []
    for sample in test_data:
        if sample > threshold:
            predictions.append(1)  # anomaly
        else:
            predictions.append(0)  # normal
            
    return predictions
