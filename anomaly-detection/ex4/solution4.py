from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from utils import binary2neg_boolean
import numpy as np


def detect_cov(data: np.ndarray, outliers_fraction: float) -> list:

    detector = EllipticEnvelope(
        contamination=outliers_fraction,
        support_fraction=0.9  # Use 90% of points for covariance estimation
    )
    
    predictions = detector.fit_predict(data)
    return binary2neg_boolean(predictions)

def detect_ocsvm(data: np.ndarray, outliers_fraction: float) -> list:

    detector = svm.OneClassSVM(
        kernel='rbf',
        nu=outliers_fraction,
        gamma='scale',  # Better automatic scaling of the RBF kernel
    )
    
    predictions = detector.fit_predict(data)
    return binary2neg_boolean(predictions)

def detect_iforest(data: np.ndarray, outliers_fraction: float) -> list:

    detector = IsolationForest(
        contamination=outliers_fraction,
        n_estimators=100,
        max_samples='auto',  # Automatically determine samples per tree
    )
    
    predictions = detector.fit_predict(data)
    return binary2neg_boolean(predictions)

def detect_lof(data: np.ndarray, outliers_fraction: float) -> list:

    detector = LocalOutlierFactor(
        contamination=outliers_fraction,
        n_neighbors=500,  # Increased significantly for better density estimation
        metric='minkowski',  # Using Minkowski distance
        p=1,  # Using Manhattan distance (p=1) instead of Euclidean (p=2)
        leaf_size=20,  # Reduced for more precise neighbor searches
        novelty=False,
        algorithm='auto'  # Let sklearn choose the best algorithm
    )
    
    predictions = detector.fit_predict(data)
    return binary2neg_boolean(predictions)
