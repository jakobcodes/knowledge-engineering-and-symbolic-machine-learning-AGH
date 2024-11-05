from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from utils import binary2neg_boolean
import numpy as np

SEED = 1


def detect_cov(data: np.ndarray, outliers_fraction: float) -> list:
    pass


def detect_ocsvm(data: np.ndarray, outliers_fraction: float) -> list:
    pass


def detect_iforest(data: np.ndarray, outliers_fraction: float) -> list:
    pass


def detect_lof(data: np.ndarray, outliers_fraction: float) -> list:
    pass
