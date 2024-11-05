import numpy as np


def reconstruction_errors(inputs: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """Calculate reconstruction errors.

    :param inputs: Numpy array of input images
    :param reconstructions: Numpy array of reconstructions
    :return: Numpy array (1D) of reconstruction errors for each pair of input and its reconstruction
    """
    pass


def calc_threshold(reconstr_err_nominal: np.ndarray) -> float:
    """Calculate threshold for anomaly-detection

    :param reconstr_err_nominal: Numpy array of reconstruction errors for examples drawn from nominal class.
    :return: Anomaly-detection threshold
    """
    pass


def detect(reconstr_err_all: np.ndarray, threshold: float) -> list:
    """Recognize anomalies using given reconstruction errors and threshold.

    :param reconstr_err_all: Numpy array of reconstruction errors.
    :param threshold: Anomaly-detection threshold
    :return: list of 0/1 values
    """
    pass
