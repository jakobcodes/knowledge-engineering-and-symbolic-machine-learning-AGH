import numpy as np


def reconstruction_errors(inputs: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """Calculate reconstruction errors.

    :param inputs: Numpy array of input images
    :param reconstructions: Numpy array of reconstructions
    :return: Numpy array (1D) of reconstruction errors for each pair of input and its reconstruction
    """
    # Calculate Mean Squared Error (MSE) for each image
    # We use mean() along axis=1 to get average error per image
    return np.mean((inputs - reconstructions) ** 2, axis=1)


def calc_threshold(reconstr_err_nominal: np.ndarray) -> float:
    """Calculate threshold for anomaly-detection

    :param reconstr_err_nominal: Numpy array of reconstruction errors for examples drawn from nominal class.
    :return: Anomaly-detection threshold
    """
    # Calculate threshold as mean + 2 * standard deviation
    # This is a common statistical approach that covers ~95% of the normal distribution
    mean = np.mean(reconstr_err_nominal)
    std = np.std(reconstr_err_nominal)
    return mean + 2 * std


def detect(reconstr_err_all: np.ndarray, threshold: float) -> list:
    """Recognize anomalies using given reconstruction errors and threshold.

    :param reconstr_err_all: Numpy array of reconstruction errors.
    :param threshold: Anomaly-detection threshold
    :return: list of 0/1 values
    """
    # Return 1 for anomalies (errors above threshold) and 0 for normal examples
    return [1 if err > threshold else 0 for err in reconstr_err_all]
