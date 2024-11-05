The source codes for subsequent tasks are located in the following directories:  
`ex1`, `ex2`, `ex3`, `ex4`, and `ex5`.

- The scripts `exN/scriptN.py` are programs that perform anomaly detection.
- Each of the above scripts uses functions that students need to write — their placeholders are located in the files `exN/solutionN.py`.
- Anomalies should be assigned class 1, while normal examples should be assigned class 0.
- The file `utils.py` contains, among other things, a useful function `binary2neg_boolean` for converting (with negation) binary results -1/1 to 0/1.
- To pass the class, you must:
  - Completed files `exN/solutionN.py`
  - A report containing:
    - The achieved F1-scores
    - The obtained graphs
    - Observations/conclusions/answers to the questions asked.


**Ex 1**  
**1D Statistical Model**

1. Assume the type of statistical distribution for the training data.
2. Estimate its parameters based on the training data.
3. Determine the anomaly detection threshold relative to the calculated statistical distribution parameters.
4. Calculate detection results for the test data.

**Ex 2**
**2D Statistical Model**

1. Estimate the covariance matrix of the statistical distribution of the training data, e.g., using the `sklearn.covariance.MinCovDet` package.
2. Calculate the maximum distance of the training examples from the expected value (mean) using the Mahalanobis metric — see: `MinCovDet.mahalanobis()`.
3. Compute the Mahalanobis distances for the test examples and determine if they are outliers based on these distances.

**Ex 3**
**Statistical Model vs. OC-SVM**

1. Perform anomaly detection on the test set using the Mahalanobis distance, as in Task 2 (reuse the function).
2. Compare the results with those obtained on the test set using the OneClass-SVM algorithm: `sklearn.svm.OneClassSVM`. Use an appropriate kernel.
3. Comment on the results — what tendencies do these algorithms have? In what situations are they suitable?
4. How does manipulating the parameters of the OC-SVM algorithm affect the results?

**Ex 4**
**Learning in the Presence of Anomalies**  
*Comparison of selected unsupervised algorithms on a dataset containing anomalies.*  
The ratio of anomalies to the size of the training set is known.

1. Covariance estimation and Mahalanobis distance — a ready-made implementation can be used: `sklearn.covariance.EllipticEnvelope`
2. OneClass-SVM: `sklearn.svm.OneClassSVM`
3. Isolation Forest: `sklearn.ensemble.IsolationForest`
4. Local Outlier Factor: `sklearn.neighbors.LocalOutlierFactor`

Ad 1. Use each of the methods listed.
Ad 2. Comment on the choice of parameters for each method — what do they change?
Ad 3. Test on the training set — provide the results.
Ad 4. Compare the behavior of each method, i.e., what tendencies they have and what they are suitable for.

**Ex 5**
**Normal Examples and Anomalies**

Normal examples are records from the MNIST handwritten digits database.  
The test set consists of records from MNIST (considered as the nominal class) and the Fashion MNIST dataset, containing images of clothing (considered as unknown anomalies at the training stage).

1. Propose a measure for reconstruction quality (`ex5/solution5/reconstruction_errors()`).
2. Propose a method for determining the reconstruction error threshold indicating an anomaly (`ex5/solution5/calc_threshold()`).
3. Detect anomalies based on this threshold (`ex5/solution5/predict()`).
4. Discuss the reconstruction error histograms for both datasets.
5. Describe other observations related to the experiment.

**Optional:**

- Investigate the effect of increasing the number of neurons in the hidden (latent) layer on the effectiveness of anomaly detection.
- Investigate the effect of adding more layers to the autoencoder on the effectiveness of anomaly detection.
