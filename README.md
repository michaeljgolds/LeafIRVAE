# LeafIRVAE
Impulse response cVAE-GAN

## Requirements
+Tensorflow >= 2.0
+Matplotlib
+Numpy
+Scikit-learn

Database of leaf impulses and labels can be found here: https://dx.doi.org/10.6084/m9.figshare.24437920

## Files
+RealLeafTypeClassifier.py:

   This script makes a simple CNN classifier to distinguish the IRs by leaf species.
+CVG-All-Beta.py:

   This is the main model file. Running the script will make a model and train it on the real IRs. Uncommenting the last line make a file with sample generated IRs that can be used by the azimuth regressor.
+azimuthRegressor.py:

   This script makes two single layer dense network regressor that predicts the azimuth angles of generated and real IRs, then tests the networks. It outputs the errors in a mat file.
