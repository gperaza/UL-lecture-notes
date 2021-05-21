# Dimensionality Reduction and PCA

## Introduction

Feature extraction refers to a set of techniques to build derivative features from the original features present in the data set. The new features can be linear or non-linear combinations of the original ones. The new features aim to capture the patterns on the data, while removing noise and redundancy, improving the performance of learning algorithms. Also, some algorithms work better with particular representations of a data set.

In the case where the number of extracted features is less than the number of original features, we are said to be performing dimensionality reduction. A reduced number of dimensions alleviates the **curse of dimensionality**, improving learning performance, as well as reducing the size of data set, improving computation time.

## Principal Component Analysis (PCA)

The most popular method for feature extraction is Principal Component Analysis (PCA). In simple terms, PCA searches for a rotation (orthogonal linear transformation) of the axes in data space (features) in which the transformed features are uncorrelated, i.e., the rotated covariance matrix is diagonal. The diagonal entries of the covariance matrix are, then, the variance along each of the new axes. Finally, dimensionality reduction is performed by dropping the features with the lowest variance, under the assumption that those are the least informative, or contribute less to the reconstruction error.

```{raw} html
<script src="pca-rot.js" id="65f7bd09-d4f1-469e-b724-4de2189c2e59"></script>
```
There are several ways to justify that specific transformation.

## References

```{bibliography}
:style: unsrt
:filter: docname in docnames
```
