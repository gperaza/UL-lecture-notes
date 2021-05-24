# Dimensionality Reduction and PCA

## Introduction

Feature extraction refers to a set of techniques to build derivative features from the original features present in the data set. The new features can be linear or non-linear combinations of the original ones. The new features aim to capture the patterns on the data, while removing noise and redundancy, improving the performance of learning algorithms. Also, some algorithms work better with particular representations of a data set.

In the case where the number of extracted features is less than the number of original features, we are said to be performing dimensionality reduction. A reduced number of dimensions alleviates the **curse of dimensionality**, improving learning performance, as well as reducing the size of data set, improving computation time.

## Principal Component Analysis (PCA)

The most popular method for feature extraction is Principal Component Analysis (PCA). In simple terms, PCA searches for a rotation (orthogonal linear transformation) of the axes in data space (features) in which the transformed features are uncorrelated, i.e., the rotated covariance matrix is diagonal. The diagonal entries of the covariance matrix are, then, the variance along each of the new axes. Finally, dimensionality reduction is performed by dropping the features with the lowest variance, under the assumption that those are the least informative, or contribute less to the reconstruction error.

```{raw} html
<script src="pca-rot.js" id="65f7bd09-d4f1-469e-b724-4de2189c2e59"></script>
```
### PCA through variance maximization

There are several ways to justify that specific transformation. Let\'s to approach the problem directly, and find the set of orthogonal directions for which the variance is maximized. We will call this directions the **loading vectors**, the projections onto them the **principal components**, and we order them from larges variance to smallest. The first principal component is, thus, the direction along which the variance is larger.

To find the variance along an arbitrary direction $w$, we first take the projection of each centered observation along $w$, then take the variance. During this lecture, we will denote the centered data matrix by $X$ to simplify notation.

```{math}
\begin{align}
Var(w^T x) =& \frac{1}{n}\sum_i \left(w^T(x_i - \mu)\right)^{2}\\
=& \frac{1}{n}\sum_i w^T \tilde{x}_i \tilde{x}_i^T w\\
=& \frac{1}{n} w^T \sum_i\left(\tilde{x}_i \tilde{x}_i^T\right) w\\
=& \frac{1}{n} w^T X_c^T X_c w\\
=&  w^T \Sigma  w
\end{align}
```
So, to find the first principal component, we need to solve

$$
w^{*} = \underset{\mid w \mid=1}{\operatorname{argmax}}\ w^T\Sigma w
$$

Which is easier than it seems {cite}`brent_pca`. Let\'s assume $\Sigma$ is already diagonal, with entries $\lambda_i$ along the diagonal, and $\lambda_1 \geq \ldots \geq \lambda_d$. For any unit vector $w$

$$
w^T \Sigma w = \sum_{l,i,j} w_i x_{li} x_{lj} w_j
= \sum_l \lambda_{l} w_i^2 \leq \lambda_1\sum w_i^2 = \lambda_1 = e_1^T\Sigma e_1
$$

Then it\'s easy to see that $w^* = e_i$, the first basis versor. In the general case of a non-diagonal matrix $\Sigma$, we can always apply the eigen-decomposition

$$
\Sigma = W \Lambda W^T
$$

where $W$ is the orthogonal matrix with the eigenvectors of $\Sigma$ as columns, and $\Lambda$ is a diagonal matrix with . So, plugin the decomposition into the previous solution

$$
(e_1^T W^T) \Sigma (W e_1)
$$

from which we obtain the general loading $w^* = W e_1$, the first column of $W$, i.e., the first eigenvector of $\Sigma$.

For the next loading vectors and components, we add the restriction that the new loading must be perpendicular to the previously vectors.

$$
w_k = \underset{\mid w \mid=1, w \perp w_1,...,w_{k-1}}{\operatorname{argmax}}\ w^T\Sigma w
$$

Finding $w_{k}$ is analogous to finding $w_1$, working in the reduced space after removing the previous $k-1$, dimensions. So the solution must be also an eigenvector of $\Sigma$. In fact, the whole set of eigenvectors of $\Sigma$ define the directions of the principal components, and the eigenvalues are the variances along that very directions. As required, all the eigenvalues are positive, since the covariance matrix is positive semi-definite.

The principal components of a centered observation vector $x_c$ are the coordinates of that vector in the space spanned by the eigenvectors, $w_i^T x_c$, with the projected vector given by $W^T x_{c}$, and the variance along that direction given by the corresponding eigenvalue $\lambda$. The data matrix of the principal components is then the rotated centered data matrix,

$$
X_{PC} = X_c W
$$

To summarize, from an optimization point of view, we seek to maximize $w^T\Sigma w$ subject to the restriction $w^T w = 1$. Using Lagrange multipliers, we need to maximize $w^T\Sigma w - \lambda (w^T w - 1)$. Differentiating with respect to $w$ and equating the derivative to 0, we obtain

$$
\Sigma w - \lambda w = 0 \iff \Sigma w = \lambda w
$$

which is precisely the eigenproblem equation.

In the previous derivation we focused on PCA as a linear transformation that identifies directions of maximal variance. Next, we explore two derivations focusing on identifying useful sub-spaces to perform dimensionality reduction.

### Minimizing the least-square reconstruction error

In the previous section, a natural interpretation of the procedure is that of fitting a multivariate Gaussian, defined by $\mu$ and $\Sigma$ to the data. The idea here is to reinterpret the same process as of fitting a linear model to the data, where the fitted hyperplane is of dimension $q<d$. We follow the discussion in {cite}`hastie2009elements`, section 14.5.

The equation of the fitting hyper-plane of dimension $q<d$ is

$$
f(\lambda) = \mu + W_{:q}\lambda,
$$

where $\lambda$ is the q-dimensional vector with the reduced parametric coordinates of a point in the plane, $\mu$ is the mean vector, a point in the plane, and $W_{:q}$ is an orthogonal matrix with $q$ unit vectors as columns. The $q$ columns of $W_{:q}$ are vectors parallel to the plane, so the product $W_{:q}\lambda$ is a linear combination of those vectors that explore the plane as we change the values of $\lambda$.

To fit the model, we seek to minimize the **reconstruction** error, by optimizing for $\mu$, the set of coordinate vectors $\{\lambda_i\}$ and the matrix $W_{:q}$

$$
\underset{\mu,\{\lambda_i\},W_{:q}}{\min}
\sum_{i=1}^n \left | x_i - \mu - W_{:q} \lambda_i \right |^2
$$

Derivating with respect to $\mu$ and $\lambda_{i}$ allows us to optimize jointly for (exercise)

```{math}
\begin{align}
\mu^{*} =& \bar{x}\\
\lambda_i^{*} =& W_{:q}^T(x_i - \bar{x}).
\end{align}
```
As you may have already guessed, the optimal matrix $W_{:q}$ will turn out to be the matrix of eigenvectors of $\Sigma$. This makes the $\lambda_i$ the first $q$ principal components of $x$. Now we find $W_{:q}$,

$$
\underset{W_{q:}}{\min} \sum_i^n
\left |
\tilde{x}_i - W_{:q}W_{:q}^T\tilde{x}_i
\right |^2
$$

where $\tilde{x}_i = x_i - \bar{x}$. The $d\times d$ matrix $W W^T$ is a projection matrix. It first projects each point $\tilde{x}_i$ into the hyper-plane by obtaining each component along each column of $W$. This a q-dimensional representation in the sub-space. Next, we move back into the original d-dimensional space by multiplying by $W$, effectively taking a linear combination of the unit vectors, with each projected component as the weights.

Exercise: Show the minimizing the reconstruction error is equivalent as maximizing the variance along the first q directions of $W_{:q}$. Use matrix algebra to transform into equivalent expressions.

The above exercise shows that the solution is the same as in the previous section.

We can connect the solution with another matrix decomposition, namelu the Singular Value Decomposition (SVD) of the data matrix $X$,

$$
X = UDW^T
$$

where $U$ and $V$ are orthonormal matrices

### Maximizing the projected dispersion (variance)

### Considerations

It is a good idea to standardize the data matrix before applying PCA, since the particular choice of units may artificially inflate one feature variance with respect to others, thus biasing the first principal component along that direction. This is equivalent to diagonalize the correlation matrix.

Also, components associated with the smallest eigenvalues (variances), especially is the difference with previous eigenvalues is large, indicate possible linear relations in the data set. This small variance components can be regarded as random noise on top of a linear model.

## References

```{bibliography}
:style: unsrt
:filter: docname in docnames
```
