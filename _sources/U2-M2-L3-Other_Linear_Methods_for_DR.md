# Other Dimensionality Reduction Methods

## Multidimensional Scaling (MDS)

Multidimensional Scaling works directly on the distance or dissimilarity matrix $D$ and tries to find a set of vectors $z_i$ that reproduces the distance matrix (preserves pairwise distance) as close as possible, with the liberty of choosing the dimensions of the $z_i$.

MDS can be classified onto metric, and non metric MDS. Metric MDS tries to preserve the actual distance values in the distance matrix, while non-metric MDS just preserves the rank. Classical MDS and the least squares scalings are metric scalings, and Shephard-Kruskal is a non-metric scaling.

### Classical MDS

Classical MDS is a linear dimensionality reduction method. To perform classical MDS, we first transform the distance matrix into a **centered** Gram matrix, through the equation

$$
G = -\frac{1}{2}(I - \frac{1}{N}\vec{1}\vec{1}^T) D^2 (I - \frac{1}{N}\vec{1}\vec{1}^T)
$$

where $M = \frac{1}{N}\vec{1}\vec{1}^T$ is the mean operator, and $D^2$ is the squared distance matrix, not $DD$. As en exercise, prove the previous statement.

The Gram matrix, being positive semi-definite, can be factorized as $G=XX^T$, so that $g_{ij} = x_i^T x_j$. Classical MDS seeks to minimize

$$
S_C(\{z_i\}) = \sum_{ij} \left( g_{ij} - z_i^T z_j  \right)^2
= \frac{1}{2}|G - ZZ^T|_F^2
$$ But this is same objective as PCA, only applied to the Gram matrix instead of the covariance matrix. The best low rank approximation is given by the eigen-decomposition of the Gram matrix, $G=U S^2 U^T$, such that $Z = U_q S_q$. From this discussion, it is clear that classical MDS is equivalent to PCA if euclidean distances are used. In fact, if the distances are Euclidean, the full rank model recovers the exact original configuration of points.

Note that classical MDS assumes euclidean distances, so as to have a positive semi-definite Gram matrix. For non-euclidean distances, negative eigenvalues arise from the decomposition, which mean we cannot take $S = \sqrt(S^2)$ on the diagonal elements. In such cases, we can restrict the solution to the coordinates associated with positive eigenvalues, still obtaining a solution, but the solution is no longer equivalent to PCA.

### Least squares or Kruskal-Shephard scaling

In metric MDS, the coordinates $z_i$ are found by minimizing the stress function,

$$
S_M({z_i}) = \sum_{i \neq j} \left(d_{ij} - |z_i - z_j|\right)^2
$$

This objective does not have a closed form solution, and must be optimized numerically. The solution are no longer the same as those obtained from PCA, and can be non-linear.

The stress can be minimized with a number of optimizes, including the ones based on gradient descent. If the Euclidean distance is used, the SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm can be used. SMACOF is a [majorize-minimization](https://en.wikipedia.org/wiki/MM_algorithm) (MM) algorithm, in which instead of minimizing the objective directly, we minimize a lower bound, the majorizing function.

A general MM algorithm works as follows. Let $f(x)$ be the function to be minimized. The majorized function $g(x, x_m)$ must satisfy

```{math}
\begin{align}
g(x | x_m) \leq f(x) \forall x\\
g(x_m | x_m) = f(x_m)
\end{align}
```
The point $x_{m}$ is where $g(x|x_{m})$ touches $f(x)$. At each iteration we find the minimum of $g$ and move $x_m$ to this new minimum.

$$
x _{m+1} = \arg \min _{x}g(x |x_{m})}
$$

The above iterative method will guarantee that $f(x_{m})$ will converge to a local optimum or a saddle point as $m$ goes to infinity.

$$
f(x _{m+1}) \leq g(x _{m+1}|x _{m}) \leq g(x_{m}|x_{m}) = f(x_{m})
$$

In the following figure from Wikipedia illustrates the process for maximization.

![Source: <https://en.wikipedia.org/wiki/File:Mmalgorithm.jpg>](Figures/Mmalgorithm.jpg)

A possible majorizing function for the stress can be obtained from the stress function,

$$
S_M({z_i}) = \sum_{i > j} \left(d_{ij} - |z_i - z_j|\right)^2
= \sum_{i > j} d_{ij}^2 + \sum_{i > j} |z_i - z_j|^2 - 2\sum_{i > j} d_{ij}|z_i - z_j|
$$

The first term is constant. The second term is squared in the $z_i$ and can be written as

```{math}
\begin{align}
|z_i - z_j|^2 = |z_i|^2 + |z_j|^2 - 2\left<z_i, z_j\right>
\end{align}
```
so,

```{math}
\begin{align}
\sum_{i>j} |z_i - z_j|^2 =& (n-1)\sum_{i}|z_i|^2 + - 2\sum_{i>j}\left<z_i, z_j\right>\\
=& (n-1)\sum_{i}|z_i|^2  - \sum_{i\neq j}\left<z_i, z_j\right>\\
=& n\sum_{i}|z_i|^2  - \sum_{i, j}\left<z_i, z_j\right>\\
=& n\left(\sum_{i}|z_i|^2  - \frac{1}{n}\sum_{i, j}\left<z_i, z_j\right>\right)\\
=& tr\left(Z^T Z - \frac{1}{n} Z^T \vec{1}\vec{1}^T Z  \right)\\
=& tr\left(nZ^T (I - M) Z \right)\\
=& tr\left(Z^T (nP) Z \right)\\
=& tr\left(Z^T V Z \right)
\end{align}
```
This second term is quadratic and easy to optimize. The third term can be written as

```{math}
\begin{align}
\sum_{i>j} d_{ij}|z_i - z_j|
=& \sum_{i>j} \frac{d_{ij}}{|z_i - z_j|}|z_i - z_j|^{2}\\
=& \sum_{i>j} s_{ij}|z_i - z_j|^{2}
\end{align}
```
where

```{math}
\begin{align}
s_{ij} =
\begin{cases}
\frac{d_{ij}}{|z_i - z_j|} & |z_i - z_j| > 0\\
0 & |z_i - z_j| = 0
\end{cases}
\end{align}
```
### Shephard-Kruskal non-metric scaling

### Example:

## Non-negative matrix factorization (NNMF)

## Random Projection

## References

```{bibliography}
:style: unsrt
:filter: docname in docnames
```
