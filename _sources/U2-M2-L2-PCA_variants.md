# Some linear and non-linear variants of PCA

## Sparse PCA

One disadvantage of PCA is that the PC are linear combinations of all original variables. This hurts interpretability. An variant that aims to recover new features as limited linear combinations of original features, and aid interpretability of the new components and loadings, is sparse PCA. There are many variants of sparse PCA, but all share the property that the new basis vectors are sparse, i.e., contain many zero entries. A recent overview of Sparse PCA variants can be found in {cite}`zou2018selective`, where the optimization algorithms the different variants are also discussed, including the early methods of SCoTLASS {cite}`jolliffe2003modified` and SPCA {cite}`zou2006sparse`. In particular, note that approaches of variance maximization and reconstruction error minimization, although equivalent for PCA, are not equivalent for sparse PCA.

A particular approach that shares many characteristics with other methods in this course is called dictionary learning. We will now introduce sparse dictionary learning, but keep in mind that other approaches exist, see {cite}`zou2018selective` and references therein.

Within the dictionary learning framework, the PCA loadings (base vectors) are called a dictionary, and each base vector is called and atom. The principal components are called scores. The PCA algorithm learns a dictionary and scores the solve a particular optimization problem, minimizing the reconstruction error of a low rank representation of the data set. We can code the PCA objective within the dictionary learning framework as follows

$$
\underset{Y\in \mathbf{R}^{n \times q}, D \in \mathbf{R}^{d \times q}}{\operatorname{argmin}}
\left| X - Y D^{T}  \right |_2^2
,\quad DD^T = 1,
$$

where $D$ is the dictionary, consisting of $q$ column unit vectors, $Y$ is the score matrix, containing the coefficients of the dictionary columns. The matrix $YD^T$ is a low rank reconstruction of the matrix $X$. The condition on $D$ is required to avoid trivial solutions with arbitrarily large entries in $D$, and consequently small entries in $Y$.

It can be shown that, for $q\leq d$, the solution that minimizes the Frobenius norm of the difference between the original matrix and its low rank reconstruction is obtained from the SVD of X, by keeping only the first $q$ singular vectors, i.e., leads to the PCA solution. This is not surprising, as the Forbenius norm condition is another way to write the least square reconstruction error. Still, the dictionary learning framework is more general, since we can choose any value for dimension $q$, allowing for overcomplete dictionaries ($q > d$), useful in some computer vision applications.

To enforce sparse entries in the solution, we just a LASSO penalty to the objective function

```{math}
\begin{align}
\underset{Y\in \mathbf{R}^{n \times q}, D \in \mathbf{R}^{d \times q}}{\operatorname{argmin}}
\frac{1}{2}\left| X - Y D^{T}  \right |_2^2
- \lambda_D \left| D \right|_1
- \lambda_Y \left | Y \right |_1
\end{align}
```
where the $l_{1}$ norm is taken for the sum of all entries of each matrix. We distinguish two important cases of the penalized problem:

1.  Sparse PCA, $\lambda_{Y} = 0$, and sparsity is only enforced in the dictionary. A normalization must be enforced in $Y$, $|Y|_2 < Const$ for example.
2.  Sparse dictionary learning, $\lambda_{D} = 0$, and sparsity is only enforced in the coefficients. A normalization must be enforced in $D$, $|D|_2 < Const$ for example.

Let\'s focus on sparse PCA, sparse dictionary learning being similar. The question is now how to optimize

```{math}
\begin{align}
\underset{Y\in \mathbf{R}^{n \times q}, D \in \mathbf{R}^{d \times q}}{\operatorname{argmin}}
\frac{1}{2}\left| X - Y D^{T}  \right |_2^2
- \lambda_D \left| D \right|_1
\end{align}
```
Many methods have been proposed, reviews can be found in {cite}`bach2011optimization` and {cite}`mairal2014sparse`. We will discuss an alternating coordinate descent approach, which, although not the fastest, has proven to be effective in practice and relative easy to implement.

The objective function is not convex jointly in both $D$ and $Y$, making it hard to optimize jointly. An popular option, that does not guarantees a global minimum, is to alternate optimization with respect to $D$, keeping $Y$ fixed, and optimizing with respect to $Y$, keeping $D$ fixed, repeating both steps until convergence. Both sub-problems are convex, thus solvable by coordinate descent.

Let\'s focus first on finding the optimal solution for $D$, keeping $Y$ fixed. The optimization problem can be split into independent sub-problems for the rows of $D$

```{math}
\begin{align}
\frac{1}{2}\left| X - Y D^{T}  \right |_2^2
- \lambda \left| D \right|_1
= \sum_{i=1}^q\left[\frac{1}{2} \left| X_{:,i} - Y D_{i}  \right |_2^2
- \lambda \left| D_i \right|_1 \right]
\end{align}
```
and each individual objective inside the summation is in the form of a standard LASSO regression ($|y-X\beta|_2^2 - \lambda|\beta|_1$), which can be optimized using sub-derivatives. Differentiating with respect to $D_{ij}$ we obtain the Karush-Kuhn-Tucker (KKT) conditions,

```{math}
\begin{align}
0 \in [-(X_{:,i} - Y D_i)^T Y_{:,j} - \lambda, -(X_{:,i} - Y D_i)^T Y_{:,j} + \lambda] \quad D_{ij} = 0\\
-(X_{:,i} - Y D_i)^T Y_{:,j} + \lambda\operatorname{sgn}(D_{ij}) = 0 \quad D_{ij} \neq 0
\end{align}
```
or,

```{math}
\begin{align}
\left| (X_{:,i} - Y D_i)^T Y_{:,j}\right| \leq \lambda \quad D_{ij} = 0\\
(X_{:,i} - Y D_i)^T Y_{:,j} = \lambda\operatorname{sgn}(D_{ij}) \quad D_{ij} \neq 0
\end{align}
```
Now, coordinate descent acts by optimizing along one coordinate at a time in an iterative manner until convergence. We will minimize for a single $D_{ij}$, while keeping all other variables. To simplify notation for the sub-problem, well use the definitions $X_{:,i} = x$ and $D_i = d$, do the objective, row wise, becomes

```{math}
\begin{align}
\frac{1}{2}\left| x - Yd  \right|_2^2 - \lambda|d|_1
=&\frac{1}{2}\sum_{i=1}^n\left(x_i - \sum_{j=1}^d Y_{ij} d_j   \right)^2 - \lambda\sum_{j=1}^d |d_j|\\
=&\frac{1}{2}\sum_{i=1}^n\left(x_i - \sum_{j \neq k} Y_{ij} d_j - Y_{ik}d_k   \right)^2 - \lambda|d_k| - \lambda\sum_{j\neq k} |d_j|\\
=&\frac{1}{2}\sum_{i=1}^n\left(r_{ik}  - Y_{ik} d_k   \right)^2 - \lambda|d_k| - \lambda\sum_{j\neq k} |d_j|\\
\end{align}
```
where $r_{ik}=x_i - \sum_{j\neq k} Y_{ij}d_j$ is the partial residual with respect $d_k$. Differentiating with respect to $D_{ik} = d_k$, we obtain the KTT conditions for the single variable problem

```{math}
\begin{align}
\left|\sum_{i=1}^n\left(r_{ik}  - Y_{ik} d_k   \right)Y_{ik}\right| \leq \lambda \quad d_{k} = 0\\
\sum_{i=1}^n\left(r_{ik}  - Y_{ik} d_k   \right)Y_{ik} = \lambda\operatorname{sgn}(d_k) \quad d_{k} \neq 0
\end{align}
```
which, after some algebra, lead to the solutions

```{math}
\begin{align}
d_k =
\begin{cases}
\frac{z_{k} - \lambda}{|Y_{:,k}|_2^2} & z_{k} > \lambda\\
0 & |z_{k}| \leq \lambda\\
\frac{z_{k} + \lambda}{|Y_{:,k}|_2^2} & z_{k} < -\lambda
\end{cases}
\end{align}
```
where $z_{k}= \sum_{i} r_{ik}Y{ik}$. The solution can be written in terms of soft-thresholding function,

```{math}
\begin{align}
S_{\lambda}(x) =
\begin{cases}
x - \lambda & x > \lambda\\
0 & |x| \leq \lambda\\
x + \lambda & x < -\lambda
\end{cases}
\end{align}
```
as

$$
d_k = S_{\frac{\lambda}{|Y_{:,k}|_2^2}}\left(\frac{z_{ik}}{|Y_{:,k}|_2^2}\right)
$$

Going back to the original representation, we obtain for row $l$,

$$
D_{lk} = S_{\frac{\lambda}{|Y_{:,k}|_2^2}}\left(\frac{z^{(l)}_{ik}}{|Y_{:,k}|_2^2}\right)
$$

with

```{math}
\begin{align}
z^{(l)}_{ik} =& \sum_i r^{(l)}_{ik} Y_{ik} \\
=& \sum_i
       \left(
         X_{il} - \sum_j Y_{ij}D_{lj} + Y_{ik}D_{lk}
       \right) Y_{ik} \\
=& \sum_i \left(
            X_{il} - \sum_j Y_{ij}D_{lj}
          \right) Y_{ik}
     + \sum_i Y_{ik}^2 D_{lk}\\
=& \sum_i \left(
            X_{il} - Y_{i:}^T D_{l:}
          \right) Y_{ik}
     + \sum_i Y_{ik}^2 D_{lk}\\
=& \left( X_{:l} - Y D_{l:}
          \right)^T Y_{:k}
     + \sum_i Y_{ik}^2 D_{lk}
\end{align}
```
so that

$$
D_{lk} = S_{\frac{\lambda}{|Y_{:,k}|_2^2}}
\left( D_{lk} +  \frac{\left( X_{:l} - Y D_{l:} \right)^T Y_{:k}}
{|Y_{:,k}|_2^2}\right)
$$

where the $D_{lk}$ on the right hand side is the value before the update. The reason for the rearrangement is that we want to define a block coordinate descent algorithm over the columns of $D$. Notice that the update only depends on values on each row $l$, for $D$. This means we can $D_{kl}$ for all rows $l$ simultaneously, this is called block coordinate descent, with columns as blocks.

$$
D_{:k} = S_{\frac{\lambda}{|Y_{:,k}|_2^2}}
\left( D_{:k} +  \frac{\left( X^T - DY^T \right) Y_{:k}}
{|Y_{:,k}|_2^2}\right)
$$

where the soft-thresholding function is applied element-wise to the vector. To optimize with respect to $D$, we need to iteratively update the columns of $D$ until convergence. Notice that pre-computing matrices $X^T Y$ and $Y^T Y$ allows to save a lot of repeated calculations. We leave the implementation for the assignments.

Now, we move forward to the optimization step with respect to $Y$. To optimize $Y$, we will employ a block coordinate descent algorithm, and enforce the restriction that the $l_2$ norm of each column of $Y$ is equal or less than 1. The solution is very similar to the solution for $D$, now splitting the objective function into rows of $Y$.

Take the objective function

```{math}
\begin{align}
\frac{1}{2}\left| X - Y D^{T}  \right |_2^2
- \lambda \left| D \right|_1
= \sum_{i=1}^q\left[\frac{1}{2} \left| X_{i:} - D Y_i  \right |_2^2  \right]
- \lambda \left| D \right|_1
\end{align}
```
Realizing that the problems decomposes row-wise on $Y$, we can derive the whole objective by the column, akin to taking each row and deriving with respect a single coordinate. Since derivating with respect to $Y$ will zero out the regularization term, we ignore it from now on. We can expand the Frobenius norm,

```{math}
\begin{align}
L = \frac{1}{2}\left| X - Y D^{T}  \right |_2^2
 =& \frac{1}{2}\sum_{ij}\left(X_{ij} - \sum_m Y_{im}D_{jm}\right)^{2}
\end{align}
```
Taking the derivative

```{math}
\begin{align}
0 = \frac{\partial L}{\partial Y_{lk}}
=& \sum_{ij}\left(X_{ij} - \sum_m Y_{im}D_{jm}\right)D_{jk}\delta_{il}\\
=& \sum_{j}\left(X_{lj} - \sum_m Y_{lm}D_{jm}\right)D_{jk}\\
=& \sum_{j}\left(X_{lj} - \sum_{m \neq k} Y_{lm}D_{jm} - Y_{lk}D_{jk} \right)D_{jk}\\
=& \sum_{j}\left(X_{lj} - \sum_{m \neq k} Y_{lm}D_{jm}\right)D_{jk} - \left(\sum_{j}D_{jk}^2 \right)Y_{lk}
\end{align}
```
Solving

```{math}
\begin{align}
Y_{lk} =& \frac{\sum_{j}\left(X_{lj} - \sum_{m \neq k} Y_{lm}D_{jm}\right)D_{jk}}{\sum_{j}D_{jk}^2}\\
=& \frac{\sum_{j}\left(X_{lj} - \sum_{m} Y_{lm}D_{jm}\right)D_{jk}}{\sum_{j}D_{jk}^2}
+ \frac{Y_{lk} \sum_{j}D_{jk}^2}{\sum_{j}D_{jk}^2}\\
=& Y_{lk} + \frac{\sum_{j}\left(X_{lj} -  Y_{l}^TD_{j} \right) D_{jk}} {|D_{:k}|^2}\\
=& Y_{lk} + \frac{\left(X_{l} -  D Y_l\right)^T D_{:k}} {|D_{:k}|^2}
\end{align}
```
Finally, for the whole column

```{math}
\begin{align}
Y_{:k} = Y_{:k} + \frac{\left(X -  YD^{T}\right) D_{:k}} {|D_{:k}|^2}
\end{align}
```
Again note that pre-computing $XD$ and $D^TD$ allows to avoid repeated calculations.

We still need to enforce the normalization on columns of $Y$. It can be shown that re-projecting back to the unit ball is enough to ensure convergence.

```{math}
\begin{align}
Y_{:k} = \frac{Y_{:k}}{\max(1,|Y_{:k}|_2)}
\end{align}
```
The algorithm then repeats optimizing each column iteratively until convergence. Again, we leave the implementation for the assignments.

Further restrictions can be imposed, for example, to enforce structure on the sparse loadings {cite}`jenatton2010structured`. Other applications for SPCA include image de-noising by compressed sensing.

We can explore how sparse PCA works using the implementation from sklearn, which implements a different optimization algorithm ({cite}`mairal2009online`), more efficient, but also harder to implement and less general.

### Example: Faces

We take the faces example from the scikit-learn documentation. You have already explored faces principal components using standard PCA. Now, let\'s take a look at the components using sparse PCA.

``` python
# Code from
#https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import MiniBatchSparsePCA

n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)

faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True,
                                random_state=0)

n_samples, n_features = faces.shape

faces_centered = faces - faces.mean(axis=0)
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)
```

``` example
Dataset consists of 400 faces
```

``` python
def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=cmap,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


plot_gallery("First centered Olivetti faces", faces_centered[:n_components])
```

![](./.ob-jupyter/78f1763fc24a5923870969e7214bfcbf1ef7c28d.png)

``` python
estimator = MiniBatchSparsePCA(n_components=n_components, alpha=0.8,
                               n_iter=100, batch_size=3,
                               random_state=0)

estimator.fit(faces_centered)
components_ = estimator.components_

plot_gallery("First Sparse Components", components_[:n_components])
```

![](./.ob-jupyter/c692b93d0ef9ad043cfff0b03d4aac1a5d724494.png)

### Example: Sparse news data

We take an example from {cite}`zhang2012sparse`, dealing with news data. The data set, obtained from <http://cs.nyu.edu/~roweis/data.html>, records word appearances in 16242 news postings. Each feature represents one of a hundred words, and takes the value 1 is the word is present in the post, and 0 otherwise.

``` python
import numpy as np
import pandas as pd

news = pd.read_csv('Data/20news_w100.csv')
news.head()
```

|     | aids | baseball | bible | bmw | cancer | car | card | case | children | christian | computer | course | data | dealer | disease | disk | display | doctor | dos | drive | driver | earth | email | engine | evidence | fact | fans | files | food | format | ftp | games | god | government | graphics | gun | health | help | hit | hockey | honda | human | image | insurance | israel | jesus | jews | launch | law | league | lunar | mac | mars | medicine | memory | mission | moon | msg | nasa | nhl | number | oil | orbit | patients | pc  | phone | players | power | president | problem | program | puck | question | religion | research | rights | satellite | science | scsi | season | server | shuttle | software | solar | space | state | studies | system | team | technology | university | version | video | vitamin | war | water | win | windows | won | world |
|-----|------|----------|-------|-----|--------|-----|------|------|----------|-----------|----------|--------|------|--------|---------|------|---------|--------|-----|-------|--------|-------|-------|--------|----------|------|------|-------|------|--------|-----|-------|-----|------------|----------|-----|--------|------|-----|--------|-------|-------|-------|-----------|--------|-------|------|--------|-----|--------|-------|-----|------|----------|--------|---------|------|-----|------|-----|--------|-----|-------|----------|-----|-------|---------|-------|-----------|---------|---------|------|----------|----------|----------|--------|-----------|---------|------|--------|--------|---------|----------|-------|-------|-------|---------|--------|------|------------|------------|---------|-------|---------|-----|-------|-----|---------|-----|-------|
| 0   | 0    | 0        | 0     | 0   | 0      | 0   | 0    | 0    | 0        | 0         | 0        | 0      | 0    | 0      | 0       | 0    | 0       | 0      | 0   | 0     | 0      | 0     | 1     | 0      | 0        | 0    | 0    | 0     | 0    | 0      | 0   | 0     | 0   | 0          | 0        | 0   | 0      | 0    | 0   | 0      | 0     | 0     | 0     | 0         | 0      | 0     | 0    | 0      | 0   | 0      | 0     | 0   | 0    | 0        | 0      | 0       | 0    | 0   | 0    | 0   | 0      | 0   | 0     | 0        | 0   | 0     | 0       | 0     | 0         | 0       | 0       | 0    | 0        | 0        | 1        | 0      | 0         | 0       | 0    | 0      | 0      | 0       | 1        | 0     | 0     | 0     | 0       | 1      | 0    | 0          | 0          | 0       | 1     | 0       | 0   | 0     | 0   | 0       | 0   | 0     |
| 1   | 0    | 0        | 0     | 0   | 0      | 0   | 0    | 0    | 0        | 0         | 0        | 0      | 0    | 0      | 0       | 0    | 0       | 0      | 0   | 0     | 0      | 0     | 0     | 0      | 0        | 0    | 0    | 0     | 0    | 0      | 0   | 0     | 0   | 0          | 0        | 0   | 0      | 0    | 0   | 0      | 0     | 0     | 0     | 0         | 0      | 0     | 0    | 0      | 0   | 0      | 0     | 0   | 0    | 0        | 0      | 0       | 0    | 0   | 0    | 0   | 0      | 0   | 0     | 0        | 0   | 0     | 0       | 0     | 0         | 0       | 0       | 0    | 0        | 0        | 0        | 0      | 0         | 0       | 0    | 0      | 0      | 0       | 0        | 0     | 0     | 1     | 0       | 0      | 0    | 0          | 0          | 0       | 0     | 0       | 0   | 0     | 0   | 0       | 0   | 0     |
| 2   | 0    | 0        | 0     | 0   | 0      | 0   | 0    | 0    | 0        | 0         | 0        | 0      | 0    | 0      | 0       | 0    | 0       | 0      | 0   | 0     | 0      | 0     | 0     | 0      | 0        | 0    | 0    | 1     | 0    | 0      | 1   | 0     | 0   | 0          | 0        | 0   | 0      | 0    | 0   | 0      | 0     | 0     | 0     | 0         | 0      | 0     | 0    | 0      | 0   | 0      | 0     | 0   | 0    | 0        | 0      | 0       | 0    | 0   | 0    | 0   | 0      | 0   | 0     | 0        | 0   | 0     | 0       | 0     | 0         | 0       | 0       | 0    | 0        | 0        | 0        | 0      | 0         | 0       | 0    | 0      | 0      | 0       | 0        | 0     | 0     | 0     | 0       | 0      | 0    | 0          | 0          | 1       | 0     | 0       | 0   | 0     | 0   | 0       | 0   | 0     |
| 3   | 0    | 0        | 0     | 0   | 0      | 0   | 0    | 0    | 0        | 0         | 0        | 0      | 0    | 0      | 0       | 0    | 0       | 0      | 0   | 0     | 0      | 0     | 1     | 0      | 0        | 0    | 0    | 0     | 0    | 0      | 0   | 0     | 0   | 0          | 0        | 0   | 0      | 0    | 0   | 0      | 0     | 0     | 0     | 0         | 0      | 0     | 0    | 0      | 0   | 0      | 0     | 0   | 0    | 0        | 0      | 0       | 0    | 0   | 0    | 0   | 0      | 0   | 0     | 0        | 1   | 0     | 0       | 0     | 0         | 0       | 0       | 0    | 0        | 0        | 0        | 0      | 0         | 0       | 0    | 0      | 0      | 0       | 0        | 0     | 0     | 0     | 0       | 0      | 0    | 0          | 1          | 0       | 0     | 0       | 0   | 0     | 0   | 0       | 0   | 0     |
| 4   | 0    | 0        | 0     | 0   | 0      | 0   | 0    | 0    | 0        | 0         | 0        | 0      | 0    | 0      | 0       | 0    | 0       | 0      | 0   | 0     | 0      | 0     | 0     | 0      | 0        | 0    | 0    | 0     | 0    | 0      | 1   | 0     | 0   | 0          | 0        | 0   | 0      | 0    | 0   | 0      | 0     | 0     | 1     | 0         | 0      | 0     | 0    | 0      | 0   | 0      | 0     | 0   | 0    | 0        | 0      | 0       | 0    | 0   | 0    | 0   | 0      | 0   | 0     | 0        | 0   | 0     | 0       | 0     | 0         | 0       | 1       | 0    | 0        | 0        | 0        | 0      | 0         | 0       | 0    | 0      | 0      | 0       | 0        | 0     | 0     | 0     | 0       | 0      | 0    | 0          | 0          | 1       | 0     | 0       | 0   | 0     | 0   | 0       | 0   | 0     |

Note the data is already sparse, so we will not standardize it to preserve sparseness. We try regular PCA first.

``` python
from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(news)

pca_comp = pca.components_
plt.figure(figsize=(10,10))
plt.imshow(pca_comp);
```

![](./.ob-jupyter/8f55435b690de5072d90c469e578c6e446eb6a94.png)

As you can appreciate, the first principal components are dense, hurting interpretability. We now apply sparse PCA.

``` python
from sklearn.decomposition import SparsePCA

spca = SparsePCA(alpha=2, n_components=20)
X_spca = spca.fit_transform(news)

spca_comp = spca.components_
plt.figure(figsize=(10,10))
plt.imshow(spca_comp);
```

![](./.ob-jupyter/a7aed399cdba281bfc8a5b0685c0b45fa76f6184.png)

We can play with the number of components and the degree of sparseness of the components. The idea is that now each component represents sets of words, that may be associated with specific topics.

``` python
words = news.columns
for i, c in enumerate(spca_comp):
    print(f"Set of words positively associated with {i+1} component:")
    print(list(words[c > 0]))
    print()
```

``` example
Set of words positively associated with 1 component:
[]

Set of words positively associated with 2 component:
['card', 'data', 'disk', 'display', 'dos', 'driver', 'files', 'format', 'ftp', 'graphics', 'image', 'mac', 'memory', 'pc', 'program', 'server', 'software', 'version', 'video', 'win', 'windows']

Set of words positively associated with 3 component:
['phone', 'research', 'science', 'university']

Set of words positively associated with 4 component:
['question']

Set of words positively associated with 5 component:
['help']

Set of words positively associated with 6 component:
['bible', 'children', 'christian', 'earth', 'evidence', 'god', 'human', 'jesus', 'jews', 'religion', 'science']

Set of words positively associated with 7 component:
['problem']

Set of words positively associated with 8 component:
[]

Set of words positively associated with 9 component:
[]

Set of words positively associated with 10 component:
['memory', 'system']

Set of words positively associated with 11 component:
[]

Set of words positively associated with 12 component:
['data', 'earth', 'format', 'ftp', 'image', 'launch', 'mars', 'mission', 'moon', 'nasa', 'orbit', 'program', 'research', 'satellite', 'science', 'shuttle', 'solar', 'space', 'technology']

Set of words positively associated with 13 component:
[]

Set of words positively associated with 14 component:
['case']

Set of words positively associated with 15 component:
[]

Set of words positively associated with 16 component:
['car', 'disk', 'drive', 'mac', 'pc', 'scsi', 'software']

Set of words positively associated with 17 component:
[]

Set of words positively associated with 18 component:
['power']

Set of words positively associated with 19 component:
[]

Set of words positively associated with 20 component:
[]
```

## Robust PCA

Robust PCA is a recent method for matrix separation that has received increasing attention, with applications in Machine Learning and Computer Vision among others. From a matrix separation perspective, classical PCA aims to solve

$$
M = L_0 + N_0
$$

where $M$ is the observed data matrix, $L_0$ is a low rank matrix and $N_0$ is dense perturbation matrix whose entries are assumed small. One way to solve this problem is to find the solution to the optimization problem

$$
\underset{S.T.\ rank(L) \leq k}{\operatorname{minimize}} |M - L|_F
$$

for which an analytic solution is obtained by truncating the SVD of $M=U\Sigma V^T$, such that $L = U_{q}\Sigma_{q} V^T_{q}$.

A problem with classical PCA is that the solution is extremely sensible to outliers. The presence of a single highly corrupted observation takes the solution far away from the true solution. Since real data sets are likely to be contaminated (e.g., sensor errors, adversarial attacks), this is a real issue for PCA.

```{figure} Figures/rpca-004.png
Image taken from a presentation by Yuxin Chen (<http://www.princeton.edu/~yc5/ele520_math_data/lectures/robust_PCA.pdf>)
```

Robust PCA is a modification of the objective of classical PCA that aims to address the presence of outliers and large perturbations. Robust PCA assumes the data matrix $M$ is composed as a sum of a **low rank** component $L_{0}$ and a sparse component of arbitrarily large perturbations or corruptions $S_{0}$,

$$
M = L_0 + S_0
$$

```{figure} Figures/rpca_sum.png
Image taken from a presentation by Yuxin Chen (<http://www.princeton.edu/~yc5/ele520_math_data/lectures/robust_PCA.pdf>)
```

In a seminal paper by Candes et. al. {cite}`candes2011robust`, it was proven that, under broad conditions, the recovery of $L_0$ and $S_0$ is possible, and the solution is **exact**. This a rather surprising result, since the problem seems intractable at first sight. Twice the number of unknowns than entries in $M$, we **do not know the rank** of $L_0$ **nor the locations (or number)** of the non-zero entries of $S_{0}$. The recovery is, of course, not always possible, since one think of examples where it would not make sense.

The objective of an idealized RPCA can be expressed as

$$
\underset{S.T. L + S = M}{\min} rank(L) + \lambda |S|_{0}
$$

where $|S|_0$ is the zero-norm, which counts the number of non-zero elements of $S$. Unfortunately this is a very hard problem to optimize, since it is not convex, and both the rank and the 0 norm are discontinuous. In {cite}`candes2011robust`, an relaxed approach through Principal Projection Pursuit is proposed,

$$
\underset{S.T. L + S = M}{\min} |L|_{*} + \lambda |S|_{1}
$$

where $|L|_{*}| = \sum_i \sigma_i(L)$ is the nuclear norm of $L$, i.e., the sum of its singular values, and we use the 1-norm instead of the 0-norm, i.e., a LASSO penalization, to encourage sparsity. The reason the nuclear norm works instead of the rank, is that number of non-zero singular values is exactly the rank of the matrix. The new objective is a convex relaxation of the original one, and can be solved by a number of optimizations algorithms, and is able to recover the exact solution $L = L_0$ and $S = S_0$, given a few conditions discussed below.

First, $L_0$ must not be sparse, otherwise the identification of the sparse component is undetermined. Also, a low rank sparse model is largely orthogonal, meaning most observations are independent, which makes it impossible to use correlation information to impute the missing values. This is, reconstruction is possible, because a low rank dense matrix necessarily is redundant, meaning information from the un-corrupted entries can be used to reconstruct corrupted entries.

Second, the non-zero components of $S_0$ must be spread out, as not to be low rank, again, to avoid identifiability issues. An spread out $S_0$ means the corruptions are not targeted, and cannot delete single rows, for example, so the corrupted values can be recovered from the correlations in the data matrix. A good model for $S_{0}$ is to required that the locations of its non-zero entries are sampled from a uniform distribution, or that each entry has a constant probability of being zeroed, independently of others (Bernoulli trials.)

A way to measure spareness of $L_0$ is through the concept of incoherence. Incoherence is based on the correlation between the principal components and the basis vectors. With the SVD given by $L_0 = U \Sigma V^{*}$, the incoherence condition demands that

```{math}
\begin{align}
\underset{i}{\max} |U^T e_i|_2^2 \leq \frac{\mu_1 r}{n}\\
\underset{i}{\max} |V^T e_i|_2^2 \leq \frac{\mu_1 r}{n}\\
|UV^T|_\infty \leq \sqrt{\frac{\mu_2 r}{n^2}}
\end{align}
```
where $\mu$ are the coherence parameters, i.e., the minimum values that satisfies the requirements. Requiring that the projections are small means that the principal components are spread out among all basis vectors. If this was not the case, and the PC lie along a few basic vectors, then $L_0$ is sparse.

Besides dimensionality reduction, RPCA have found applications in:

-   Video surveillance, separation of background (low rank) and foreground (sparse).

```{figure} Figures/rpca-video.png
Images from {cite}`candes2011robust`.
```

-   Face recognition, where shadows and occlusions are interpreted as the sparse component. This works because the column space for images of a face is very low rank.

```{figure} Figures/rpca-faces.png
Images from {cite}`candes2011robust`.
```

-   Latent Semantic Indexing, where $L_0$ could capture the common words across documents, while $S_0$ would capture the words that best represent each individual document.

-   Graph clustering, where the low rank component corresponds to edges between elements in a cluster, and the sparse component to elements across clusters.

```{figure} Figures/rpca-graph.png
Image taken from a presentation by Yuxin Chen (<http://www.princeton.edu/~yc5/ele520_math_data/lectures/robust_PCA.pdf>)
```

-   Gaussian graphical models with sparse covariance matrix and latent variables.

```{figure} Figures/rpca-gaussian.png
Image taken from a presentation by Yuxin Chen (<http://www.princeton.edu/~yc5/ele520_math_data/lectures/robust_PCA.pdf>)
```

The covariance and precision ($\Lambda = \Sigma^{-1}$) matrices can be blocked partitioned as

```{math}
\begin{align}
\Sigma =
\begin{pmatrix}
\Sigma_{o} & \Sigma_{o,h} \\
\Sigma_{o,h}^T & \Sigma_{h}
\end{pmatrix}
=
\begin{pmatrix}
\Lambda_{o} & \Lambda_{o,h} \\
\Lambda_{o,h}^T & \Lambda_{h}
\end{pmatrix}^{-1}
\end{align}
```
It is known from linear algebra (short complement formula) that

```{math}
\begin{align}
\underbrace{\Sigma_{o}^{-1}}_{observed}
= \underbrace{\Lambda_{o}}_{sparse}
- \underbrace{\Lambda_{o,h}\Lambda_{h}^{-1}\Lambda_{h,o}}_{\text{low rank if # hv is small}}
\end{align}
```
Thus, we can recover the hidden component of the graph given that number of hidden variables is small.

-   Recommendation models where user data is deliberately corrupt, and the recommendation matrix is sparse, the Netflix problem.

## Kernel PCA

## References

```{bibliography}
:style: unsrt
:filter: docname in docnames
```
