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

The main result from {cite}`candes2011robust` proves that Principal Component Pursuit recovers the exact solutions $L = L_0$, $S = S_0$, with high probability ($1 - O(n^{-10})$), given that the following conditions are satisfied:

-   $rank(L_0) \lesssim \frac{n}{\max{\mu_1,\mu_2}\log^2 n}$
-   The non-zero entries of $S_0$ are randomly located, and the number of entries $|S_0|_0 \leq \rho_s n^2$, where $\rho_s$ is some constant, the non-vanishing fraction of allowed zeros.

More even, the value of the regularization parameter is universal, given by $\lambda=\max(n,d)}^{-1/2}$. This conditions turn out to be quite broad, the rank of $L$ can be quite high, up to $n/polylog(n)$, the magnitude of the corruptions can be arbitrarily large and be of any signs, and the proportion of corrupted entries is finite. Numerical simulations show that successful recovery displays a phase transition in the combination of paramters $rank(L)$ and $\rho_S$, i.e., there is a region where the recovery is always successful, and a region where it always fails, see figure below.

```{figure} Figures/rpca-phase.png
Images from {cite}`candes2011robust`.
```

More recent work have extended the results above, allowing recovery from a larger fraction of errors {cite}`chen2013low,ganesh2010dense`.

Finally, Principal Components Pursuit can also extended to matrix completion with corrupted data. Additionally to sparse corruptions, consider also that only a subset $\Omega$ of the data is observed, the objective is now

$$
\underset{S.T.\quad \mathcal{P}_{\Omega}(L + S) = \mathcal{P}_{\Omega}(M)}{\min} |L|_{*} + \lambda |S|_{1}
$$

where $\mathcal{P}_{\Omega}$ is the projection operator onto the set of observed entries, meaning that only the correspondence between observed entries is enforced.

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

For further reading, refer to {cite}`chandrasekaran2011rank`, {cite}`chandrasekaran2010latent`, and {cite}`chen2015incoherence`.

We still haven\'t discuss any algorithm to solve Principal Projection Pursuit. Several options have been explored in the literature, a nice algorithm due to its applicability to a wide range of problems is Alternating Direction Method of Multipliers (ADMM). There is a nice video on ADMM [here](https://www.youtube.com/watch?v=ixBjVgGITHc). Basically, you use a Lagrange multiplier plus a quadratic term that is proved to increase convergence properties (augmented Lagrangian), then optimize iteratively for block of variables and the multiplier.

The augmented Lagrangian for PPC objective is

```{math}
\begin{align}
\mathcal{L}(L,S,Y) = |L|_{*} + \lambda|S|_1
+ \left<Y, M - L - S\right>
+ \frac{\rho}{2}|M - L - S|_F^2
\end{align}
```
where $Y$ is a matrix Lagrangian multipliers, each $y_{ij}$ enforcing the constraint on each entry of $M - L - S = 0$, and $\rho$ is the penalty term of the augmented quadratic term, also the step-size of the gradient update of $Y$ (see below) as to ensure duality (watch video). Note that the inner product for matrices is defined as $\left< A,B\right> = tr(A^T B)$.

The steps of the ADMM algorithm for RPCA are

```{math}
\begin{align*}
\mathbf{L}_{k+1} &= \mathop{\mathrm{arg\,min}}_{L} \,\,\mathcal{L}(\mathbf{L}, \mathbf{S}_k, \mathbf{Y}_k)\\
\mathbf{S}_{k+1} &= \mathop{\mathrm{arg\,min}}_{S} \,\, \mathcal{L}(\mathbf{L}_{k+1}, \mathbf{S}, \mathbf{Y}_k)\\
\mathbf{Y}_{k+1} &= \mathbf{Y}_k + \rho(\mathbf{M} - \mathbf{L}_{k+1} - \mathbf{S}_{k+1}),
\end{align*}
```
where the update for $Y$ is just a gradient ascent step with step-size $\rho$. By completing the square, the Lagrangian takes the form

```{math}
\begin{align}
\mathcal{L}(L,S,Y) = |L|_{*} + \lambda|S|_1
+ \frac{\rho}{2}|M - L - S + \frac{Y}{\rho}|_F^2 - \frac{|Y|_F^2}{2}
\end{align}
```
for which the ADMM steps become, (ignoring constants with respect to each step),

```{math}
\begin{align*}
\mathbf{L}_{k+1} &= \mathop{\mathrm{arg\,min}}_{L} \,\,|L|_{*}
+ \frac{\rho}{2}\left |M - L - S_k + \frac{Y_k}{\rho}\right|_F^2\\
\mathbf{S}_{k+1} &= \mathop{\mathrm{arg\,min}}_{S} \,\, \lambda|S|_1
+ \frac{\rho}{2}\left|M - L_{k+1} - S + \frac{Y_k}{\rho}\right|_F^2\\
\mathbf{Y}_{k+1} &= \mathbf{Y}_k + \rho(\mathbf{M} - \mathbf{L}_{k+1} - \mathbf{S}_{k+1}),
\end{align*}
```
The first two steps have exact solutions, the second one is given by the soft-thresholding function, just a in the discussion of SPCA. Let\'s discuss the minimizing with respect to $L$ first. To deal with the first objective, we need the gradient of the nuclear norm, so let\'s begin there. The nuclear norm can be given either as the trace of the singular value matrix, or the trace of the square root of $X^TX$ ($XX^T$ is $d > n$). This is the reason is also known as the trace norm.

```{math}
\begin{align}
tr(\sqrt{X^T X})
=& tr(\sqrt{V \Sigma U U^T \Sigma V})
= tr(\sqrt{V \Sigma \Sigma V^T})\\
=& tr(\sqrt{V \Sigma V^T V \Sigma V^T})
= tr(\sqrt{(V \Sigma V^T)^2})\\
=& tr(V \Sigma V^T)
= tr(\Sigma V^T V)
= tr(\Sigma)
= \sum_i \sigma_{i}
\end{align}
```
For a direct derivation of the full rank case, see [here](https://math.stackexchange.com/questions/701062/derivative-of-the-nuclear-norm/1016743#701104), since we know $L$ is not full rank, the derivative is discontinuous along the direction where $L$ changes rank. Remember, for a low rank matrix, points lie in a hyper-plan, if a infinitesimal change takes a point out of the hyperplane (into the tangent sub-space), the rank of $L$ changes, and the rate of change of the nuclear norm is discontinuous. This means we are dealing with sub-differentials again. An intuitive way to find the sub-differential of the nuclear norm is to use its dual norm. A dual norm, measures the *size* of a matrix by its effect on another matrix, so the dual of the spectral norm is defined as

```{math}
\begin{align}
|A|_{dual} = \mathop{\mathrm{sup}}_{ |Q|_2 \leq 1} \left< Q, A \right>
\end{align}
```
This is, find the maximum possible value of $\left< Q, A \right>$ over all possible matrices $Q$, such that $|Q|_2\leq 1$. Here $|Q|_2 = \max(\sigma_i(Q)) = \sigma_1(Q)$ is the spectral norm of $Q$, i.e., its maximum singular value. We need to prove that the dual of the spectral norm is the nuclear norm,

```{math}
\begin{align}
|A|_{*} = \mathop{\mathrm{sup}}_{ \sigma_1(Q) \leq 1} \left< Q, A \right>
\end{align}
```
A full prove is given [here](https://math.stackexchange.com/questions/1158798/show-that-the-dual-norm-of-the-spectral-norm-is-the-nuclear-norm), the idea is that to make the inner product as large as possible, we need a matrix $Q$ with all singular values equal to 1, \"parallel\" to $A$, so a good choice is $Q=UV^T$, where $U$ and $V$ come from the SVD of $A=U\Sigma V^T$. Under this conditions

```{math}
\begin{align}
|A|_{*} =& \mathop{\mathrm{sup}}_{ \sigma_1(Q) \leq 1} \left< Q, A \right>\\
=& \left< UV^T, A \right> = tr(VU^T U \Sigma V^T)\\
=& tr(V \Sigma V^T) = tr(V^T V \Sigma ) = tr(\Sigma) = \sum_i\sigma_i(A)
\end{align}
```
So, taking the sub-gradient,

```{math}
\begin{align}
\partial_{L} |L|_{*} = \partial \left<Q, L \right> = {Q}
\end{align}
```
equal to the set of $Q$, with $|Q|_2<1$, that maximize the inner product. A possible solution, we have seen, is $Q = UV^T$, but it is not the only one. Any matrix $UV^T + W$, with $U^T W= W V^{T} = 0$ and $|W|_2\leq 1$ also maximizes the inner product, as $W$ lies of the tangent space of $L$, and thus $\left<W, L \right>$ vanishes.

$$
tr(W^T L) = tr(L^T W) = tr(V\Sigma U^T W) = 0
$$

Making

```{math}
\begin{align}
\partial_{L} |L|_{*} = \{UV^T + W : U^T W = WV^T = 0, |W|_{2}\leq 1\}
\end{align}
```
The first objective

```{math}
\begin{align}
|L|_{*} + \frac{\rho}{2}\left |M - L - S_k + \frac{Y_k}{\rho}\right|_F^2
\end{align}
```
is in the form (with $\tau = 1/\rho$, and $A = M - S_k + \frac{Y_k}{\rho}$)

```{math}
\begin{align}
\tau |L|_{*} + \frac{1}{2}\left |L - A\right|_F^2
\end{align}
```
Taking the sub-gradient, and require that zero is part of the set

```{math}
\begin{align}
0 \in \tau \partial|L|_{*} + L^{*} - A
\end{align}
```
or

```{math}
\begin{align}
A - L^{*} \in \tau \partial|L|_{*}
\end{align}
```
It is shown in {cite}`cai2010singular`, that the solution is given by the singular value thresholding (SVT) of $A$, defined as

$$
L^{*} = \mathcal{D}_{\tau}(A) = U S_{\tau}(\Sigma) V^T
= U\ diag(\max(\sigma_i - \tau), 0)\ V^T
$$

where $S_{\tau}$ is the soft-thresholding function $sgn(x)\max(|x|-\tau, 0)$, as before, and we take into account that singular values are always positive. SVT sets to zero all singular values of $A$ less than $\tau$. To see that $\mathcal{D_{\tau}(L)}$ is indeed a solution we can follow {cite}`cai2010singular`, and do $A = U_0\Sigma_{0}V^T_0 +  U_1\Sigma_{1}V^T_1$ where $U_0$, $V_{0}$ (resp. $U_1$, $V_1$) are the singular vectors associated with singular values greater than $\tau$ (resp. smaller than or equal to $\tau$). Then,

$$
L^{*} = \mathcal{D}_{\tau}(A) = U_0(\Sigma_{0} - \tau I) V_0^{T}
$$

and

```{math}
\begin{align}
A - L^{*}
=& U_0\Sigma_{0}V^T_0 +  U_1\Sigma_{1}V^T_1 - U_0(\Sigma_{0} - \tau I) V_0^{T}\\
=& \tau(U_0 V_0 + \frac{U_1 \Sigma_1 V^T_1}{\tau})\\
=& \tau(U_0 V_0 + W)
\end{align}
```
where $|W|_{2} = |\frac{U_1 \Sigma_1 V^T_1}{\tau}|_{2} < 1$ since all singular values of $\Sigma_{1}$ are bounded by $\tau$. This automatically shows that $A-L^{*} \in \tau\partial|L|_{*}$, since by definition $U_0^TW = WV_0^T = 0$.

So, the solution of the first step is given by

$$
L_{k+1} = \mathcal{D}_{\frac{1}{\rho}}\left( M - S_k + \frac{Y_k}{\rho}  \right)
$$

Now, onto the second step, the objective

```{math}
\begin{align}
\lambda|S|_1 + \frac{\rho}{2}\left|M - L_{k+1} - S + \frac{Y_k}{\rho}\right|_F^2
\end{align}
```
is in the form (with $\tau = \lambda/\rho$ and $A = M - L_{k+1} + Y_k/\rho$)

```{math}
\begin{align}
\tau|S|_1 + \frac{1}{2}\left|S - A \right|_F^2
\end{align}
```
obtaining the sub-gradient, it is easy to find the solution in terms of the soft-thresholding function as

$$
S_{k+1} = S_{\frac{\lambda}{\rho}}\left( M - L_{k+1} + \frac{Y_k}{\rho}  \right)
$$

A nice example on using RPCA implemented in TensorLy can be found [here](http://jeankossaifi.com/blog/rpca.html).

## Kernel PCA

### PCA from the Gram Matrix

We have shown that PCA can be obtained from the SVD of a centered data matrix $X=UDV^T$, and the principal components are $Z = UD$. Now suppose we do not have access to the data matrix, only to the matrix of inner products, the Gram matrix $G = X X^T$. We can still obtain the principal components from the eigen decomposition of the Gram matrix $G=U D^2 U^{T}$. If the inner products of the Gram matrix are not centered, we can use the double centered Gram matrix $(I - M)G(1 -M)$, with $M=\frac{1}{N}1 1^{T}$ being the mean operator, so that the centered data matrix is $X_c = (I - M)X$, so that $(I - M)G(1 -M) = (I - M)XX^T(1 -M) = X_c X_c^T$.

If we were to know the data matrix, the loadings or eigenvectors of the covariance matrix can be obtained from the following relation, considering the eigen value equation for $G$

```{math}
\begin{align}
G u_i =& \lambda_i u_i\\
X X^T u_i =& \lambda_i u_i\\
X^T X X^T u_i = & X^T \lambda_i u_i\\
\Sigma (X^T u_i) =& \lambda (X^T u_i)\\
\Sigma v_i =& \lambda v_i
\end{align}
```
so, the loadings are $v_i = X^Tu_i$. Unfortunately, if the data matrix is unknown, the loadings cannot be recovered. This alternatively is useful is $d >> n$, so we avoid the computation of large covariance matrix.

In the case we cannot recover the loadings, we can still van project new points into the loadings space, as long as we have the inner products with the original data points. Consider a n-dimensional vector $g_0$ of inner products between a new point $x_0$ and all observations $x_i$. $g_{0i} = x_0^T x_i$. The centered projection of $x_0$ onto the principal components is given by (homework)

$$
z_0 = D^{-1}U^T\left(I-M\right)\left(k_0 - \frac{1}{N} K 1\right)
$$

### The Kernel trick

Many learning algorithms take the Gram matrix as input, rather then the data matrix. The Gram matrix is the matrix of pairwise inner products of observation vectors, defined as $G = X X^T$, such that $G_{ij}=\left<x_i, x_j\right>$.

For such cases, a simple way to make those algorithms non-linear us to substitute the Gram matrix for a Kernel matrix, where each entry $K_{ij} = k(x_i, x_j)$ is a kernel function of observation vector pairs, usually a similarity metric.

A kernel function can be interpreted as an inner product in a different vector space (possibly of infinite dimension), for which the coordinate vectors are not known. By using the kernel matrix, the computation of the coordinates in the new space can be avoided, since for well defined kernels, finding the inner product directly is much more efficient.

$$
k(x_i, x_j) =\left<\phi(x_i), \phi(x_j)\right>_{\mathcal{V}}
$$

where $\phi(x)$ is a feature map from the original vector space to the new space $\mathcal{V}$.

```{figure} Figures/Kernel_trick_idea.svg
SVM with kernel given by ϕ(a, b) = (a, b, a² + b²) and thus K(x, y) = x.T y + \|x\|² \|y\|². The training points are mapped to a 3-dimensional space where a separating hyperplane can be easily found. Source: <https://en.wikipedia.org/wiki/File:Kernel_trick_idea.svg>
```

Popular kernels are:

-   Gaussian: $\exp\left(-\beta|x_i-x_j|^2\right)$
-   Polynomial: $(1 + x_i^T x_j)^p$
-   Hyperbolic tangent: $\tanh(x_i^T x_j + \delta)$

The idea of Kernel PCA is to use the Kernel matrix instead of the Gram matrix, so we are interested in the eigen decomposition of $(I - M)K(1 -M) = UD^2U^T$, where we are using the doubled centered kernel matrix, since PCA needs centered data to work. Note that even if the original data matrix is centered, we still need to center the Kernel matrix, since nothing assures that the new features $\phi(x)$ are centered.

Note that the right eigenvectors of the decomposition $\phi(X) = UDV^T$, i.e., the eigenvectors (loadings) in feature space, can be expanded in terms of the basis of observations,

$$
v_m = \sum_{j=1}^n \alpha_{jm}\phi(x_j)
$$

With this expansion, the projection of any observation onto the mth principal component is given by

$$
z_{im} = v_m^T \phi(x_i) = \sum_{j=1}^n \alpha_{jm}\phi(x_j)^T \phi(x_i)
= \sum_{j=1}^n \alpha_{jm}K(x_i, x_j)
$$

As an exercise, show that the coefficients are given by $\alpha_{jm} = u_{jm}/d_m$, assume a centered K matrix. This is, the vector of coefficients, $\alpha_m$ is equal to the $m$ eigenvector divided by the square of the eigenvalue, $\alpha_m = u_m/d_m$.

If the K matrix is not centered, we can still project any existing observation by centering K before, and any new observation $x_0$, by applying (show in assignments)

$$
\vec{z}_{m} = A (I - M)\left(\vec{k}_0 - \frac{1}{N} K \vec{1}\right)
$$

where $A$ is the matrix of column vectors $\alpha_m$ given by $A=U D^{-1}$, and $k_0$ is a vector of kernel products $\phi(x_0)^T\phi(x_i)$ with all observations $x_i$.

### Example: Concentric clusters

``` python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import  KernelPCA

np.random.seed(42)

rads = [1 , 2.8, 5]
n = 150

X = []
y = [0]*150 + [1]*150 + [2]*150

for r in rads:
    theta = np.random.uniform(0, 2*np.pi, size=n)
    X.append(np.hstack([r*np.cos(theta)[:,None],
                        r*np.sin(theta)[:,None]]))

X = np.concatenate(X)
X = X + np.random.normal(loc=0, scale=0.25, size=X.shape)

fig, ax = plt.subplots(2, 2, figsize=(16,16))
ax[0,0].scatter(X[:,0], X[:,1], c=y)
ax[0,0].axis('equal')

kpca = KernelPCA(kernel="rbf", gamma=0.5, n_components=2)
X_kpca = kpca.fit_transform(X)
ax[0,1].set_title('RBF, $\gamma=1/2$')
ax[0,1].scatter(X_kpca[:,0], X_kpca[:,1], c=y)
ax[0,1].axis('equal')

kpca = KernelPCA(kernel="rbf", gamma=0.2, n_components=2)
X_kpca = kpca.fit_transform(X)
ax[1,0].set_title('RBF, $\gamma=1/5$')
ax[1,0].scatter(X_kpca[:,0], X_kpca[:,1], c=y)
ax[1,0].axis('equal')

# Precompute Kernel
# phi(x1,x2) = (x1, x2, x1^2 + x2^2)
# K(x,y) = x.y + |x|^2 |y|^2

X2 = X**2
Xs = X2.sum(axis=1)
K = X{cite}`X`.T + Xs[:,None] @ Xs[None,:]

kpca = KernelPCA(kernel='precomputed', n_components=2)
X_kpca = kpca.fit_transform(K)
ax[1,1].set_title('Custom kernel: $K(x,y) = x^T y + |x|^2 |y|^2$')
ax[1,1].scatter(X_kpca[:,0], X_kpca[:,1], c=y)
ax[1,1].axis('equal');
```

![](./.ob-jupyter/82d19181d3b6c26c6b9a3628cdf59cc8640ade24.png)

### Example: De-noising

## References

```{bibliography}
:style: unsrt
:filter: docname in docnames
```
