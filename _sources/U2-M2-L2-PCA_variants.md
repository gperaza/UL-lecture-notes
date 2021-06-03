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
Y_{:k} = \frac{Y_{:k}}{|Y_{:k}|_2}
\end{align}
```
The algorithm then repeats optimizing each column iteratively until convergence. Again, we leave the implementation for the assignments.

We can explore how sparse PCA works using the implementation from sklearn, which implements a different optimization algorithm ({cite}`mairal2009online`), more efficient, but also harder to implement and less general.

### Example: Sparse eigen-faces

Further restrictions can be imposed, for example, to enforce structure on the sparse loadings {cite}`jenatton2010structured`. Other applications for SPCA include image de-noising by compressed sensing.

## Robust PCA

## Kernel PCA

## References

```{bibliography}
:style: unsrt
:filter: docname in docnames
```
