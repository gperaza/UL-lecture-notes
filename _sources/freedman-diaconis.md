# Freedman-Diaconis Rule

There is a rule of thumb for determining the optimal number of bins in a
histogram. The Freedman-Diaconis rule states that the optimal bin width
can be estimated as

$$
h = \frac{2\ IQR}{n^{1/3}}
$$

The asymptotic (large $n$) optimal bin width was derived by Scott
{cite}`scott1979optimal`, yet, its value depends on the derivative of the
theoretical distribution, often not known. Freedman and Diaconis claimed
{cite}`freedman1981histogram` the bin width can be robustly estimated by the
formula above, which works well most of the time, under the requirements
that the true distribution has squared integrable and continuous first
and second derivatives.

It is worth it to provide a rough derivation of the FD rule, as it is an
nice exercise in the art of approximation.

To find the optimal bin width, we minimize the Mean Integrated Squared
Error (MISE),

$$
MISE = E\left[ \int \left\{ H(x) - f(x)  \right\}^{2}\right],
$$

where $f(x)$ is the underlying probability density function and $H(x)$
is the histogram function, which gives the height bin $j$ containing
$x$,

$$
H(x) = \frac{N_{j}(x)}{N h}
$$

The MISE gives the expected area between the estimated density
(histogram) and the true density ($f(x)$).

Within each bin, the count number $N_j$ is a binomial random variable
with parameters $p=hf_j$ and $n=N$ (\# of trials), where $p$ is the
probability of an observation lying in bin $j$ and

$$
f_j = \frac{1}{h}\int_{\text{bin }j}f(u) du.
$$

Thus we obtain

$$
E\left[ H(x) \right] = \frac{E\left[ N_{j}  \right]}{Nh} = f_j(x)
$$

and

$$
Var\left(H(x)\right) = \frac{Var\left( N_{j}  \right)}{N^2 h^2}
= \frac{f_j\left( 1 - h f_j \right)}{N h}
$$

so

```{math}
\begin{align}
E\left[ \right\{ H(x) - f(x)\left\}^{2}  \right]
=& E\left[ H^2(x) \right] - 2f(x) E\left[H(x)\right] + f^2(x) \\
=& Var(H(x)) + E[H(x)]^2 - 2f(x) E\left[H(x)\right] + f^2(x) \\
=& \frac{f_j\left( 1 - h f_j \right)}{N h} + f_j^2 - 2 f f_j + f^2\\
=& \frac{f_j}{Nh} - \frac{f_j^2}{N} + (f_j - f)^2
\end{align}
```
and integrating over the whole interval, with $\int p_j = h\sum p_j =
h = h\int f_j \rightarrow \int f_j = 1$

$$
MISE = \frac{1}{Nh} - \frac{1}{N}\int f_j^2(x)dx + \int (f_j(x) - f(x))^2 dx.
$$

The first term above refers to the sampling error made, and grows as
$h\rightarrow 0$ since bin counts stop reflecting the true density, some
even being left empty. The last term refers to the bias from
discretization, and goes to zero with $h$, as we show below. The mid
term is proportional to $\sim \frac{1}{N}$, and vanishes in the
asymptotic limit $N\rightarrow \infty$, which is the limit that
interests us.

To deal with third term in the MISE, we expand $f(u)$ around $x$,
keeping the linear terms, which amounts to discarding contributions of
order at most $O(h^{2})$

$$
f(u) \approx f(x) + f'(x)(u-x)
$$

The expression for $f_j(x)$ can now be approximated

```{math}
\begin{align}
f_j \approx& \frac{1}{h}\int_{\text{bin }j}\left( f(x) + f'(x)(u-x)  \right) du\\
=& \frac{1}{h} \left( hf(x) + \frac{f'(x)}{2}\left( u-x \right)^{2} \Big|_{x_j}^{x_{j+1}} \right)\\
=& f(x) + \frac{f'(x)}{2h}\left[\left( x_{j+1} -x \right)^{2} - \left( x_{j} -x \right)^{2}\right]\\
=& f(x) + \frac{f'(x)}{2h}\left[\left( (x_{j+1} - x_j) - (x - x_j) \right)^{2} - \left( x - x_j \right)^{2}\right]\\
=& f(x) + \frac{f'(x)}{2h}\left[\left( h - (x - x_j) \right)^{2} - \left( x - x_j \right)^{2}\right]\\
=& f(x) + \frac{f'(x)}{2h}\left[ h^2 - 2h(x-x_j) \right]\\
=& f(x) + \frac{hf'(x)}{2} - (x-x_j)f'(x),
\end{align}
```
where $x_{j}$ is the lower limit of bin $j$. We can identify the bias of
$E[H(x)]$ as $\frac{hf'(x)}{2} - (x-x_j)f'(x)$.

Plugging this expansion into the MISE, remebering the second term
vanishes,

```{math}
\begin{align}
MISE \approx& \frac{1}{Nh} + \int \left(\frac{hf'(x)}{2} - (x-x_j)f'(x)\right)^2 dx\\
=& \frac{1}{Nh} + \frac{h^2}{4}\int f'^2(x)dx + \int (x-x_j)^2f'^2(x) dx - h\int (x-x_j)f'^2(x) dx
\end{align}
```
We can split the second a third integrals into the sum of integrals for
each bin $j$, for which $x_j$ are constants. With a change of variable,
$y=x-x_j$, we have

```{math}
\begin{align}
\int (x-x_j)^2f'^2(x) dx = \sum_j \int_0^h  y^2f'^2(y + x_j) dy\\
\int (x-x_j)f'^2(x) dx = \sum_j \int_0^h y f'^2(y + x_j) dy.
\end{align}
```
Expanding yet again $f'(y + x_{j}) \approx f'(x_{j}) + O(h)$

```{math}
\begin{align}
\sum_j \int_0^h  y^2(f'^2(x_j) +  O(h))dy = \sum_j \frac{h^{3}}{3} f'^2(x_j) + O(h^4) \\
\sum_j \int_0^h y (f'^2(x_j) +  O(h))dy = \sum_j \frac{h^{2}}{2} f'^2(x_j) +  O(h^{3}).
\end{align}
```
Dropping the higher order terms, we can approximate the sums by
integrals, by identifying $h = \Delta x$,

```{math}
\begin{align}
\sum_j \frac{h^{3}}{3} f'^2(x_j) \approx \frac{h^{2}}{3} \int  f'^2(x)dx  \\
\sum_j \frac{h^{2}}{2} f'^2(x_j) \approx \frac{h}{2}\int f'^2(x)dx.
\end{align}
```
Plugging back into the MISE equation

```{math}
\begin{align}
MISE \approx& \frac{1}{Nh} + \left(\frac{h^2}{4} + \frac{h^2}{3} - \frac{h^2}{2}\right)\int f'^2(x)dx
= \frac{1}{Nh} + \frac{h^2}{12}\int f'^2(x)dx.
\end{align}
```
And optmizing with respect to $h$, one obtains the optimal bin width
$h^{*}$

$$
-\frac{1}{N (h^{*})^{2}} + \frac{h^{*}}{6} \int f'^2(x)dx = 0.
$$

Which yields

$$
h^* = \left(\frac{6}{N \int f'^2(x)dx}\right)^\frac{1}{3}
$$

This optimal bin width depends on the density $f(x)$. Assuming a normal
distribution gives

$$
h^* = \left(\frac{24\sqrt{\pi}}{N}\right)^{\frac{1}{3}}s
$$

Another choice is to approximate the integral more robustly using the
IQR, leading to the FD rule.

## References

```{bibliography}
:style: unsrt
:filter: docname in docnames
```

