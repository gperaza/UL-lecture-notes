# Hierarchical Clustering

Hierarchical clustering algorithms they hierarchical representations in which the clusters at each level of the hierarchy are created by merging clusters at the next lower level. At the lowest level, each cluster contains a single observation. At the highest level there is only one cluster containing all of the data.

Two basic paradigms:

1.  Agglomerative (bottom-up): Start at the bottom and at each level recursively merge a selected pair of clusters into a single cluster. This produces a grouping at the next higher level with one less cluster. The pair chosen for merging consist of the two groups with the smallest intergroup dissimilarity.

2.  Divisive (top-down): Divisive methods start at the top and at each level recursively split one of the existing clusters at that level into two new clusters. The split is chosen to produce two new groups with the largest between-group dissimilarity.

With both paradigms there are $N - 1$ levels in the hierarchy. Each level represents a different partition. The entire hierarchy represents an ordered sequence of such groupings. The choice of a particular partition depends on the characteristics of each particular dataset, and often requires domain knowledge, though some metrics, such as the gap statistic can aid the decision.

## Agglomerative clustering: Linkage Schemes

### Single Linkage

A nice discussion is found in [Wikipedia](https://en.wikipedia.org/wiki/Single-linkage_clustering).

In single-linkage clustering, the distance between two clusters is determined by a single element pair, namely those two elements (one in each cluster) that are closest to each other. The shortest of these links that remains at any step causes the fusion of the two clusters whose elements are involved. The method is also known as nearest neighbor clustering. A drawback of this method is that it tends to produce long thin clusters in which nearby elements of the same cluster have small distances, but elements at opposite ends of a cluster may be much farther from each other than two elements of other clusters. This may lead to difficulties in defining classes that could usefully subdivide the data.

Mathematically, the linkage function -- the distance $D(X,Y)$ between clusters $X$ and $Y$ -- is described by the expression

$$
D(X,Y)=\min _{x\in X,y\in Y}d(x,y),
$$

where $X$ and $Y$ are any two sets of elements considered as clusters, and $d(x,y)$ denotes the distance between the two elements $x$ and $y$.

The following algorithm is an agglomerative scheme that erases rows and columns in a proximity matrix as old clusters are merged into new ones. The $N\times N$ proximity matrix $D$ contains all distances $d(i,j)$. The clusterings are assigned sequence numbers $0,1,\ldots,n-1$ and $L(k)$ is the level of the $k$-th clustering. A cluster with sequence number $m$ is denoted $(m)$ and the proximity between clusters $(r)$ and $(s)$ is denoted $d[(r),(s)]$.

The single linkage algorithm is composed of the following steps:

1.  Begin with the disjoint clustering having level $L(0)=0$ and sequence number $m=0$.
2.  Find the most similar pair of clusters in the current clustering, say pair $(r),(s)$, according to $d[(r),(s)]=\min d[(i),(j)]$ where the minimum is over all pairs of clusters in the current clustering.
3.  Increment the sequence number: $m=m+1$. Merge clusters $(r)$ and $(s)$ into a single cluster to form the next clustering $m$. Set the level of this clustering to $L(m)=d[(r),(s)]$.
4.  Update the proximity matrix, $D$, by deleting the rows and columns corresponding to clusters $(r)$ and $(s)$ and adding a row and column corresponding to the newly formed cluster. The proximity between the new cluster, denoted $(r,s)$ and old cluster $(k)$ is defined as $d[(r,s),(k)]=\min\{d[(k),(r)],d[(k),(s)]\}$.
5.  If all objects are in one cluster, stop. Else, go to step 2.

We now need a function to find the minimum distance in a distance matrix:

``` python
def find_merge(D):
    """ Find clusters to merge.

    INPUTS:
        - D: square distance matrix.
    OUTPUTS:
        - d_min: minimum distance in D.
        - i: row index of d_min
        - j: column index of d_min
    """


    d = D.shape[0]

    # You must return the following values correctly
    d_min = 0
    i = -1
    j = -1

    # Find minimum of the distance matrix excluding the diagonal
    ### BEGIN SOLUTION
    mask = np.ones(D.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    valid_idx = np.where(mask)[0]
    d_min = D[mask].min()
    i, j = np.where(D == d_min)[0]
    ### END SOLUTION

    return d_min, i, j
```

Now, test it on the example Data Matrix D, you should obtain 17:

``` python
D = np.array([
    [ 0, 17, 21, 31, 23],
    [17,  0, 30, 34, 21],
    [21, 30,  0, 28, 39],
    [31, 34, 28,  0, 43],
    [23, 21, 39, 43,  0]
])

d, i, j = find_merge(D)
print(f'Minimum distance in D: {d}, at position ({i},{j})')
### BEGIN HIDDEN TESTS
assert find_merge(D) == (17, 0, 1)
### END HIDDEN TESTS
```

``` example
Minimum distance in D: 17, at position (0,1)
```

Now we need a way to delete the corresponding row/columns in D, and add a new pair of row-column for the new cluster formed my merging. Adding and deleting rows/columns from numpy arrays is SLOW, so a better approach is to use masks on the original data matrix. If you are not familiar with masks, you may take the slow approach of modifying the D matrix at each iteration.

``` python
def merge_clusters(D, i, j):
    """ Merge clusters (i) (j) with distances given in D.
    INPUTS:
        - D: distance matrix for clusters.
        - i: index of the first cluster to merge.
        - j: index of the second cluster to merge.

    OUPUT
        - D_new: new data matrix with """

    mask = mask = np.ones(D.shape, dtype=bool)
    # Remove row i and column j
    mask[i,:] = False
    mask[:,j] = False

    # Replace contents in row j and column i with
    # the distances to the new cluster.
```

### Complete Linkage

### Average Linkage

### Ward\'s Criterion

## Divisive clustering

## Dendogram

Recursive binary splitting/agglomeration can be represented by a rooted binary tree. The nodes of the trees represent groups. The root node represents the entire data set. The N terminal nodes each represent one of the individual observations (singleton clusters). Each non-terminal node (\"parent\") has two daughter nodes. For divisive clustering the two daughters represent the two groups resulting from the split of the parent; for agglomerative clustering the daughters represent the two groups that were merged to form the parent.

Most agglomerative and some divisive methods (when viewed bottom-up) possess a monotonicity property. That is, the dissimilarity between merged clusters is monotone increasing with the level of the merger. Thus the binary tree can be plotted so that the height of each node is proportional to the value of the intergroup dissimilarity between its two daughters. The terminal nodes representing individual observations are all plotted at zero height. This type of graphical display is called a dendrogram.

## Example: Human Tumor Microarray Data

This example is taken from {cite}`hastie2009elements`, the implementation is my own.

``` python
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import numpy as np
import pickle as pkl
```

We apply Hierarchical clustering to human tumor microarray data. The data are a $6830\times 64$ matrix of real numbers, each representing an expression measurement for a gene (row) and sample (column). We cluster the samples (64 columns), each represented as a 6830 dimensional vector of gene expressions. Each sample is labeled according to the tumor type. We want to explore if the clustering process groups similar tumor types into the same cluster.

``` python
# Import micro-array data
with open('Data/nci.micro.data.pkl', 'rb') as f:
    data_dic = pkl.load(f)
    X = data_dic['data'].T  # tranpose to get samples in rows
    labels = data_dic['labels']

fig, ax = plt.subplots(figsize=(15,15))
# Show a sample of genes
gens_idx = np.random.randint(len(X), size=100)
im = ax.imshow(X.T[gens_idx,:], cmap='inferno')

gens = [f'gen{x:03}' for x in gens_idx]

# We want to show all ticks
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(gens)))
ax.set_xticklabels(labels)
ax.set_yticklabels(gens)
ax.set_ylabel('gens')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")

ax.set_title("DNA microrray data. Sample of 100 gens (rows) and 64 tumors (columns). Color meassures gen expression.")
fig.tight_layout()
```

![](./.ob-jupyter/6b425db3697be9c980d733afed5d867c8deeb433.png)

To compare HC with a base model, first we use K-means clustering on the data. To determine the number of clusters, we try to look for a kink, or elbow, in the sum of squares curve. Since there is no clear elbow, we follow the original reference and arbitrarily choose $K=3$.

``` python
from sklearn.cluster import KMeans

# How many clusters?
sum_sq = []
for K in range(1,11):
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    sum_sq.append(kmeans.inertia_)
plt.plot(range(1,11), sum_sq, '-o')
plt.xlabel('Number of clusters K')
plt.ylabel('Sum of Squares');
```

![](./.ob-jupyter/70640174722769dc273a70f63f01de861e4a83a8.png)

``` python
# No clear elbow, choose K=3.
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
C = kmeans.labels_

# Build table with label count
labels = np.array(labels)
unique_labels = np.unique(labels)
l_counts = []
for k in range(3):
    k_counts = []
    # Look at members of cluaster k
    y_k = labels[C==k]
    # Count # isntances of each label in cluster k
    for label in unique_labels:
        k_counts.append((y_k == label).sum())
    l_counts.append(k_counts)

df = pd.DataFrame(data=l_counts, columns=unique_labels)
df
```

|     | BREAST | CNS | COLON | K562A-repro | K562B-repro | LEUKEMIA | MCF7A-repro | MCF7D-repro | MELANOMA | NSCLC | OVARIAN | PROSTATE | RENAL | UNKNOWN |
|-----|--------|-----|-------|-------------|-------------|----------|-------------|-------------|----------|-------|---------|----------|-------|---------|
| 0   | 2      | 0   | 7     | 1           | 1           | 6        | 1           | 1           | 0        | 0     | 0       | 0        | 0     | 0       |
| 1   | 2      | 0   | 0     | 0           | 0           | 0        | 0           | 0           | 7        | 0     | 0       | 0        | 0     | 0       |
| 2   | 3      | 5   | 0     | 0           | 0           | 0        | 0           | 0           | 1        | 9     | 6       | 2        | 9     | 1       |

We see that the procedure is successful at grouping together samples of the same cancer. In fact, the two breast cancers in the second cluster were later found to be misdiagnosed and were melanomas that had metastasized. However, K-means clustering has shortcomings in this application. For one, it does not give a linear ordering of objects within a cluster. Secondly, as the number of clusters K is changed, the cluster membership can change in arbitrary ways. That is, with say four clusters, the clusters need not to be nested with the three clusters above. For these reason, hierarchical clustering is probably preferable for this application.

We now apply agglomerative clustering, testing different linkage strategies.

``` python
# We'll try different linkage methods and compare them side by side.

# Single linkage
Z_sl = linkage(X, 'single', optimal_ordering='True')

# Complete linkage
Z_cl = linkage(X, 'complete', optimal_ordering='True')

# Average linkage
Z_al = linkage(X, 'average', optimal_ordering='True')

# Ward linkage
Z_ward = linkage(X, 'ward', optimal_ordering='True')
```

What is $Z$? $Z$ is the ****linkage matrix****. Each row of $Z$ tells us what clusters are merged in each of the $N-1$ iterations of the algorithm. Each row is composed of

\[$cluster_i$, $cluster_j$, $d_{ij}$, Observation in new cluster \]

``` python
Z_sl[0]
```

``` example
array([50.        , 49.        , 38.23033267,  2.        ])
```

So, at the first iteration, observations 50 and 49 are merged into a cluster of 2 elements. The single linkage distance of these observations in 38.23.

``` python
Z_sl[20]
```

``` example
array([10.        , 81.        , 63.57607597,  7.        ])
```

On iteration 21, clusters 10 (original obs) and cluster 81 are merged. Note that we had 64 original samples, so the index 81 must refer to a new cluster created in a previous step. Let\'s print a couple of rows. Specifically, all indices greater than `len(X)` refer to the cluster formed in step `idx - len(X)`, so index 81 refers to the cluster created in step $81-64 = 17$. Also note that the distance increases monotonically with each step, which allows the interpretation of branch lengths in the dendrogram.

``` python
Z_sl[:20]
```

|     | 0   | 1   | 2       | 3   |
|-----|-----|-----|---------|-----|
| 0   | 50  | 49  | 38.2303 | 2   |
| 1   | 48  | 64  | 38.596  | 3   |
| 2   | 56  | 57  | 39.1056 | 2   |
| 3   | 20  | 21  | 45.1516 | 2   |
| 4   | 34  | 35  | 45.3534 | 2   |
| 5   | 68  | 36  | 45.443  | 3   |
| 6   | 1   | 0   | 51.4382 | 2   |
| 7   | 61  | 60  | 56.7802 | 2   |
| 8   | 71  | 59  | 56.9683 | 3   |
| 9   | 11  | 12  | 57.9173 | 2   |
| 10  | 73  | 13  | 60.3505 | 3   |
| 11  | 41  | 43  | 60.4965 | 2   |
| 12  | 16  | 74  | 60.8044 | 4   |
| 13  | 38  | 39  | 61.5543 | 2   |
| 14  | 30  | 31  | 61.6375 | 2   |
| 15  | 75  | 45  | 61.9293 | 3   |
| 16  | 15  | 14  | 62.1781 | 2   |
| 17  | 76  | 80  | 62.4295 | 6   |
| 18  | 79  | 44  | 63.2641 | 4   |
| 19  | 29  | 82  | 63.4286 | 5   |

Another thing you can do is calculate the [Cophenetic Correlation Coefficient](https://en.wikipedia.org/wiki/Cophenetic_correlation) of your hierarchy. This correlates the actual pairwise distances of all your samples to those implied by the hierarchical clustering. The closer the value is to 1, the better the clustering preserves the original distances.

Now, lets visualize the dendrograms:

``` python
# Condensed distance matrix:
D = pdist(X)

fig, ax = plt.subplots(2,2, figsize=(20,20))
ax[0,0].set_title(f'Single linkage, CCC: {cophenet(Z_sl, D)[0]:0.4f}')
dendrogram(Z_sl, ax=ax[0,0], labels=y);

ax[0,1].set_title(f'Complete linkage, CCC: {cophenet(Z_cl, D)[0]:0.4f}')
dendrogram(Z_cl, ax=ax[0,1], labels=y);

ax[1,0].set_title(f'Average linkage, CCC: {cophenet(Z_al, D)[0]:0.4f}')
dendrogram(Z_al, ax=ax[1,0], labels=y);

ax[1,1].set_title(f'Ward, CCC: {cophenet(Z_ward, D)[0]:0.4f}')
dendrogram(Z_ward, ax=ax[1,1], labels=y);
```

![](./.ob-jupyter/c175b8d3f2c427b0ee347f9d61366e75bec0fb80.png)

Average and complete linkage gave similar results, while single linkage produced unbalanced groups with long thin clusters. We focus on the average linkage clustering, which reports the largest Cophenetic Correlation Coefficient.

``` python
fig = plt.figure(figsize=(16,8))
dendrogram(Z_al, labels=y, leaf_font_size=12);
plt.title('Average linkage');
```

![](./.ob-jupyter/98edd69540eb28b1545f68c7d925068d905dd112.png)

Like K-means clustering, hierarchical clustering is successful at clustering simple cancers together. However it has other nice features. By cutting off the dendrogram at various heights, different numbers of clusters emerge, and the sets of clusters are nested within one another. Secondly, it gives some partial ordering information about the samples.

We can cut the dendrogram at some height to obtain a specific partition. The choice of the number of clusters $k$ is still ill-defined. Techniques useful with K-means, such as the elbow method or the gap statistic, can be used here as well, but expert knowledge is still often required to find a meaningful partition.

We can use the `fcluster` function to specify the cut on the tree in several ways (look at the documentation).

``` python
from scipy.cluster.hierarchy import fcluster

# Specifying the distance at which to cut
labels_1 = fcluster(Z_al, 90, criterion='distance')
print(labels_1)

# Specifying the number of cluster
labels_2 = fcluster(Z_al, 3, criterion='maxclust')
print(labels_2)
```

``` example
[ 6  6  6  6  9  6  6  6  6 10  6  6  6  6  6  6  6  6  6  7  6  6  6  6
  6  6  6  6  6  6  6  6  6  1  1  1  1  1  2  2  3  5  5  5  5  5  5  5
  5  5  5  5  6  4  4  8  8  8  8  8  8  8  8  8]
[2 2 2 2 3 2 2 2 2 3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1
 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
```

We can apply clustering to both rows and columns for a nice visualization.

``` python
# Perform clustering on rows
Z_rows = linkage(X.T, 'average', optimal_ordering='True')

# Compute and plot first dendrogram.
fig = plt.figure(figsize=(8,32))
# x ywidth height
ax1 = fig.add_axes([0.05,0.1,0.4,0.7])
D1 = dendrogram(Z_rows, orientation='left', distance_sort=True, color_threshold=0, link_color_func=lambda k: 'grey')
ax1.set_yticks([])
ax1.set_xticks([])

#Compute and plot second dendrogram.
ax2 = fig.add_axes([0.5,0.81,0.4,0.1])
D2 = dendrogram(Z_al, orientation='top', distance_sort=True, color_threshold=0, link_color_func=lambda k: 'grey')
ax2.set_yticks([])
ax2.set_xticks([])

#Compute and plot the heatmap
ax3 = fig.add_axes([0.5,0.1,0.4,0.7])
im = ax3.imshow(X[D2['leaves']].T[D1['leaves']], cmap='inferno', aspect='auto')
ax3.set_yticks([])
ax3.set_xticks([])
```

![](./.ob-jupyter/46bbab74f82670d19e7ca6109334cf5aa6d109fb.png)

****Note****: There exists a far superior implementation in terms of memory usage called `fastcluster` useful for large datasets. Learn more about the fastcluster module <http://danifold.net/fastcluster.html>. SciKit-Learn also provides an implementation compliant with their API for use in complex ML pipelines.

## References

```{bibliography}
:style: unsrt
:filter: docname in docnames
```
