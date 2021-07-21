# Assignment

#### Imports

```python
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
```

#### Data

Get data here: [Wine Dataset](https://archive.ics.uci.edu/ml/datasets/wine)

#### About

This data is the result of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines. Do not use the 1st column, it's the label or target. You can use this to compare your results against.

I want you to explore the sklearn and scipy libraries, plus the information provided in the lecture to complete this assignment.

#### Directions

- Load the data, throw out the 1st column
- The remaining 13 columns are the predictors
- Scale the data (clustering is sensitive to scale)
- Find the distance between pairs of all 13-dimensional vectors
  - Use `pdist`
  - This results in a single large vector
  - Convert this to a squre matrix using `squareform`
- After you have all the distances from all points to all points, then use `linkage` to create the clusters
- To visualize your clusters use `dendrogram`
- To get the top 3 clusters, use `fcluster`
- Compare your top 3 clusters to the actual data (col 1) and see how close it is

#### Fin

