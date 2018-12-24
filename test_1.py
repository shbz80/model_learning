from scipy.spatial.distance import pdist, squareform
import numpy as np

X=np.array([[1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12]
            ],dtype=np.float)
length_scale = np.array([6,7,8],dtype=np.float)
dists = pdist(X / length_scale, metric='sqeuclidean')
K = np.exp(-.5 * dists)
K = squareform(K)
np.fill_diagonal(K, 1)
print 'done'