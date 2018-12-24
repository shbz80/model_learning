import numpy as np
import matplotlib.pyplot as plt
from utilities import get_N_HexCol
from utilities import plot_ellipse

mu = np.array([[  0.64351776,   0.3707491 ],
               [  4.41717054, -10.7437802 ],
               [  3.102827,    -2.48863657],
               [  1.88864293,   7.73979623]])

sigma = np.array([[[ 0.1699102,   0.115086],
                   [ 0.115086,    6.4948015]],
                  [[ 0.13628531, -0.95244736],
                   [-0.95244736, 20.91734011]],
                  [[ 0.13403647,  2.62345651],
                   [ 2.62345651, 61.65984628]],
                  [[ 0.13541353, -0.84032525],
                   [-0.84032525, 17.00480621]]])

K = mu.shape[0]
T = 5.
N = 100 # 100

x = np.linspace(0.,T,N)
x = x.reshape((N,1))

mu_y_x = np.zeros((N,K,1))

for k in range(K):
    mu_x = mu[k][0]
    mu_y = mu[k][1]
    sig = sigma[k]
    sig_xx = sig[0,0]
    inv_sig_xx = 1./sig_xx
    sig_xy = sig[0,1]
    sig_yx = sig[1,0]
    sig_yy = sig[1,1]
    for n in range(N):
        mu_y_x[n][k] = mu_y + sig_yx*inv_sig_xx*(x[n] - mu_x)

colors = get_N_HexCol(K)
colors = np.asarray(colors)
fig, ax = plt.subplots()
plt.scatter(mu[:,0],mu[:,1])
plt.ylim(ymax=25, ymin=-30)
plt.xlim(xmax=6, xmin=-1)
for k in range(K):
    plot_ellipse(ax, mu[k], sigma[k], color=colors[k]/255.0)
    plt.plot(x,mu_y_x[:,k,:],color=colors[k]/255.0)
plt.show()
