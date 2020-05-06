import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dyn_func(x_t, u_t):
    a1 = 2
    b1 = 1.
    # a2 = .75
    # b2 = 1.
    a2 = 2
    b2 = 1
    if x_t >= 0 and x_t < .5 and u_t < 0.5:
        x_t1 = a1*x_t**2 + b1*u_t
    elif x_t > .5 and u_t < 0.5:
        x_t1 = a2 * np.sqrt(x_t) + b2 * u_t
    # elif x_t > .5 and u_t > 0.5:
    else:
        x_t1 = a2 * np.sqrt(x_t) + b2 * u_t**2
    return x_t1

x = np.linspace(0.1, 1, 30)
u = np.linspace(-.1, 1, 30)

X, U = np.meshgrid(x, u)

X1 = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X1[i,j] = dyn_func(X[i,j], U[i,j])

fig = plt.figure()
plt.rcParams.update({'font.size': 20})
fig.subplots_adjust(left=0.00, bottom=0.00, right=1., top=1.)
ax = plt.axes(projection='3d')
ax.plot_surface(X, U, X1, cmap='bwr')
# ax.set_xlabel('$x(t)$')
# ax.set_ylabel('$u(t)$')
# ax.set_zlabel('$x(t+1)$')
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
plt.show()

None