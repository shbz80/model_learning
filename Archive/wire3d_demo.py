'''
=================
3D wireframe plot
=================

A very basic demonstration of a wireframe plot.
'''

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# Grab some test data.
X, Y, Z = axes3d.get_test_data(0.05)

# Plot a basic wireframe.
# ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
surf = ax.plot_surface(X, Y, Z, cmap='Blues',
                       linewidth=0, antialiased=False)
ax.set_xlabel('$x_t$', fontsize=50)
ax.set_ylabel('$u_t$', fontsize=50)
ax.set_zlabel('$x_{t+1}$', fontsize=50)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
surf = ax.plot_surface(X, Y, Z, cmap='Reds',
                       linewidth=0, antialiased=False)
ax.set_xlabel('$x_t$', fontsize=50)
ax.set_ylabel('$u_t$', fontsize=50)
ax.set_zlabel('$x_{t+1}$', fontsize=50)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])


plt.show()
