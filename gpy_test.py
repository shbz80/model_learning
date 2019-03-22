import GPy, numpy as np
from matplotlib import pyplot as plt
np.random.seed(1)

X = np.random.uniform(-3.,3.,(50,2))
Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(50,1)*0.05




kernel = GPy.kern.RBF(input_dim = 2, ARD = True) + GPy.kern.White(2)
# k.K(X)
m = GPy.models.GPRegression(X,Y,kernel)
m.rbf.lengthscale[:] =1.0
# m.rbf.lengthscale.constrain_bounded(0.1, 0.5)
m.rbf.variance[:] = 1.0
m.Gaussian_noise[:] = 1.0
# m.Gaussian_noise.fix()
m.optimize_restarts(optimizer='lbfgs', num_restarts=10)    # This selects random (drawn from  N(0,1)N(0,1) ) initializations for the parameter values
m.predict()
print(m)

