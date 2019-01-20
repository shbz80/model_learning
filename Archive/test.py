from discont_function import DiscontinuousFunction
import numpy as np
import matplotlib.pyplot as plt

discon_params = {
            'dt': 0.01,
            'Nsam': 1,
            'Nsec': 10,
            'noise_gain': 3.,
            # 'noise_gain': 0,
            'disc_flag': True,
            'lin_m': -5.,
            'lin_o': 10.,
            'quad_o': -10.,
            # 'quad_o': -15.,
            # 'quad_a': 10.,
            'quad_a': 5.,
            'sin_o': -10.,
            'sin_a': 2.5,
            'offset': 5.,
            }

discFunc = DiscontinuousFunction(discon_params)
xt,yt,yt_n = discFunc.genNsamplesNew()
plt.figure()
plt.scatter(xt,yt_n)
plt.plot(xt,yt,color='r')
plt.show()
