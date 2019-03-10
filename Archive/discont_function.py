import numpy as np
import matplotlib.pyplot as plt
from Archive.utilities import get_N_HexCol
from Archive.utilities import plot_ellipse
from gmm import GMM
from copy import deepcopy
from sklearn.gaussian_process import GaussianProcessRegressor


class DiscontinuousFunction(object):
    def __init__(self, params):
        self._params = params
        self._dt = self._params['dt'] # sample time
        self._Nsam = self._params['Nsam'] # Number of samples
        self._Nsec = self._params['Nsec'] # NUmber of sections
        self._noise_gain = self._params['noise_gain']
        self._disc_flag = self._params['disc_flag']
        self._lin_m = self._params['lin_m']
        self._lin_o = self._params['lin_o']
        self._quad_o = self._params['quad_o']
        self._quad_a = self._params['quad_a']
        self._sin_o = self._params['sin_o']
        self._sin_a = self._params['sin_a']
        self._offset = self._params['offset']

    def genMeanFunc(self, X, disc_flag = True, noisy=False):
        dt = self._dt
        Nsam = self._Nsam
        N = X.shape[0]
        T = X[-1]
        Ns = int(N/4.)
        Ts = T/4.
        Nfl = X[(X<Ts)].shape[0]
        Nln = X[(X<2.*Ts)][(X[(X<2.*Ts)]>=Ts)].shape[0]
        Nqd = X[(X<3.*Ts)][(X[(X<3.*Ts)]>=2.*Ts)].shape[0]
        Nsn = X[(X<=T)][(X[(X<=T)]>=3.*Ts)].shape[0]
        Nln = Nln + Nfl
        Nqd = Nqd + Nln
        Nsn = Nsn + Nqd
        # Nfl = N - Ns*3
        # Nln = Nfl + Ns
        # Nqd = Nln + Ns
        # Nsn = Nqd + Ns
        # print Nsn, N
        assert(Nsn==N)
        lin_m = self._lin_m
        lin_o = self._lin_o
        quad_o = self._quad_o
        quad_a = self._quad_a
        sin_o = self._sin_o
        sin_a = self._sin_a

        x_flat = X[:Nfl]
        x_lin = X[Nfl:Nln] - X[Nfl]
        x_quad = X[Nln:Nqd] - X[Nln]
        x_sin = X[Nqd:Nsn] - X[Nqd]
        x_sin = x_sin/x_sin[-1]
        x_sin = x_sin*2.*np.pi
        y_flat = np.zeros(Nfl)
        # y_flat = 3.*np.sin(x_flat*10.)
        lin_o = self._lin_o if disc_flag else 0.
        y_lin = lin_m*x_lin + lin_o + 3.*np.sin(x_lin*20,)
        quad_o = self._quad_o if disc_flag else y_lin[-1]
        y_quad = quad_a*x_quad**2 + quad_o
        sin_o = self._sin_o if disc_flag else y_quad[-1]
        y_sin = sin_a*np.sin(x_sin) + sin_o
        x_lin = X[Nfl:Nln]
        x_quad = X[Nln:Nqd]
        x_sin = X[Nqd:Nsn]
        if noisy:
            noise_gain = self._noise_gain
            y_flat = y_flat + np.random.normal(size=Nfl)*noise_gain*0.5
            y_lin = y_lin + np.random.normal(size=Nln-Nfl)*noise_gain
            y_quad = y_quad + np.random.normal(size=Nqd-Nln)*noise_gain
            y_sin = y_sin + np.random.normal(size=Nsn-Nqd)*noise_gain*2.
        Y = np.concatenate((y_flat,y_lin,y_quad,y_sin))
        Xout = np.concatenate((x_flat,x_lin,x_quad,x_sin))
        assert(Xout.shape==Y.shape)
        return Xout, Y

    def genNoisyFunc(self, X, disc_flag=True):
        X, Y = self.genMeanFunc(X, disc_flag, noisy=True)
        noise_gain = self._noise_gain
        N = Y.shape[0]
        # Yo = Y + np.random.normal(size=N)*noise_gain
        Yo = Y
        return X, Yo

    def genRealFunc(self,T,disc_flag=True,plot=False):
        dt = self._dt
        N = int(T/dt)
        xr = np.linspace(0.,T,N)
        xr,yr = self.genMeanFunc(xr,disc_flag)
        if plot:
            plt.figure()
            plt.plot(xr,yr)
            plt.plot()
            # plt.waitforbuttonpress(0)
        return xr, yr

    def genNsamplesFunc(self,T,disc_flag=True,plot=False):
        dt = self._dt
        Nsam = self._Nsam
        N = int(T/dt) # number of time steps
        Xi = np.zeros((N,Nsam))
        Xo = np.zeros((N,Nsam))
        Yo = np.zeros((N,Nsam))
        for i in range(Nsam):
            Xi[:,i] = np.sort(np.random.uniform(0.,T,N))
            Xo[:,i], Yo[:,i] = self.genNoisyFunc(Xi[:,i],disc_flag=disc_flag)
        X = np.expand_dims(Xo.T, axis=2)
        Y = np.expand_dims(Yo.T, axis=2)
        xr, yr = self.genRealFunc(T,disc_flag,plot=False)
        if plot:
            plt.figure()
            plt.scatter(X,Y)
            plt.plot(xr,yr,color='r')
            # plt.waitforbuttonpress(0)
        return X, Y

    def genNsamplesNew(self, sec_list=None, plot=False):
        dt = self._dt
        sec_types = ['flat','lin','quad','sin']
        if sec_list is None:
            Nsec = self._Nsec # each sec 1 seconds
            sec_list = []
            for sec_i in range(Nsec):
                sec_type = np.random.choice(sec_types)
                sec_list.append(sec_type)

        Nsec = len(sec_list)
        T = Nsec*1.
        N = int(T/dt)
        Ns = N/Nsec

        xf = np.sort(np.random.uniform(0.,1.,Ns))
        xl = np.sort(np.random.uniform(-0.5,0.5,Ns))
        xq = np.sort(np.random.uniform(0.,1.,Ns))
        xs = np.sort(np.random.uniform(0.,2.*np.pi,Ns))
        xs_w = np.sort(np.random.uniform(0.,5.*np.pi,Ns))

        xf_r = np.linspace(0.,1.,Ns)
        xl_r = np.linspace(-0.5,0.5,Ns)
        xq_r = np.linspace(0.,1.,Ns)
        xs_r = np.linspace(0.,2.*np.pi,Ns)
        xs_w_r = np.linspace(0.,5.*np.pi,Ns)

        o = self._offset#5.
        m = self._lin_m # -5.
        a = self._quad_a # 5
        ap = self._sin_a # 2.5

        yf = np.zeros(Ns)
        yl = xl*m
        yq = a*xq**2
        ys = ap*np.sin(xs)

        yf_r = np.zeros(Ns)
        yl_r = xl_r*m
        yq_r = a*xq_r**2
        ys_r = ap*np.sin(xs_r)

        xl = xl + 0.5
        xs = xs/(2.*np.pi)
        xs_w = xs_w/(5.*np.pi)

        xl_r = xl_r + 0.5
        xs_r = xs_r/(2.*np.pi)
        xs_w_r = xs_w_r/(5.*np.pi)

        yq =  yq - a/2.
        yq_r =  yq_r - a/2.
        noise_gain = self._noise_gain
        i=0
        for sec_type in sec_list:
            p = i%2
            if sec_type == 'flat':
                x = xf + float(i)
                y = yf+o if p else yf-o
                y_n = y + np.random.normal(size=Ns)*noise_gain#*0.3
                x_r = xf_r + float(i)
                y_r = yf_r+o if p else yf_r-o
            elif sec_type == 'lin':
                x = xl + float(i)
                y = yl+o if p else yl-o
                y_n = y + np.random.normal(size=Ns)*noise_gain#*1.5
                x_r = xl_r + float(i)
                y_r = yl_r+o if p else yl_r-o
            elif sec_type == 'quad':
                x = xq + float(i)
                y = yq+o if p else yq-o
                y_n = y + np.random.normal(size=Ns)*noise_gain
                x_r = xq_r + float(i)
                y_r = yq_r+o if p else yq_r-o
            elif sec_type == 'sin':
                x = xs + float(i)
                y = ys+o if p else ys-o
                y_n = y + np.random.normal(size=Ns)*noise_gain#*2.
                x_r = xs_r + float(i)
                y_r = ys_r+o if p else ys_r-o
            else:
                raise NotImplementedError()
            if i==0:
                xt = x.reshape(-1,1)
                yt = y.reshape(-1,1)
                yt_n = y_n.reshape(-1,1)
                xt_r = x_r.reshape(-1,1)
                yt_r = y_r.reshape(-1,1)
            else:
                x = x.reshape(-1,1)
                y = y.reshape(-1,1)
                y_n = y_n.reshape(-1,1)
                xt = np.concatenate((xt,x),axis=0)
                yt = np.concatenate((yt,y),axis=0)
                yt_n = np.concatenate((yt_n,y_n),axis=0)

                x_r = x_r.reshape(-1,1)
                y_r = y_r.reshape(-1,1)
                xt_r = np.concatenate((xt_r,x_r),axis=0)
                yt_r = np.concatenate((yt_r,y_r),axis=0)
            i += 1
        if plot:
            plt.figure()
            plt.scatter(xt,yt_n)
            plt.plot(xt_r,yt_r,color='r')
        return xt_r,yt_r,xt,yt_n



    def clusterGmmFunc(self,X,Y,xr,yr,T,K,restarts,plot=False):
        dt = self._dt
        Nsam = self._Nsam
        N = int(T/dt) # number of time steps
        data = np.c_[X,Y]
        # data = np.reshape(data,[N*Nsam,data.shape[2]])

        Gmm = []
        ll = np.zeros(restarts)
        for it in range(restarts):
            gmm = GMM(dxu=1,dxux=2)
            gmm.update(data,K)
            Gmm.append(deepcopy(gmm))
            ll[it] = gmm.ll
            print('GMM log likelihood:',ll[it])
            del gmm
        best_gmm = np.argmax(ll)
        self._best_gmm = Gmm[best_gmm]
        w = Gmm[best_gmm].w
        wn = Gmm[best_gmm].wn
        mu = Gmm[best_gmm].mu
        sigma = Gmm[best_gmm].sigma
        mass = Gmm[best_gmm].mass

        if plot:
            colors = get_N_HexCol(K)
            colors = np.asarray(colors)
            col = np.zeros([data.shape[0],3])

            idx = np.argmax(w,axis=1)
            for i in range(K):
                col[(idx==i)] = colors[i]

            # plt.figure()
            # fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
            fig, ax = plt.subplots()
            plt.scatter(data[:,0],data[:,1],c=col/255.0)
            # plt.plot(xr,yr,color='r')
            plt.plot()
            for k in range(K):
                plot_ellipse(ax, mu[k], sigma[k], color=colors[k]/255.0)
        return w,wn,mu,sigma,idx,mass

    def gmmPredictFunc(self,T,MGP=None,x_test=None):
        dt = self._dt
        N = int(T/dt) # number of time steps
        if self._best_gmm:
            K = self._best_gmm.mu.shape[0]
            if x_test is None:
                # Mesh the input space for evaluations of the real function, the prediction
                x_test = np.linspace(0.,T,N)
                x_test = np.reshape(x_test, (-1,1))
            if MGP is None:
                y_xs = self._best_gmm.predict(x_test)
            else:
                assert(len(MGP)==K)
                y_xs = np.zeros((N,K,1))
                for n in range(N):
                    x = x_test[n]
                    for k in range(K):
                        gp = MGP[k]
                        y_m,y_cov  = gp.predict(x.reshape(1,1),return_cov=True)
                        y_xs[n,k,0] = y_m
                        # y_xs[n,k,0] = 0
                #         ys.append([y_m,np.sqrt(y_cov)])
                #         log_prob = stats.norm.logpdf(y_m, y_m, np.sqrt(y_cov))+np.log(mass[k])

            y_x = np.zeros((N,1))
            clsidx = np.zeros(N,dtype=int)
            for n in range(N):
                x = np.tile(x_test[n],(K,1))
                y = y_xs[n].reshape(K,1)
                xy = np.c_[x,y]
                # Compute probability of each point under each cluster.
                logobs = self._best_gmm.estep(xy)
                # Renormalize to get cluster weights.
                idx = np.argmax(logobs)
                r = idx//K
                c = idx%K
                clsidx[n] = c
                y_x[n] = y_xs[n][c]
            return x_test,y_x,clsidx

    def gpFitFunc(self,x,y,xr,yr,gp_params):
        alpha = gp_params['alpha']
        K_C = gp_params['K_C']
        K_RBF = gp_params['K_RBF']
        K_W = gp_params['K_W']
        normalize_y = gp_params['normalize_y']
        restarts = gp_params['restarts']
        # kernel = K_C + K_RBF + K_W
        kernel = K_RBF + K_W
        self._gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=restarts, normalize_y=normalize_y)
        # Fit to data using Maximum Likelihood Estimation of the parameters
        self._gp.fit(x, y)
        # get estimated kernal paramaters
        gp_theta = self._gp.kernel_.theta
        hyperparams = self._gp.get_params()
        # K = self._gp.kernel_(self._gp.X_train_)
        # print np.diag(K).shape, np.diag(K), K[0][0]
        # print K
        # get gp fit score
        y_true = np.reshape(yr,(-1,1))
        x_true = np.reshape(xr,(-1,1))
        score = self._gp.score(x_true,y_true)
        return np.exp(gp_theta), score, hyperparams

    def gpScoreFunc(self,x,y):
        if self._gp:
            # Mesh the input space for evaluations of the real function, the prediction and
            # its MSE
            y_pred = self._gp.predict(x)
            score = np.linalg.norm(y_pred-y)
            return score

    def gpPredictFunc(self,T,xr,yr,plot=False):
        dt = self._dt
        N = int(T/dt) # number of time steps
        if self._gp:
            # Mesh the input space for evaluations of the real function, the prediction and
            # its MSE
            x_test = np.linspace(0.,T,N)
            x_test = np.reshape(x_test, (-1,1))
            y_mean, y_cov = self._gp.predict(x_test, return_cov=True)
            x_test = np.reshape(x_test,(-1))
            y_mean = np.reshape(y_mean,(-1))
            y_std = np.reshape(np.sqrt(np.diag(y_cov)),(-1))
            # plot GP learning
            if plot:
                plt.figure()
                plt.plot(x_test,y_mean, color='b')
                plt.fill_between(x_test, y_mean - y_std, y_mean + y_std, alpha=0.2, color='b')
                plt.plot(xr,yr,color='r')
                plt.plot()
                # plt.waitforbuttonpress(0)
            return y_mean, y_std
