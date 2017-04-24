import numpy as np
from pyloess import stl
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
from rpca import robust_pca


def plot_verticle(data, scatter_x=None, scatter_y=None):
    rows = len(data)
    for i in range(rows):
        plt.subplot(rows, 1, i + 1)
        plt.plot(data[i])
    if scatter_x is not None and scatter_y is not None:
        plt.subplot(rows, 1, 1)
        plt.plot(scatter_x, scatter_y, 'o', mfc='none')
    plt.show()


class StlDtector:
    def __init__(self, period, interval=14 * 1440, esd_rate=0.01, no=2, nt=1.2):
        self.esd_rate = esd_rate
        self.period = period
        self.no = no
        self.nt = self.period * nt
        self.interval = interval

    def detect_anom_local(self, x, plot=False):
        rate = 0.01
        decomp = stl(x, np=self.period, no=self.no, nt=self.nt)          # STL decomposition
        seasonal = decomp['seasonal']
        trend = decomp['trend']

        # pick anomaly
        median = np.median(x)
        residual = x - seasonal - trend
        ret = pyasl.generalizedESD(residual, int(x.shape[0] * self.esd_rate))
        anom_ind = ret[1]
        anom_val = np.array([x[k] for k in anom_ind])
        if plot is True:
            plot_verticle([x], anom_ind, anom_val)
        return anom_ind

    def detect_anom(self, data, ret_set=True, plot=False):
        anom_ind = set()
        for i in range(data.shape[0] / self.interval):
            anom_ind_list = self.detect_anom_local(data[i * self.interval: (i + 1) * self.interval], plot=plot)
            anom_ind |= set(np.array(anom_ind_list) + i * self.interval)
        if ret_set is True:
            return anom_ind
        else:
            anom_ind_list = np.zeros([data.shape[0]], dtype=np.int16)
            for num in anom_ind:
                anom_ind_list[num] = 1
            return anom_ind_list


class RpcaDetector:
    def __init__(self, period, interval=14 * 1440, lamb_rate=0.8, esd_rate=0.01):
        self.esd_rate = esd_rate
        self.period = period
        self.interval = interval
        self.lamb_rate = lamb_rate

    def detect_anom_local(self, x, plot=False):
        assert x.shape[0] % self.period == 0

        X = x.reshape([self.period, x.shape[0] / self.period])

        # rpca
        lamb_base = max(x.shape) ** -0.5
        L, S = robust_pca(X, lamb=lamb_base * self.lamb_rate)
        L = L.reshape([x.shape[0]])
        S = S.reshape([x.shape[0]])

        # select anomaly
        ret = pyasl.generalizedESD(S, int(x.shape[0] * self.esd_rate))
        anom_ind = ret[1]
        anom_val = np.array([x[k] for k in anom_ind])

        if plot is True:
            plot_verticle([x, L, S], anom_ind, anom_val)

        return anom_ind

    def detect_anom(self, data, ret_set=True, plot=False):
        anom_ind = set()
        for i in range(data.shape[0] / self.interval):
            anom_ind_list = self.detect_anom_local(data[i * self.interval: (i + 1) * self.interval], plot=plot)
            anom_ind |= set(np.array(anom_ind_list) + i * self.interval)
        if ret_set is True:
            return anom_ind
        else:
            anom_ind_list = np.zeros([data.shape[0]], dtype=np.int16)
            for num in anom_ind:
                anom_ind_list[num] = 1
            return anom_ind_list
