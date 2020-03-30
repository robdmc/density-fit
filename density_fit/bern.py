import numpy as np
import pandas as pd
from scipy.special import comb, erf
import wrapt
import holoviews as hv
from scipy.interpolate import interp1d
from scipy import stats
import copy
import easier as ezr
np.set_printoptions(precision=4)


class Compress:
    def compress(self, x):
        """
        Compresses an sorted array to have min 0 and max 1
        """
        x = x.flatten()
        xmin = x[0]
        xmax = x[-1]
        x = (x - xmin) / (xmax - xmin)

        self.xmin = xmin
        self.xmax = xmax

        return x

    def expands(self, x):
        """
        Expands an array to go from [0, 1] interval to original interval
        """
        x = x.flatten()
        xmin = self.xmin
        xmax = self.xmax

        x = xmin + (xmax - xmin) * x
        return x


class Bern(Compress):

    def __init__(self, N):
        self.N = N

    def bern_term(self, n, k, x):
        """
        Returns the kth order term of a nth degree
        Bernstein polynomial
        """
        return comb(n, k) * (1 - x) ** (n - k) * x ** k

    def get_bern_sum(self, N, infunc):
        """
        Performs the appropriate berstein approximation
        expansion and sums up the terms
        """
        def bern_sum(xin):
            k_vec = np.arange(N + 1)
            coeff_vec = infunc(k_vec / N)


            # import pdb; pdb.set_trace()

            if len(xin.shape) == 1:
                xin = np.expand_dims(xin, 0)

            xout = np.zeros_like(xin)

            for row_ind in range(xin.shape[0]):
                x = xin[row_ind, :].flatten()
                X, K = np.meshgrid(x, k_vec)
                _, C = np.meshgrid(x, coeff_vec)
                B = self.bern_term(N, K, X)

                terms = C * B
                out = np.sum(terms, axis=0)
                xout[row_ind, :] = out
            if len(xin.shape) == 1:
                return xout.flatten()
            else:
                return xout
        return bern_sum

    def get_fit_func(self, func):
        return self.get_bern_sum(self.N, func)


class Tester:
    def __init__(self):
        N = 100
        np.random.seed(10)
#         dist = stats.t(df=100)
#         self.data = dist.rvs(500)
        self.data = np.concatenate([np.random.randn(5 * N), 20 * np.random.randn(N)])

    @ezr.cached_property
    def ecdf(self):
        x = pd.Series(self.data).value_counts().sort_index()
        dx = pd.Series(x.index.values).diff().median()
        x[x.index[0] - dx] = 0
        x = x.sort_index()

        df = pd.DataFrame(x)
        df = df.reset_index().rename(columns={'index': 'value', 0: 'num'})
        df['ecdf'] = df.num.cumsum() / df.num.sum()
        df = df.rename(columns={'value': 'x', 'ecdf': 'y'})[['x', 'y']]
        df.loc[:, 'x'] = (df.x - df.x.iloc[0]) / (df.x.iloc[-1] - df.x.iloc[0])
        return df

    @ezr.cached_property
    def ecdf_func(self):
        return interp1d(self.ecdf.x.values, self.ecdf.y.values, fill_value=(0, 1), bounds_error=False)


    def test__plot(self):
        x = np.linspace(self.ecdf.x.min(), self.ecdf.x.max(), 300)
        yt = self.ecdf_func(x)

        B = Bern(600)
        yf = B.get_fit_func(self.ecdf_func)(x)

        c1 = hv.Curve((x, yt), label='ecdf')
        c2 = hv.Curve((x, yf), label='fit')
        display(c1 * c2)

    def test_delta(self):
        x = np.linspace(self.ecdf.x.min(), self.ecdf.x.max(), 300)
        yt = self.ecdf_func(x)

        B = Bern(600)
        fit_func = B.get_fit_func(self.ecdf_func)
        yf = fit_func(x)

        delta = yt - yf
        print(f'mean={np.mean(delta)},  std={np.std(delta)}')