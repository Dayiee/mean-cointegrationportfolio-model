

import numpy as np
import pandas as pd

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

import sklearn
from sklearn.linear_model import LinearRegression

import arch
import arch.unitroot as unitroot

import multiprocessing
from multiprocessing import Pool

import time

from Leybourne import Leybourne
import traceback

class Portfolio(object):
    def __init__(self, *args, **kwargs):
        #TODO
        pass

    def __repr__(self):
        #TODO
        pass


class MeanVariancePortfolio(Portfolio):
    def __init__(self):
        pass

    def fit(self, return_hist, E, **kwargs):
        assert isinstance(return_hist, pd.DataFrame)
        self.stocknames = return_hist.columns

        Sigma = return_hist.cov()
        r = return_hist.mean()
        ef = EfficientFrontier(r, Sigma)
        ef.efficient_return(E)
        w = pd.Series(ef.clean_weights())
        self.w = w

        return {'portfolio':w, 'Sigma':Sigma}

    def eval(self, eval_price_hist, **kwargs):
        assert isinstance(eval_price_hist, pd.DataFrame)
        """ data preparation """
        eval_price_hist = eval_price_hist.sort_index()
        # remove the stocks which have less than 5 valid values
        eval_price_hist = eval_price_hist.dropna(axis=1, thresh=5)
        # remove the stocks whose prices are NaN in the first row
        firstrow_nan = pd.isna(eval_price_hist.iloc[0])
        eval_price_hist = eval_price_hist.loc[:,~firstrow_nan]

        """ make stocknames consistent """
        stocknames = set(self.w.index).intersection(set(eval_price_hist.columns))
        stocknames = list(stocknames)
        print("Number of stocks: %d"%len(stocknames))
        eval_price_hist = eval_price_hist[stocknames]
        w = self.w.loc[stocknames]
        # rescale the weights so that they sum up to 1
        w = w / w.sum()

        """ cumulative individual price ratio """
        ratio_cum = eval_price_hist / eval_price_hist.iloc[0]
        # fill nan records with the element before it.
        ratio_cum = ratio_cum.fillna(method='ffill')
        
        """ cumulative portfolio value ratio:
            ratio_cum (num_timesteps,num_stocks) * w (num_stocks,) 
            -> portf_valueratio_realtime (num_timesteps,1)
        """
        portf_valueratio_cum = ratio_cum @ w # dot product, cumulative value ratio

        """ portfolio value ratio between 2 consecutive time steps """
        portf_valueratio_realtime = \
            (portf_valueratio_cum / portf_valueratio_cum.shift(1)).iloc[1:]
        
        assert isinstance(portf_valueratio_cum, pd.Series)
        assert isinstance(portf_valueratio_realtime, pd.Series)
        
        return {'gain': portf_valueratio_cum.iloc[-1] - 1,
                'portfolio_ret': portf_valueratio_realtime - 1} 


class MeanVariancePortfolioCor(MeanVariancePortfolio):
    """ (override the `fit` method of MeanVariancePortfolio)
        Use Pearson's correlation coeffcient of returns,
        instead of covariance of returns.
    """
    def fit(self, return_hist, E, **kwargs):
        assert isinstance(return_hist, pd.DataFrame)
        self.stocknames = return_hist.columns

        Sigma = return_hist.corr()
        r = return_hist.mean()
        ef = EfficientFrontier(r, Sigma)
        ef.efficient_return(E)
        w = pd.Series(ef.clean_weights())
        self.w = w

        return {'portfolio':w, 'Sigma':Sigma}

    
# class MeanVariancePortfolioCor(Portfolio):
#     def __init__(self):
#         pass

#     def fit(self, return_hist, E, **kwargs):
#         assert isinstance(return_hist, pd.DataFrame)
#         # return_hist = price_hist.dropna(axis=1, how='any')
#         self.stocknames = return_hist.columns

#         Sigma = return_hist.corr()
#         r = return_hist.mean()
#         ef = EfficientFrontier(r, Sigma)
#         ef.efficient_return(E)
#         w = pd.Series(ef.clean_weights())
#         self.w = w

#         return {'portfolio':w, 'Sigma':Sigma}

#     def eval(self, eval_price_hist, **kwargs):
#         assert isinstance(eval_price_hist, pd.DataFrame)
#         eval_price_hist = eval_price_hist.sort_index()[self.stocknames]
#         eval_price_hist = eval_price_hist.dropna(axis=1, how='all').dropna(axis=0, how='any')
#         ret = eval_price_hist.iloc[-1] / eval_price_hist.iloc[0]
#         stocknames = set(self.w.index).intersection(set(ret.index))
#         print(len(stocknames))
#         portfolio_gain = (self.w.loc[stocknames] * ret.loc[stocknames]).sum() - 1
#         return {'gain': portfolio_gain}    
    


class MeanVariancePortfolioCorVol(MeanVariancePortfolio):
    """ (override the `fit` method of MeanVariancePortfolio)
        Use Pearson's correlation coeffcient of volatility,
        instead of covariance of returns.
    """
    def fit(self, return_hist, vol_hist, E, **kwargs):
        assert isinstance(return_hist, pd.DataFrame)
        assert isinstance(vol_hist, pd.DataFrame)
        # make stocks of `return_hist` and `vol_hist` consistent
        self.stocknames = set(vol_hist.columns).intersection(set(return_hist.columns))
        self.stocknames = list(self.stocknames)
        return_hist = return_hist[self.stocknames]
        vol_hist = vol_hist[self.stocknames]

        Sigma = vol_hist.corr()
        r = return_hist.mean()
        ef = EfficientFrontier(r, Sigma)
        ef.efficient_return(E)
        w = pd.Series(ef.clean_weights())
        self.w = w

        return {'portfolio':w, 'Sigma':Sigma}


class MeanVariancePortfolioCovVol(MeanVariancePortfolio):
    """ (override the `fit` method of MeanVariancePortfolio)
        Use covariance of volatility,
        instead of covariance of returns.
    """
    def fit(self, return_hist, vol_hist, E, **kwargs):
        assert isinstance(return_hist, pd.DataFrame)
        assert isinstance(vol_hist, pd.DataFrame)
        # make stocks of `return_hist` and `vol_hist` consistent
        self.stocknames = set(vol_hist.columns).intersection(set(return_hist.columns))
        self.stocknames = list(self.stocknames)
        return_hist = return_hist[self.stocknames]
        vol_hist = vol_hist[self.stocknames]

        Sigma = vol_hist.cov()
        r = return_hist.mean()
        ef = EfficientFrontier(r, Sigma)
        ef.efficient_return(E)
        w = pd.Series(ef.clean_weights())
        self.w = w

        return {'portfolio':w, 'Sigma':Sigma}

# class MeanVariancePortfolioCorVol(Portfolio):
#     def __init__(self):
#         pass

#     def fit(self, return_hist,vol_hist, E, **kwargs):
#         assert isinstance(return_hist, pd.DataFrame)
#         # return_hist = price_hist.dropna(axis=1, how='any')
#         self.stocknames = vol_hist.columns

#         Sigma = vol_hist.corr()
#         r = return_hist.mean()
#         ef = EfficientFrontier(r, Sigma)
#         ef.efficient_return(E)
#         w = pd.Series(ef.clean_weights())
#         self.w = w
     
#         return {'portfolio':w, 'Sigma':Sigma}

#     def eval(self, eval_price_hist, **kwargs):
#         assert isinstance(eval_price_hist, pd.DataFrame)
#         eval_price_hist = eval_price_hist.sort_index()[self.stocknames]
#         eval_price_hist = eval_price_hist.dropna(axis=1, how='all').dropna(axis=0, how='any')
#         ret_rate=(eval_price_hist/eval_price_hist.shift(1)).iloc[1:]
#         f=lambda x: np.cumprod(x)
#         ret_rate_cprod=ret_rate.apply(f,axis=0)

#         stocknames = set(self.w.index).intersection(set(ret_rate_cprod.columns))
#         portfolio0=self.w.loc[stocknames]*100

#         ret_rate_cprod=ret_rate_cprod[list(stocknames)]
#         f=lambda x: np.asarray(x)*np.asarray(portfolio0)
#         eval_ret=ret_rate_cprod.apply(f,axis=1)
#         portfolio_ret=eval_ret.apply(lambda x: x.sum(),axis=1)
#         portfolio_ret=(portfolio_ret/portfolio_ret.shift(1)).iloc[1:]
#         portfolio_ret=portfolio_ret.dropna(axis=0, how='any')
#         ret = eval_price_hist.iloc[-1] / eval_price_hist.iloc[0]
#         stocknames = set(self.w.index).intersection(set(ret.index))
#         print(len(stocknames))
#         portfolio_gain = (self.w.loc[stocknames] * ret.loc[stocknames]).sum() - 1
#         return {'gain': portfolio_gain, 'portfolio_ret': portfolio_ret}  
    
    
# class MeanVariancePortfolioCovVol(Portfolio):
#     def __init__(self):
#         pass

#     def fit(self, return_hist,vol_hist, E, **kwargs):
#         assert isinstance(return_hist, pd.DataFrame)
#         # return_hist = price_hist.dropna(axis=1, how='any')
#         self.stocknames = vol_hist.columns

#         Sigma = vol_hist.cov()
#         r = return_hist.mean()
#         ef = EfficientFrontier(r, Sigma)
#         ef.efficient_return(E)
#         w = pd.Series(ef.clean_weights())
#         self.w = w

#         return {'portfolio':w, 'Sigma':Sigma}

#     def eval(self, eval_price_hist, **kwargs):
#         assert isinstance(eval_price_hist, pd.DataFrame)
#         eval_price_hist = eval_price_hist.sort_index()[self.stocknames]
#         eval_price_hist = eval_price_hist.dropna(axis=1, how='all').dropna(axis=0, how='any')
#         ret = eval_price_hist.iloc[-1] / eval_price_hist.iloc[0]
#         stocknames = set(self.w.index).intersection(set(ret.index))
#         print(len(stocknames))
#         portfolio_gain = (self.w.loc[stocknames] * ret.loc[stocknames]).sum() - 1
#         return {'gain': portfolio_gain}     
    
    
    
    
def linear_regression_residual(x, y):
    lr = LinearRegression(fit_intercept=True)
    x = x.reshape(-1,1)
    lr.fit(x, y)
    residual = y - lr.predict(x)
    return residual


'''
def linear_regression_residual(x, y):
    
    residual = y - x
    return residual

'''

def cointegration_pvalue(x,y,test_method):
    if np.all(x == y):
        return 0
    residual = linear_regression_residual(x,y)
    # print('residual:', residual)
    if test_method == 'ADF':
        pvalue = unitroot.ADF(residual).pvalue
    elif test_method == 'DFGLS':
        pvalue = unitroot.DFGLS(residual).pvalue 
    elif test_method == 'PP':
        pvalue = unitroot.PhillipsPerron(residual).pvalue
    elif test_method == 'KPSS':
        pvalue = 1 - unitroot.KPSS(residual).pvalue
    elif test_method == 'LMC':
        ll=Leybourne()
        Sigma[i,j] = 1 - ll.run(residual, arlags=None, regression='c', method='ols', varest='var94')[1]
        
    else:
        raise ValueError("Unknown method '%s'"%test_method)
    return pvalue


class CointegPortfolio(MeanVariancePortfolio):

    def fit(self, return_hist, price_hist, E, test_method='ADF',
            multithreads=False, n_threads=None, **kwargs):
        if multithreads and (n_threads is None):
            # 如果没有指定使用线程的数量，那么默认使用最大线程数的一半
            n_threads = int(os.cpu_count() / 2)
        assert isinstance(price_hist, pd.DataFrame)
        self.stocknames = price_hist.columns

        n_stocks = len(self.stocknames)
        Sigma = np.zeros((n_stocks, n_stocks))
        price_hist = price_hist.dropna(axis=0, how='any')
        price = price_hist.values
        self.price = price

        if multithreads:
            # multi-thread
            with Pool(n_threads) as p:
                Sigma = p.starmap(cointegration_pvalue, 
                    [(price[:,i], price[:,j], test_method) for i in range(n_stocks) for j in range(n_stocks)])
            Sigma = np.array(Sigma).reshape(n_stocks, n_stocks)
            Sigma = 1 - Sigma
            Sigma = pd.DataFrame(Sigma, columns=self.stocknames, index=self.stocknames)
        else:
            # single-thread
            for i in range(n_stocks):
                for j in range(n_stocks):
                    if i==j:
                        Sigma[i,j] = 1
                    else:
                        # t0 = time.time()
                        residual = linear_regression_residual(price[:,i], price[:,j])
                        # times['regression'] = times.get('regression', 0) + time.time() - t0
                        
                        # t0 = time.time()
                        if test_method == 'ADF':
                            Sigma[i,j] = 1 - unitroot.ADF(residual).pvalue
                        elif test_method == 'DFGLS':
                            Sigma[i,j] = 1 - unitroot.DFGLS(residual).pvalue 
                        elif test_method == 'PP':
                            Sigma[i,j] = 1 - unitroot.PhillipsPerron(residual).pvalue
                        elif test_method == 'KPSS':
                            Sigma[i,j] = unitroot.KPSS(residual).pvalue
                        elif test_method == 'LMC':
                            ll = Leybourne()
                            Sigma[i,j] = ll.run(residual, arlags=None, regression='c', method='ols', varest='var94')[1]
                     # times['stationarity_test'] = times.get('stationarity_test', 0) + time.time() - t0

        
        r = return_hist.mean()
        ef = EfficientFrontier(r, Sigma)
        ef.efficient_return(E)
        w = pd.Series(ef.clean_weights())
        self.w = w

        return {'portfolio':w, 'Sigma':Sigma}
    
#     def eval(self, eval_price_hist, **kwargs):
#         assert isinstance(eval_price_hist, pd.DataFrame)
#         eval_price_hist = eval_price_hist.sort_index()
#         # remove the stocks which have less than 5 valid values
#         eval_price_hist = eval_price_hist.dropna(axis=1, thresh=5)
#         # remove the stocks whose prices are NaN in the first row
#         firstrow_nan = pd.isna(eval_price_hist.iloc[0])
#         eval_price_hist = eval_price_hist.loc[:,~firstrow_nan]

#         stocknames = set(self.w.index).intersection(set(eval_price_hist.columns))
#         stocknames = list(stocknames)
#         print("Number of stocks: %d"%len(stocknames))
#         eval_price_hist = eval_price_hist[stocknames]
#         w = self.w.loc[stocknames]
#         # rescale the weights so that they sum up to 1
#         w = w / w.sum()

#         ratio_cum = eval_price_hist / eval_price_hist.iloc[0]
#         ratio_cum = ratio_cum.fillna(method='ffill')

#         portf_valueratio_cum = ratio_cum @ w
#         portf_valueratio_realtime = \
#             (portf_valueratio_cum / portf_valueratio_cum.shift(1)).iloc[1:]
        
#         assert isinstance(portf_valueratio_cum, pd.Series)
#         assert isinstance(portf_valueratio_realtime, pd.Series)
        
#         return {'gain': portf_valueratio_cum.iloc[-1] - 1,
#                 'portfolio_ret': portf_valueratio_realtime - 1} 
    



'''    
class CointegPortfolioVol(Portfolio):
    def __init__(self):
        pass

    def fit(self, return_hist, vol_hist, E, test_method='ADF',
              **kwargs):
        
        assert isinstance(vol_hist, pd.DataFrame)
        self.stocknames = vol_hist.columns

        n_stocks = len(self.stocknames)
        Sigma = np.zeros((n_stocks, n_stocks))
        vol_hist = vol_hist.dropna(axis=0, how='any')
        vol = vol_hist.values
        self.vol = vol

        
            # single-thread
        for i in range(n_stocks):
                for j in range(n_stocks):
                    if i==j:
                        Sigma[i,j] = 1
                    else:
                        # t0 = time.time()
                        residual = linear_regression_residual(vol[:,i], vol[:,j])
                        # times['regression'] = times.get('regression', 0) + time.time() - t0
                        
                        # t0 = time.time()
                        if test_method == 'ADF':
                            Sigma[i,j] = 1 - unitroot.ADF(residual).pvalue
                        elif test_method == 'DFGLS':
                            Sigma[i,j] = 1 - unitroot.DFGLS(residual).pvalue 
                        elif test_method == 'PP':
                            Sigma[i,j] = 1 - unitroot.PhillipsPerron(residual).pvalue
                        elif test_method == 'KPSS':
                            Sigma[i,j] = unitroot.KPSS(residual).pvalue
                        elif test_method == 'LMC':
                            ll = Leybourne()
                            Sigma[i,j] = ll.run(residual, arlags=None, regression='c', method='ols', varest='var94')[1]
                     # times['stationarity_test'] = times.get('stationarity_test', 0) + time.time() - t0

        
        r = return_hist.mean()
        ef = EfficientFrontier(r, Sigma)
        ef.efficient_return(E)
        w = pd.Series(ef.clean_weights())
        self.w = w

        return {'portfolio':w, 'Sigma':Sigma}

  
    
    def eval(self, eval_price_hist, **kwargs):
        assert isinstance(eval_price_hist, pd.DataFrame)
        eval_price_hist = eval_price_hist.sort_index()[self.stocknames]
        eval_price_hist = eval_price_hist.dropna(axis=1, how='all').dropna(axis=0, how='any')
        ret = eval_price_hist.iloc[-1] / eval_price_hist.iloc[0]
        stocknames = set(self.w.index).intersection(set(ret.index))
        print(len(stocknames))
        portfolio_gain = (self.w.loc[stocknames] * ret.loc[stocknames]).sum() - 1
        return {'gain': portfolio_gain} 
    
    
class CointegPortfolioRet(Portfolio):
    def __init__(self):
        pass

    def fit(self, return_hist, vol_hist, E, test_method='ADF',
             **kwargs):
        
        assert isinstance(return_hist, pd.DataFrame)
        self.stocknames = return_hist.columns

        n_stocks = len(self.stocknames)
        Sigma = np.zeros((n_stocks, n_stocks))
        return_hist = return_hist.dropna(axis=0, how='any')
        rett = return_hist.values
        self.rett = rett

        
            # single-thread
        for i in range(n_stocks):
                for j in range(n_stocks):
                    if i==j:
                        Sigma[i,j] = 1
                    else:
                        # t0 = time.time()
                        residual = linear_regression_residual(rett[:,i], rett[:,j])
                        # times['regression'] = times.get('regression', 0) + time.time() - t0
                        
                        # t0 = time.time()
                        if test_method == 'ADF':
                            Sigma[i,j] = 1 - unitroot.ADF(residual).pvalue
                        elif test_method == 'DFGLS':
                            Sigma[i,j] = 1 - unitroot.DFGLS(residual).pvalue 
                        elif test_method == 'PP':
                            Sigma[i,j] = 1 - unitroot.PhillipsPerron(residual).pvalue
                        elif test_method == 'KPSS':
                            Sigma[i,j] = unitroot.KPSS(residual).pvalue
                        elif test_method == 'LMC':
                            ll = Leybourne()
                            Sigma[i,j] = ll.run(residual, arlags=None, regression='c', method='ols', varest='var94')[1]
                     # times['stationarity_test'] = times.get('stationarity_test', 0) + time.time() - t0

        
        r = return_hist.mean()
        ef = EfficientFrontier(r, Sigma)
        ef.efficient_return(E)
        w = pd.Series(ef.clean_weights())
        self.w = w

        return {'portfolio':w, 'Sigma':Sigma}

  
    
    def eval(self, eval_price_hist, **kwargs):
        assert isinstance(eval_price_hist, pd.DataFrame)
        eval_price_hist = eval_price_hist.sort_index()[self.stocknames]
        eval_price_hist = eval_price_hist.dropna(axis=1, how='all').dropna(axis=0, how='any')
        ret = eval_price_hist.iloc[-1] / eval_price_hist.iloc[0]
        stocknames = set(self.w.index).intersection(set(ret.index))
        print(len(stocknames))
        portfolio_gain = (self.w.loc[stocknames] * ret.loc[stocknames]).sum() - 1
        return {'gain': portfolio_gain} 
    
    
'''
        

