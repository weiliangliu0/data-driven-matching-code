#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:54:07 2024

This code generates the data to produce Figure 2, Tables 1 and 2.

@author: weiliang liu
"""
#from DDMP_MILP import *
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as s
import scipy
import warnings
import statsmodels.api as sm
from scipy.stats import t
from scipy.special import gammaincc, gamma,gammainccinv,lambertw

# Suppress all warnings
warnings.filterwarnings("ignore")

def moderated_quantile_err(distribution, mean, sample_size, a=0,plotting = 1):
    
    if distribution == 'uniform':
        data = np.random.uniform(0, 2*mean, sample_size)
        sorted_data = np.sort(data)
            
    elif distribution == 'weibull':
        # The mean of Weibull = b * Gamma(1 + 1/a)
        scale = mean / gamma(1 + 1/a)
        data = s.weibull_min.rvs(a, scale=scale, size=sample_size)   
        sorted_data = np.sort(data)
        #print("max data:",np.max(sorted_data))
        
    elif distribution == 'trunc_normal':
        c_v = 1
        loc = mean
        scale = c_v*mean
        lower_end, upper_end = (0 - loc) / scale, np.inf
        data = s.truncnorm.rvs(lower_end, upper_end, loc=loc,scale=scale, size=sample_size)   
        sorted_data = np.sort(data)
    
    
    if distribution == "uniform":
        # True quantile function for a uniform distribution in [0, 2 * mean]
        G_inv = lambda x: x * 2 * mean
        # Moderating function f(u) = 1 - G(u) for uniform distribution
        f = lambda u: 1 - (u / (2 * mean))
    elif distribution == "weibull":
        # True quantile function for a Weibull distribution
        G_inv = lambda x: s.weibull_min.ppf(x, a, scale=scale)
        # Moderating function f(u) = 1 - G(u) for Weibull distribution
        f = lambda u: 1 - s.weibull_min.cdf(u, a, scale=scale)
    elif distribution == "trunc_normal":
        loc = mean
        #std of the normal
        scale = c_v*mean
        ## truncate below loc - lower_end*std and above loc + upper_end*std
        lower_end, upper_end = (0 - loc) / scale, np.inf
        G_inv = lambda x: s.truncnorm.ppf(x, lower_end, upper_end, loc=loc, scale=scale)
        f = lambda u: 1 - s.truncnorm.cdf(u, lower_end, upper_end, loc=loc, scale=scale)
    
    # Empirical quantile function
    G_hat_inv = lambda x: np.quantile(sorted_data, x, method='inverted_cdf')
    # Calculate the moderated uniform error
    max_error = 0
    max_error_point = 0
    for x in np.linspace(0, 1, 1000):  # Sample points in [0, 1]
        G_inv_x = G_inv(x)
        G_hat_inv_x = G_hat_inv(x)
        ### calculate \int_{hG^-1}^{G^-1} 1-G(u) du
        if distribution == 'weibull':
            v1 = (G_hat_inv_x / scale) ** a
            v0 = (G_inv_x / scale) ** a
            integral = (scale / a) * gamma(1/a) * (gammaincc(1/a, v0)-gammaincc(1/a, v1))
        elif distribution == 'uniform':
            integral = (G_inv_x-G_inv_x**2/(4*mean)) - (G_hat_inv_x-G_hat_inv_x**2/(4*mean))
        else:
            integral,_ = scipy.integrate.quad(f, G_hat_inv_x, G_inv_x)
            
        error = np.abs(integral)
        if error > max_error:
            max_error = error
            max_error_point = x
    
    if plotting == 1:
        x_vals = np.linspace(0, 1, 1000)
        G_inv_vals = [G_inv(x) for x in x_vals]
        G_hat_inv_vals = [G_hat_inv(x) for x in x_vals]
        plt.figure()
        plt.plot(x_vals, G_inv_vals, label='True quantile')
        plt.plot(x_vals, G_hat_inv_vals, label='Empirical sample size: '+str(sample_size), linestyle='--')
        plt.scatter([max_error_point], [G_hat_inv(max_error_point)], color='red', zorder=5, label='Max Error Point:'+str(max_error_point))
        plt.xlabel('Moderated Quantile Error: '+str(max_error))
        plt.ylabel('Quantile Value')
        plt.title('True vs Empirical Quantiles:'+distribution+'; shape:'+str(a))
        plt.legend()
        plt.grid()
        plt.show()

    return max_error, -1,max_error_point


if __name__ == '__main__':

    mean = 1
    rep = 100
    Numerical_Result = {}
    sample_sizes = [10,50,100,500,1000,5000,10000,50000,100000,500000,1000000]
    a_list = [0.1,0.125,0.15,0.175,0.2,0.75,1,1.25,2,4,8,16,32]
  
    
    start_time = time.time()
    for sample_size in sample_sizes:
        print("-------------------------------------------------")
        print(f"Sample size: {sample_size}")
        Numerical_Result[sample_size,'uniform','qt_err'] = []
        Numerical_Result[sample_size,'uniform','cdf_err'] = []
        Numerical_Result[sample_size,'uniform','max_err_pt'] = []
        for i in range(rep):
            max_error,cdf_error,max_error_point = moderated_quantile_err('uniform', mean, sample_size, a=0,plotting = 0)
            Numerical_Result[sample_size,'uniform','qt_err'].append(max_error)
            Numerical_Result[sample_size,'uniform','cdf_err'].append(cdf_error)
            Numerical_Result[sample_size,'uniform','max_err_pt'].append(max_error_point)
            
        for a in a_list:
            Numerical_Result[sample_size,'weibull',a,'qt_err'] = []
            Numerical_Result[sample_size,'weibull',a,'cdf_err'] = []
            Numerical_Result[sample_size,'weibull',a,'max_err_pt'] = []
            for i in range(rep):
                if (i+1) % 10 ==0:
                    print(f'shape: {a}; Rep: {i+1}/{rep}')
                max_error,cdf_error,max_error_point = moderated_quantile_err('weibull', mean, sample_size, a,plotting = 0)
                Numerical_Result[sample_size,'weibull',a,'qt_err'].append(max_error)
                Numerical_Result[sample_size,'weibull',a,'cdf_err'].append(cdf_error)
                Numerical_Result[sample_size,'weibull',a,'max_err_pt'].append(max_error_point)
    # End the timer
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.4f} seconds")            
    np.save('Numeric_Result_qt_cdf_err_range100w_rep100_a_all.npy', Numerical_Result)        
    
