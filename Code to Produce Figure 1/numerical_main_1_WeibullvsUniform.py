#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 07:06:22 2025

@author: weiliang liu

This code generates the data for producing Figure 1.
"""

from DDMP_MILP import *
import time
import warnings


### network parameter
c1 = 3.5    
la2 = 10
c = [c1,4,1]
lambdas = [1,la2,1]
J = 2
K = 1
v = [[1],[1]]
thetas = [1,1,1] 
sample_sizes = [10,50,100,500,1000,2000]
#### number of parameters to experiment
#a_list = [1,1.05,1.1,1.15,1.2] # shape parameter for Weibull; a = 1 corresponds to expo
a_list = [0.1,0.125,0.15,0.175,0.2,0.75,1,1.25,2,4,8,16,32]
#################### gtmp solution
m_star_dict = {}
F_star_dict = {}
m_star, F_star = gtmp_uniform(J, K, v, c, lambdas, thetas)
m_star_dict['uniform'] = m_star
F_star_dict['uniform'] = F_star
print('c1:',c1)
print('---------------------------------')
print('Uniform m_star:')
print(m_star)

for a in a_list:
    m_star, F_star = gtmp_weibull(J, K, v, c, lambdas, thetas,a)
    m_star_dict['weibull',a] = m_star
    F_star_dict['weibull',a] = F_star
    print('---------------------------------')
    print('Weibull-'+str(a)+' m_star:')
    print(m_star)
    print('F_star:')
    print(F_star)
###### calculating optimal value gap #################
rep = 100

start_time = time.time()
Numerical_Result = {}
for sample_size in sample_sizes:
    print("-------------------------------------------------")
    print(f"Sample size: {sample_size}")
    Numerical_Result[sample_size,'uniform','DDMP_sol'] = []
    Numerical_Result[sample_size,'uniform','m_gap'] = []
    Numerical_Result[sample_size,'uniform','F_gap'] = []
    for i in range(rep):
        B = sample_size * np.ones(J + K, dtype=int)
        r = generate_r(J, K, thetas, B,'uniform',0)
        m_sol = ddmp_milp(J, K, v, c, lambdas, B, r)
        Numerical_Result[sample_size,'uniform','DDMP_sol'].append(m_sol)
        
        m_gap, F_gap = optimizer_gap_uniform(J, K, v, c, lambdas, thetas,m_sol, m_star_dict['uniform'],F_star_dict["uniform"])
        Numerical_Result[sample_size,'uniform','m_gap'].append(m_gap)
        Numerical_Result[sample_size,'uniform','F_gap'].append(F_gap)
        
    for a in a_list:
        Numerical_Result[sample_size,'weibull',a,'DDMP_sol'] = []
        Numerical_Result[sample_size,'weibull',a,'m_gap'] = []
        Numerical_Result[sample_size,'weibull',a,'F_gap'] = []
        for i in range(rep):
            if (i+1) % 10 ==0:
                print(f'shape: {a}; Rep: {i+1}/{rep}')
            B = sample_size * np.ones(J + K, dtype=int)
            r = generate_r(J, K, thetas, B,'weibull',a)
            m_sol = ddmp_milp(J, K, v, c, lambdas, B, r)
            Numerical_Result[sample_size,'weibull',a,'DDMP_sol'].append(m_sol)
            
            m_gap, F_gap = optimizer_gap_weibull(J, K, v, c, lambdas, thetas,a, m_sol, m_star_dict['weibull',a],F_star_dict['weibull',a])
            Numerical_Result[sample_size,'weibull',a,'m_gap'].append(m_gap)
            Numerical_Result[sample_size,'weibull',a,'F_gap'].append(F_gap)
 
np.save('Numeric_Result_DDMP_Gap_Weibull_vs_Uniform_c1_'+str(c1)+'.npy', Numerical_Result)   
# End the timer
end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime:.4f} seconds")   

###### calculating mean estimation gap #################
sample_sizes = [10,50,100,500,1000,2000]
a_list = [0.1,0.125,0.15,0.175,0.2,0.75,1,2,4,8,16,32]
mean_est_result = {}
for ss in sample_sizes:
    for a in a_list:
        mean_est_result[ss, 'Weibull',a, 'abs_mean_gap'] = []
        mean_est_result[ss, 'Weibull',a, 'rela_mean_gap'] = []
    mean_est_result[ss, 'Uniform', 0,'abs_mean_gap'] = []
    mean_est_result[ss, 'Uniform', 0,'rela_mean_gap'] = []

a_list = a_list + [0]
rep = 100
J = 1
K = 0
thetas = [1]
for ss in sample_sizes:
    for a in a_list:
        B = ss * np.ones(J + K, dtype=int)
        if a>0:
            distribution = 'weibull'
            distribution1 = 'Weibull'
        else:
            distribution = 'uniform'
            distribution1 = 'Uniform'
        for i in range(rep):
            r_arr = generate_r(J, K, thetas, B,distribution,a)
            for w in range(0, J + K):
                mean_value = np.mean(r_arr[w][1:])  # Calculate mean for b=1,...,B
                abs_gap = abs(mean_value-1/thetas[0])
                rela_gap = abs(mean_value-1/thetas[0])*thetas[0]
                mean_est_result[ss, distribution1,a, 'abs_mean_gap'].append(abs_gap)
                mean_est_result[ss, distribution1,a, 'rela_mean_gap'].append(rela_gap)
np.save('mean_est_result_Weibull_vs_Uniform.npy', mean_est_result)