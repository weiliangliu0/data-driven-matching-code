#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 06:39:02 2025

This code visualize data obtained in "numerical_main_1_WeibullvsUniform" to produce Figure 1
@author: leonliu
"""

from DDMP_MILP import *
import time
import warnings

                            
def plot_results_all_a_mean_gap(rep, sample_sizes, Numeric_Result, a_list,weibull_color='#4682B4',marker_move=0,log_scale_y=0,log_scale=1,uniform =1,save_path=0):
    # Create an empty DataFrame
    data = pd.DataFrame(columns=['abs_gap', 'rela_gap','distribution', 'sample size'])
    
    # Initialize lists to store data
    abs_gap_list = []
    rela_gap_list = []
    distribution_list = []
    sample_size_list = []
    
    # Populate lists with data from Numeric_Result
    for ss in sample_sizes:
        for a in a_list:
            abs_gap_list += Numeric_Result[ss, 'Weibull',a, 'abs_mean_gap']
            rela_gap_list += Numeric_Result[ss, 'Weibull',a, 'rela_mean_gap']
            distribution_list += ['Weibull: '+str(a)] * rep
        
        if uniform == 1:
            abs_gap_list += Numeric_Result[ss, 'Uniform',0, 'abs_mean_gap']
            rela_gap_list += Numeric_Result[ss, 'Uniform',0, 'rela_mean_gap']
            distribution_list += ['Uniform'] * rep
            sample_size_list += [ss] * (len(a_list)+1) * rep
        else:
            sample_size_list += [ss] * len(a_list) * rep

    # Populate the DataFrame with the lists
    data['abs_gap'] = abs_gap_list
    data['rela_gap'] = rela_gap_list
    data['distribution'] = distribution_list
    data['sample size'] = sample_size_list
    
    #legend_label_mapping = {f'weibull: {a}': f'Weibull: {a}' for a in a_list}
    #if uniform == 1:
    #    legend_label_mapping['uniform'] = 'Uniform'
    # Replace 'distribution' column values with renamed labels
    #data['distribution'] = data['distribution'].replace(legend_label_mapping)
    # Define the new hue_order for consistent ordering
    hue_order = [f'Weibull: {a}' for a in a_list]
    if uniform == 1:
        hue_order.append('Uniform')
    
    sns.set(style="whitegrid")
    if uniform == 1:
        palette =  [weibull_color] * len(a_list) + ['r'] 
    else:
        palette =  [weibull_color] * len(a_list)

    # Plot mean value with 95% CI for 'qt_err'
    weibull_markers = ['^', 's', 'P', 'o', 'D', 'v', '*']
    if marker_move == 1:
        weibull_markers = ['s', 'P', 'o', 'D', 'v', '*','^']
    weibull_markers = weibull_markers[:len(a_list)]
    marker_sizes = [10] * len(a_list) + [6]
    '''
    fig, ax = plt.subplots(1, figsize=(6, 4))
    sns.lineplot(
        x='sample size',
        y='abs_gap',
        hue='distribution',
        data=data,
        palette=palette,
        estimator=np.mean,
        ci=95,
        ax=ax,
        err_style="band",
        style='distribution',
        dashes=[(2, 2)] * len(a_list) + [(2, 2)],
        markers= weibull_markers+['X'],
        markersize= 10
    )
    if log_scale == 1:
        ax.set_xscale('log')  # Set x-axis to logarithmic scale
        ax.set_xlabel('Sample size (log scale)', fontsize=16)
    else:
        ax.set_xlabel('Sample size', fontsize=16)
    #ax.set_ylabel('Relative Mean Est. Gap (%)', fontsize=16)
    ax.set_ylabel('Absolute Mean Est. Gap', fontsize=16)
    #plt.ylim([-0.05,1.05])
    ax.tick_params(axis='both', labelsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=12, loc='upper right')
    ''' 
    fig, ax = plt.subplots(1, figsize=(6, 4))
    sns.lineplot(
        x='sample size',
        y='rela_gap',
        hue='distribution',
        data=data,
        palette=palette,
        estimator=np.mean,
        ci=95,
        ax=ax,
        err_style="band",
        style='distribution',
        dashes=[(2, 2)] * len(a_list) + [(2, 2)],
        markers= weibull_markers+['X'],
        markersize= 10
    )
    if log_scale == 1:
        ax.set_xscale('log')  # Set x-axis to logarithmic scale
        ax.set_xlabel('Sample size (log scale)', fontsize=16)
    else:
        ax.set_xlabel('Sample size', fontsize=16)
        
    if log_scale_y == 1:
        ax.set_yscale('log')  # Set x-axis to logarithmic scale
        #ax.set_ylabel('Relative Mean Est. Err. (%)', fontsize=16)
        ax.set_ylabel('Estimation Error in Mean', fontsize=16)
    else:   
        #ax.set_ylabel('Relative Mean Est. Err. (%)', fontsize=16)
        ax.set_ylabel('Estimation Error in Mean', fontsize=16)
    #ax.set_ylabel('Absolute Mean Est. Gap', fontsize=16)
    #plt.ylim([-0.05,1.05])
    ax.tick_params(axis='both', labelsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=12, loc='upper right')
    if save_path != 0:
        plt.savefig(save_path, format='png', dpi=600, bbox_inches='tight')


def plot_results_all_a_DDMP_gap(rep, sample_sizes, Numeric_Result, a_list,weibull_color='#4682B4',marker_move=0,log_scale_y=0, log_scale=1,gap_perc = 1,uniform =1,save_path=0):
    # Create an empty DataFrame
    data = pd.DataFrame(columns=['m_gap', 'F_gap','distribution', 'sample size'])
    
    # Initialize lists to store data
    m_gap_list = []
    F_gap_list = []
    distribution_list = []
    sample_size_list = []
    
    # Populate lists with data from Numeric_Result
    for ss in sample_sizes:
        for a in a_list:
            m_gap_list += Numeric_Result[ss, 'weibull',a, 'm_gap']
            F_gap_list += Numeric_Result[ss, 'weibull',a, 'F_gap']
            distribution_list += ['weibull: '+str(a)] * rep
        
        if uniform == 1:
            m_gap_list += Numeric_Result[ss, 'uniform', 'm_gap']
            F_gap_list += Numeric_Result[ss, 'uniform', 'F_gap']
            distribution_list += ['uniform'] * rep

            sample_size_list += [ss] * (len(a_list)+1) * rep
        else:
            sample_size_list += [ss] * len(a_list) * rep

    # Populate the DataFrame with the lists
    data['m_gap'] = m_gap_list
    data['F_gap'] = F_gap_list
    data['distribution'] = distribution_list
    data['sample size'] = sample_size_list
    
    legend_label_mapping = {f'weibull: {a}': f'Weibull: {a}' for a in a_list}
    if uniform == 1:
        legend_label_mapping['uniform'] = 'Uniform'
    # Replace 'distribution' column values with renamed labels
    data['distribution'] = data['distribution'].replace(legend_label_mapping)
    # Define the new hue_order for consistent ordering
    hue_order = [f'Weibull: {a}' for a in a_list]
    if uniform == 1:
        hue_order.append('Uniform')
    
    sns.set(style="whitegrid")
    if uniform == 1:
        palette =  [weibull_color] * len(a_list) + ['r'] 
    else:
        palette =  [weibull_color] * len(a_list)

    # Plot mean value with 95% CI for 'qt_err'
    weibull_markers = ['^', 's', 'P', 'o', 'D', 'v', '*']
    if marker_move == 1:
        weibull_markers = ['s', 'P', 'o', 'D', 'v', '*','^']
    weibull_markers = weibull_markers[:len(a_list)]
    marker_sizes = [10] * len(a_list) + [6]
    
    fig, ax = plt.subplots(1, figsize=(6, 4))
    sns.lineplot(
        x='sample size',
        y='F_gap',
        hue='distribution',
        data=data,
        palette=palette,
        estimator=np.mean,
        ci=95,
        ax=ax,
        err_style="band",
        style='distribution',
        dashes=[(2, 2)] * len(a_list) + [(2, 2)],
        markers= weibull_markers+['X'],
        markersize= 10
    )
    if log_scale == 1:
        ax.set_xscale('log')  # Set x-axis to logarithmic scale
        ax.set_xlabel('Sample size (log scale)', fontsize=16)
    else:
        ax.set_xlabel('Sample size', fontsize=16)
        
        
    if gap_perc == 1:
        if log_scale_y == 1:
            ax.set_yscale('log')  # Set x-axis to logarithmic scale
            ax.set_ylabel('Relative Optimal Value Gap (%)', fontsize=16)
        else:
            ax.set_ylabel('Relative Optimal Value Gap (%)', fontsize=16)
    else:
        #ax.set_ylabel('Optimal Value Gap', fontsize=16)
        #ax.set_ylabel(r'$F(\bm{m}^\star) - F(\hat{\bm{m}})$', fontsize=16)
        ax.set_ylabel(r'$F(\mathbf{m^*}) - F(\mathbf{\hat{m}}^{KA})$', fontsize=20)
    #plt.ylim([-0.05,1.05])
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xticks(sample_sizes)
    handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles, labels, fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend(handles, labels, fontsize=12, loc='upper right')
    if save_path != 0:
        plt.savefig(save_path, format='png', dpi=600, bbox_inches='tight')
    '''
    fig, ax = plt.subplots(1, figsize=(6, 4))
    sns.lineplot(
        x='sample size',
        y='m_gap',
        hue='distribution',
        data=data,
        palette=palette,
        estimator=np.mean,
        ci=95,
        ax=ax,
        err_style="band",
        style='distribution',
        dashes=[(2, 2)] * len(a_list) + [(2, 2)],
        markers= weibull_markers+['X'],
        markersize= 10
    )
    if log_scale == 1:
        ax.set_xscale('log')  # Set x-axis to logarithmic scale
        ax.set_xlabel('Sample Size (log scale)', fontsize=16)
    else:
        ax.set_xlabel('Sample size', fontsize=16)
    ax.set_ylabel(r'Optimizer Gap ($L_1$ Distance)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    #plt.ylim([-0.05,1.05])
    #ax.set_xticks(sample_sizes)
    handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles, labels, fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend(handles, labels, fontsize=12, loc='upper right')
    plt.show() 
    '''
#############  #############   ############# ############# 

plot_results_all_a_mean_gap(rep, sample_sizes, mean_est_result, [0.15,0.2],log_scale_y=0)

c1 = 3.5
rep = 100
mean_est_result = np.load('mean_est_result_Weibull_vs_Uniform.npy',allow_pickle=True).item()
Numeric_Result = np.load('Numeric_Result_DDMP_Gap_Weibull_vs_Uniform_c1_'+str(c1)+'.npy',allow_pickle=True).item()
sample_sizes = [10,50,100,500,1000,2000]

######## plot patience time mean estimation error ##########
a_list_sub  = [0.15,1,4,8]
sp_est = 'mean_est_gap_full_Weibull_Uniform.png'
plot_results_all_a_mean_gap(rep, sample_sizes, mean_est_result, a_list_sub,log_scale_y=0,save_path=sp_est)


a_list_sub  = [1,4,8]
sp_est = 'mean_est_gap_sub_Weibull_Uniform.png'
plot_results_all_a_mean_gap(rep, sample_sizes, mean_est_result, a_list_sub,marker_move = 1,log_scale_y=0,save_path=sp_est)

######## plot optimal value gap ##########
a_list = [0.1,0.125,0.15,0.175,0.2,0.75,1,1.25,2,4,8,16,32]
a_list_1 = [0.1,0.125,0.15,0.175,0.2,0.75,1]
a_list_2 =[1.25,2,4,8,16,32] 

la2 = 10
c = [c1,4,1]
lambdas = [1,la2,1]
J = 2
K = 1
v = [[1],[1]]
thetas = [1,1,1]

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
    
for sample_size in sample_sizes:
    print("-------------------------------------------------")
    print(f"Sample size: {sample_size}")
    Numeric_Result[sample_size,'uniform','m_gap'] = []
    Numeric_Result[sample_size,'uniform','F_gap'] = []
    for i in range(rep):
        m_sol = Numeric_Result[sample_size,'uniform','DDMP_sol'][i]
        m_gap, F_gap = optimizer_gap_uniform_abs(J, K, v, c, lambdas, thetas,m_sol, m_star_dict['uniform'],F_star_dict["uniform"])
        Numeric_Result[sample_size,'uniform','m_gap'].append(m_gap)
        Numeric_Result[sample_size,'uniform','F_gap'].append(F_gap)
        
    for a in a_list:
        Numeric_Result[sample_size,'weibull',a,'m_gap'] = []
        Numeric_Result[sample_size,'weibull',a,'F_gap'] = []
        for i in range(rep):
            m_sol = Numeric_Result[sample_size,'weibull',a,'DDMP_sol'][i]
            m_gap, F_gap = optimizer_gap_weibull_abs(J, K, v, c, lambdas, thetas,a, m_sol, m_star_dict['weibull',a],F_star_dict['weibull',a])
            Numeric_Result[sample_size,'weibull',a,'m_gap'].append(m_gap)
            Numeric_Result[sample_size,'weibull',a,'F_gap'].append(F_gap)

a_list_sub  = [0.15,1,4,8]
sp_val = 'v1_abs_val_gap_full_Weibull_Uniform_015_1_4_8.png'
plot_results_all_a_DDMP_gap(rep, sample_sizes, Numeric_Result, a_list_sub,log_scale_y=0, log_scale=1,gap_perc = 0,uniform = 1,save_path=sp_val)



