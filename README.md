# data-driven-matching-code
This file is an online supplement to the paper "Data-Driven Matching for Impatient and Heterogeneous Demand and Supply" by Weiliang Liu, Amy Ward, and Xun Zhang.

It contains codes for re-producing all the figures and tables in the paper.

Codes are categorized into two folders by their purpose.


1. Folder "Code to Produce Figure 1":
  This folder contains codes for generating Figure 1 of the paper. Specifically,

	"numerical_main_1_WeibullvsUniform.py": it generates the data for producing Figure 1. The data output is "mean_est_result_Weibull_vs_Uniform.npy" and 						"Numeric_Result_DDMP_Gap_Beta_vs_Uniform_c1_3.5.npy".

	"visualize_main_1_OptimizationError": it visualizes data obtained in "numerical_main_1_WeibullvsUniform" to produce Figure 1.

	"DDMP_MILP.py": it defines functions that are invoked in the above codes.
	

2. Folder "Code to Produce Figure 2, Tables 1 and 2":
  This folder contains codes for generating Figure 2, Tables 1 and 2 in the paper. Specifically,

	"numerical_main_2": it generates the data for producing Figure 2, Tables 1 and 2. The data output is "Numeric_Result_qt_cdf_err_range100w_rep100_a_all.npy".

	"visualize_numerical_main_2_convergence_rate.py": it visualizes data obtained in "numerical_main_2" to produce Figure 2, Tables 1 and 2.

	










