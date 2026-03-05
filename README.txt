This file is an online supplement to the paper "Data-Driven Matching for Impatient and Heterogeneous Demand and Supply" by Weiliang Liu, Amy Ward, and Xun Zhang.

It contains codes for re-producing all the figures and tables in the paper.

Codes are categorized into four folders by their purpose.


1. Folder "Code to Produce Figure 1":
  This folder contains codes for generating Figure 1 of the paper. Specifically,

	"numerical_main_1_WeibullvsUniform.py": it generates the data for producing Figure 1. The data output is "mean_est_result_Weibull_vs_Uniform.npy" and 						"Numeric_Result_DDMP_Gap_Beta_vs_Uniform_c1_3.5.npy".

	"visualize_main_1_OptimizationError.py": it visualizes data obtained in "numerical_main_1_WeibullvsUniform" to produce Figure 1.

	"DDMP_MILP.py": it defines functions that are invoked in the above codes.
	


2. Folder "Code to Produce Figure 3, Tables 1 and 2":
  This folder contains codes for generating Figure 3, Tables 1 and 2 in the paper. Specifically,

	"numerical_main_2.py": it generates the data for producing Figure 3, Tables 1 and 2. The data output is "Numeric_Result_qt_cdf_err_range100w_rep100_a_all.npy".

	"visualize_numerical_main_2_convergence_rate.py": it visualizes data obtained in "numerical_main_2" to produce Figure 3, Tables 1 and 2.



3. Folder "Code to Produce Figure 4":
   "visualize_thm7.py": it generates Figure 4 in the paper. The generated figure is "p_gamma_VaryB_FixEps1_Thm7.png". This code also analyzes the relevant functions that 			appear in Theorem 7 of the paper.



4. Folder "Code to Produce Table 4":
  "DDMP_Computation_Time.py": it generates the data for producing Table 4 in Appendix E.5 of the paper. The data output is "Weibull_4_timing_experiment_median_pivot.csv" and          				"Weibull_4_timing_experiment_raw.csv"
	










