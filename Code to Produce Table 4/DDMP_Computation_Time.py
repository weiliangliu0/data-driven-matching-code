#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 09:56:58 2026

@author: leonliu
"""

import time
import math
import warnings
import numpy as np
from scipy.optimize import minimize
import scipy.stats as s
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from gurobipy import Model, GRB, quicksum, LinExpr, Env
from statsmodels.stats.diagnostic import lilliefors
from scipy.special import gammainc

# Suppress all warnings
warnings.filterwarnings("ignore")

def generate_r(J, K, theta, B, distribution='uniform', a=1,beta_para = (2,2)):
    """
    Generate random variables for Weibull or Gamma distributions with the same mean.
    Parameters:
    J (int): Number of demand nodes.
    K (int): Number of supply nodes.
    theta (list): Reneging mean for each node.
    B (list): Number of samples to generate for each node.
    distribution (str): 'gamma' or 'weibull' to specify which distribution to use.
    a (float): Shape parameter for both Gamma and Weibull distributions.
    Returns:
    dict: Generated random variables for each node.
    """
    r = {}
    for w in range(0, J + K):
        r_w = [0]  # place holder
        if distribution == 'uniform':
            r_w.extend(sorted(np.random.uniform(0, 2 / theta[w], B[w])))
            
        elif distribution == 'gamma':
            # Gamma mean = shape * scale = 1/theta[w]
            # Set shape to `a` and solve for scale to match the mean
            scale = (1 / theta[w]) / a
            r_w.extend(sorted(s.gamma.rvs(a, scale=scale, size=B[w])))
            
        elif distribution == 'weibull':
            # The mean of Weibull = b * Gamma(1 + 1/a)
            # Set b such that the mean matches 1/theta[w]
            b_weibull = (1 / theta[w]) / math.gamma(1 + 1/a)
            r_w.extend(sorted(s.weibull_min.rvs(a, scale=b_weibull, size=B[w])))
        
        elif distribution == 'beta':
            alpha, beta = beta_para
            # Scale Beta distribution to have the desired mean
            scale_factor = (1 / theta[w]) / (alpha / (alpha + beta))
            r_w.extend(sorted(s.beta.rvs(alpha, beta, size=B[w]) * scale_factor))
            
        r_w[0] = r_w[1] # r_{w0} = r_{w1}
        r[w] = r_w
    return r

def ddmp_milp(J, K, v, c, lambdas, B, r): ## check this code
    kappa = 1e-7
    # Step 1: Compute coefficients I_wb
    I = {}
    for w in range(0, J + K):
        I[w] = []
        for b in range(B[w] + 1):
            I_wb = (lambdas[w] / B[w]) * (sum(r[w][i] for i in range(1, b)) + (B[w] - max(b - 1,0)) * r[w][b])
            I[w].append(I_wb)
            
    env = Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    model = Model(env=env)        

    # Step 3: Define decision variables
    # ct decision variables {m_{jk}: j=1,..,J; k=J+1,...,J+K}
    m = model.addVars(J, K, lb=0, vtype=GRB.CONTINUOUS, name='m')

    # Binary decision variables {z_{wb}: w=1,...,J+K; b=0,...,B_w}
    z = {}
    for w in range(0, J + K):
        z[w] = model.addVars(range(B[w] + 1), vtype=GRB.BINARY, name=f'z_{w}')

    # Step 4: Set objective function
    model.setObjective(
        quicksum(v[j][k] * m[j,k] for j in range(J) for k in range(K)) -
        quicksum(c[w] * quicksum(I[w][b] * z[w][b] for b in range(B[w] + 1)) for w in range(0, J + K)),
        GRB.MAXIMIZE
    )

    # Step 5: Add constraints
    # Constraint 53b and 53c for j in J
    for j in range(J):
        for b in range(B[j] + 1):
            model.addConstr(1 - quicksum(m[j,k] for k in range(K)) / lambdas[j] <= b / B[j] + (1 - z[j][b]))
            model.addConstr(1 - quicksum(m[j,k] for k in range(K)) / lambdas[j] >= (b - 1) / B[j]+kappa - (1 - z[j][b]))

    # Constraint 53d and 53e for k in K
    for k in range(K):
        for b in range(B[J + k] + 1):
            model.addConstr(1 - quicksum(m[j,k] for j in range(J)) / lambdas[J + k] <= b / B[J + k] + (1 - z[J + k][b]))
            model.addConstr(1 - quicksum(m[j,k] for j in range(J)) / lambdas[J + k] >= (b - 1) / B[J + k]+kappa - (1 - z[J + k][b]))

    # Constraint 53f for w in W
    for w in range(0, J + K):
        model.addConstr(quicksum(z[w][b] for b in range(B[w] + 1)) == 1)

    # Additional constraints for m in M_hat
    for k in range(K):
        model.addConstr(quicksum(m[j,k] for j in range(J)) <= lambdas[J + k], name=f'm_constraint_k_{k}')
    for j in range(J):
        model.addConstr(quicksum(m[j,k] for k in range(K)) <= lambdas[j], name=f'm_constraint_j_{j}')

    # Optimize model
    #  impose a 3600-second time limit
    model.setParam("TimeLimit", 3600)
    
    model.optimize()

    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        m_solution = np.array([m[j, k].x for j in range(J) for k in range(K)]).reshape(J, K)
        #z_solution = {w: np.array([z[w][b].x for b in range(B[w] + 1)]) for w in range(0, J + K)}
        return m_solution
    else:
        return None





def run_timing_experiment(
    network_sizes=None,
    sample_sizes=None,
    n_rep=5,
    a=4,
    dist_name="weibull",
    seed=808,
):
    """
    Run timing experiments for ddmp_milp under randomized parameters.

    Parameters
    ----------
    network_sizes : list of tuple
        Each tuple is (J, K).
    sample_sizes : list of int
        Sample sizes to test.
    n_rep : int
        Number of random replications per (J, K, sample_size).
    a : float
        Parameter passed into generate_r(...).
    dist_name : str
        Distribution name passed into generate_r(...).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    raw_df : pd.DataFrame
        Raw replication-level results.
    median_table : pd.DataFrame
        Table with median (25%, 75%) runtime and number of binaries.
    mean_table : pd.DataFrame
        Table with mean runtime and 95% normal-approximation CI.
    """
    if network_sizes is None:
        network_sizes = [(2,1), (2, 2), (3, 3), (4, 4), (5, 5)]
    if sample_sizes is None:
        sample_sizes = [100,250, 500, 1000, 1500]

    rng = np.random.default_rng(seed)
    results = []

    for (J, K) in network_sizes:
        for sample_size in sample_sizes:
            for rep in range(1, n_rep + 1):
                # Randomize parameters
                # c: size J+K, each entry ~ Uniform[1,5]
                c = rng.uniform(1.0, 5.0, size=J + K).tolist()

                # lambdas: size J+K, each entry ~ Uniform[1,10]
                lambdas = rng.uniform(1.0, 10.0, size=J + K).tolist()

                # v: size J x K, each entry ~ Uniform[1,5]
                v = rng.uniform(1.0, 5.0, size=(J, K)).tolist()

                # thetas: keep fixed as in your prototype
                thetas = np.ones(J + K).tolist()

                # B: all equal to sample_size
                B = sample_size * np.ones(J + K, dtype=int)

                # Generate r
                r = generate_r(J, K, thetas, B, dist_name, a)
                
                # Print parameters
                print("=" * 80)
                print(f"network = {J}*{K}, sample_size = {sample_size}, rep = {rep}")
                print(f"c = {c}")
                print(f"lambdas = {lambdas}")
                print(f"v = {v}")
                # Time the solve
                start_time = time.perf_counter()
                m_sol = ddmp_milp(J, K, v, c, lambdas, B, r)
                end_time = time.perf_counter()

                runtime = end_time - start_time
                num_binary = (J + K) * (sample_size + 1)
                num_ct = J*K
                num_constr = 4*(J+K)

                results.append({
                    "J": J,
                    "K": K,
                    "network_size": f"{J}*{K}",
                    "sample_size": sample_size,
                    "rep": rep,
                    "runtime_sec": runtime,
                    "num_binary": num_binary,
                    "num_ct": num_ct,
                    "num_constr": num_constr,
                    # optional: keep for debugging / reproducibility
                    "m_sol": m_sol,
                })

                print(
                    f"runtime={runtime:.4f}s, binaries={num_binary}, continuous={num_ct}, constr={num_constr}"
                )

    raw_df = pd.DataFrame(results)

    # -----------------------------------------------------
    # Table 1: median (25%, 75%) + number of binaries
    # -----------------------------------------------------
    grouped = raw_df.groupby(["network_size", "sample_size"], as_index=False)

    median_summary = grouped["runtime_sec"].agg(
        median_runtime="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )

    binaries_summary = grouped["num_binary"].first()

    median_table = median_summary.merge(
        binaries_summary, on=["network_size", "sample_size"], how="left"
    )

    median_table["runtime_summary"] = median_table.apply(
        lambda row: f"{row['median_runtime']:.4f} ({row['q25']:.4f}, {row['q75']:.4f})",
        axis=1
    )

    median_table = median_table[
        ["network_size", "sample_size", "num_binary", "runtime_summary"]
    ].sort_values(["network_size", "sample_size"]).reset_index(drop=True)

    # -----------------------------------------------------
    # Table 2: mean + 95% CI using normal approximation
    # -----------------------------------------------------
    mean_summary = grouped["runtime_sec"].agg(
        mean_runtime="mean",
        std_runtime="std",
        n="count",
    )

    z = 1.96
    mean_summary["se"] = mean_summary["std_runtime"] / np.sqrt(mean_summary["n"])
    mean_summary["ci_lower"] = mean_summary["mean_runtime"] - z * mean_summary["se"]
    mean_summary["ci_upper"] = mean_summary["mean_runtime"] + z * mean_summary["se"]

    mean_table = mean_summary.merge(
        binaries_summary, on=["network_size", "sample_size"], how="left"
    )

    mean_table["runtime_mean_ci"] = mean_table.apply(
        lambda row: f"{row['mean_runtime']:.4f} ({row['ci_lower']:.4f}, {row['ci_upper']:.4f})",
        axis=1
    )

    mean_table = mean_table[
        ["network_size", "sample_size", "num_binary", "runtime_mean_ci"]
    ].sort_values(["network_size", "sample_size"]).reset_index(drop=True)

    return raw_df, median_table, mean_table


def make_pivot_table(summary_df, value_col):
    """
    Convert a long summary table into a pivot table:
    rows = network_size, columns = sample_size
    """
    pivot = summary_df.pivot(
        index="network_size",
        columns="sample_size",
        values=value_col
    )
    return pivot


# =========================================================
# Run the experiment
# =========================================================
if __name__ == "__main__":
    raw_df, median_table, mean_table = run_timing_experiment(
        network_sizes=[(2,1),(2,2),(3,3),(4, 4),(5,5)],
        sample_sizes=[100, 250, 500, 1000, 1500],
        n_rep=5,
        a=4,
        dist_name="weibull",
        seed=808,
    )

    # Print raw summaries
    print("\n=== Table 1: Median (25%, 75%) Runtime + Number of Binaries ===")
    print(median_table.to_string(index=False))

    # Optional: pivot tables for a cleaner paper-style layout
    median_pivot = make_pivot_table(median_table, "runtime_summary")
   

    print("\n=== Pivot: Median (25%, 75%) Runtime ===")
    print(median_pivot.to_string())
    
    raw_df.to_csv("Weibull_4_timing_experiment_raw.csv", index=False)
    median_pivot.to_csv("Weibull_4_timing_experiment_median_pivot.csv")