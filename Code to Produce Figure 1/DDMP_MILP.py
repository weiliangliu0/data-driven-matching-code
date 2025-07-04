#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:28:36 2024

@author: leonliu
"""
import math
import lin.polyhedron as ph
import numpy as np
from scipy.optimize import minimize
import scipy.stats as s
#import pulp as p
#import copy
#import csv
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from gurobipy import Model, GRB, quicksum, LinExpr, Env
from statsmodels.stats.diagnostic import lilliefors
from scipy.special import gammainc



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



def ddmp_exp(J, K, v, c, lambdas, B, r):
    htheta = []
    for w in range(0, J + K):
        mean_value = np.mean(r[w][1:])  # Calculate mean for b=1,...,B
        htheta.append(1/mean_value)
    
    env = Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    model = Model(env=env) 
    
    m = model.addVars(J, K, lb=0, vtype=GRB.CONTINUOUS, name='m')
    
    model.setObjective(
        quicksum(v[j][k] * m[j,k] for j in range(J) for k in range(K)) -
        quicksum(c[j]*lambdas[j]/htheta[j] * (1-quicksum(m[j,k] for k in range(K))/lambdas[j]) for j in range(J))-
        quicksum(c[J+k]*lambdas[J+k]/htheta[J+k] * (1-quicksum(m[j,k] for j in range(J))/lambdas[J+k]) for k in range(K)),
        GRB.MAXIMIZE
    )
    # Additional constraints for m in M_hat
    for k in range(K):
        model.addConstr(quicksum(m[j,k] for j in range(J)) <= lambdas[J + k], name=f'm_constraint_k_{k}')
    for j in range(J):
        model.addConstr(quicksum(m[j,k] for k in range(K)) <= lambdas[j], name=f'm_constraint_j_{j}')

    # Optimize model
    model.optimize()
    model.Params.TimeLimit = 60 #60 seconds
    model.optimize()

    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        m_solution = np.array([m[j, k].x for j in range(J) for k in range(K)]).reshape(J, K)
        return m_solution
    else:
        return None  
    return 0

def ddmp_milp_with_test(J, K, v, c, lambdas, B, r, test_result):
    # test_result = 1 if reject the null-hypothesis that the data follows exponential distribution
    kappa = 1e-7
    
    # Step 1: Compute coefficients I_wb and htheta
    I = {}
    for w in range(0, J + K):
        I[w] = []
        for b in range(B[w] + 1):
            I_wb = (lambdas[w] / B[w]) * (sum(r[w][i] for i in range(1, b)) + (B[w] - max(b - 1,0)) * r[w][b])
            I[w].append(I_wb)
    htheta = []
    for w in range(0, J + K):
        mean_value = np.mean(r[w][1:])  # Calculate mean for b=1,...,B
        htheta.append(1/mean_value)

    env = Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    model = Model(env=env) 
    obj  = LinExpr() 
    
    # Step 3: Define decision variables, 
    # Continuous decision variables {m_{jk}: j=1,..,J; k=J+1,...,J+K}
    m = model.addVars(J, K, lb=0, vtype=GRB.CONTINUOUS, name='m')
    obj +=  quicksum(v[j][k] * m[j,k] for j in range(J) for k in range(K))
    # Binary decision variables {z_{wb}: w=1,...,J+K; b=0,...,B_w}
    z = {}
    for w in range(0, J + K):
        if test_result[w] == 1: #reject the null that data follows an exponential distribution
            z[w] = model.addVars(range(B[w] + 1), vtype=GRB.BINARY, name=f'z_{w}')
            obj -= c[w] * quicksum(I[w][b] * z[w][b] for b in range(B[w] + 1))
            model.addConstr(quicksum(z[w][b] for b in range(B[w] + 1)) == 1)
            if w < J:
                j = w
                for b in range(B[j] + 1):
                    model.addConstr(1 - quicksum(m[j,k] for k in range(K)) / lambdas[j] <= b / B[j] + (1 - z[j][b]))
                    model.addConstr(1 - quicksum(m[j,k] for k in range(K)) / lambdas[j] >= (b - 1) / B[j]+kappa - (1 - z[j][b]))
            elif w >= J:
                k = w - J
                for b in range(B[J + k] + 1):
                    model.addConstr(1 - quicksum(m[j,k] for j in range(J)) / lambdas[J + k] <= b / B[J + k] + (1 - z[J + k][b]))
                    model.addConstr(1 - quicksum(m[j,k] for j in range(J)) / lambdas[J + k] >= (b - 1) / B[J + k]+kappa - (1 - z[J + k][b]))
        elif test_result[w] != 1: # accept the null that data follows an exponential distribution
            if w < J:
                j = w
                obj -= c[j]*lambdas[j]/htheta[j] * (1-quicksum(m[j,k] for k in range(K))/lambdas[j])
            elif w >= J:
                k = w - J
                obj -= c[J+k]*lambdas[J+k]/htheta[J+k] * (1-quicksum(m[j,k] for j in range(J))/lambdas[J+k])
    # Additional constraints for m in M_hat
    for k in range(K):
        model.addConstr(quicksum(m[j,k] for j in range(J)) <= lambdas[J + k], name=f'm_constraint_k_{k}')
    for j in range(J):
        model.addConstr(quicksum(m[j,k] for k in range(K)) <= lambdas[j], name=f'm_constraint_j_{j}')
        
    # Step 4: Set objective function
    model.setObjective(obj,GRB.MAXIMIZE)
    model.Params.TimeLimit = 60 #60 seconds
    model.optimize()

    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        m_solution = np.array([m[j, k].x for j in range(J) for k in range(K)]).reshape(J, K)
        #z_solution = {w: np.array([z[w][b].x for b in range(B[w] + 1)]) for w in range(0, J + K)}
        return m_solution
    else:
        return None

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
    # Continuous decision variables {m_{jk}: j=1,..,J; k=J+1,...,J+K}
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
    model.optimize()
    model.Params.TimeLimit = 60 #60 seconds
    model.optimize()

    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        m_solution = np.array([m[j, k].x for j in range(J) for k in range(K)]).reshape(J, K)
        #z_solution = {w: np.array([z[w][b].x for b in range(B[w] + 1)]) for w in range(0, J + K)}
        return m_solution
    else:
        return None

def find_extreme_pt(J, K,lambdas):
    # Construct matrix A
    lambdaD = lambdas[:J]  # Demand node arrival rates
    lambdaS = lambdas[J:]
    A1 = np.zeros((J + K, J * K), dtype=int)
    for j in range(0, J):
        for k in range(0, K):
            A1[j][j * K + k] = 1
            A1[J + k][j * K + k] = 1
    A2 = np.diag(np.full(J * K, 1))
    A = np.concatenate((A1, A2))
    # Construct vector b
    b = []
    for i in range(0, J):
        b.append(lambdaD[i])
    for i in range(0, K):
        b.append(lambdaS[i])
    for i in range(0, J * K):
        b.append(0)

    # Construct condition string d
    d1 = []
    for i in range(0, J + K + 1):
        if i == 0:
            d1.append('[')
        else:
            d1.append('<''='',')
    for i in range(J + K + 2, J + K + J * K + 2):
        if i == J + K + J * K + 1:
            d1.append('>''=')
        else:
            d1.append('>''='',')
    d1.append(']')
    d = ''.join(d1)

    # Compute extreme points
    extremes1 = ph.extreme_points(A, b, d)
    extremes2 = set(tuple(x) for x in extremes1)
    extremes = list(list(x) for x in extremes2)
    return extremes

def queue_len_uniform(J, lambdas, thetas, m):
    K = len(lambdas) - J
    lambdaD = lambdas[:J]  # Demand node arrival rates
    lambdaS = lambdas[J:]  # Supply node arrival rates
    thetaD = thetas[:J]  # Demand reneging mean
    thetaS = thetas[J:]   # Supply reneging mean
    qD = []
    qS = []
    for j in range(0, J):
        xD = 1 - max(min((sum(m[h] for h in range(j * K, (j + 1) * K)) / lambdaD[j]),1),0)**2
        qD_j = (lambdaD[j] / thetaD[j]) * xD
        qD.append(qD_j)
    for k in range(0, K):
        xS = 1 - max(min((sum(m[h] for h in range(k, J * K, K)) / lambdaS[k]),1),0)**2
        qS_k = (lambdaS[k] / thetaS[k]) * xS
        qS.append(qS_k)
    return qD,qS

def queue_len_exp(J,lambdas, thetas, m):
    K = len(lambdas) - J
    lambdaD = lambdas[:J]  # Demand node arrival rates
    lambdaS = lambdas[J:]  # Supply node arrival rates
    thetaD = thetas[:J]  # Demand reneging mean
    thetaS = thetas[J:]   # Supply reneging mean
    qD = []
    qS = []
    for j in range(0, J):
        xD = 1 - sum(m[h] for h in range(j * K, (j + 1) * K)) / lambdaD[j]
        qD_j = (lambdaD[j] / thetaD[j]) * xD
        qD.append(qD_j)
    for k in range(0, K):
        xS = 1 - sum(m[h] for h in range(k, J * K, K)) / lambdaS[k]
        qS_k = (lambdaS[k] / thetaS[k]) * xS
        qS.append(qS_k)
    return qD,qS

### a is the shape parameter of gamma distribution; a > 1
#Take in the scale (a) and the rate parameters of the gamma distribution
#Return the function as the integrand of the integration function below
def integrand_gamma(u, a, mean_renege_rate):
	return mean_renege_rate * (1 - s.gamma.cdf(u, a, 0, 1/(a*mean_renege_rate)))

#Take in the scale and the rate parameters of the gamma distribution
#Calculate the integration int_0^x \theta [1-G(u)] du
def integration_gamma(a, mean_renege_rate, x):
	return scipy.integrate.quad(integrand_gamma, 0, x, args = (a, mean_renege_rate))[0]

def queue_len_gamma(J,lambdas, thetas, m,a):
    K = len(lambdas) - J
    lambdaD = lambdas[:J]  # Demand node arrival rates
    lambdaS = lambdas[J:]  # Supply node arrival rates
    thetaD = thetas[:J]  # Demand reneging mean
    thetaS = thetas[J:]   # Supply reneging mean
    qD = []
    qS = []
    for j in range(0, J):
        xD_prime = max(min(1-(sum(m[h] for h in range(j*K,(j+1)*K)))/lambdaD[j],1),0)
        xD = s.gamma.ppf(xD_prime, a, 0, 1/(a*thetaD[j]))
        qD_j = (lambdaD[j]/thetaD[j]) * integration_gamma(a, thetaD[j], xD)
        qD.append(qD_j)
    for k in range(0, K):
        xS_prime = max(min(1-(sum(m[h] for h in range(k, J * K, K)))/lambdaS[k],1),0)
        xS = s.gamma.ppf(xS_prime, a, 0, 1/(a*thetaS[k]))
        qS_k = (lambdaS[k] / thetaS[k]) * integration_gamma(a, thetaS[k], xS)
        qS.append(qS_k)
    return qD,qS

def integrand_beta(u, beta_para, mean_renege_rate):
    alpha, beta = beta_para
    scale_factor = (1/mean_renege_rate) / (alpha / (alpha + beta))
    return mean_renege_rate * (1 - s.beta.cdf(u / scale_factor, alpha, beta))

def integration_beta(beta_para, mean_renege_rate, x):
	return scipy.integrate.quad(integrand_beta, 0, x, args = (beta_para, mean_renege_rate))[0]

def queue_len_beta(J,lambdas, thetas, m,beta_para):
    K = len(lambdas) - J
    lambdaD = lambdas[:J]  # Demand node arrival rates
    lambdaS = lambdas[J:]  # Supply node arrival rates
    thetaD = thetas[:J]  # Demand reneging mean
    thetaS = thetas[J:]   # Supply reneging mean
    qD = []
    qS = []
    alpha, beta = beta_para
    for j in range(0, J):
        scale_factor = (1/thetaD[j]) / (alpha / (alpha + beta))
        xD_prime = max(min(1-(sum(m[h] for h in range(j*K,(j+1)*K)))/lambdaD[j],1),0)
        xD =  s.beta.ppf(xD_prime, alpha, beta) * scale_factor
        qD_j = (lambdaD[j]/thetaD[j]) * integration_beta(beta_para, thetaD[j], xD)
        qD.append(qD_j)
    for k in range(0, K):
        scale_factor = (1/thetaS[k]) / (alpha / (alpha + beta))
        xS_prime = max(min(1-(sum(m[h] for h in range(k, J * K, K)))/lambdaS[k],1),0)
        xS = s.beta.ppf(xS_prime, alpha, beta) * scale_factor
        qS_k = (lambdaS[k] / thetaS[k]) * integration_beta(beta_para, thetaS[k], xS)
        qS.append(qS_k)
    return qD,qS

def obj_beta(J, K, v, c, lambdas, thetas, m, beta_para):
    reward = [item for row in v for item in row]  # Matching rewards
    costD = c[:J]  # Demand per-unit-time waiting cost
    costS = c[J:]  # Supply per-unit-time waiting cost
    obj_queue_costD = 0
    obj_queue_costS = 0
    qD,qS = queue_len_beta(J,lambdas, thetas, m,beta_para)
    obj_match_value = np.vdot(reward, m)
    for j in range(0, J):
        obj_queue_costD += costD[j] * qD[j]
    for k in range(0, K):
        obj_queue_costS += costS[k] * qS[k]
    objective = obj_match_value - obj_queue_costD - obj_queue_costS
    return objective

def gtmp_beta_GridSearch(J, K, v, c, lambdas, thetas, grid, beta_para):
    obj = []
    for m in grid:
        objective = obj_beta(J, K, v, c, lambdas, thetas, m, beta_para)
        obj.append(objective)
    # Find the optimal value and position
    optimal_value = max(obj)
    position = obj.index(optimal_value)
    m_star = grid[position]
    m_star = [m_star[x:x+K] for x in range(0, len(m_star), K)]
    return m_star, optimal_value 

def optimizer_gap_beta(J, K, v, c, lambdas, thetas,beta_para, m_sol, m_star,F_star):
    m_sol_flat = np.array([item for row in m_sol for item in row])
    m_star_flat = np.array([item for row in m_star for item in row])
    F_sol = obj_beta(J, K, v, c, lambdas, thetas, m_sol_flat,beta_para)
    F_gap_perc = (F_star - F_sol)/np.abs(F_star)*100
    m_gap = np.sum(np.abs(m_sol_flat - m_star_flat))
    return m_gap,F_gap_perc

def optimizer_gap_beta_abs(J, K, v, c, lambdas, thetas,beta_para, m_sol, m_star,F_star):
    m_sol_flat = np.array([item for row in m_sol for item in row])
    m_star_flat = np.array([item for row in m_star for item in row])
    F_sol = obj_beta(J, K, v, c, lambdas, thetas, m_sol_flat,beta_para)
    F_gap_abs = F_star - F_sol
    m_gap = np.sum(np.abs(m_sol_flat - m_star_flat))
    return m_gap,F_gap_abs

#Take in the scale and the rate parameters of the weibull distribution; a > 1
def integrand_weibull(u, a, mean_renege_rate):
    scale = (1 / mean_renege_rate) / math.gamma(1 + 1/a)
    return mean_renege_rate * (1 - s.weibull_min.cdf(u, a, scale=scale))

def integration_weibull(a, mean_renege_rate, x):
	return scipy.integrate.quad(integrand_weibull, 0, x, args = (a, mean_renege_rate))[0]

def integration_weibull_closedform(a, mean_renege_rate, x):
    scale = (1 / mean_renege_rate) / math.gamma(1 + 1/a) 
    return scale/a*math.gamma(1/a)*gammainc(1/a,(x/scale)**a)
#x = 100
#val1 = integration_weibull(a, mean_renege_rate, 3)
#va12 = integration_weibull_closedform(a, mean_renege_rate, 3)

def queue_len_weibull(J, lambdas, thetas, m,a):
    K = len(lambdas) - J
    lambdaD = lambdas[:J]  # Demand node arrival rates
    lambdaS = lambdas[J:]  # Supply node arrival rates
    thetaD = thetas[:J]  # Demand reneging mean
    thetaS = thetas[J:]   # Supply reneging mean
    qD = []
    qS = []
    for j in range(0, J):
        scale = (1 / thetaD[j]) / math.gamma(1 + 1/a)
        xD_prime = max(min(1-(sum(m[h] for h in range(j*K,(j+1)*K)))/lambdaD[j],1),0)
        xD = s.weibull_min.ppf(xD_prime, a, scale= scale)
        #qD_j = (lambdaD[j]/thetaD[j]) * integration_weibull(a, thetaD[j], xD)
        qD_j = lambdaD[j] * integration_weibull_closedform(a, thetaD[j], xD)
        qD.append(qD_j)
    for k in range(0, K):
        scale = (1 / thetaS[k]) / math.gamma(1 + 1/a)
        xS_prime = max(min(1-(sum(m[h] for h in range(k, J * K, K)))/lambdaS[k],1),0)
        xS = s.weibull_min.ppf(xS_prime, a, scale = scale)
        #qS_k = (lambdaS[k] / thetaS[k]) * integration_weibull(a, thetaS[k], xS)
        qS_k = lambdaS[k] * integration_weibull_closedform(a, thetaS[k], xS)
        qS.append(qS_k)
    return qD,qS

def obj_weibull(J, K, v, c, lambdas, thetas, m,a):
    reward = [item for row in v for item in row]  # Matching rewards
    costD = c[:J]  # Demand per-unit-time waiting cost
    costS = c[J:]  # Supply per-unit-time waiting cost
    obj_queue_costD = 0
    obj_queue_costS = 0
    qD,qS = queue_len_weibull(J,lambdas, thetas, m,a)
    obj_match_value = np.vdot(reward, m)
    for j in range(0, J):
        obj_queue_costD += costD[j] * qD[j]
    for k in range(0, K):
        obj_queue_costS += costS[k] * qS[k]
    objective = obj_match_value - obj_queue_costD - obj_queue_costS
    return objective
        

def obj_uniform(J, K, v, c, lambdas, thetas, m):
    reward = [item for row in v for item in row]  # Matching rewards
    costD = c[:J]  # Demand per-unit-time waiting cost
    costS = c[J:]  # Supply per-unit-time waiting cost
    obj_queue_costD = 0
    obj_queue_costS = 0
    qD,qS = queue_len_uniform(J,lambdas, thetas, m)
    obj_match_value = np.vdot(reward, m)
    for j in range(0, J):
        obj_queue_costD += costD[j] * qD[j]
    for k in range(0, K):
        obj_queue_costS += costS[k] * qS[k]
    objective = obj_match_value - obj_queue_costD - obj_queue_costS
    return objective

def obj_exp(J, K, v, c, lambdas, thetas, m):
    reward = [item for row in v for item in row]  # Matching rewards
    costD = c[:J]  # Demand per-unit-time waiting cost
    costS = c[J:]  # Supply per-unit-time waiting cost
    obj_queue_costD = 0
    obj_queue_costS = 0
    qD,qS = queue_len_exp(J,lambdas, thetas, m)
    obj_match_value = np.vdot(reward, m)
    for j in range(0, J):
        obj_queue_costD += costD[j] * qD[j]
    for k in range(0, K):
        obj_queue_costS += costS[k] * qS[k]
    objective = obj_match_value - obj_queue_costD - obj_queue_costS
    return objective

def obj_gamma(J, K, v, c, lambdas, thetas, m,a):
    reward = [item for row in v for item in row]  # Matching rewards
    costD = c[:J]  # Demand per-unit-time waiting cost
    costS = c[J:]  # Supply per-unit-time waiting cost
    obj_queue_costD = 0
    obj_queue_costS = 0
    qD,qS = queue_len_gamma(J,lambdas, thetas, m,a)
    obj_match_value = np.vdot(reward, m)
    for j in range(0, J):
        obj_queue_costD += costD[j] * qD[j]
    for k in range(0, K):
        obj_queue_costS += costS[k] * qS[k]
    objective = obj_match_value - obj_queue_costD - obj_queue_costS
    return objective

    
def gtmp_uniform(J, K, v, c, lambdas, thetas):
    extremes = find_extreme_pt(J, K,lambdas)
    # Evaluate objectives
    obj = []
    for i in range(len(extremes)):
        m = extremes[i]
        objective = obj_uniform(J, K, v, c, lambdas, thetas, m)
        obj.append(objective)
    # Find the optimal value and position
    optimal_value = max(obj)
    position = obj.index(optimal_value)
    m_star=extremes[position]
    m_star = [m_star[x:x+K] for x in range(0, len(m_star), K)]
    return m_star, optimal_value  

def gtmp_exp(J, K, v, c, lambdas, thetas):
    extremes = find_extreme_pt(J, K,lambdas)
    # Evaluate objectives
    obj = []
    for i in range(len(extremes)):
        m = extremes[i]
        objective = obj_exp(J, K, v, c, lambdas, thetas, m)
        obj.append(objective)
    # Find the optimal value and position
    optimal_value = max(obj)
    position = obj.index(optimal_value)
    m_star=extremes[position]
    m_star = [m_star[x:x+K] for x in range(0, len(m_star), K)]
    return m_star, optimal_value      

def gtmp_gamma(J, K, v, c, lambdas, thetas,a):
    extremes = find_extreme_pt(J, K,lambdas)
    # Evaluate objectives
    obj = []
    for i in range(len(extremes)):
        m = extremes[i]
        objective = obj_gamma(J, K, v, c, lambdas, thetas, m,a)
        obj.append(objective)
    # Find the optimal value and position
    optimal_value = max(obj)
    position = obj.index(optimal_value)
    m_star=extremes[position]
    m_star = [m_star[x:x+K] for x in range(0, len(m_star), K)]
    return m_star, optimal_value  

#### for shape a>1, increasing hazard rate
####  : there exists an optimal extreme point solution
def gtmp_weibull_extr_pt(J, K, v, c, lambdas, thetas,a):
    extremes = find_extreme_pt(J, K,lambdas)
    # Evaluate objectives
    obj = []
    for i in range(len(extremes)):
        m = extremes[i]
        objective = obj_weibull(J, K, v, c, lambdas, thetas, m,a)
        obj.append(objective)
    # Find the optimal value and position
    optimal_value = max(obj)
    position = obj.index(optimal_value)
    m_star = extremes[position]
    m_star = [m_star[x:x+K] for x in range(0, len(m_star), K)]
    return m_star, optimal_value  

#### for shape a<1, decreasing hazard rate
####  : convex optimization problem
def gtmp_weibull_cvx(J, K, v, c, lambdas, thetas, a):
    # Objective function to minimize (negative of original objective to maximize)
    def objective(m):
        return - obj_weibull(J, K, v, c, lambdas, thetas, m,a)
    # Constraints to ensure the feasibility of the decision variables
    def demand_constraints(m, j):
        return lambdas[j] - sum(m[j * K:(j + 1) * K])
    def supply_constraints(m, k):
        return lambdas[J + k] - sum(m[k:J * K:K])
    # Initial guess for the decision variables
    m0 = [min(lambdas)/max(J,K)]* (J * K)  #[0] * (J * K)
    # Define bounds for each element of m (m >= 0)
    bounds = [(0, None) for _ in range(J * K)]
    # Define constraints for demand and supply
    constraints = []
    for j in range(J):
        constraints.append({'type': 'ineq', 'fun': demand_constraints, 'args': (j,)})
    for k in range(K):
        constraints.append({'type': 'ineq', 'fun': supply_constraints, 'args': (k,)})
    result = minimize(objective, m0, method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_value = -result.fun
    optimal_solution = result.x
    optimal_solution  = [list(optimal_solution[x:x+K]) for x in range(0, len(optimal_solution), K)]
    return optimal_solution, optimal_value

#### a unifying gtmp for weibull distribution based on shape parameter a
def gtmp_weibull(J, K, v, c, lambdas, thetas, a):
    if a>= 1:
        m_star, optimal_value = gtmp_weibull_extr_pt(J, K, v, c, lambdas, thetas,a)
    elif a < 1:
        m_star, optimal_value = gtmp_weibull_cvx(J, K, v, c, lambdas, thetas,a)
    return m_star, optimal_value  

def optimizer_gap_weibull(J, K, v, c, lambdas, thetas,a, m_sol, m_star,F_star):
    m_sol_flat = np.array([item for row in m_sol for item in row])
    m_star_flat = np.array([item for row in m_star for item in row])
    F_sol = obj_weibull(J, K, v, c, lambdas, thetas, m_sol_flat,a)
    F_gap_perc = (F_star - F_sol)/np.abs(F_star)*100
    m_gap = np.sum(np.abs(m_sol_flat - m_star_flat))
    return m_gap,F_gap_perc

def optimizer_gap_uniform(J, K, v, c, lambdas, thetas, m_sol, m_star,F_star):
    m_sol_flat = np.array([item for row in m_sol for item in row])
    m_star_flat = np.array([item for row in m_star for item in row])
    F_sol = obj_uniform(J, K, v, c, lambdas, thetas, m_sol_flat)
    F_gap_perc = (F_star - F_sol)/np.abs(F_star)*100
    m_gap = np.sum(np.abs(m_sol_flat - m_star_flat))
    return m_gap,F_gap_perc


def optimizer_gap_weibull_abs(J, K, v, c, lambdas, thetas,a, m_sol, m_star,F_star):
    m_sol_flat = np.array([item for row in m_sol for item in row])
    m_star_flat = np.array([item for row in m_star for item in row])
    F_sol = obj_weibull(J, K, v, c, lambdas, thetas, m_sol_flat,a)
    F_gap_perc = (F_star - F_sol)
    m_gap = np.sum(np.abs(m_sol_flat - m_star_flat))
    return m_gap,F_gap_perc

def optimizer_gap_uniform_abs(J, K, v, c, lambdas, thetas, m_sol, m_star,F_star):
    m_sol_flat = np.array([item for row in m_sol for item in row])
    m_star_flat = np.array([item for row in m_star for item in row])
    F_sol = obj_uniform(J, K, v, c, lambdas, thetas, m_sol_flat)
    F_gap_perc = (F_star - F_sol)
    m_gap = np.sum(np.abs(m_sol_flat - m_star_flat))
    return m_gap,F_gap_perc

