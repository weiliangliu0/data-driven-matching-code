#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 14:05:29 2025

@author: leonliu
"""


import time
import seaborn as sns
import pandas as pd
import scipy.stats as s
import scipy
import warnings
import statsmodels.api as sm
from scipy.stats import t
from scipy.special import gammaincc, gamma,gammainccinv,lambertw
import matplotlib.ticker as ticker

# Visualizing Theorem 7: helper functions and a runnable demo.
# - Implements Φ, Ψ (three parts), Λ and ζ following the paper's definitions.
# - Lets you plug in your own distribution G (cdf), tail-integral inverse y(ε), and ρ(δ).
# - Demo uses a simple exponential distribution as a placeholder to generate p(β) and γ(β) curves
#   for sample sizes B=100 and B=1000 with ε=0.01. You can swap in your own pieces.

import numpy as np
import math
import matplotlib.pyplot as plt

# ---------- Numerics helpers ----------
def _binom_coeff_log(B, k):
    """log of binomial coefficient: ln(C(B,k)) using lgamma to avoid overflow."""
    if k < 0 or k > B:
        return -np.inf
    return math.lgamma(B + 1) - math.lgamma(k + 1) - math.lgamma(B - k + 1)

def _safe_pos(x):
    return x if x > 0 else 0.0

# ---------- Distribution example (you can replace this) ----------
class ExponentialDist:
    """
    Exponential distribution with rate theta.
    - G(u) = 1 - exp(-theta u)    (cdf)
    - y_eps solves ∫_y^∞ (1 - G(u)) du = ε, which gives y = -(1/theta) ln(theta ε)
    - ρ(δ) is a lower bound on g(G^{-1}(x)) over x∈[δ/2,1-δ/2]; for Exp(θ), g(G^{-1}(x)) = \theta*(1-x) ≥ \theta*\delta/2
    """
    def __init__(self, theta=1.0):
        self.theta = float(theta)

    def G(self, u):
        u = np.maximum(u, 0.0)
        return 1.0 - np.exp(-self.theta * u)

    def y_eps(self, eps):
        # y satisfying (1/theta) * exp(-theta*y) = eps  ->  y = -(1/theta) * ln(theta*eps)
        arg = self.theta * max(eps, 1e-300)
        y = -(1.0 / self.theta) * math.log(arg)
        return max(0.0, y)

    def rho(self, delta):
        delta = float(delta)
        return self.theta *delta/2
    
class UniformDist:
    """
    Uniform reneging distribution on [0,a].
    - G(u) = x/a    (cdf)
    - y_eps solves ∫_y^∞ (1 - G(u)) du = ε, which gives y = a - sqrt(2a*eps)
    - ρ(δ) is a lower bound on g(G^{-1}(x)), which is a constant 1/a
    """
    def __init__(self, a=2.0):
        self.a = float(a)

    def G(self, u):
        u = np.minimum(np.maximum(u, 0.0),self.a)
        return u/self.a

    def y_eps(self, eps):
        y = self.a - math.sqrt(2*self.a*eps)
        return max(0.0, y)

    def rho(self, delta):
        return 1/self.a

# ---------- Paper functions ----------
#### for L1 Wasserstein distance between G and hG ####
def Phi(B, eps, a1, a2):
    """Φ(B; ε) := a1 * exp(-a2 * B * ε^2). Eq. (13)
       Exp(theta): a1 = 8, a2 = theta/32, for eps leq 1
       (Fournier-Guilin 2019 explicit for Exp(theta),d=1 (univariate distribution), p =1 (L1-Wasserstein))
       Unif([0, 2/theta]): a1 = 2, a2 = theta^2/2; from DKW
    """
    return float(a1) * math.exp(-float(a2) * B * (eps ** 2))

def _exp_from_log(logx):
    # guard overflow/underflow thresholds for double precision
    if logx < -745:  # ~np.log(np.finfo(float).tiny)
        return 0.0
    if logx > 709:   # ~np.log(np.finfo(float).max)
        return float("inf")
    return math.exp(logx)

def Psi1(B, delta, eps, G_cdf, y_func):
    """
    Ψ_1(B; δ, ε) := (Bδ + 1) * C(B, ceil(B(1-δ))) * [ G(y(ε)) ]^{B(1-δ)}. Eq. (16)
    """
    k = int(math.ceil(B * (1.0 - delta)))
    logC = _binom_coeff_log(B, k)
    logval = math.log(B * delta + 1.0) + logC + B * (1.0 - delta) * math.log(float(G_cdf(y_func(eps))))
    return _exp_from_log(logval)

def Psi2(B, delta, eps, G_cdf, C_f2=1.0):
    """
    Ψ_2(B; δ, ε) := (Bδ + 2) * C(B, ceil(Bδ)) * [ 1 - G(ε / C_f2) ]^{B - Bδ - 1}. Eq. (16)
    For f = 1 - G we have C_f2 = 1.
    """
    k = int(math.ceil(B * delta))
    logC = _binom_coeff_log(B, k)
    term = 1.0 - float(G_cdf(eps / float(C_f2)))
    power = B - B * delta - 1.0
    logval = math.log(B * delta + 2.0) + logC + power * math.log(term)
    return _exp_from_log(logval)

def Psi3(B, delta, eps, rho_func):
    """
    Ψ_3(B; δ, ε) := 2 * exp( - [ B * min(ε ρ(δ), δ) ]^2 / 2 ). Eq. (16)
    """
    a = min(float(eps) * float(rho_func(delta)), float(delta))
    return 2.0 * math.exp(-B*(a ** 2) / 2)

def logPsi1(B, delta, eps, G_cdf, y_func):
    """
    Ψ_1(B; δ, ε) := (Bδ + 1) * C(B, ceil(B(1-δ))) * [ G(y(ε)) ]^{B(1-δ)}. Eq. (16)
    """
    k = int(math.ceil(B * (1.0 - delta)))
    logC = _binom_coeff_log(B, k)
    logval = math.log(B * delta + 1.0) + logC + B * (1.0 - delta) * math.log(float(G_cdf(y_func(eps))))
    return logval

def logPsi2(B, delta, eps, G_cdf, C_f2=1.0):
    """
    Ψ_2(B; δ, ε) := (Bδ + 2) * C(B, ceil(Bδ)) * [ 1 - G(ε / C_f2) ]^{B - Bδ - 1}. Eq. (16)
    For f = 1 - G we have C_f2 = 1.
    """
    k = int(math.ceil(B * delta))
    logC = _binom_coeff_log(B, k)
    term = 1.0 - float(G_cdf(eps / float(C_f2)))
    power = B - B * delta - 1.0
    logval = math.log(B * delta + 2.0) + logC + power * math.log(term)
    return logval

def logPsi3(B, delta, eps, rho_func):
    """
    Ψ_3(B; δ, ε) := 2 * exp( - B*[ min(ε ρ(δ), δ) ]^2 / 2 ). Eq. (16)
    """
    a = min(float(eps) * float(rho_func(delta)), float(delta))
    return math.log(2.0) - B*(a ** 2) / 2

def logsumexp(vals):
    m = max(vals)
    if not np.isfinite(m):
        return -np.inf
    s = sum(math.exp(v - m) for v in vals)
    return m + math.log(s)

def Psi_sum_log(B, delta, eps, dist):
    return logsumexp([
        logPsi1(B, delta, eps, dist.G, dist.y_eps),
        logPsi2(B, delta, eps, dist.G),
        logPsi3(B, delta, eps, dist.rho),
    ])

def choose_delta_optimal(B, eps1, dist, tol=1e-10, max_iter=2000):
    #eps1 = nu1*(1-beta)*eps/6
    U = min(0.5, 1.0 - dist.G(dist.y_eps(eps1)), dist.G(eps1))
    if U <= 0.0:
        return 0.0, float("inf")
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = 0.0, U
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = Psi_sum_log(B, c, eps1, dist)
    fd = Psi_sum_log(B, d, eps1, dist)
    it = 0
    while (b - a) > tol and it < max_iter:
        if fc > fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = Psi_sum_log(B, d, eps1, dist)
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = Psi_sum_log(B, c, eps1, dist)
        it += 1
    delta_star = 0.5 * (a + b)
    f_star = Psi_sum_log(B, delta_star, eps1, dist)
    return delta_star, f_star


def Lambda(B, eps, lam, d):
    """
    Λ_w(B; ε) = 2 * exp( - (B/2) * min( ε/(λ_w d_w), (ε/(λ_w d_w))^2 ) ), Eq(30)
    Poisson arrival with rate lam: d_w = 2/la_w (Berstein inequality)
    """
    scale = float(eps) / (float(lam) * float(d))
    return 2.0 * math.exp(- (B / 2.0) * min(scale, scale ** 2))

def zeta(e1, e2, e3):
    """
    ζ(ε1, ε2, ε3) := min{ ε1, ε2/(1+ε2), ε2/[1-ε2]_+, ε3/(1+ε3), ε3/[1-ε3]_+ }. Eq (31)
    """
    candidates = [
        float(e1),
        float(e2) / (1.0 + float(e2)),
        (float(e2) / _safe_pos(1.0 - float(e2))) if _safe_pos(1.0 - float(e2)) > 0 else float("inf"),
        float(e3) / (1.0 + float(e3)),
        (float(e3) / _safe_pos(1.0 - float(e3))) if _safe_pos(1.0 - float(e3)) > 0 else float("inf"),
    ]
    # Filter potential NaNs/infs gracefully
    candidates = [c for c in candidates if np.isfinite(c) and c >= 0]
    return min(candidates) if candidates else 0.0

def epsilon_bar(beta, eps, nu2, nu3):
    """
    ε̄(β) := ζ( min((1-β)/2, 3β/2) * ν2 ε,  min(ν2(1-β)ε, ν3(1-β)ε, 1),  min(1, ν3(1-β)ε) ), below Eq (42)
    """
    a1 = min((1.0 - beta) / 2.0, 3.0 * beta / 2.0) * float(nu2) * float(eps)
    a2 = min(float(nu2) * (1.0 - beta) * float(eps), float(nu3) * (1.0 - beta) * float(eps), 1.0)
    a3 = min(1.0, float(nu3) * (1.0 - beta) * float(eps))
    return zeta(a1, a2, a3)

def gamma_beta(beta, eps, L_F, sum_lambda_hat):
    """
    γ(β) = min( β ε / (sum_w \hat{λ}_w L_F), 1 ). Eq (42)
    """
    return min((float(beta) * float(eps)) / (float(sum_lambda_hat) * float(L_F)), 1.0)


def p_of_beta(beta, eps, B_G, B_lambda, dists, a1s, a2s, nu1, nu2, nu3, lambdas, ds):
    if not isinstance(B_G, (list, tuple, np.ndarray)):
        B_G = [int(B_G)] * len(dists)
    if not isinstance(B_lambda, (list, tuple, np.ndarray)):
        B_lambda = [int(B_lambda)] * len(dists)

    eps1 = (float(nu1) / 6.0) * (1.0 - float(beta)) * float(eps)
    ebar = epsilon_bar(beta, eps, nu2, nu3)

    total = 0.0
    for BGi, BLi, dist,  a1, a2, lam, d in zip(
        B_G, B_lambda, dists, a1s, a2s, lambdas, ds
    ):
        delta, _ = choose_delta_optimal(BGi, eps1, dist, tol=1e-10, max_iter=2000)
        total += Phi(BGi, eps1, a1, a2)
        total += Psi1(BGi, delta, eps1, dist.G, dist.y_eps)
        total += Psi2(BGi, delta, eps1, dist.G, C_f2=1.0)
        total += Psi3(BGi, delta, eps1, dist.rho)
        total += Lambda(BLi, ebar, lam, d)
    return max(0.0, 1.0 - total), delta #max(0.0, min(1.0, 1.0 - total))

# ---------- For the simple network in Figure 1(a): 2*1 network ----------
J, K = 2,1
thetas = [1,1,1] # reneging rates 
lambdas = [1,1,1] # poisson arrival rates
cs = [3.5,4,1] # holding costs
vs = [[1],[1]] # matching values

nu1 = 1/(4*(J+K)*max([cs[i]*lambdas[i] for i in range(len(cs))]))
L  = min([1/theta for theta in thetas])
L_F = max(vs)[0] + L*sum(cs)
nu2 = 1/(3*L_F*sum(lambdas))
nu3 = 1/(6*sum([cs[i]*lambdas[i]*(1/thetas[i]+L) for i in range(len(cs))]))

dist_list = [UniformDist(a=2/thetas[i]) for i in range(len(thetas))]
a1_list = [2]*len(thetas)
a2_list = [theta**2/2 for theta in thetas]

d_list = [2/la for la in lambdas] 

eps = 1
sum_lambda_hat = sum(lambdas) # set to true arr rates for simplicity

betas = np.linspace(0.01, 0.99, 99)
#betas = [0.5]

B = 1e9
beta = betas[0] #
theta = thetas[0]
dist = dist_list[0]
a1, a2 = a1_list[0], a2_list[0]
lam, d = lambdas[0], d_list[0]

eps1 = (float(nu1) / 6.0) * (1.0 - float(beta)) * float(eps)
ebar = epsilon_bar(beta, eps, nu2, nu3)

total = 0.0
#delta = choose_delta(dist, nu1, beta, eps) ## if delta is too large, Psi1 and Psi2 will explode
delta, f_star = choose_delta_optimal(B, eps1, dist, tol=1e-10, max_iter=2000)

########### psi1 #########
k = int(math.ceil(B * (1.0 - delta)))
logC = _binom_coeff_log(B, k)
logval = math.log(B * delta + 1.0) + logC + B * (1.0 - delta) * math.log(float(1-theta*eps1))
Psi1_val = _exp_from_log(logval)
print('Psi_1:',Psi1_val)

########### psi2 #########
k = int(math.ceil(B * delta))
logC = _binom_coeff_log(B, k)
term = 1.0 - float(dist.G(eps1))
power = B - B * delta - 1.0
logval = math.log(B * delta + 2.0) + logC + power * math.log(term)
Psi2_val = _exp_from_log(logval)
print('Psi_2:',Psi2_val)

########### psi3 #########
"""
Ψ_3(B; δ, ε) := 2 * exp( - [ min(ε ρ(δ), δ) ]^2 / (2B) ). Eq. (16)
"""
a = min(float(eps1) * float(dist.rho(delta)), float(delta))
Psi3_val = 2.0 * math.exp(-B*(a ** 2) / 2)
print('Psi_3:',Psi3_val)
print('sanity check:', np.exp(Psi_sum_log(B, delta, eps1, dist)))

#### Phi #########
Phi_val = float(8) * math.exp(-float(theta/32) * B * (eps1 ** 2))
print('Phi:',Phi_val)
#### Lambda
scale = float(eps1) / 2
Lamda_val = 2.0 * math.exp(- (B / 2.0) * min(scale, scale ** 2))
print('Lambda:',Lamda_val)
#########


total += Phi(B, eps1, a1, a2)
total += Psi1(B, delta, eps1, dist.G, dist.y_eps)
total += Psi2(B, delta, eps1, dist.G, C_f2=1.0)
total += Psi3(B, delta, eps1, dist.rho)
total += Lambda(B, ebar, lam, d)

def fmt_sci(n, precision=1):
    """Format number n in scientific notation with given precision."""
    s = f"{n:.{precision}e}"     # scientific notation, e.g. '3.5e+08'
    mant, exp = s.split('e')
    return f"{mant}e{int(exp)}"  # -> '3.5e8'


### \Psi_i under varying delta ########
fs = 14
B = 3e8
#beta = 0.1
beta = 0
eps1 = 0.003 #(float(nu1) / 6.0) * (1.0 - float(beta)) * float(eps)
U = min(0.5, 1.0 - dist.G(dist.y_eps(eps1)), dist.G(eps1))
deltas = np.linspace(0, 0.0002, 100000)
B_list = [3e8, 3.5e8, 4e8]

deltas_large = np.linspace(0, 0.04, 100000)
fig, ax = plt.subplots(1)
for B in B_list:
    vals1 = [logPsi1(B, d, eps1, dist.G, dist.y_eps) for d in deltas_large]
    ax.plot(deltas_large, vals1, label=f"B={fmt_sci(B)}")
ax.set_xlabel(r'$\delta$',fontsize=fs)
plt.ylabel(r'$\log[\Psi_1(B;\delta,\epsilon)]$',fontsize=fs)
sf = ticker.ScalarFormatter(useMathText=True)
sf.set_scientific(True)
sf.set_powerlimits((0, 0))   # always use scientific notation
sf.set_useOffset(False)      # disable the 0.+offset display
ax.xaxis.set_major_formatter(sf)
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax.minorticks_on()
ax.legend()
ax.grid(True)
#ax.grid(True, which='both', linestyle='--', alpha=0.3)
fig.tight_layout()
#plt.savefig('Psi1_varyDelta_Cross.png',dpi=200, bbox_inches='tight')

fig, ax = plt.subplots(1)
for B in B_list:
    vals1 = [logPsi1(B, d, eps1, dist.G, dist.y_eps) for d in deltas]
    ax.plot(deltas, vals1, label=f"B={fmt_sci(B)}")
ax.set_xlabel(r'$\delta$',fontsize=fs)
plt.ylabel(r'$\log[\Psi_1(B;\delta,\epsilon)]$',fontsize=fs)
sf = ticker.ScalarFormatter(useMathText=True)
sf.set_scientific(True)
sf.set_powerlimits((0, 0))   # always use scientific notation
sf.set_useOffset(False)      # disable the 0.+offset display
ax.xaxis.set_major_formatter(sf)
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax.minorticks_on()
ax.legend()
ax.grid(True)
#ax.grid(True, which='both', linestyle='--', alpha=0.3)
fig.tight_layout()
#plt.savefig('Psi1_varyDelta_noCross.png',dpi=200, bbox_inches='tight')

fig, ax = plt.subplots(1)
for B in B_list:
    vals2 = [logPsi2(B, d, eps1, dist.G) for d in deltas]
    ax.plot(deltas, vals2, label=f"B={fmt_sci(B)}")
ax.set_xlabel(r'$\delta$',fontsize=fs)
plt.ylabel(r'$\log[\Psi_2(B;\delta,\epsilon)]$',fontsize=fs)
sf = ticker.ScalarFormatter(useMathText=True)
sf.set_scientific(True)
sf.set_powerlimits((0, 0))   # always use scientific notation
sf.set_useOffset(False)      # disable the 0.+offset display
ax.xaxis.set_major_formatter(sf)
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax.minorticks_on()
ax.legend()
ax.grid(True)
#ax.grid(True, which='both', linestyle='--', alpha=0.3)
fig.tight_layout()
#plt.savefig('Psi2_varyDelta_Cross.png',dpi=200, bbox_inches='tight')


fig, ax = plt.subplots(1)
for B in B_list:
    vals3 = [logPsi3(B, d, eps1, dist.rho) for d in deltas]
    ax.plot(deltas, vals3, label=f"B={fmt_sci(B)}")
ax.set_xlabel(r'$\delta$',fontsize=fs)
plt.ylabel(r'$\log[\Psi_3(B;\delta,\epsilon)]$',fontsize=fs)
sf = ticker.ScalarFormatter(useMathText=True)
sf.set_scientific(True)
sf.set_powerlimits((0, 0))   # always use scientific notation
sf.set_useOffset(False)      # disable the 0.+offset display
ax.xaxis.set_major_formatter(sf)
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax.minorticks_on()
ax.legend()
ax.grid(True)
#ax.grid(True, which='both', linestyle='--', alpha=0.3)
fig.tight_layout()
#plt.savefig('Psi3_varyDelta.png',dpi=200, bbox_inches='tight')


#fig, ax = plt.subplots(1, figsize=(8, 5))
fig, ax = plt.subplots(1)
for B in B_list:
    sum_vals = [Psi_sum_log(B, d, eps1, dist) for d in deltas]
    ax.plot(deltas, sum_vals, label=f"B={fmt_sci(B)}")
ax.set_xlabel(r'$\delta$',fontsize=fs)
ax.set_ylabel(r'$\log[\Psi_1+\Psi_2+\Psi_3]$',fontsize=fs)
sf = ticker.ScalarFormatter(useMathText=True)
sf.set_scientific(True)
sf.set_powerlimits((0, 0))   # always use scientific notation
sf.set_useOffset(False)      # disable the 0.+offset display
ax.xaxis.set_major_formatter(sf)
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax.minorticks_on()
ax.legend()
ax.grid(True)
#ax.grid(True, which='both', linestyle='--', alpha=0.3)
fig.tight_layout()
#plt.savefig('SumPsi_varyDelta.png',dpi=200, bbox_inches='tight')

fig, ax = plt.subplots(1)
for B in B_list:
    sum_vals = [Psi_sum_log(B, d, eps1, dist) for d in deltas]
    ax.plot(deltas, sum_vals, label=f"B={fmt_sci(B)}")
ax.set_xlabel(r'$\delta$',fontsize=fs)
ax.set_ylabel(r'$\log[\Psi_1+\Psi_2+\Psi_3]$',fontsize=fs)
sf = ticker.ScalarFormatter(useMathText=True)
sf.set_scientific(True)
sf.set_powerlimits((0, 0))   # always use scientific notation
sf.set_useOffset(False)      # disable the 0.+offset display
ax.xaxis.set_major_formatter(sf)
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax.minorticks_on()
ax.set_ylim([min(sum_vals)-0.5, 7])  # ensure vals_sum defined appropriately
ax.legend()
ax.grid(True)
#ax.grid(True, which='both', linestyle='--', alpha=0.3)
fig.tight_layout()
#plt.savefig('SumPsi_varyDelta_ZoomedIn.png',dpi=200, bbox_inches='tight')

##################### Phi vs Psi_i vs Lambda under varying beta   ########################
fs = 14
B_list = [1e8, 3e8, 9e8]
eps = 1

fig, ax = plt.subplots(1)
for B in B_list:
    vals = []
    for beta in betas:
        eps1 = (float(nu1) / 6.0) * (1.0 - float(beta)) * float(eps)
        vals.append(Phi(B, eps1, a1, a2))
    ax.plot(betas, vals, label=f"B={fmt_sci(B)}")
ax.set_xlabel(r'$\beta$',fontsize=fs)
ax.set_ylabel(r'$\Phi(B;(1-\beta)\nu_1\epsilon/6)$',fontsize=fs)
sf = ticker.ScalarFormatter(useMathText=True)
sf.set_scientific(True)
sf.set_powerlimits((0, 0))   # always use scientific notation
sf.set_useOffset(False)      # disable the 0.+offset display
ax.xaxis.set_major_formatter(sf)
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax.minorticks_on()
ax.legend()
ax.grid(True)
#ax.grid(True, which='both', linestyle='--', alpha=0.3)
fig.tight_layout()
#plt.savefig('Phi_varyBeta.png',dpi=200, bbox_inches='tight')

fig, ax = plt.subplots(1)
for B in B_list:
    vals = []
    for beta in betas:
        eps1 = (float(nu1) / 6.0) * (1.0 - float(beta)) * float(eps)
        delta, _ = choose_delta_optimal(B, eps1, dist, tol=1e-10, max_iter=2000)
        val = Psi1(B, delta, eps1, dist.G, dist.y_eps)+\
                Psi2(B, delta, eps1, dist.G, C_f2=1.0)+\
                    Psi3(B, delta, eps1, dist.rho)
        vals.append(val)
    ax.plot(betas, vals, label=f"B={fmt_sci(B)}")
ax.set_xlabel(r'$\beta$',fontsize=fs)
ax.set_ylabel(r'$\Psi_1+\Psi_2+\Psi_3$',fontsize=fs)
sf = ticker.ScalarFormatter(useMathText=True)
sf.set_scientific(True)
sf.set_powerlimits((0, 0))   # always use scientific notation
sf.set_useOffset(False)      # disable the 0.+offset display
ax.xaxis.set_major_formatter(sf)
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax.minorticks_on()
ax.legend()
ax.grid(True)
#ax.grid(True, which='both', linestyle='--', alpha=0.3)
fig.tight_layout()
#plt.savefig('SumPsi_varyBeta.png',dpi=200, bbox_inches='tight')

#plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(1)
for B in B_list:
    vals = []
    for beta in betas:
        ebar = epsilon_bar(beta, eps, nu2, nu3)
        vals.append(Lambda(B, ebar, lam, d))
    ax.plot(betas, vals, label=f"B={fmt_sci(B)}")
ax.set_xlabel(r'$\beta$',fontsize=fs)
ax.set_ylabel("Λ(B; ε̲(β))",fontsize=fs)
sf = ticker.ScalarFormatter(useMathText=True)
sf.set_scientific(True)
sf.set_powerlimits((0, 0))   # always use scientific notation
sf.set_useOffset(False)      # disable the 0.+offset display
ax.xaxis.set_major_formatter(sf)
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax.minorticks_on()
ax.legend()
ax.grid(True)
#ax.grid(True, which='both', linestyle='--', alpha=0.3)
fig.tight_layout()
#plt.savefig('Lambda_UbarEps_varyBeta.png',dpi=200, bbox_inches='tight')

fig, ax = plt.subplots(1)
for B in B_list:
    vals = []
    for beta in betas:
        ebar = epsilon_bar(beta, eps, nu2, nu3)
        vals.append(ebar)
    ax.plot(betas, vals, 'k',label=f"B={fmt_sci(B)}")
ax.set_xlabel(r'$\beta$',fontsize=fs)
ax.set_ylabel("ε̲(β)",fontsize=fs)
sf = ticker.ScalarFormatter(useMathText=True)
sf.set_scientific(True)
sf.set_powerlimits((0, 0))   # always use scientific notation
sf.set_useOffset(False)      # disable the 0.+offset display
ax.xaxis.set_major_formatter(sf)
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax.minorticks_on()
#ax.legend()
ax.grid(True)
#ax.grid(True, which='both', linestyle='--', alpha=0.3)
fig.tight_layout()
#plt.savefig('UbarEps_varyBeta.png',dpi=200, bbox_inches='tight')


################## p(beta) vs beta ############3
#B_list = [1e8, 3e8, 6e8, 9e8, 12e8,5e9,1e10]
betas = np.linspace(0.0001, 0.99, 100)
gamma_vals = [gamma_beta(b, eps, L_F, sum_lambda_hat) for b in betas]
B_list = [3e8,6e8, 9e8]
#B_list = [1e8, 3e8, 9e8]
fig, ax = plt.subplots(1)
for B in B_list:
    p_vals = []
    for b in betas:
        p_val, _ = p_of_beta(b, eps, int(B), int(B),
                             dist_list, a1_list, a2_list,
                             nu1, nu2, nu3, lambdas, d_list)
        p_vals.append(p_val)
    #ax.plot(betas, p_vals,label=f"B={fmt_sci(B)}")
    ax.plot(gamma_vals, p_vals,label=f"B={fmt_sci(B)}")
ax.set_xlabel(r'$\gamma$',fontsize=fs)
ax.set_ylabel(r"$p(\gamma)$",fontsize=fs)
#ax.set_xlabel(r'$\beta$',fontsize=fs)
#ax.set_ylabel(r"$p(\beta)$",fontsize=fs)
sf = ticker.ScalarFormatter(useMathText=True)
sf.set_scientific(True)
sf.set_powerlimits((0, 0))   # always use scientific notation
sf.set_useOffset(False)      # disable the 0.+offset display
ax.xaxis.set_major_formatter(sf)
ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
ax.minorticks_on()
ax.legend()
ax.grid(True)
fig.tight_layout()
plt.savefig('p_gamma_VaryB_FixEps1_Thm7.png',dpi=200, bbox_inches='tight')




    
    