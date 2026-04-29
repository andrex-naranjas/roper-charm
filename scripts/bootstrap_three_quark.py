#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------
 Script to obtain uncertainties of heavy mass spectrum and widhts via bootstrap
 Authors: A. Ramirez-Morales (andres.ramirez.morales@cern.ch) and
          H. Garcia-Tecocoatzi
-----------------------------------------------------------------------------
"""
import sys
import os
from iminuit import Minuit
import numpy as np
import datetime
import pandas as pd
import json
# framework modules
from charmfw.baryons import data_preparation as dp
from charmfw.baryons.charm_three_quark import CharmThreeQuark


workpath = os.getcwd()

# for running batch jobs with htcondor
batch_number = None
run_baryons = None
if len(sys.argv) == 4:
    batch_number = sys.argv[1]
    workpath = sys.argv[2]
    run_baryons = sys.argv[3]

config = None
with open(workpath+"/config/three_quark_config.json", "r") as jsonfile:
    config = json.load(jsonfile)

if config is not None:
    if run_baryons is None:
        run_baryons = config["baryons"]
    n_events = config["n_events"]
    asymmetric = config["asymmetric_errors"]
    decay_width = config["decay_width"]
    decay_width_em = config["decay_width_em"]
    bootstrap = config["bootstrap_mass"]
    bootstrap_width = config["bootstrap_st_dec"]
    bootstrap_width_em = config["bootstrap_em_dec"]
    prev_params = config["previous_param"]
else:
    sys.exit('Please provide a configuration file. Try again!')

print('Getting paper results for:', run_baryons)

# input parameters
param_v,param_w,param_x,param_y,param_z,param_q1,param_q2,param_q3,\
    param_is_rho,param_is_lam,param_is_omega,param_is_cascade,param_is_sigma = dp.fetch_data_extended()

def model(q1, q2, q3, is_rho, is_lam, is_omega, is_cascade, is_sigma, v, w, x, y, z, m1, m2, m3, k, a, b, e, g):
    """
    mass model, m1 == bottom, m2== strange, m3 == light
    """
    return q1*m1 + q2*m2 + q3*m3 + \
        v*k*np.sqrt(1./(is_rho*(is_omega*m2 + is_cascade*((m2+m3)/2) + is_sigma*m3 ) + \
                        is_lam*(is_omega*((3*m2*m1)/(2.*m2+m1)) + is_cascade*((1.5*(m2+m3)*m1)/(m1+m2+m3)) + is_sigma*((3.*m3*m1)/(2.*m3+m1)) ) )) + \
                  w*a + x*b + y*e + z*g


def least_squares(m1, m2, m3, k, a, b, e, g):
    # y_var_0 = sigma_0 # best sigma_0=12.47 (bottom)
    # yvar_0 = y_var_0*np.ones(16)
    # yvar = y_errors_exp
    # yvar_2 = np.power(yvar_0, 2) + np.power(yvar, 2)
    yvar_2 = 0.001
    pred_m = model(param_q1, param_q2, param_q3, param_is_rho, param_is_lam,
                   param_is_omega, param_is_cascade, param_is_sigma, param_v,
                   param_w, param_x, param_y, param_z,
                   m1, m2, m3, k, a, b, e, g)
    yval_2 = np.power( (pred_m - exp_m), 2)
    return np.sum( np.divide(yval_2, yvar_2) )

def fit(least_squares):
    m = Minuit(least_squares, m1=1, m2=1, m3=1, k=0, a=0, b=0, e=0, g=0)#1400, m2=300, m3=250, k=0, a=0, b=0, e=0, g=0)
    m.limits['m1'] = (1500,1700)
    m.limits['m2'] = (300,500)
    m.limits['m3'] = (250,450)
    m.errordef=Minuit.LEAST_SQUARES
    m.migrad()
    return m

def sample_gauss(mu, sigma):
    return np.random.normal(mu, sigma, 10000)

def random(sample, random_n=1):
    #return np.mean(resample(sample, replace=False, n_samples=1, random_state=random_n))
    return np.random.choice(sample, size=None)


# arrays to store the sampled parameters
sampled_k,sampled_a,sampled_b,sampled_e,sampled_g = ([]),([]),([]),([]),([])
sampled_m1,sampled_m2,sampled_m3 = ([]),([]),([])

# arrays to store sampled correlation coeficients
rho_m2m1,rho_m3m1,rho_km1,rho_am1,rho_bm1,rho_em1,rho_gm1 = ([]),([]),([]),([]),([]),([]),([])
rho_m3m2,rho_km2,rho_am2,rho_bm2,rho_em2,rho_gm2,rho_km3  = ([]),([]),([]),([]),([]),([]),([])
rho_am3,rho_bm3,rho_em3,rho_gm3,rho_ak,rho_bk,rho_ek      = ([]),([]),([]),([]),([]),([]),([])
rho_gk, rho_ba, rho_ea, rho_ga, rho_eb,rho_gb,rho_ge      = ([]),([]),([]),([]),([]),([]),([])

# start bootstrap
start = datetime.datetime.now()

sigma_model = 0.5*15.42**2 # obtained with optimization (Li.Jin)
# gaussian pdf with the measured value and with experimental and model(sigma_model) uncertainties
# Omega states
gauss_2695 = sample_gauss(2695.20, np.power((1.70**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2702), PDG
gauss_2770 = sample_gauss(2765.90, np.power((2.00**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2767), PDG
gauss_3000 = sample_gauss(3000.41, np.power((0.22**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 3015), PDG
gauss_3050 = sample_gauss(3050.20, np.power((0.13**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 3044), PDG
gauss_3066 = sample_gauss(3065.46, np.power((0.28**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 3051), PDG Excluded from the fit
gauss_3090 = sample_gauss(3090.00, np.power((0.50**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 3080), PDG Excluded from the fit
gauss_3120 = sample_gauss(3119.10, np.power((1.00**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 3140), PDG Excluded from the fit
# Cascades six-plet
gauss_2578 = sample_gauss(2578.8,  np.power((0.90**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2570), PDG Average
gauss_2645 = sample_gauss(2645.97, np.power((0.71**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2635), PDG Average
gauss_2923 = sample_gauss(2923.04, np.power((0.35**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2934), LHCb PhysRevLett.124.222001
gauss_2938 = sample_gauss(2938.55, np.power((0.30**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2941), LHCb PhysRevLett.124.222001
gauss_2965 = sample_gauss(2964.88, np.power((0.33**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2969), LHCb PhysRevLett.124.222001 Excluded from the fit
gauss_3055 = sample_gauss(3055.9,  np.power((0.40**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 3060), PDG Average
gauss_3080 = sample_gauss(3078.6,  np.power((2.80**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 3096), PDG Average
# Sigma c
gauss_2453 = sample_gauss(2453.5, np.power((0.90**2 + sigma_model), 0.5 ))   # all OK (corresponds to predicted 2453), PDG Average
gauss_2518 = sample_gauss(2518.1, np.power((2.80**2 + sigma_model), 0.5 ))   # all OK (corresponds to predicted 2517), PDG Average
gauss_2801 = sample_gauss(2800,   np.power((20.0**2 + sigma_model), 0.5 ))   # all OK (corresponds to predicted 2819), PDG Average
# Lambda C
gauss_2286 = sample_gauss(2286.46, np.power((0.14**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2283), PDG
gauss_2592 = sample_gauss(2592.25, np.power((0.28**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2649), PDG
gauss_2628 = sample_gauss(2628.11, np.power((0.19**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2685), PDG
# Cascade C anti-3-plet
gauss_2469 = sample_gauss(2469.42, np.power((1.77**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2461), PDG Average
gauss_2792 = sample_gauss(2793.3,  np.power((1.40**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2796), PDG Average
gauss_2816 = sample_gauss(2818.49, np.power((2.07**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2832), PDG Average
gauss_2970 = sample_gauss(2968.6,  np.power((3.30**2 + sigma_model), 0.5 ))  # all OK (corresponds to predicted 2980), PDG Average Exluded from the fit

# plug here the sigma_0 optimization lines from data_utils.py

count = 0
# construct the simulated sampling distribution (bootstrap technique)
for i in range(n_events): # max 10000 with decays included, computationally expensive
    #if(states=='All'):
    exp_m = np.array([ # measured baryon masses
        # omegas
        random(gauss_2695), random(gauss_2770), random(gauss_3000),
        #random(gauss_3050),# random(gauss_3066),
        random(gauss_3090),
        random(gauss_3120),
        # cascades six-plet
        random(gauss_2578), random(gauss_2645),
        # random(gauss_2923),
        # random(gauss_2938),
        random(gauss_2965),
        random(gauss_3055),
        random(gauss_3080),
        # sigma
        random(gauss_2453), random(gauss_2518), random(gauss_2801),
        # lambda         
        random(gauss_2286), random(gauss_2592), random(gauss_2628),
        # cascade anti-3plet
        random(gauss_2469), random(gauss_2792), random(gauss_2816)#, random(gauss_2970)
    ])    
    # perform the parameter fitting (via minimizing squared distance)
    m = fit(least_squares)

    if type(m.covariance) != type(None):
        count += 1
    else:
        continue

    sampled_m1 = np.append(sampled_m1, m.values['m1'])
    sampled_m2 = np.append(sampled_m2, m.values['m2'])
    sampled_m3 = np.append(sampled_m3, m.values['m3'])

    sampled_k = np.append(sampled_k, m.values['k'])
    sampled_a = np.append(sampled_a, m.values['a'])
    sampled_b = np.append(sampled_b, m.values['b'])
    sampled_e = np.append(sampled_e, m.values['e'])
    sampled_g = np.append(sampled_g, m.values['g'])

    # if states != 'omega':
    # correlation matrix
    # corr = m.np_matrix(correlation=True), iminuit1.4.2
    corr = m.covariance.correlation()

    rho_m2m1 = np.append(rho_m2m1, corr['m2','m1'])
    rho_m3m1 = np.append(rho_m3m1, corr['m3','m1'])
    rho_km1  = np.append(rho_km1,  corr['k','m1'])
    rho_am1  = np.append(rho_am1,  corr['a','m1'])
    rho_bm1  = np.append(rho_bm1,  corr['b','m1'])
    rho_em1  = np.append(rho_em1,  corr['e','m1'])
    rho_gm1  = np.append(rho_gm1,  corr['g','m1'])

    rho_m3m2  = np.append(rho_m3m2, corr['m3','m2'])
    rho_km2   = np.append(rho_km2 , corr['k','m2'])
    rho_am2   = np.append(rho_am2 , corr['a','m2'])
    rho_bm2   = np.append(rho_bm2 , corr['b','m2'])
    rho_em2   = np.append(rho_em2 , corr['e','m2'])
    rho_gm2   = np.append(rho_gm2 , corr['g','m2'])

    rho_km3  = np.append(rho_km3, corr['k','m3'])
    rho_am3  = np.append(rho_am3, corr['a','m3'])
    rho_bm3  = np.append(rho_bm3, corr['b','m3'])
    rho_em3  = np.append(rho_em3, corr['e','m3'])
    rho_gm3  = np.append(rho_gm3, corr['g','m3'])

    rho_ak  = np.append(rho_ak, corr['a','k'])
    rho_bk  = np.append(rho_bk, corr['b','k'])
    rho_ek  = np.append(rho_ek, corr['e','k'])
    rho_gk  = np.append(rho_gk, corr['g','k'])

    rho_ba  = np.append(rho_ba, corr['b','a'])
    rho_ea  = np.append(rho_ea, corr['e','a'])
    rho_ga  = np.append(rho_ga, corr['g','a'])

    rho_eb  = np.append(rho_eb, corr['e','b'])
    rho_gb  = np.append(rho_gb, corr['g','b'])
    
    rho_ge  = np.append(rho_ge, corr['g','e'])


print(round(sampled_m1.mean()), "mb",  round(sampled_m1.std()) )
print(round(sampled_m2.mean()), "ms",  round(sampled_m2.std()) )
print(round(sampled_m3.mean()), "mn",  round(sampled_m3.std()) )

print("K", pow(sampled_k.mean(), 2)/(1000**3),  "KB", pow(sampled_k.std(), 2)/(1000**3))
print("A", sampled_a.mean(), " PS ",  sampled_a.std())
print("B", sampled_b.mean(), " PSL ", sampled_b.std())
print("E", sampled_e.mean(), " PI  ", sampled_e.std())
print("G", sampled_g.mean(), " PF ",  sampled_g.std())

# save bootstrap results
df = pd.DataFrame({"M1" : sampled_m1,"M2" : sampled_m2,"M3" : sampled_m3,
                   "K" : sampled_k,   "A" : sampled_a,
                   "B": sampled_b,    "E" : sampled_e, "G" : sampled_g})

if batch_number is None:
    if not os.path.exists(workpath+"/tables/"):
        os.makedirs(workpath+"/tables/")        
    df.to_csv(workpath+"/tables/bootstrap_param_"+run_baryons+".csv", index=False)
else:
    if not os.path.exists(workpath+"/batch_results/"+run_baryons+"/parameters/"):
        os.makedirs(workpath+"/batch_results/"+run_baryons+"/parameters/")
    df.to_csv(workpath+"/batch_results/"+run_baryons+"/parameters/"+str(batch_number)+".csv", index=False)

# create dictionaries
param   = {'q1':param_q1, 'q2':param_q2, 'q3':param_q3,'is_rho':param_is_rho, 'is_lam':param_is_lam,'is_omega':param_is_omega,
           'is_cascade':param_is_cascade, 'is_sigma':param_is_sigma,'V':param_v, 'W':param_w, 'X':param_x, 'Y':param_y, 'Z':param_z}
sampled = {'sampled_m1':sampled_m1,'sampled_m2':sampled_m2,'sampled_m3':sampled_m3,'sampled_k':sampled_k,
           'sampled_a':sampled_a, 'sampled_b':sampled_b, 'sampled_e':sampled_e, 'sampled_g':sampled_g}

corr_mat_ext ={'rho_m2m1':rho_m2m1, 'rho_m3m1':rho_m3m1, 'rho_km1':rho_km1, 'rho_am1':rho_am1, 'rho_bm1':rho_bm1, 'rho_em1':rho_em1, 'rho_gm1':rho_gm1,
               'rho_m3m2':rho_m3m2, 'rho_km2':rho_km2, 'rho_am2':rho_am2, 'rho_bm2':rho_bm2, 'rho_em2':rho_em2, 'rho_gm2':rho_gm2, 'rho_km3':rho_km3,
               'rho_am3':rho_am3, 'rho_bm3':rho_bm3, 'rho_em3':rho_em3, 'rho_gm3':rho_gm3, 'rho_ak':rho_ak, 'rho_bk':rho_bk, 'rho_ek':rho_ek,
               'rho_gk':rho_gk, 'rho_ba':rho_ba, 'rho_ea':rho_ea, 'rho_ga':rho_ga, 'rho_eb':rho_eb, 'rho_gb':rho_gb, 'rho_ge':rho_ge}

df = pd.DataFrame(corr_mat_ext)
if batch_number is None:
    if not os.path.exists(workpath+"/tables/"):
        os.makedirs(workpath+"/tables/")
    df.to_csv(workpath+"/tables/bootstrap_correlation_"+run_baryons+".csv", index=False)
else:
    if not os.path.exists(workpath+"/batch_results/"+run_baryons+"/correlation/"):
        os.makedirs(workpath+"/batch_results/"+run_baryons+"/correlation/")
    df.to_csv(workpath+"/batch_results/"+run_baryons+"/correlation/"+str(batch_number)+".csv", index=False)


# calculate masses and widths using the bootstraped fitted parameters
results = CharmThreeQuark(baryons=run_baryons, params=param, sampled=sampled, corr_mat=corr_mat_ext, asymmetric=asymmetric,
                           decay_width=decay_width, bootstrap_width=bootstrap_width, decay_width_em=decay_width_em, bootstrap_width_em=bootstrap_width_em, batch_number=batch_number, workpath=workpath)
#print(results)
#input()

results.fetch_values()
results.paper_results_predictions(bootstrap=bootstrap, bootstrap_width=bootstrap_width, prev_params=prev_params)
end = datetime.datetime.now()
elapsed_time = end - start
print(count, "no. successes")
print("Elapsed total time = " + str(elapsed_time))
