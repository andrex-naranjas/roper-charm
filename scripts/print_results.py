#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------
 Authors: A. Ramirez-Morales (andres.ramirez.morales@cern.ch)
          H. Garcia-Tecocoatzi
---------------------------------------------------------------
"""
import sys
from os import getcwd
# framework includes
import charmfw.common.data_visualization as dv
from charmfw.common.charm_tables import CharmTables
from charmfw.common.charm_plots import CharmPlots

# print results for journal
if len(sys.argv) <= 1:
    sys.exit('Provide bottom states group name. Try again!')

run_baryons = sys.argv[1]
workpath = getcwd()

# create summary of the results and store in a csv file
dv.paper_tables_results(run_baryons, di_three_quark='three_quark', decay_width=False,
                        asymmetric=True, prev_params=False, workpath=workpath, batch_number=True) # batch_number -> True, None ;)
print('three-quark results created')

# create summary of the results and store in a csv file
dv.paper_tables_results(run_baryons, di_three_quark='diquark', decay_width=False,
                        asymmetric=True, prev_params=False, workpath=workpath, batch_number=None)
print('diquark results created')

# # create summary tables for 
# dv.decay_indi_tables_results(run_baryons, decay_type="strong", asymmetric=True,
#                           prev_params=False, workpath=workpath, batch_number=True) # change to batch_number to True
# print('individual decays strong created')

dv.decay_indi_tables_results(run_baryons, decay_type="electro", asymmetric=True,
                             prev_params=False, workpath=workpath, batch_number=True) # change to batch_number to True
print('individual decays electro created')

# tables
charm_tables = CharmTables(run_baryons, workpath=workpath, batch_results=True) # assume diquark never come from batch jobs (FIX this)
charm_tables.single_model_table()
charm_tables.combined_model_table()
charm_tables.parameter_combined()
charm_tables.correlation_table_three()
charm_tables.correlation_table_di_flavor()

# charm_tables.decay_indi_table()

#sextet
if run_baryons == "cascades":
    # test
    #charm_tables.decay_indi_table_em_err(compare=False, charge="", n_states=9, off_set=0) # cascades
    charm_tables.decay_indi_table_em_err(compare=False, charge="zero", n_states=21, off_set=9) # cascades
    # charm_tables.decay_indi_table_em_err(compare=False, charge="positive", n_states=21, off_set=9) # cascades
elif run_baryons=="omegas": # [D-wave n_states=21, off_set=9]; [P-wave n_states=9, off_set=0]
    # test
    charm_tables.decay_indi_table_em_err(compare=False, charge="", n_states=21, off_set=9) # omegas
    #charm_tables.decay_indi_table_em_err(compare=False, charge="", n_states=9, off_set=0) # omegas
elif run_baryons == "sigmas":
    #charm_tables.decay_indi_table_em_err(compare=False, charge="", n_states=9, off_set=0) # cascades
    # charm_tables.decay_indi_table_em_err(compare=False, charge="charged", n_states=21, off_set=9)
    charm_tables.decay_indi_table_em_err(compare=False, charge="zero", n_states=21, off_set=9)
#antitriplet
if run_baryons =="cascades_anti3":
    #charm_tables.decay_indi_table_em_err(compare=False, charge="", n_states=8, off_set=0) # cascades_anti3
    #charm_tables.decay_indi_table_em_err(compare=False, charge="zero", n_states=19, off_set=8)
    charm_tables.decay_indi_table_em_err(compare=False, charge="negative", n_states=19, off_set=8)
elif run_baryons =="lambdas":
    #charm_tables.decay_indi_table_em_err(compare=False, charge="", n_states=8, off_set=0) # lambdas
    charm_tables.decay_indi_table_em_err(compare=False, charge=None, n_states=19, off_set=8)

charm_tables.comparison_three_quark_model_table()

# plots
# bottom_plots = BottomPlots(run_baryons, workpath=workpath)
# bottom_plots.load_data("diquark")
# bottom_plots.mass_spectrum_plot()
# bottom_plots.load_data("threequark")
# bottom_plots.mass_spectrum_plot()
