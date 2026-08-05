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
    sys.exit('Provide charm states group name. Try again!')

run_baryons = sys.argv[1]
workpath = getcwd()

# create summary of the results and store in a csv file
dv.paper_tables_results(run_baryons, di_three_quark='diquark', decay_width=True,
                        asymmetric=True, prev_params=False, workpath=workpath, batch_number=None)
print('diquark results created')

# create summary tables for 
dv.decay_indi_tables_results(run_baryons, decay_type="strong", asymmetric=True,
                             prev_params=False, workpath=workpath, batch_number=None) # change to batch_number to True
print('individual diquark decays strong created')

# tables
charm_tables = CharmTables(run_baryons, workpath=workpath, batch_results=False) # assume diquark never come from batch jobs (FIX this)
charm_tables.single_model_table()
charm_tables.parameter_single()
charm_tables.decay_indi_table()

# plots
# bottom_plots = BottomPlots(run_baryons, workpath=workpath)
# bottom_plots.load_data("diquark")
# bottom_plots.mass_spectrum_plot()
# bottom_plots.load_data("threequark")
# bottom_plots.mass_spectrum_plot()
