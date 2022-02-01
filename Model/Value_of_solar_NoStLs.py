"""
# Code to loop through different carbon emission reduction constraints;
# Give input table, region of interest, and year;
# Run the code as: >> python Run_Case_Example.py Case_AdvancedNuclear.csv US 2018
# Uploaded by Lei Duan on Febrary 08, 2020.
"""

from Preprocess_Input import preprocess_input
from Run_Core_Model import run_model_main_fun
from FindRegion import update_series, update_timenum
import sys, numpy as np


### Read input data
if len(sys.argv) < 3:
    print ('input less parameters than required') 
    sys.exit()
else:
    case_input_path_filename = sys.argv[1]
    single_year_index = int(sys.argv[2])

### Pre-processing
print ('Macro_Energy_Model: Pre-processing input')
case_dic,tech_list = preprocess_input(case_input_path_filename)

### Set basic information
case_name_default = case_dic['case_name']
case_dic['num_time_periods'] = 8760
case_dic['output_path'] = '/data/carnegie/leiduan/cesm_archive/MEM_AdvNuc'
solar_flag, wind_flag = False, False

### Find values
for idx in range(len(tech_list)):
    name = tech_list[idx]['tech_name']
    if name == 'demand':      demand_idx  = idx
    if name == 'natgas':      natgas_idx  = idx;  co2_emis_natgas = tech_list[idx]['var_co2']
    if name == 'solar':       solar_idx   = idx;  solar_flag = True
    if name == 'wind':        wind_idx    = idx;  wind_flag = True
    if name == 'nuclear':     nuclear_idx = idx
    if name == 'storage':     storage_idx = idx
    if name == 'shift_load':  load_shift_idx = idx

case_dic['year_start'] = single_year_index
case_dic['year_end'] = single_year_index
mean_demand = update_series(case_dic, tech_list[demand_idx])
if solar_flag == True: mean_solar = update_series(case_dic, tech_list[solar_idx])
if wind_flag == True: mean_wind = update_series(case_dic, tech_list[wind_idx])


capacity_specified = [100.0]
case_dic['co2_constraint'] = 0.0
tech_list[load_shift_idx]['var_cost'] = 1e20
tech_list[storage_idx]['fixed_cost'] = 1e20
tech_list[solar_idx]['capacity'] = capacity_specified[0] / mean_solar
case_dic['case_name'] = f'zzz_Solar100x_Year{single_year_index}_NoLsSt'
run_model_main_fun(case_dic, tech_list) 
