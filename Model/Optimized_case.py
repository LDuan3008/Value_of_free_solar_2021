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
    if name == 'natgas_ccs':  natgas_idx  = idx;  co2_emis_natgas_idx = tech_list[idx]['var_co2']
    if name == 'solar':       solar_idx   = idx;  solar_flag = True
    if name == 'wind':        wind_idx    = idx;  wind_flag = True
    if name == 'nuclear':     nuclear_idx = idx

### Set cycle values
co2_constraints_percentage = np.array([1e24, 50.0, 20.0, 10.0, 1.0, 0.1, 0.0]) 
upper_co2_emissions = 8760 * 1 * co2_emis_natgas
co2_constraints_list = upper_co2_emissions * co2_constraints_percentage / 100


if single_year_index == -1:
    for year_cycle_idx in range(4):
        year = 2016 + year_cycle_idx
        case_dic['year_start'] = year
        case_dic['year_end'] = year
        mean_demand = update_series(case_dic, tech_list[demand_idx])
        if solar_flag == True: mean_solar = update_series(case_dic, tech_list[solar_idx])
        if wind_flag == True: mean_wind = update_series(case_dic, tech_list[wind_idx])
        for idx in range(len(co2_constraints_list)):
            case_dic['co2_constraint'] = co2_constraints_list[idx]
            case_dic['case_name'] = f'{case_name_default}_Year{str(year)}_Co2Con{str(co2_constraints_percentage[idx])}_Optimized'
            run_model_main_fun(case_dic, tech_list) 
else:
    case_dic['year_start'] = single_year_index
    case_dic['year_end'] = single_year_index
    mean_demand = update_series(case_dic, tech_list[demand_idx])
    if solar_flag == True: mean_solar = update_series(case_dic, tech_list[solar_idx])
    if wind_flag == True: mean_wind = update_series(case_dic, tech_list[wind_idx])
    for idx in range(len(co2_constraints_list)):
        case_dic['co2_constraint'] = co2_constraints_list[idx]
        case_dic['case_name'] = f'{case_name_default}_Year{str(single_year_index)}_Co2Con{str(co2_constraints_percentage[idx])}_Optimized'
        run_model_main_fun(case_dic, tech_list) 