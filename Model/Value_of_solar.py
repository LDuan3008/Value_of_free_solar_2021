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
    co2_emis_natgas = 0.49

### Set cycle values
co2_constraints_percentage = np.array([1e24, 50.0, 20.0, 10.0, 1.0, 0.1, 0.0]) 
upper_co2_emissions = 8760 * 1 * co2_emis_natgas
co2_constraints_list = upper_co2_emissions * co2_constraints_percentage / 100
### Set capacity cycle
capacity_specified = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 
                      0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78,
                      0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18,
                      1.20, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34, 1.36, 1.38, 1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52, 1.54, 1.56, 1.58,
                      1.60, 1.62, 1.64, 1.66, 1.68, 1.70, 1.72, 1.74, 1.76, 1.78, 1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98,
                      2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.00, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90, 4.00]

if single_year_index == -1:
    for year_cycle_idx in range(3):
        year = 2016 + year_cycle_idx
        case_dic['year_start'] = year
        case_dic['year_end'] = year
        mean_demand = update_series(case_dic, tech_list[demand_idx])
        if solar_flag == True: mean_solar = update_series(case_dic, tech_list[solar_idx])
        if wind_flag == True: mean_wind = update_series(case_dic, tech_list[wind_idx])
        for idx in range(len(co2_constraints_list)):
            case_dic['co2_constraint'] = co2_constraints_list[idx]
            for idx_cap in range(len(capacity_specified)):
                tech_list[solar_idx]['capacity'] = capacity_specified[idx_cap] / mean_solar
                case_dic['case_name'] = f'{case_name_default}_Year{str(year)}_Co2Con{str(co2_constraints_percentage[idx])}_SoCap{str(capacity_specified[idx_cap])}'
                run_model_main_fun(case_dic, tech_list) 
else:
    case_dic['year_start'] = single_year_index
    case_dic['year_end'] = single_year_index
    mean_demand = update_series(case_dic, tech_list[demand_idx])
    if solar_flag == True: mean_solar = update_series(case_dic, tech_list[solar_idx])
    if wind_flag == True: mean_wind = update_series(case_dic, tech_list[wind_idx])
    for idx in range(len(co2_constraints_list)):
        case_dic['co2_constraint'] = co2_constraints_list[idx]
        for idx_cap in range(len(capacity_specified)):
            tech_list[solar_idx]['capacity'] = capacity_specified[idx_cap] / mean_solar
            case_dic['case_name'] = f'{case_name_default}_Year{str(single_year_index)}_Co2Con{str(co2_constraints_percentage[idx])}_SoCap{str(capacity_specified[idx_cap])}'
            run_model_main_fun(case_dic, tech_list) 