
import matplotlib.pyplot as plt 
import numpy as np
import pickle, os
from scipy import stats

category1 = ['natgas', 'natgas_ccs', 'natgas_fixed', 'natgas_ccs_fixed']
category2 = ['solar',       'wind',       'storage',       'conventional_nuclear',       'Resistant Heater',       'nuclear',
             'solar_fixed', 'wind_fixed', 'storage_fixed', 'conventional_nuclear_fixed', 'Resistant Heater_fixed', 'nuclear_fixed']
category3 = {'advanced_nuclear': ['au_reactor', 'au_generator', 'au_tes'],
             'pgp': ['pgp_electrolyzer', 'pgp_fuelcell', 'pgp_h2storage'],
             'csp': ['csp_generation', 'csp_turbine', 'csp_tes']}
category4 = ['lost_load']
category5 = ['shift_load']

def initial_keys(info, keylist):
    for key in keylist:
        info[key] = []

def nonan(input):
    input[input==0] = np.nan
    return input

def Update_info(info, category, results_tech_dic, results_case_dic, results_time_dic, input_case_dic, cost):
    if category in category1:
        cost_var = cost[category + '_var']
        cost_fix = cost[category + '_fix']
        cost_co2 = cost[category + '_co2']
        co2_price_n = input_case_dic['co2_price']
        info[category+'_cap'].append(results_tech_dic[category+' capacity'])
        info[category+'_emissions'].append( np.sum(cost_co2 * results_time_dic[category+' dispatch']) )
        info[category+'_tot'].append(cost_fix * results_tech_dic[category+' capacity']+np.mean(cost_var * results_time_dic[category+' dispatch'])+np.mean(cost_co2 * results_time_dic[category+' dispatch'] * co2_price_n))
    if category in category2:
        cost_fix = cost[category+'_fix']
        info[category+'_cap'].append(results_tech_dic[category+' capacity'])
        info[category+'_fix'].append(cost_fix * results_tech_dic[category+' capacity'])
    if category in category3.keys():
        name_list = category3[category]
        efficiency = cost[ name_list[1] + '_eff']
        # to X
        info[name_list[0]+'_cap'].append(results_tech_dic[name_list[0]+' capacity'] * efficiency)
        info[name_list[0]+'_fix'].append(results_tech_dic[name_list[0]+' capacity'] * cost[name_list[0]+'_fix'])
        # from X
        info[name_list[1]+'_cap'].append(results_tech_dic[name_list[1]+' capacity'])
        info[name_list[1]+'_fix'].append(results_tech_dic[name_list[1]+' capacity'] * cost[name_list[1]+'_fix'])
        # in X
        info[name_list[2]+'_cap'].append(results_tech_dic[name_list[2]+' capacity'] * efficiency)
        info[name_list[2]+'_fix'].append(results_tech_dic[name_list[2]+' capacity'] * cost[name_list[2]+'_fix'])
        # total cost
        info[category+'_tot'].append(results_tech_dic[name_list[0]+' capacity']*cost[name_list[0]+'_fix'] + results_tech_dic[name_list[1]+' capacity']*cost[name_list[1]+'_fix'] + results_tech_dic[name_list[2]+' capacity']*cost[name_list[2]+'_fix'])
    if category in category4:
        cost_var = cost[category+'_var']
        info[category+'_var'].append(np.mean(results_time_dic[category+' dispatch']*cost_var))
        info[category+'_mean'].append(np.mean(results_time_dic[category+' dispatch']))
    if category in category5:
        cost_var = cost[category+'_var']
        cost_fix = cost[category + '_fix']
        absolute_dispatch = np.abs(results_time_dic[category+' stored'])
        info[category+'_tot'].append(cost_fix * results_tech_dic[category+' capacity']+np.mean(cost_var * absolute_dispatch))
    if category == 'others':
        info['system_cost'].append(results_case_dic['system_cost'])
        info['co2_emissions'].append(results_case_dic['co2_emissions'])

def Get_Table(default_case_name, **kwargs):

    if 'data_path' in kwargs:
        data_path = kwargs['data_path']
    else:
        data_path = './'
    
    if 'repeat_list' in kwargs:
        repeat_list = kwargs['repeat_list']
    else:
        repeat_list = ['']
    
    if 'dividing_name_point' in kwargs:
        dividing_name_point = kwargs['dividing_name_point']
    else:
        dividing_name_point = -1
    
    if 'tech_name_list' in kwargs:
        tech_name_list = kwargs['tech_name_list']
    else:
        tech_name_list = ['natgas', 'natgas_ccs', 'solar', 'wind', 'storage', 'conventional_nuclear', 'advanced_nuclear', 'lost_load']

    info = {}
    initial_keys(info, ['system_cost', 'co2_emissions'])
    for tech_idx in tech_name_list:
        if tech_idx in category1:
            initial_keys(info, [tech_idx+'_cap', tech_idx+'_tot', tech_idx+'_emissions'])
        if tech_idx in category2:
            initial_keys(info, [tech_idx+'_cap', tech_idx+'_fix'])
        if tech_idx in category3.keys():
            name_list = category3[tech_idx]
            initial_keys(info, [name_list[0]+'_cap', name_list[0]+'_fix'])
            initial_keys(info, [name_list[1]+'_cap', name_list[1]+'_fix'])
            initial_keys(info, [name_list[2]+'_cap', name_list[2]+'_fix'])
            initial_keys(info, [tech_idx+'_tot'])
        if tech_idx in category4:
            initial_keys(info, [tech_idx+'_var'])
            initial_keys(info, [tech_idx+'_mean'])
        if tech_idx in category5:
            initial_keys(info, [tech_idx+'_tot'])

    def get_individual_case_result(case_name):

        file_list = os.listdir(data_path+case_name)
        for file in file_list:
            if file[-6:] == "pickle": 
                case_name_open = file
                break
        
        # print (case_name_open)
        with open(data_path+case_name+case_name_open, 'rb') as handle:
            [[input_case_dic,  input_tech_list,  input_time_dic],
             [results_case_dic,  results_tech_dic,  results_time_dic]] = pickle.load(handle)
        
        # if 'demand series' not in info.keys():
        #     info['demand series'] = input_time_dic['demand series']
        #     info['wind series'] = input_time_dic['wind series']
        #     info['solar series'] = input_time_dic['solar series']

        cost = {}
        for tech_idx in range(len(input_tech_list)):
            current_tech_list = input_tech_list[tech_idx]
            tech_name = current_tech_list['tech_name'] #.lower()
            if 'var_cost' in current_tech_list.keys():
                cost[tech_name+'_var'] = current_tech_list['var_cost']
            if 'fixed_cost' in current_tech_list.keys():
                cost[tech_name+'_fix'] = current_tech_list['fixed_cost'] 
            if 'var_co2' in current_tech_list.keys():
                cost[tech_name+'_co2'] = current_tech_list['var_co2']
            if 'efficiency' in current_tech_list.keys():
                cost[tech_name+'_eff'] = current_tech_list['efficiency'] 

        
        for tech_name_idx in tech_name_list:
            Update_info(info, tech_name_idx, results_tech_dic, results_case_dic, results_time_dic, input_case_dic, cost)
        Update_info(info, 'others', results_tech_dic, results_case_dic, results_time_dic, input_case_dic, cost)

    if repeat_list != ['']:
        for repeat_n in repeat_list:
            # print (repeat_n)
            if dividing_name_point > -1:
                case_name = default_case_name[:dividing_name_point] + str(repeat_n) + default_case_name[dividing_name_point:] + '/'
                # print (case_name)
                get_individual_case_result(case_name)
            else:
                case_name = default_case_name + str(repeat_n) + '/'
                get_individual_case_result(case_name)
    else:
        case_name = default_case_name + '/'
        get_individual_case_result(case_name)
    
    return info








def get_case_dispatch(case_name, data_find):

    file_list = os.listdir(data_find+case_name)
    for file in file_list:
        if file[-6:] == "pickle": 
            case_name_open = file
            break
    with open(data_find+case_name+case_name_open, 'rb') as handle:
        [[input_case_dic,  input_tech_list,  input_time_dic],
         [results_case_dic,  results_tech_dic,  results_time_dic]] = pickle.load(handle)

    cost = {}
    for tech_idx in range(len(input_tech_list)):
        current_tech_list = input_tech_list[tech_idx]
        tech_name = current_tech_list['tech_name'].lower()
        if 'var_cost' in current_tech_list.keys():
            cost[tech_name+'_var'] = current_tech_list['var_cost']
        if 'fixed_cost' in current_tech_list.keys():
            cost[tech_name+'_fix'] = current_tech_list['fixed_cost'] 
        if 'var_co2' in current_tech_list.keys():
            cost[tech_name+'_co2'] = current_tech_list['var_co2']
        if 'efficiency' in current_tech_list.keys():
            cost[tech_name+'_eff'] = current_tech_list['efficiency']

    info = {}
    info['demand_potential'] = results_time_dic['demand potential']
    # Above the zero line
    info['natgas_dispatch'] = results_time_dic['natgas dispatch']
    info['natgas_ccs_dispatch'] = results_time_dic['natgas_ccs dispatch']
    info['solar_potential']  = results_time_dic['solar potential']
    info['wind_potential']   = results_time_dic['wind potential']
    info['storage_dispatch'] = results_time_dic['storage dispatch']
    info['storage_in_dispatch'] = results_time_dic['storage in dispatch']
    info['storage_stored'] = results_time_dic['storage stored']
    info['nuclear_potential'] = results_time_dic['demand potential'] * 0 + results_tech_dic['nuclear capacity']

    # Fix Storage dispatch
    total_dispatch = info['natgas_dispatch'] + \
                     info['natgas_ccs_dispatch'] + \
                     info['solar_potential'] + \
                     info['wind_potential'] + \
                     info['nuclear_potential']
    Diff3 = info['demand_potential'] - total_dispatch
    Diff3[Diff3<=0] = 0
    Diff3[Diff3> 0] = 1
    info['storage_dispatch'] = info['storage_dispatch'] * Diff3
    Diff4 = info['demand_potential'] - total_dispatch
    Diff4[Diff4<=0] = 0
    Diff5 = info['storage_dispatch'] - Diff4
    Diff5[Diff5<0] = 0
    info['storage_dispatch'] = info['storage_dispatch'] - Diff5

    # Used solar 
    total_other = info['natgas_dispatch'] + \
                  info['natgas_ccs_dispatch'] + \
                  info['wind_potential'] + \
                  info['nuclear_potential'] + \
                  info['storage_dispatch']
    unmet = info['demand_potential'] - total_other
    unmet[unmet<0] = 0
    info['solar_used'] = unmet
    return [info]






















def for_fun(case_name, data_find):

    file_list = os.listdir(data_find+case_name)

    for file in file_list:
        if file[-6:] == "pickle": 
            case_name_open = file
            break

    with open(data_find+case_name+case_name_open, 'rb') as handle:
        [[input_case_dic,  input_tech_list,  input_time_dic],
         [results_case_dic,  results_tech_dic,  results_time_dic]] = pickle.load(handle)
    
    for idx in range(len(input_tech_list)):
        if input_tech_list[idx]['tech_name'] == 'solar':      solar_idx = idx
        if input_tech_list[idx]['tech_name'] == 'storage':    storage_idx = idx
        if input_tech_list[idx]['tech_name'] == 'shift_load': loadshift_idx = idx



    ## Energy shifted
    system_cost = results_case_dic['system_cost']
    energy_shifted = 0
    try:
        storage_fix_cost = input_tech_list[storage_idx]['fixed_cost']
        storage_var_cost = input_tech_list[storage_idx]['var_cost']
        storage_cost = results_tech_dic['storage capacity'] * storage_fix_cost + np.average(np.array(results_time_dic['storage stored']).astype(float)) * storage_var_cost
        energy_shifted = energy_shifted + np.sum(np.array(results_time_dic['storage stored']).astype(float))
    except:
        storage_cost = 0.0
        energy_shifted = energy_shifted + 0
    try:
        loadshift_fix_cost = input_tech_list[loadshift_idx]['fixed_cost']
        loadshift_var_cost = input_tech_list[loadshift_idx]['var_cost']
        loadshift_cost = results_tech_dic['shift_load capacity'] * loadshift_fix_cost + np.average(np.abs(np.array(results_time_dic['shift_load stored']).astype(float))) * loadshift_var_cost
        energy_shifted = energy_shifted + np.sum(np.abs(np.array(results_time_dic['shift_load stored']).astype(float)))
        b = np.sum(np.abs(np.array(results_time_dic['shift_load stored']).astype(float)))
    except:
        loadshift_cost = 0.0
        energy_shifted = energy_shifted + 0
    

    ## System cost considering free solar
    try:
        solar_fix_cost = input_tech_list[solar_idx]['fixed_cost']
        solar_cost = results_tech_dic['solar capacity'] * solar_fix_cost
    except:
        solar_cost = 0.0
    balancing_cost = system_cost - solar_cost

    
    ## Other generation technology costs
    wind_cap = results_tech_dic['wind capacity']
    nuclear_cap = results_tech_dic['nuclear capacity']
    cost_other = 0.015913707 * wind_cap + 0.07707777 * nuclear_cap


    ## Storage capacity
    storage_cap = results_tech_dic['storage capacity']


    ## Demand met by solar
    natgas_dispatch = results_time_dic['natgas dispatch']
    natgas_ccs_dispatch = results_time_dic['natgas_ccs dispatch']
    solar_potential = results_time_dic['solar potential']
    wind_potential = results_time_dic['wind potential']
    nuclear_potential = results_time_dic['demand potential'] * 0 + results_tech_dic['nuclear capacity']
    tot_dispatch = natgas_dispatch + natgas_ccs_dispatch + wind_potential + nuclear_potential
    try:
        demand_potential = results_time_dic['demand potential'] + results_time_dic['storage in dispatch'] - results_time_dic['storage dispatch'] + results_time_dic['shift_load dispatch'] 
    except:
        try:
            demand_potential = results_time_dic['demand potential'] + results_time_dic['storage in dispatch'] - results_time_dic['storage dispatch']
        except:
            demand_potential = results_time_dic['demand potential']
    # The first way: solar last:
    diff = demand_potential - tot_dispatch
    diff[diff<=0] = 0
    solar_dispatch_last = np.sum(diff) / np.sum(demand_potential) * 100
    # The second way: solar first:
    diff = solar_potential - demand_potential
    diff[diff<0] = 0
    solar_met_demand = np.copy(solar_potential) - diff
    solar_dispatch_first = np.sum(solar_met_demand) / np.sum(demand_potential) * 100


    return energy_shifted, balancing_cost, solar_dispatch_last, solar_dispatch_first, cost_other, storage_cap





