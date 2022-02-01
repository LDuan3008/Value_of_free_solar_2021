import numpy as np
from scipy import stats
from Shared_fun import update_series

import matplotlib
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.path as mpath


def nonan(input):
    input[input==0] = np.nan
    return input

def create_table(test_year, series_file, normalization=-1):
    case_dic, tech_dic = {}, {}
    case_dic['year_start'] = test_year
    case_dic['month_start'] = 1
    case_dic['day_start'] = 1
    case_dic['hour_start'] = 1
    case_dic['year_end'] = test_year
    case_dic['month_end'] = 12
    case_dic['day_end'] = 31
    case_dic['hour_end'] = 24
    case_dic['data_path'] = '.'
    tech_dic['series_file'] = series_file
    if normalization > 0:
        tech_dic['normalization'] = 1
    update_series(case_dic, tech_dic)
    mean_series = np.average(tech_dic['series'])
    return mean_series

def Fig_cost1_main(table, table_op, tech_cost, co2_cons, cap_list, name=''):
    SS = create_table(2019, 'US_capacity_solar_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
    color_list = ['#723C70', '#5C4D7D', '#2E6F95', '#34A0A4', '#76C893', '#99D98C', '#D9ED92']
    ax1 = plt.subplot(111)
    for idx in range(len(co2_cons)):
        system_cost = np.array(table[idx]['system_cost']).astype(float)
        techno_cost = np.array(table[idx][tech_cost]).astype(float)
        differ_cost = system_cost - techno_cost
        margin_cost = (differ_cost[:-1] - differ_cost[1:]) / (cap_list[1:]-cap_list[:-1])
        pre_name = 'Value_to_System_Cost'
        right = 2; upper = 0.12
        ax1.plot(np.array(cap_list), differ_cost, color=color_list[idx])
        optmized_solar_cap = table_op[idx]['solar_cap'][0] * SS
        optmized_diff_cost = table_op[idx]['system_cost'][0] - table_op[idx][tech_cost][0]
        ax1.scatter(optmized_solar_cap, optmized_diff_cost, marker='*', s=25, color='firebrick')
    ax1.plot(np.r_[0, right], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
    ax1.set_xticks(np.r_[cap_list[::17], 4])
    ax1.set_xlim(0, right)
    ax1.set_ylim(0, upper)
    # plt.show()
    plt.savefig(f'{pre_name}_{name}.ps')
    plt.clf()

    

def Fig_cost1(table, tech_cost, co2_cons, cap_list, plot_type=0, name=''):

    SS = create_table(2019, 'US_capacity_solar_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
    WS = create_table(2019, 'US_capacity_wind_25pctTop_unnormalilzed_19800101_20200930.csv', -1)

    if tech_cost == 'solar_fix':
        fix_cost = 0.014772123 / SS
    elif tech_cost == 'wind_fix':
        fix_cost = 0.015913707 / WS
    elif tech_cost == 'nuclear_fix':
        fix_cost = 0.07742812
    color_list = ['#723C70', '#5C4D7D', '#2E6F95', '#34A0A4', '#76C893', '#99D98C', '#D9ED92']

    ax1 = plt.subplot(111)
    for idx in range(len(co2_cons)):
        system_cost = np.array(table[idx]['system_cost']).astype(float)
        techno_cost = np.array(table[idx][tech_cost]).astype(float)
        differ_cost = system_cost - techno_cost
        margin_cost = (differ_cost[:-1] - differ_cost[1:]) / (cap_list[1:]-cap_list[:-1])
        if plot_type == 0:
            pre_name = 'System_Cost'
            right = 4; upper = 0.18
            ax1.plot(np.array(cap_list), system_cost, color=color_list[idx])
        if plot_type == 1:
            pre_name = 'Value_to_System_Cost'
            right = 4; upper = 0.12
            ax1.plot(np.array(cap_list), differ_cost, color=color_list[idx])
        if plot_type == 2:
            pre_name = 'Marginal_Value'
            right = 4; upper = 0.18
            ax1.plot(np.array(cap_list[:-1]), margin_cost, color=color_list[idx])
            ax1.plot(np.array(cap_list), np.ones(len(cap_list))*fix_cost, color='black', linestyle='--')
    ax1.plot(np.r_[0, right], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
    ax1.set_xticks(np.r_[cap_list[::17], 4])
    ax1.set_xlim(0, right)
    ax1.set_ylim(0, upper)
    # plt.show()
    plt.savefig(f'{pre_name}_{name}.ps')
    plt.clf()



def Fig_cost1_separete(table, co2_idx, tech_list, cap_list, name=''):
    color_dic = {'natgas_cap':'black',     'natgas_tot':'black',
                 'natgas_ccs_cap':'grey',  'natgas_ccs_tot':'grey',
                 'solar_cap':'wheat',      'solar_fix':'wheat',
                 'wind_cap':'skyblue',     'wind_fix':'skyblue',
                 'nuclear_cap':'brown',    'nuclear_fix':'brown',
                 'storage_cap':'indigo',   'storage_fix':'indigo',
                 'advanced_nuclear_tot': 'tomato',
                 'pgp_tot': 'limegreen',
                 'csp_tot': 'violet',
                 'system_cost':'black',
                 'lost_load_var':'cadetblue'}
    stack_list = []
    color_list = []
    for tech_idx in tech_list:
        stack_list.append(np.array(table[co2_idx][tech_idx]).astype(float))
        color_list.append(color_dic[tech_idx])
    ax1 = plt.subplot(111)
    ax1.stackplot(np.array(cap_list), stack_list, colors=color_list)
    ax1.set_xticks(np.r_[cap_list[::17], 4])
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 0.12)
    # plt.show()
    plt.savefig(f'{name}.ps')
    plt.clf()





def plot_dispatch(DicNew):

    def ReshapeData(TS, choice):
        TS_reshape = TS.reshape(-1, 24)
        if choice == 0:
            return np.mean(TS_reshape, axis=0)
        if choice == 1:
            return np.mean(TS_reshape, axis=1)
        if choice == 2:
            return np.array(TS)
    
    choice_lengh = {0:24, 1:365, 2:8760}
    choice = 2
    xaxisn = choice_lengh[choice]
    FirstDayofMonth = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    for idx in range(len(DicNew)):
        plt.stackplot( np.arange(xaxisn), 
                       ReshapeData(DicNew[idx]['nuclear_potential'], choice), 
                       ReshapeData(DicNew[idx]['solar_potential'], choice), 
                       ReshapeData(DicNew[idx]['wind_potential'], choice), 
                       ReshapeData(DicNew[idx]['natgas_dispatch'], choice), 
                       ReshapeData(DicNew[idx]['natgas_ccs_dispatch'], choice), 
                       ReshapeData(DicNew[idx]['storage_dispatch'], choice),
                       colors = ['brown', 'wheat', 'skyblue', 'black', 'grey', 'violet'] )
        plt.plot( np.arange(xaxisn), ReshapeData(DicNew[idx]['demand_potential'], choice), color='black' )
        
        # plt.xlim(0, xaxisn-1)
        # plt.xticks(FirstDayofMonth)
        # plt.xlim(15*24, 20*24)
        # plt.xticks([15*24, 16*24, 17*24, 18*24, 19*24, 20*24])
        plt.xlim((181+15)*24, (181+20)*24)
        plt.xticks([4704, 4728, 4752, 4776, 4800, 4824])

        plt.ylim(0, 2)
        plt.yticks([0, 0.5, 1, 1.5, 2.0])
        plt.show() 
        # plt.savefig('Dispatch'+str(idx)+'.ps') 
        plt.clf()



def main_figure2(storage_amount, Additional_Ls0, solar_dispatch):

    ax0 = plt.subplot(111)
    ax0.plot(storage_amount, Additional_Ls0[0, 2, :], color='black')
    ax0.plot(storage_amount, solar_dispatch[0, :, 10], color='#A50026')
    ax0.plot(storage_amount, solar_dispatch[0, :, 15], color='#FDB366')
    ax0.plot(storage_amount, solar_dispatch[0, :, 20], color='#EAECCC')
    ax0.plot(storage_amount, solar_dispatch[0, :, 25], color='#6EA6CD')
    ax0.plot(storage_amount, solar_dispatch[0, :, 30], color='#364B9A')
    ax0.set_xscale('log', basex=10)
    ax0.set_xlim(10**0, 10**3)
    ax0.set_ylim(0, 100)
    # plt.show() 
    plt.savefig('f2a.ps') 
    plt.clf()

    ax0 = plt.subplot(111)
    ax0.plot(storage_amount, Additional_Ls0[1, 2, :], color='black')
    ax0.plot(storage_amount, solar_dispatch[1, :, 10], color='#A50026')
    ax0.plot(storage_amount, solar_dispatch[1, :, 15], color='#FDB366')
    ax0.plot(storage_amount, solar_dispatch[1, :, 20], color='#EAECCC')
    ax0.plot(storage_amount, solar_dispatch[1, :, 25], color='#6EA6CD')
    ax0.plot(storage_amount, solar_dispatch[1, :, 30], color='#364B9A')
    ax0.set_xscale('log', basex=10)
    ax0.set_xlim(10**0, 10**3)
    ax0.set_ylim(0, 100)
    # plt.show() 
    plt.savefig('f2b.ps') 
    plt.clf()

    ax0 = plt.subplot(111)
    ax0.plot(storage_amount, Additional_Ls0[2, 2, :], color='black')
    ax0.plot(storage_amount, solar_dispatch[2, :, 10], color='#A50026')
    ax0.plot(storage_amount, solar_dispatch[2, :, 15], color='#FDB366')
    ax0.plot(storage_amount, solar_dispatch[2, :, 20], color='#EAECCC')
    ax0.plot(storage_amount, solar_dispatch[2, :, 25], color='#6EA6CD')
    ax0.plot(storage_amount, solar_dispatch[2, :, 30], color='#364B9A')
    ax0.set_xscale('log', basex=10)
    ax0.set_xlim(10**0, 10**3)
    ax0.set_ylim(0, 100)
    # plt.show() 
    plt.savefig('f2c.ps') 
    plt.clf()

    # ax0 = plt.subplot(111)
    # ax0.plot(storage_amount, Additional_Ls0[3, 2, :], color='black')
    # ax0.plot(storage_amount, solar_dispatch[3, :, 10], color='#A50026')
    # ax0.plot(storage_amount, solar_dispatch[3, :, 15], color='#FDB366')
    # ax0.plot(storage_amount, solar_dispatch[3, :, 20], color='#EAECCC')
    # ax0.plot(storage_amount, solar_dispatch[3, :, 25], color='#6EA6CD')
    # ax0.plot(storage_amount, solar_dispatch[3, :, 30], color='#364B9A')
    # ax0.set_xscale('log', basex=10)
    # ax0.set_xlim(10**0, 20)
    # # ax0.set_xlim(0, 20)
    # ax0.set_ylim(0, 100)
    # # plt.show() 
    # plt.savefig('f2d.ps') 
    # plt.clf()


def contour_plot(flixble_amount, storage_amount, energy_shifted, solar_dispatch):

    aaaa = plt.get_cmap('viridis_r', 100) # 'Reds'

    ax1 = plt.subplot(111)
    mp = ax1.pcolormesh(flixble_amount, storage_amount, energy_shifted[0]/8760, cmap=aaaa, norm=colors.LogNorm(vmin=10**-2,vmax=10**2))
    ax1.contour(flixble_amount, storage_amount, solar_dispatch[0], ['10', '30', '50'], colors='black')
    # ax1.contour(flixble_amount, storage_amount, solar_dispatch[0], ['10', '30', '50', '70', '90'], colors='black')
    plt.colorbar(mp, extend='both')
    ax1.set_xscale('log', basex=10)
    ax1.set_yscale('log', basey=10)
    ax1.set_xlim(10**-3, 10**3)
    ax1.set_ylim(10**0, 10**3)
    plt.show() 
    # plt.savefig('contour05x.ps') 
    plt.clf()

    # ax1 = plt.subplot(111)
    # mp = ax1.pcolormesh(flixble_amount, storage_amount, energy_shifted[1]/8760, cmap=aaaa, norm=colors.LogNorm(vmin=10**-2,vmax=10**2))
    # ax1.contour(flixble_amount, storage_amount, solar_dispatch[1], ['10', '30', '50', '70', '90'], colors='black')
    # plt.colorbar(mp, extend='both')
    # ax1.set_xscale('log', basex=10)
    # ax1.set_yscale('log', basey=10)
    # ax1.set_xlim(10**-3, 10**3)
    # ax1.set_ylim(10**0, 10**3)
    # # plt.show() 
    # plt.savefig('contour1x.ps') 
    # plt.clf()

    # ax1 = plt.subplot(111)
    # mp = ax1.pcolormesh(flixble_amount, storage_amount, energy_shifted[2]/8760, cmap=aaaa, norm=colors.LogNorm(vmin=10**-2,vmax=10**2))
    # ax1.contour(flixble_amount, storage_amount, solar_dispatch[2], ['10', '30', '50', '70', '90'], colors='black')
    # plt.colorbar(mp, extend='both')
    # ax1.set_xscale('log', basex=10)
    # ax1.set_yscale('log', basey=10)
    # ax1.set_xlim(10**-3, 10**3)
    # ax1.set_ylim(10**0, 10**3)
    # # plt.show() 
    # plt.savefig('contour2x.ps') 
    # plt.clf()


def shift_load_main(flixble_amount, Additional_St0, solar_dispatch):

    ax0 = plt.subplot(111)
    ax0.plot(flixble_amount, Additional_St0[1, 2, :], color='black')
    ax0.plot(flixble_amount, solar_dispatch[1, 0, :])  #-3, -2, -1, 0, 1
    ax0.plot(flixble_amount, solar_dispatch[1, 5, :])
    ax0.plot(flixble_amount, solar_dispatch[1, 10, :])
    ax0.plot(flixble_amount, solar_dispatch[1, 15, :])
    # ax0.plot(flixble_amount, solar_dispatch[1, 25, :])
    ax0.set_xscale('log', basex=10)
    ax0.set_xlim(10**-3, 10**3)
    ax0.set_ylim(0, 100)
    # plt.show() 
    plt.savefig('LSa.ps') 
    plt.clf()

    ax0 = plt.subplot(111)
    ax0.plot(flixble_amount, Additional_St0[2, 2, :], color='black')
    ax0.plot(flixble_amount, solar_dispatch[2, 0, :])
    ax0.plot(flixble_amount, solar_dispatch[2, 5, :])
    ax0.plot(flixble_amount, solar_dispatch[2, 10, :])
    ax0.plot(flixble_amount, solar_dispatch[2, 15, :])
    # ax0.plot(flixble_amount, solar_dispatch[2, :, :])
    ax0.set_xscale('log', basex=10)
    ax0.set_xlim(10**-3, 10**3)
    ax0.set_ylim(0, 100)
    # plt.show() 
    plt.savefig('LSb.ps') 
    plt.clf()






















"""

def cal_mean_uncertainty(table_in, VarName, idx, num_of_sub_case=11, co2_constraint=3):
    total_ensemble_number = int(len(table_in)/co2_constraint)
    table_out = np.zeros([total_ensemble_number, num_of_sub_case])
    for sub_idx in range(total_ensemble_number):
        table_out[sub_idx] = np.array(table_in[sub_idx*co2_constraint+idx][VarName]).astype(float)
    return table_out

def Fig_EachPoint(summary_table, type, plot_type=0, posfix=''):
    demand_year = np.arange(2016, 2020, 1)
    weather_year = np.arange(1980, 2020, 1)
    co2_constraints = [1e24, 50.0, 1.0]
    capacity_specified = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    color_list = ['black', 'firebrick', 'royalblue']
    total_cases = len(summary_table)
    total_constraints = len(co2_constraints)
    ensemble_number_pCase = len(demand_year) * len(weather_year)
    ax1 = plt.subplot(111)
    # System cost, technology cost, and value of adding technology X
    if type == 'wind':    tech_costs_idx = 'wind_fix';    fix_cost = 0.015913707 / 0.4298325797320377
    if type == 'solar':   tech_costs_idx = 'solar_fix';   fix_cost = 0.014772123 / 0.2780627243401388
    if type == 'nuclear': tech_costs_idx = 'nuclear_fix'; fix_cost = 0.07742812

    for idx in range(total_constraints):
        if plot_type == 0:
            pre_name = 'System_Cost'
            right = 2; upper = 0.18
            system_cost = cal_mean_uncertainty(summary_table, 'system_cost', idx)
            mean_system_cost = np.mean(system_cost, axis=0)
            std_system_cost = np.std(system_cost, axis=0)
            LB, UP = np.min(system_cost, axis=0), np.max(system_cost, axis=0)
            ax1.plot(np.array(capacity_specified), mean_system_cost, color=color_list[idx])
            ax1.fill_between(np.array(capacity_specified), LB, UP, color=color_list[idx], alpha=0.3)
        if plot_type == 1:
            pre_name = 'Value_to_System_Cost'
            # right = 2; upper = 0.14
            # right = 2; upper = 0.18
            right = 2; upper = 0.11
            system_cost = cal_mean_uncertainty(summary_table, 'system_cost', idx)
            techno_cost = cal_mean_uncertainty(summary_table, tech_costs_idx, idx)
            diff_mean_cost = np.mean(system_cost, axis=0) - np.mean(techno_cost, axis=0)
            diff_std_cost = np.std(system_cost-techno_cost, axis=0)
            mean_techno_cost = np.mean(techno_cost, axis=0)
            LB, UP = np.min(system_cost-techno_cost, axis=0), np.max(system_cost-techno_cost, axis=0)
            LB_tech, UP_tech = np.min(techno_cost, axis=0), np.max(techno_cost, axis=0)
            ax1.plot(np.array(capacity_specified), diff_mean_cost, color=color_list[idx])
            ax1.fill_between(np.array(capacity_specified), LB, UP, color=color_list[idx], alpha=0.3)
        if plot_type == 2:
            pre_name = 'Marginal_Value'
            # right = 1.8; upper = 0.14
            right = 1.8; upper = 0.25
            system_cost = cal_mean_uncertainty(summary_table, 'system_cost', idx)
            techno_cost = cal_mean_uncertainty(summary_table, tech_costs_idx, idx)
            diff_mean_cost = np.mean(system_cost, axis=0) - np.mean(techno_cost, axis=0)
            marginal_cost = (diff_mean_cost[:-1] - diff_mean_cost[1:]) / (capacity_specified[1:]-capacity_specified[:-1]) 
            ax1.plot(np.array(capacity_specified[:-1]), marginal_cost, color=color_list[idx])
            ax1.plot(np.array(capacity_specified[:-1]), np.ones(len(capacity_specified[:-1]))*fix_cost, color='black', linestyle='--')
        if plot_type == 3:
            pre_name = 'Mean_Value'
            upper = 0.14
            system_cost = cal_mean_uncertainty(summary_table, 'system_cost', idx)
            techno_cost = cal_mean_uncertainty(summary_table, tech_costs_idx, idx)
            diff_mean_cost = np.mean(system_cost, axis=0) - np.mean(techno_cost, axis=0)
            marginal_cost = (diff_mean_cost[0] - diff_mean_cost[:-1]) / (capacity_specified[1:]-capacity_specified[0]) 
            ax1.plot(np.array(capacity_specified[:-1]), marginal_cost, color=color_list[idx])
    ax1.plot(np.r_[0, right], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
    ax1.set_xlim(0, right)
    ax1.set_ylim(0, upper)
    ax1.set_xticks(capacity_specified)
    # plt.show()
    # plt.savefig(f'{pre_name}_{type}{posfix}.png', dpi=500)
    plt.savefig(f'{pre_name}_{type}{posfix}.ps')
    plt.clf()


def Fig_separate_costs(summary_table, rest_tech, PreTech, posfix=''):
    demand_year = np.arange(2016, 2020, 1)
    weather_year = np.arange(1980, 2020, 1)
    co2_constraints = [1e24, 50.0, 1.0]
    capacity_specified = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    total_cases = len(summary_table)
    total_constraints = len(co2_constraints)
    ensemble_number_pCase = len(demand_year) * len(weather_year)
    total_tech_to_plot = len(rest_tech)
    upper = 0.11
    pre_name = 'TechCost'
    color_dic = { 'natgas_cap':'black',    'natgas_tot':'black',
                  'natgas_ccs_cap':'grey', 'natgas_ccs_tot':'grey',
                  'solar_cap':'wheat',     'solar_fix':'wheat',
                  'wind_cap':'skyblue',    'wind_fix':'skyblue',
                  'nuclear_cap':'brown',   'nuclear_fix':'brown',
                  'storage_cap':'indigo',  'storage_fix':'indigo',
                  'system_cost':'black',
                  'lost_load_var':'cadetblue'}
    # System cost, technology cost, and value of adding technology X
    if PreTech == 'wind':    tech_costs_idx = 'wind_fix';    fix_cost = 0.015913707 / 0.4298325797320377
    if PreTech == 'solar':   tech_costs_idx = 'solar_fix';   fix_cost = 0.014772123 / 0.2780627243401388
    if PreTech == 'nuclear': tech_costs_idx = 'nuclear_fix'; fix_cost = 0.07742812

    for idx in range(total_constraints):
        ax1 = plt.subplot(111)
        system_cost = cal_mean_uncertainty(summary_table, 'system_cost', idx)
        techno_cost = cal_mean_uncertainty(summary_table, tech_costs_idx, idx)
        mean_cost = np.mean(system_cost, axis=0) - np.mean(techno_cost, axis=0)
        ax1.plot(np.array(capacity_specified), mean_cost, color='black')
        stack_list = []
        color_list = []
        for tech_idx in range(total_tech_to_plot):
            tech_costs = rest_tech[tech_idx]
            techno_cost = cal_mean_uncertainty(summary_table, tech_costs, idx)
            mean_techno_cost = np.mean(techno_cost, axis=0)
            stack_list.append(mean_techno_cost)
            color_list.append( color_dic[tech_costs] )
        ax1.stackplot(np.array(capacity_specified), stack_list, colors=color_list)
        ax1.plot(np.r_[0, 2], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
        ax1.set_xlim(0, 2)
        ax1.set_ylim(0, upper)
        ax1.set_xticks(capacity_specified)
        # plt.show()
        plt.savefig(f'{pre_name}_{PreTech}{idx}{posfix}.png', dpi=500)
        # plt.savefig(f'{pre_name}_{PreTech}{idx}{posfix}.ps')
        plt.clf()



def Fig_Half(TableSo_All, TableSo_noW, TableSo_noN, TableWi_All, TableWi_noS, TableWi_noN, TableNu_All, TableNu_noS, TableNu_noW):

    demand_year = np.arange(2016, 2020, 1)
    weather_year = np.arange(1980, 2020, 1)
    co2_constraints = [1e24, 50.0, 1.0]
    capacity_specified = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    # color_list = ['#8b0724', '#bd9500', '#90fa6e']
    color_list = ['black', 'firebrick', 'royalblue']

    total_constraints = len(co2_constraints)
    ensemble_number_pCase = len(demand_year) * len(weather_year)


    # Calculate_array 
    Plot_Array = np.zeros([4, 3, 2, 160])
    idx = 2

    def GetResults(table, tech_costs_idx, idx):
        system_cost = cal_mean_uncertainty(table, 'system_cost', idx) # (ensemble, cap_lev)
        techno_cost = cal_mean_uncertainty(table, tech_costs_idx, idx)

        diff_cost = system_cost - techno_cost
        diff_FiHalf = (diff_cost[:, 5] - diff_cost[:, 0]) / -1.0
        diff_LaHalf = (diff_cost[:, 10] - diff_cost[:, 5]) / -1.0
        return diff_FiHalf, diff_LaHalf

    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(223, sharex=ax1, sharey=ax1)
    ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)

    # Solar:
    S11, S12 = GetResults(TableSo_All, 'solar_fix', idx)
    ax1.boxplot(S11, positions=[0-3], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    ax1.boxplot(S12, positions=[13-3], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    S21, S22 = GetResults(TableSo_noW, 'solar_fix', idx)
    ax3.boxplot(S21, positions=[0-3], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    ax3.boxplot(S22, positions=[13-3], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    S31, S32 = GetResults(TableSo_noN, 'solar_fix', idx)
    ax4.boxplot(S31, positions=[0-3], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    ax4.boxplot(S32, positions=[13-3], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))

    # Wind:
    W11, W12 = GetResults(TableWi_All, 'wind_fix', idx)
    ax1.boxplot(W11, positions=[0], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    ax1.boxplot(W12, positions=[13], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    W21, W22 = GetResults(TableWi_noS, 'wind_fix', idx)
    ax2.boxplot(W21, positions=[0], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    ax2.boxplot(W22, positions=[13], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    W31, W32 = GetResults(TableWi_noN, 'wind_fix', idx)
    ax4.boxplot(W31, positions=[0], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    ax4.boxplot(W32, positions=[13], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))

    # Nuclear:
    N11, N12 = GetResults(TableNu_All, 'nuclear_fix', idx)
    ax1.boxplot(N11, positions=[0+3], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    ax1.boxplot(N12, positions=[13+3], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    N21, N22 = GetResults(TableNu_noS, 'nuclear_fix', idx)
    ax2.boxplot(N21, positions=[0+3], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    ax2.boxplot(N22, positions=[13+3], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    N31, N32 = GetResults(TableNu_noW, 'nuclear_fix', idx)
    ax3.boxplot(N31, positions=[0+3], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))
    ax3.boxplot(N32, positions=[13+3], widths=1.8, flierprops=dict(marker='o', markersize=4), medianprops=dict(color='green'))

    ax1.plot(np.r_[-5, 18], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
    ax2.plot(np.r_[-5, 18], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
    ax3.plot(np.r_[-5, 18], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
    ax4.plot(np.r_[-5, 18], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
    ax1.set_xlim(-5, 18)
    ax1.set_ylim(-0.02, 0.25)

    # plt.show()  
    plt.savefig(f'fig2.ps')
    plt.clf()




# -------------------------------------------------------------------------------------------------------------------------------------------------



def Fig_battery(summary_table, type, repeat_list, co2_constraint, posfix=''):
    demand_year = np.arange(2016, 2020, 1)
    weather_year = np.arange(1980, 2020, 1)
    color_list = ['black', 'firebrick', 'royalblue', 'green']
    total_cases = len(summary_table)
    total_constraints = len(co2_constraint)
    ensemble_number_pCase = len(demand_year) * len(weather_year)
    capacity_specified = np.array(repeat_list)
    fix_cost = 0.007351353

    ax1 = plt.subplot(111)
    for idx in range(total_constraints):
        if type == 'BsCap':
            pre_name = 'Value_to_System_Cost'
            tech_costs_idx = 'storage_fix'
            system_cost = cal_mean_uncertainty(summary_table, 'system_cost', idx, len(repeat_list), len(co2_constraint))
            techno_cost = cal_mean_uncertainty(summary_table, tech_costs_idx, idx, len(repeat_list), len(co2_constraint))
            mean_system_cost = np.mean(system_cost, axis=0); std_system_cost = np.std(system_cost, axis=0)
            mean_techno_cost = np.mean(techno_cost, axis=0); std_techno_cost = np.std(techno_cost, axis=0)
            diff_mean_cost = mean_system_cost - mean_techno_cost
            diff_std_cost = np.sqrt(std_system_cost**2 + std_techno_cost**2)
            ConInt_50p = stats.norm.interval(0.95, loc=diff_mean_cost, scale=diff_std_cost)
            ax1.plot(np.array(repeat_list), diff_mean_cost, color=color_list[idx])
            # ax1.fill_between(np.array(repeat_list), ConInt_50p[0], ConInt_50p[1], color=color_list[idx], alpha=0.3)
        if type == 'BsDecay':
            pre_name = 'System_Cost'
            system_cost = cal_mean_uncertainty(summary_table, 'system_cost', idx, len(repeat_list))
            mean_system_cost = np.mean(system_cost, axis=0)
            std_system_cost = np.std(system_cost, axis=0)
            ConInt_50p = stats.norm.interval(0.5, loc=mean_system_cost, scale=std_system_cost)
            ax1.plot(np.array(repeat_list), mean_system_cost, color=color_list[idx])
            ax1.fill_between(np.array(repeat_list), ConInt_50p[0], ConInt_50p[1], color=color_list[idx], alpha=0.3)
        if type == '2':
            pre_name = 'Marginal_Value'
            tech_costs_idx = 'storage_fix'
            system_cost = cal_mean_uncertainty(summary_table, 'system_cost', idx, len(repeat_list))
            techno_cost = cal_mean_uncertainty(summary_table, tech_costs_idx, idx, len(repeat_list))
            diff_mean_cost = np.mean(system_cost, axis=0) - np.mean(techno_cost, axis=0)
            marginal_cost = (diff_mean_cost[:-1] - diff_mean_cost[1:]) / (capacity_specified[1:]-capacity_specified[:-1]) 
            ax1.plot(np.array(capacity_specified[:-1]), marginal_cost, color=color_list[idx])
            ax1.plot(np.array(capacity_specified[:-1]), np.ones(len(capacity_specified[:-1]))*fix_cost, color='black', linestyle='--')
    ax1.plot(np.r_[repeat_list[0], repeat_list[-1]], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
    ax1.set_xlim(2**0, 2**14)
    ax1.set_xscale('log', basex=2)
    ax1.set_ylim(0, 0.11)
    # ax1.set_ylim(0, 0.007)
    ax1.set_xticks([2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14])
    # plt.show()
    # plt.savefig(f'{pre_name}{posfix}.png', dpi=500)
    plt.savefig(f'{pre_name}{posfix}_{type}.ps')
    plt.clf()


def Fig_separate_costs_storage(summary_table, rest_tech, repeat_list, posfix=''):
    demand_year = np.arange(2016, 2020, 1)
    weather_year = np.arange(1980, 2020, 1)
    co2_constraints = [1e24, 50.0, 1.0, 0]
    total_cases = len(summary_table)
    total_constraints = len(co2_constraints)
    ensemble_number_pCase = len(demand_year) * len(weather_year)
    total_tech_to_plot = len(rest_tech)
    pre_name = 'TechCost'
    color_dic = { 'natgas_cap':'black',    'natgas_tot':'black',
                  'natgas_ccs_cap':'grey', 'natgas_ccs_tot':'grey',
                  'solar_cap':'wheat',     'solar_fix':'wheat',
                  'wind_cap':'skyblue',    'wind_fix':'skyblue',
                  'nuclear_cap':'brown',   'nuclear_fix':'brown',
                  'storage_cap':'indigo',  'storage_fix':'indigo',
                  'system_cost':'black',
                  'lost_load_var':'cadetblue'}
    # System cost, technology cost, and value of adding technology X
    tech_costs_idx = 'storage_fix'
    fix_cost = 0.007351353
    for idx in range(total_constraints):
        ax1 = plt.subplot(111)
        system_cost = cal_mean_uncertainty(summary_table, 'system_cost', idx, len(repeat_list), len(co2_constraints))
        techno_cost = cal_mean_uncertainty(summary_table, tech_costs_idx, idx, len(repeat_list), len(co2_constraints))
        mean_cost = np.mean(system_cost, axis=0) - np.mean(techno_cost, axis=0)
        ax1.plot(np.array(repeat_list), mean_cost, color='black')
        stack_list = []
        color_list = []
        for tech_idx in range(total_tech_to_plot):
            tech_costs = rest_tech[tech_idx]
            techno_cost = cal_mean_uncertainty(summary_table, tech_costs, idx, len(repeat_list), len(co2_constraints))
            mean_techno_cost = np.mean(techno_cost, axis=0)
            stack_list.append(mean_techno_cost)
            color_list.append( color_dic[tech_costs] )
        ax1.stackplot(np.array(repeat_list), stack_list, colors=color_list)
        ax1.set_xlim(2**0, 2**14)
        ax1.set_xscale('log', basex=2)
        ax1.set_ylim(0, 0.11)
        # plt.show()
        # plt.savefig(f'Storage{posfix}{idx}.png', dpi=500)
        plt.savefig(f'Storage{posfix}{idx}.ps')
        plt.clf()








def Check_Opt(Table_Opt):
    demand_year = np.arange(2016, 2020, 1)
    weather_year = np.arange(1980, 2020, 1)
    co2_constraints = [1e24, 50.0, 1.0]
    total_constraints = len(co2_constraints)
    ensemble_number_pCase = len(demand_year) * len(weather_year)
    for idx in range(total_constraints):
        techno_cap = cal_mean_uncertainty(Table_Opt, 'storage_cap', idx, 1)   
        techno_cost = cal_mean_uncertainty(Table_Opt, 'storage_fix', idx, 1)
        print (np.mean(techno_cap))
        print (np.mean(techno_cost))
        print (np.mean(techno_cost)/np.mean(techno_cap))
        print ('--')









def battery_test(table, cap_list, posfix=''):
    def get_table(table, cap_list, VarName):
        table_out = np.zeros([len(table), len(cap_list)])
        for sub_idx in range(len(table)):
            print ('check_here', sub_idx)
            table_out[sub_idx] = np.array(table[sub_idx][VarName]).astype(float)
        return table_out

    pre_name = 'Value_to_System_Cost'
    tech_costs_idx = 'storage_fix'
    total_scenarios = len(table)
    system_cost = get_table(table, cap_list, 'system_cost')
    techno_cost = get_table(table, cap_list, tech_costs_idx)
    diff_mean_cost = system_cost - techno_cost
    cl = ['black', 'green', 'royalblue', 'firebrick']
    ax1 = plt.subplot(111)
    for idx in range(len(table)):
        ax1.plot(np.array(cap_list), diff_mean_cost[idx], color=cl[idx])
    ax1.set_xlim(2**0, 2**14)
    ax1.set_xscale('log', basex=2)
    ax1.set_ylim(0, 0.11)
    # plt.show()
    # plt.savefig(f'{pre_name}{posfix}.png', dpi=500)
    plt.savefig(f'{pre_name}{posfix}.ps')
    plt.clf()









def battery_value(table, cap_list, posfix=''):
    def get_table(table, cap_list, VarName):
        table_out = np.zeros([len(table), len(cap_list)])
        for sub_idx in range(len(table)):
            table_out[sub_idx] = np.array(table[sub_idx][VarName]).astype(float)
        return table_out
    pre_name = 'Value_to_System_Cost'
    tech_costs_idx = 'storage_fix'
    total_scenarios = len(table)
    system_cost = get_table(table, cap_list, 'system_cost')
    techno_cost = get_table(table, cap_list, tech_costs_idx)
    diff_mean_cost = np.array(system_cost - techno_cost)
    x_axis_array = np.array(cap_list)
    cl = ['black', 'green', 'royalblue', 'firebrick']
    ax1 = plt.subplot(111)
    for idx in range(len(table)):
        value_to_plot = (diff_mean_cost[idx, :-1] - diff_mean_cost[idx, 1:]) / (x_axis_array[1:]-x_axis_array[:-1]) 
        ax1.plot(np.array(cap_list)[:-1], value_to_plot, color=cl[idx])
    ax1.set_xlim(2**-5, 2**10)
    # ax1.set_xscale('log', basex=2)
    # ax1.set_ylim(0, 0.11)
    ax1.set_ylim(0, 0.035)
    # ax1.set_ylim(0, 6)
    # plt.show()
    # plt.savefig(f'{pre_name}{posfix}.png', dpi=500)
    plt.savefig(f'{pre_name}{posfix}.ps')
    plt.clf()


def Separate_costs_storage2(table, rest_tech, cap_list, idx, type, posfix=''):
    total_cases = len(table)
    total_tech_to_plot = len(rest_tech)
    pre_name = 'TechCost'
    color_dic = { 'natgas_cap':'black',    'natgas_tot':'black',
                  'natgas_ccs_cap':'grey', 'natgas_ccs_tot':'grey',
                  'solar_cap':'wheat',     'solar_fix':'wheat',
                  'wind_cap':'skyblue',    'wind_fix':'skyblue',
                  'nuclear_cap':'brown',   'nuclear_fix':'brown',
                  'storage_cap':'indigo',  'storage_fix':'indigo',
                  'system_cost':'black',
                  'lost_load_var':'cadetblue'}
    if type == 'cost':
        ax1 = plt.subplot(111)
        system_cost = np.array(table[idx]['system_cost']).astype(float)
        techno_cost = np.array(table[idx]['storage_fix']).astype(float)
        mean_cost = system_cost - techno_cost
        ax1.plot(np.array(cap_list), mean_cost, color='black')
        stack_list = []
        color_list = []
        for tech_idx in range(total_tech_to_plot):
            tech_costs = rest_tech[tech_idx]
            techno_cost = np.array(table[idx][tech_costs]).astype(float)  
            stack_list.append(techno_cost)
            color_list.append(color_dic[tech_costs])
        ax1.stackplot(np.array(cap_list), stack_list, colors=color_list)
        ax1.set_xlim(2**0, 2**14)
        ax1.set_xscale('log', basex=2)
        ax1.set_ylim(0, 0.11)
        # ax1.set_ylim(0, 0.14)
        # plt.show()
        # plt.savefig(f'Storage{posfix}.png', dpi=500)
        plt.savefig(f'Storage{posfix}{idx}.ps')
        plt.clf()


def Separate_costs_storage3(table, rest_tech, cap_list, posfix=''):

    fig = plt.figure(figsize=(12,4))
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.98, hspace=0.7, wspace=0.4)

    total_cases = len(table)
    total_tech_to_plot = len(rest_tech)
    pre_name = 'TechCost'
    color_dic = { 'natgas_cap':'black',    'natgas_tot':'black',
                  'natgas_ccs_cap':'grey', 'natgas_ccs_tot':'grey',
                  'solar_cap':'wheat',     'solar_fix':'wheat',
                  'wind_cap':'skyblue',    'wind_fix':'skyblue',
                  'nuclear_cap':'brown',   'nuclear_fix':'brown',
                  'storage_cap':'indigo',  'storage_fix':'indigo',
                  'system_cost':'black',
                  'lost_load_var':'cadetblue'}
    
    def plot_sub_plot(ax, table, idx):
        system_cost = np.array(table[idx]['system_cost']).astype(float)
        techno_cost = np.array(table[idx]['storage_fix']).astype(float)
        mean_cost = system_cost - techno_cost
        ax.plot(np.array(cap_list), mean_cost, color='black')
        stack_list = []
        color_list = []
        for tech_idx in range(total_tech_to_plot):
            tech_costs = rest_tech[tech_idx]
            techno_cost = np.array(table[idx][tech_costs]).astype(float)  
            stack_list.append(techno_cost)
            color_list.append(color_dic[tech_costs])
        ax.stackplot(np.array(cap_list), stack_list, colors=color_list)
    
    ax1 = plt.subplot(151);                         plot_sub_plot(ax1, table, 0)
    ax2 = plt.subplot(152, sharex=ax1, sharey=ax1); plot_sub_plot(ax2, table, 1)
    ax3 = plt.subplot(153, sharex=ax1, sharey=ax1); plot_sub_plot(ax3, table, 2)
    ax4 = plt.subplot(154, sharex=ax1, sharey=ax1); plot_sub_plot(ax4, table, 3)
    ax5 = plt.subplot(155, sharex=ax1, sharey=ax1); plot_sub_plot(ax5, table, 4)

    ax1.set_xlim(2**0, 2**14)
    ax1.set_xscale('log', basex=2)
    ax1.set_ylim(0, 0.15)
    ax1.set_xticks([2**0, 2**2, 2**4, 2**6, 2**8, 2**10, 2**12, 2**14])
    # plt.show()
    # plt.savefig(f'Storage{posfix}{str(idx)}.png', dpi=500)
    plt.savefig(f'Storage{posfix}.ps')
    plt.clf()
"""