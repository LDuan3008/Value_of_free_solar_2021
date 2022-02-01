
import numpy as np
import pickle, os
from Postprocess_func import Get_Table
from Postprocess_func import get_case_dispatch
from Postprocess_func import for_fun
from scipy import stats
from Shared_fun import update_series
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


#### Definition zone:
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
    mean_series = update_series(case_dic, tech_dic)
    return mean_series, tech_dic['series']


def Fig1_sub1(table, table_op, cap_op, tech_cost, co2_cons, cap_list, year, xupper, MSS):
    color_list = ['#723C70', '#5C4D7D', '#2E6F95', '#34A0A4', '#76C893', '#99D98C', '#D9ED92']
    costs_value = np.zeros([len(co2_cons), 2])
    ax1 = plt.subplot(121)
    for idx in range(len(co2_cons)):
        system_cost = np.array(table[idx]['system_cost']).astype(float)
        techno_cost = np.array(table[idx][tech_cost]).astype(float)
        differ_cost = system_cost - techno_cost
        ax1.plot(np.array(cap_list), differ_cost, color=color_list[idx])
        optmized_solar_cap = table_op[idx][cap_op][0] * MSS
        optmized_diff_cost = table_op[idx]['system_cost'][0] - table_op[idx][tech_cost][0]
        ax1.scatter(optmized_solar_cap, optmized_diff_cost, marker='*', s=25, color='firebrick')
        costs_value[idx][0], costs_value[idx][1] = np.array(differ_cost)[0], np.array(differ_cost)[100]
    print (costs_value[-3][1], costs_value[0][0], (costs_value[-3][1]-costs_value[0][0])/costs_value[0][0]*100)
    print (costs_value[-1][1], costs_value[0][0], (costs_value[-1][1]-costs_value[0][0])/costs_value[0][0]*100)
    ax1.plot(np.r_[0, xupper], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
    ax1.set_xticks(np.arange(0, 4.4, 0.4))
    ax1.set_xlim(0, xupper)
    ax1.set_ylim(0, 0.12)
def Fig1_sub2(DS, SS, MSS):
    SS_axDS = 0.4 / MSS * SS
    SS_bxDS = 0.8 / MSS * SS
    SS_cxDS = 1.2 / MSS * SS
    SS_dxDS = 1.6 / MSS * SS
    SS_2xDS = 2 / MSS * SS
    hourly_DS = np.average(DS.reshape(-1, 24), axis=0)
    hourly_SS_axDS = np.average(SS_axDS.reshape(-1, 24), axis=0)
    hourly_SS_bxDS = np.average(SS_bxDS.reshape(-1, 24), axis=0)
    hourly_SS_cxDS = np.average(SS_cxDS.reshape(-1, 24), axis=0)
    hourly_SS_dxDS = np.average(SS_dxDS.reshape(-1, 24), axis=0)
    hourly_SS_2xDS = np.average(SS_2xDS.reshape(-1, 24), axis=0)
    ax2 = plt.subplot(122)
    ax2.plot(np.arange(24), np.r_[hourly_DS[5:], hourly_DS[:5]], color='black')
    ax2.plot(np.arange(24), np.r_[hourly_SS_axDS[5:], hourly_SS_axDS[:5]], color='mistyrose')
    ax2.plot(np.arange(24), np.r_[hourly_SS_bxDS[5:], hourly_SS_bxDS[:5]], color='pink')
    ax2.plot(np.arange(24), np.r_[hourly_SS_cxDS[5:], hourly_SS_cxDS[:5]], color='red')
    ax2.plot(np.arange(24), np.r_[hourly_SS_dxDS[5:], hourly_SS_dxDS[:5]], color='orange')
    ax2.plot(np.arange(24), np.r_[hourly_SS_2xDS[5:], hourly_SS_2xDS[:5]], color='firebrick')
    ax2.set_xlim(0, 23)
    ax2.set_ylim(0, 5.2)

def fig1(table, table_op, cap_op, tech_cost, co2_cons, cap_list, year, xupper):
    MSD, DS = create_table(year, 'US_demand_unnormalized.csv', 1)
    if tech_cost == 'solar_fix':
        MSS, SS = create_table(year, 'US_capacity_solar_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
    elif tech_cost == 'wind_fix':
        MSS, SS = create_table(year, 'US_capacity_wind_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
    else:
        MSS = 1
    Fig1_sub1(table, table_op, cap_op, tech_cost, co2_cons, cap_list, year, xupper, MSS)
    Fig1_sub2(DS, SS, MSS)
    # plt.show()
    plt.savefig(f'fig{year}.ps')
    plt.clf()



# Marginal value plot
def fig_marginal_value(table, tech_cost, co2_cons, cap_list, year):
    if tech_cost == 'solar_fix':
        MSS, SS = create_table(year, 'US_capacity_solar_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
        fix_cost = 0.014772123
    elif tech_cost == 'wind_fix':
        MSS, SS = create_table(year, 'US_capacity_wind_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
        fix_cost = 0.015913707
    else:
        MSS = 1
        fix_cost = 0.07707777
    color_list = ['#723C70', '#5C4D7D', '#2E6F95', '#34A0A4', '#76C893', '#99D98C', '#D9ED92']
    ax1 = plt.subplot(111)
    for idx in range(len(co2_cons)):
        system_cost = np.array(table[idx]['system_cost']).astype(float)
        techno_cost = np.array(table[idx][tech_cost]).astype(float)
        differ_cost = system_cost - techno_cost
        marginal_cost = (differ_cost[:-1] - differ_cost[1:]) / (cap_list[1:]-cap_list[:-1]) 
        ax1.plot(np.array(cap_list[:-1]), marginal_cost, color=color_list[idx])
        ax1.plot(np.array(cap_list[:-1]), np.ones(len(cap_list[:-1]))*fix_cost/MSS, color='black', linestyle='--')
    ax1.plot(np.r_[0, 2], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
    ax1.set_xticks(np.arange(0, 4.4, 0.4))
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 0.2)
    # plt.show()
    plt.savefig('test.ps')
    plt.clf()




# Cost investment plot
def fig_capchg(table, cap_list, idx):
    stack_list = []
    color_list = []
    cap_nuclear = np.array(table['nuclear_fix']).astype(float)   ; stack_list.append(cap_nuclear)   ;   color_list.append('brown')
    cap_wind = np.array(table['wind_fix']).astype(float)         ; stack_list.append(cap_wind)      ;   color_list.append('skyblue')
    cap_storage = np.array(table['storage_fix']).astype(float)   ; stack_list.append(cap_storage)   ;   color_list.append('indigo')
    cap_natgas = np.array(table['natgas_tot']).astype(float)     ; stack_list.append(cap_natgas)    ;   color_list.append('black')
    cap_ccs = np.array(table['natgas_ccs_tot']).astype(float)    ; stack_list.append(cap_ccs)       ;   color_list.append('grey')
    # cap_wind = np.array(table['wind_fix']).astype(float)         ; stack_list.append(cap_wind)      ;   color_list.append('skyblue')
    # cap_nuclear = np.array(table['nuclear_fix']).astype(float)   ; stack_list.append(cap_nuclear)   ;   color_list.append('brown')
    # cap_storage = np.array(table['storage_fix']).astype(float)   ; stack_list.append(cap_storage)   ;   color_list.append('indigo')
    ax1 = plt.subplot(111)
    ax1.stackplot(np.array(cap_list), stack_list, colors=color_list)
    ax1.set_xticks(np.arange(0, 4.4, 0.4))
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 0.12)
    plt.show()
    # plt.savefig(f'inves{idx}.ps')
    plt.clf()


# Cost investment plot
def fig_capchg_cap(table, cap_list, idx):
    cap_nuclear = np.array(table['nuclear_cap']).astype(float) 
    cap_wind = np.array(table['wind_cap']).astype(float)       
    cap_storage = np.array(table['storage_cap']).astype(float)  
    cap_natgas = np.array(table['natgas_cap']).astype(float)   
    cap_ccs = np.array(table['natgas_ccs_cap']).astype(float)  
    ax1 = plt.subplot(111)
    # ax1.plot(np.array(cap_list), cap_nuclear, color='bro
    ax1.plot(np.array(cap_list), cap_storage, color='indigo')
    ax1.set_xticks(np.arange(0, 4.4, 0.4))
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 0.8)
    # plt.show()
    plt.savefig(f'storagecap99p{idx}.ps')
    plt.clf()




# Dispatch plot
def dispatch(table_in, idx):

    natgas_dispatch     = table_in[0]['natgas_dispatch']
    natgas_ccs_dispatch = table_in[0]['natgas_ccs_dispatch']
    wind_potential      = table_in[0]['wind_potential']
    nuclear_potential   = table_in[0]['nuclear_potential']
    storage_dispatch    = table_in[0]['storage_dispatch']
    solar_potential     = table_in[0]['solar_used']
    demand_potential    = table_in[0]['demand_potential']

    # Daily averaged
    stack_list, color_list = [], []
    nuclear_potential_daily   = np.mean(nuclear_potential.reshape(-1,24), axis=1)    ; stack_list.append(nuclear_potential_daily)    ; color_list.append('brown')
    wind_potential_daily      = np.mean(wind_potential.reshape(-1,24), axis=1)       ; stack_list.append(wind_potential_daily)       ; color_list.append('skyblue')
    storage_dispatch_daily    = np.mean(storage_dispatch.reshape(-1,24), axis=1)     ; stack_list.append(storage_dispatch_daily)     ; color_list.append('indigo')
    solar_potential_daily     = np.mean(solar_potential.reshape(-1,24), axis=1)      ; stack_list.append(solar_potential_daily)      ; color_list.append('wheat')
    natgas_dispatch_daily     = np.mean(natgas_dispatch.reshape(-1,24), axis=1)      ; stack_list.append(natgas_dispatch_daily)      ; color_list.append('black')
    natgas_ccs_dispatch_daily = np.mean(natgas_ccs_dispatch.reshape(-1,24), axis=1)  ; stack_list.append(natgas_ccs_dispatch_daily)  ; color_list.append('grey')
    demand_potential_daily    = np.mean(demand_potential.reshape(-1,24), axis=1)
    ax1 = plt.subplot2grid((1,5), (0,0), colspan=3)
    ax1.stackplot(np.arange(365), stack_list, colors=color_list)
    ax1.plot(np.arange(365), demand_potential_daily, color='black', linewidth=1)
    ax1.set_xlim(0, 364)
    plt.xticks([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    ax1.set_ylim(0, 2)
    # Hourly1
    ax2 = plt.subplot2grid((1,5), (0,3), colspan=1)
    ax2.stackplot( np.arange(8760)[15*24:20*24], 
                   nuclear_potential[15*24:20*24], 
                   wind_potential[15*24:20*24], 
                   storage_dispatch[15*24:20*24], 
                   solar_potential[15*24:20*24],
                   natgas_dispatch[15*24:20*24], 
                   natgas_ccs_dispatch[15*24:20*24], 
                   colors = ['brown', 'skyblue', 'indigo', 'wheat', 'black', 'grey'] )
    ax2.plot(np.arange(8760)[15*24:20*24], demand_potential[15*24:20*24], color='black', linewidth=1)
    ax2.set_xlim(15*24, 20*24)
    ax2.set_xticks([15*24, 16*24, 17*24, 18*24, 19*24, 20*24])
    ax2.set_ylim(0, 2)
    ax2.set_yticks([])
    # Hourly2
    ax3 = plt.subplot2grid((1,5), (0,4), colspan=1)
    ax3.stackplot( np.arange(8760)[(181+15)*24:(181+20)*24], 
                   nuclear_potential[(181+15)*24:(181+20)*24], 
                   wind_potential[(181+15)*24:(181+20)*24], 
                   storage_dispatch[(181+15)*24:(181+20)*24], 
                   solar_potential[(181+15)*24:(181+20)*24],
                   natgas_dispatch[(181+15)*24:(181+20)*24], 
                   natgas_ccs_dispatch[(181+15)*24:(181+20)*24], 
                   colors = ['brown', 'skyblue', 'indigo', 'wheat', 'black', 'grey'] )
    ax3.plot(np.arange(8760)[(181+15)*24:(181+20)*24], demand_potential[(181+15)*24:(181+20)*24], color='black', linewidth=1)
    ax3.set_xlim((181+15)*24, (181+20)*24)
    ax3.set_xticks([4704, 4728, 4752, 4776, 4800, 4824])
    ax3.set_ylim(0, 2)
    ax3.set_yticks([])
    # plt.show()
    plt.savefig(f'test{idx}.ps')
    plt.clf()








if __name__ == "__main__":

    # %%
    """
    ### Check solar/wind capacity factors here:
    year = 2019
    MSS1, SS1 = create_table(year, 'US_capacity_solar_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
    MSS2, SS2 = create_table(year, 'US_capacity_wind_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
    # print (0.2/MSS1, 0.2/MSS2)
    # print (2/MSS1, 2/MSS2)
    print (MSS1)
    print (MSS2)

    # for year in [2016, 2017, 2018, 2019]:
    #     MSS1, SS1 = create_table(year, 'US_capacity_solar_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
    #     MSS2, SS2 = create_table(year, 'US_capacity_wind_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
    #     print (2/MSS1, 2/MSS2)
    print ('done')
    stop 
    #"""


    # %%
    """
    ### Value of Solar
    data_find = '/Volumes/My Passport/MEM_AdvNuc/Solar_Power_Night/'
    capacity_specified = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 
                          0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78,
                          0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18,
                          1.20, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34, 1.36, 1.38, 1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52, 1.54, 1.56, 1.58,
                          1.60, 1.62, 1.64, 1.66, 1.68, 1.70, 1.72, 1.74, 1.76, 1.78, 1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98,
                          2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.00, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90, 4.00]
    co2_constraints = np.array([1e24, 50.0, 20.0, 10.0, 1.0, 0.1, 0.0])

    # year = 2016
    # TableS, TableW, TableN, TableOpt = [], [], [], []
    # tech_list = ['natgas', 'natgas_ccs', 'solar', 'wind', 'storage', 'nuclear']
    # for co2_idx in co2_constraints:
    #     print (co2_idx)
    #     fname1 = f'NgCcsSoWiStNu_Year{str(year)}_Co2Con{co2_idx}_SoCap'
    #     TableS += [Get_Table(fname1, data_path = data_find, tech_name_list = tech_list, repeat_list = capacity_specified)] 
    #     fname2 = f'NgCcsSoWiStNu_Year{str(year)}_Co2Con{co2_idx}_WiCap'
    #     TableW += [Get_Table(fname2, data_path = data_find, tech_name_list = tech_list, repeat_list = capacity_specified)] 
    #     fname3 = f'NgCcsSoWiStNu_Year{str(year)}_Co2Con{co2_idx}_NuCap'
    #     TableN += [Get_Table(fname3, data_path = data_find, tech_name_list = tech_list, repeat_list = capacity_specified)] 
    #     fname4 = f'NgCcsSoWiStNu_Year{str(year)}_Co2Con{co2_idx}_Optimized'
    #     TableOpt += [Get_Table(fname4, data_path = data_find, tech_name_list = tech_list)] 
    # with open('save_ValueOfSolar_2016_0710.pickle', 'wb') as handle:
    #     pickle.dump([TableS, TableW, TableN, TableOpt], handle, protocol=pickle.HIGHEST_PROTOCOL) 
    # print ('done')
    # stop

    year = 2019
    with open(f'save_ValueOfSolar_{year}_0710.pickle', 'rb') as handle:
        TableS, TableW, TableN, TableOpt = pickle.load(handle) 
    # ------------------------------------------------------------------------------------------------------------------------
    fig1(TableS, TableOpt, 'solar_cap', 'solar_fix', co2_constraints, np.array(capacity_specified), year, 2)
    # fig1(TableW, TableOpt, 'wind_cap', 'wind_fix', co2_constraints, np.array(capacity_specified), year, 4)
    # fig1(TableN, TableOpt, 'nuclear_cap', 'nuclear_fix', co2_constraints, np.array(capacity_specified), year, 4)
    # ------------------------------------------------------------------------------------------------------------------------
    # fig_marginal_value(TableS, 'solar_fix', co2_constraints, np.array(capacity_specified), year)
    # fig_marginal_value(TableW, 'wind_fix', co2_constraints, np.array(capacity_specified), year)
    # fig_marginal_value(TableN, 'nuclear_fix', co2_constraints, np.array(capacity_specified), year)
    # ------------------------------------------------------------------------------------------------------------------------
    # for idx in range(len(co2_constraints)):
    #     # fig_capchg(TableS[idx], np.array(capacity_specified), idx)
    #     if idx == 4: 
    #         # fig_capchg_cap(TableS[idx], np.array(capacity_specified), '_s')
    #         # fig_capchg_cap(TableW[idx], np.array(capacity_specified), '_w')
    #         # fig_capchg_cap(TableN[idx], np.array(capacity_specified), '_n')
    #         fig_capchg(TableS[idx], np.array(capacity_specified), '_s')
    #         fig_capchg(TableW[idx], np.array(capacity_specified), '_w')
    #         fig_capchg(TableN[idx], np.array(capacity_specified), '_n')

    # ------------------------------------------------------------------------------------------------------------------------
    # with open(f'save_ValueOfSolar_new_0901.pickle', 'rb') as handle:
    #     TableS_new1, TableS_new2 = pickle.load(handle) 
    # MSD, DS = create_table(year, 'US_demand_unnormalized.csv', 1)
    # MSS, SS = create_table(year, 'US_capacity_solar_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
    # def sub_fig_plot(table_used, ax):
    #     color_list = ['#723C70', '#5C4D7D', '#2E6F95', '#34A0A4', '#76C893', '#99D98C', '#D9ED92']
    #     for idx in range(len(co2_constraints)):
    #         system_cost = np.array(table_used[idx]['system_cost']).astype(float)
    #         techno_cost = np.array(table_used[idx]['solar_fix']).astype(float)
    #         differ_cost = system_cost - techno_cost
    #         length_solution = len(differ_cost)
    #         ax.plot(np.array(capacity_specified)[(-1*length_solution):], differ_cost, color=color_list[idx])
    #     ax.plot(np.r_[0, 4], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
    #     ax.set_xticks(np.arange(0, 4.4, 0.4))
    #     ax.set_xlim(0, 4)
    #     ax.set_yticks([0.0, 0.05, 0.1, 0.15, 0.2])
    #     ax.set_ylim(0, 0.2)
    # sub_fig_plot(TableS,      plt.subplot(131))
    # sub_fig_plot(TableS_new1, plt.subplot(132))
    # sub_fig_plot(TableS_new2, plt.subplot(133))
    # # plt.show()
    # plt.savefig('new_tableS.ps')
    # plt.clf()
    # print ('done')


    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------
    # # Test jagginess 
    # data_find = '/Volumes/My Passport/MEM_AdvNuc/test_solar_jag/'
    # capacity_specified = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40]
    # fname1 = f'test_NgCcsSoWiStNu_Year2019_Co2Con0.0_SoCap'
    # tech_list = ['natgas', 'natgas_ccs', 'solar', 'wind', 'storage', 'nuclear']
    # TableS_check = []
    # TableS_check += [Get_Table(fname1, data_path = data_find, tech_name_list = tech_list, repeat_list = capacity_specified)] 

    # table_concerin = TableS[-1]
    # # table_used = TableS[-1]
    # table_used = TableS_check[0]

    # ax1 = plt.plot(111)
    # stack_list = []
    # color_list = []
    # cap_natgas = np.array(table_used['natgas_tot']).astype(float)     ; stack_list.append(cap_natgas[:len(capacity_specified)])    ;   color_list.append('black')
    # cap_ccs = np.array(table_used['natgas_ccs_tot']).astype(float)    ; stack_list.append(cap_ccs[:len(capacity_specified)])       ;   color_list.append('grey')
    # cap_wind = np.array(table_used['wind_fix']).astype(float)         ; stack_list.append(cap_wind[:len(capacity_specified)])      ;   color_list.append('skyblue')
    # cap_nuclear = np.array(table_used['nuclear_fix']).astype(float)   ; stack_list.append(cap_nuclear[:len(capacity_specified)])   ;   color_list.append('brown')
    # cap_storage = np.array(table_used['storage_fix']).astype(float)   ; stack_list.append(cap_storage[:len(capacity_specified)])   ;   color_list.append('indigo')
    # ax1 = plt.subplot(111)
    # ax1.stackplot(np.array(capacity_specified), stack_list, colors=color_list)
    # ax1.set_xticks(np.arange(0, 4.4, 0.4))
    # ax1.set_xlim(0, 0.4)
    # ax1.set_ylim(0, 0.12)
    # plt.show()
    # # plt.savefig(f'inves{idx}.ps')
    # plt.clf()

    stop 
    # """


    # %% 
    # """
    ### Test storage cost and load-shifting
    data_find = '/Volumes/My Passport/MEM_AdvNuc/Temporal_lsst/'
    year = 2019 
    name_prefix = 'zzz_Solar100x_Year2019_'

    scale_factor_St = np.arange(0.2, 1.02, 0.02) 
    scale_factor_Ls = np.arange(0.0, 1.02, 0.02)

    # energy_shifted = np.zeros([len(scale_factor_Ls), len(scale_factor_St)])
    # balancing_cost = np.zeros([len(scale_factor_Ls), len(scale_factor_St)])
    # solar_dispatch_last = np.zeros([len(scale_factor_Ls), len(scale_factor_St)])
    # solar_dispatch_first = np.zeros([len(scale_factor_Ls), len(scale_factor_St)])
    # cost_other = np.zeros([len(scale_factor_Ls), len(scale_factor_St)])
    # storage_cap = np.zeros([len(scale_factor_Ls), len(scale_factor_St)])
    # for idx1 in range(len(scale_factor_Ls)):
    #     for idx2 in range(len(scale_factor_St)):
    #         case_name = f'{name_prefix}LS{str(round(scale_factor_Ls[idx1], 2))}ST{str(round(scale_factor_St[idx2], 2))}'
    #         energy_shifted[idx1, idx2], balancing_cost[idx1, idx2], solar_dispatch_last[idx1, idx2], solar_dispatch_first[idx1, idx2], cost_other[idx1, idx2], storage_cap[idx1, idx2] = for_fun(case_name + '/', data_find)
    # with open('save_LsSt_0719.pickle', 'wb') as handle:
    #     pickle.dump([energy_shifted, balancing_cost, solar_dispatch_last, solar_dispatch_first, cost_other, storage_cap], handle, protocol=pickle.HIGHEST_PROTOCOL) 
    # stop 

    # with open('save_LsSt_0719_100p.pickle', 'rb') as handle:
    # with open('save_LsSt_0903_99p.pickle', 'rb') as handle:
    with open('save_LsSt_1221_99p.pickle', 'rb') as handle:
        energy_shifted, balancing_cost, solar_dispatch_last, solar_dispatch_first, cost_other, storage_cap = pickle.load(handle) 
    print (solar_dispatch_last.min())
    # stop 

    # Contour plot
    aaaa = plt.get_cmap('Oranges', 100)
    ax1 = plt.subplot(111)
    mp = ax1.pcolormesh(scale_factor_St.astype(float), scale_factor_Ls.astype(float), solar_dispatch_last, cmap=aaaa, norm=mcolors.Normalize(vmin=0,vmax=100))
    cp = ax1.contour(scale_factor_St.astype(float), scale_factor_Ls.astype(float), solar_dispatch_last, ['10', '30', '50', '70', '90'], colors='black') # ['20', '40', '60', '80']
    # lp = ax1.clabel(cp, inline=True, fmt=r'%r', fontsize=10)
    plt.colorbar(mp)
    ax1.set_xlim(0.2, 1); plt.xticks([     0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_ylim(0.0, 1); plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.show() 
    # plt.savefig('fig2new.ps')
    plt.clf()
    stop 
    # """




    # %%
    """
    # Solar dispatch 
    data_find = '/Volumes/My Passport/MEM_AdvNuc/Solar_Power_Night/'
    capacity_specified = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 
                          0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78,
                          0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18,
                          1.20, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34, 1.36, 1.38, 1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52, 1.54, 1.56, 1.58,
                          1.60, 1.62, 1.64, 1.66, 1.68, 1.70, 1.72, 1.74, 1.76, 1.78, 1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98,
                          2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.00, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90, 4.00]
    co2_constraints = np.array([1e24, 50.0, 20.0, 10.0, 1.0, 0.1, 0.0])
    year = 2019 

    # energy_shifted       = np.zeros([len(co2_constraints), len(capacity_specified)])
    # balancing_cost       = np.zeros([len(co2_constraints), len(capacity_specified)])
    # solar_dispatch_last  = np.zeros([len(co2_constraints), len(capacity_specified)])
    # solar_dispatch_first = np.zeros([len(co2_constraints), len(capacity_specified)])
    # cost_other           = np.zeros([len(co2_constraints), len(capacity_specified)])
    # storage_cap          = np.zeros([len(co2_constraints), len(capacity_specified)])
    # for idx1 in range(len(co2_constraints)):
    #     for idx2 in range(len(capacity_specified)):
    #         case_name = f'NgCcsSoWiStNu_Year{str(year)}_Co2Con{co2_constraints[idx1]}_SoCap{capacity_specified[idx2]}'
    #         energy_shifted[idx1, idx2], balancing_cost[idx1, idx2], solar_dispatch_last[idx1, idx2], solar_dispatch_first[idx1, idx2], cost_other[idx1, idx2], storage_cap[idx1, idx2] = for_fun(case_name + '/', data_find)
    # with open('save_sp_0719.pickle', 'wb') as handle:
    #     pickle.dump([energy_shifted, balancing_cost, solar_dispatch_last, solar_dispatch_first, cost_other, storage_cap], handle, protocol=pickle.HIGHEST_PROTOCOL) 


    # with open('save_sp_0719.pickle', 'rb') as handle:
    #     energy_shifted, balancing_cost, solar_dispatch_last, solar_dispatch_first, cost_other, storage_cap = pickle.load(handle) 

    with open(f'save_ValueOfSolar_2019_1127_Sensitivity7.pickle', 'rb') as handle:
        energy_shifted, balancing_cost, solar_dispatch_last, solar_dispatch_first, cost_other, storage_cap = pickle.load(handle) 

    print (solar_dispatch_last[4][100])
    print (solar_dispatch_last[6][100])
    color_list = ['#723C70', '#5C4D7D', '#2E6F95', '#34A0A4', '#76C893', '#99D98C', '#D9ED92']
    ax1 = plt.subplot(121)
    for idx in range(len(co2_constraints)):
        ax1.plot(np.array(capacity_specified), solar_dispatch_last[idx], color=color_list[idx])
    ax1.plot(np.r_[0, 4], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
    ax1.set_xticks(np.arange(0, 4.4, 0.4))
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 60)
    plt.show()
    # plt.savefig('figx.ps')
    plt.clf()
    stop 
    # """



    # %%
    """
    # Dispatch plot
    # Check dispatch under 0% and 100% level, with solar at 2 hours of annual mean demand
    data_find = '/Volumes/My Passport/MEM_AdvNuc/Solar_Power_Night/' 
    capacity_specified = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 
                          0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78,
                          0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18,
                          1.20, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34, 1.36, 1.38, 1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52, 1.54, 1.56, 1.58,
                          1.60, 1.62, 1.64, 1.66, 1.68, 1.70, 1.72, 1.74, 1.76, 1.78, 1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98,
                          2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.00, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90, 4.00]
    co2_constraints = np.array([1e24, 50.0, 20.0, 10.0, 1.0, 0.1, 0.0])
    year = 2019 
    case_name1 = f'NgCcsSoWiStNu_Year2019_Co2Con{co2_constraints[0]}_SoCap2.0/'
    case_name2 = f'NgCcsSoWiStNu_Year2019_Co2Con{co2_constraints[4]}_SoCap2.0/'
    dispatch_table1 = get_case_dispatch(case_name1, data_find)
    dispatch_table2 = get_case_dispatch(case_name2, data_find)
    dispatch(dispatch_table1, '1')
    dispatch(dispatch_table2, '2')
    case_name3 = f'NgCcsSoWiStNu_Year2019_Co2Con{co2_constraints[0]}_SoCap0.2/'
    case_name4 = f'NgCcsSoWiStNu_Year2019_Co2Con{co2_constraints[4]}_SoCap0.2/'
    dispatch_table3 = get_case_dispatch(case_name3, data_find)
    dispatch_table4 = get_case_dispatch(case_name4, data_find)
    dispatch(dispatch_table3, '3')
    dispatch(dispatch_table4, '4')
    case_name5 = f'NgCcsSoWiStNu_Year2019_Co2Con{co2_constraints[0]}_SoCap0.0/'
    case_name6 = f'NgCcsSoWiStNu_Year2019_Co2Con{co2_constraints[4]}_SoCap0.0/'
    dispatch_table5 = get_case_dispatch(case_name5, data_find)
    dispatch_table6 = get_case_dispatch(case_name6, data_find)
    dispatch(dispatch_table5, '5')
    dispatch(dispatch_table6, '6')

    stop 
    # """


    # %%
    """
    # Bar 1 (case a):  No emission constraints, no free solar:  Natural gas dominated system (show fixed and variable costs separately)
    # Bar 2 (case e):  No emission constraints, abundant free solar: Natural gas dominated at night; small reduction in nat gas capacity cost, reduction in nat gas variable cost
    # Bar 3 (case f):  99% emission constraint, abundant free solar: Now you need to replace the natural gas with a wind and battery system that will be more expensive than the natural gas.
    # The idea is that Bar 2 shows that the solar saves you mostly just the daytime variable cost of the natural gas, which is about 25% of the total natural gas cost.

    capacity_specified = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 
                          0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78,
                          0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18,
                          1.20, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34, 1.36, 1.38, 1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52, 1.54, 1.56, 1.58,
                          1.60, 1.62, 1.64, 1.66, 1.68, 1.70, 1.72, 1.74, 1.76, 1.78, 1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98,
                          2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.00, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90, 4.00]
    co2_constraints = np.array([1e24, 50.0, 20.0, 10.0, 1.0, 0.1, 0.0])

    year = 2019
    with open(f'save_ValueOfSolar_{year}_0710.pickle', 'rb') as handle:
        TableS, TableW, TableN, TableOpt = pickle.load(handle) 

    def separate_cost(table_in, idx):
        table_out = {}
        natgas_fix = 0.010163177
        natgas_ccs_fix = 0.026770145
        # System cost
        table_out['system_cost'] = table_in['system_cost'][idx]
        # Only fix cost
        table_out['solar_fix'] = table_in['solar_fix'][idx]
        table_out['wind_fix'] = table_in['wind_fix'][idx]
        table_out['nuclear_fix'] = table_in['nuclear_fix'][idx]
        table_out['storage_fix'] = table_in['storage_fix'][idx]
        # Fixed and variable cost
        table_out['natgas_tot'] = table_in['natgas_tot'][idx]
        table_out['natgas_fix'] = table_in['natgas_cap'][idx] * natgas_fix
        table_out['natgas_var'] = table_out['natgas_tot'] - table_out['natgas_fix']
        table_out['natgas_ccs_tot'] = table_in['natgas_ccs_tot'][idx]
        table_out['natgas_ccs_fix'] = table_in['natgas_ccs_cap'][idx] * natgas_ccs_fix
        table_out['natgas_ccs_var'] = table_out['natgas_ccs_tot'] - table_out['natgas_ccs_fix']
        return table_out

    table_1e24_00x_out = separate_cost(TableS[0], 0)
    table_1e24_02x_out = separate_cost(TableS[0], 10)
    table_1e24_20x_out = separate_cost(TableS[0], 100)
    table_0_00x_out = separate_cost(TableS[-3], 0)
    table_0_02x_out = separate_cost(TableS[-3], 10)
    table_0_20x_out = separate_cost(TableS[-3], 100)
    print ('start')
    import matplotlib.pyplot as plt
    def make_plot(table_in, x, ax):
        bot = 0.0
        ax.bar(x, np.array(table_in['natgas_fix']), bottom=bot, color='black'); bot = bot + table_in['natgas_fix']
        ax.bar(x, np.array(table_in['natgas_var']), bottom=bot, color='orange'); bot = bot + table_in['natgas_var']
        ax.bar(x, np.array(table_in['natgas_ccs_fix']), bottom=bot, color='grey'); bot = bot + table_in['natgas_ccs_fix']
        ax.bar(x, np.array(table_in['natgas_ccs_var']), bottom=bot, color='green'); bot = bot + table_in['natgas_ccs_var']
        ax.bar(x, np.array(table_in['wind_fix']), bottom=bot, color='skyblue'); bot = bot + table_in['wind_fix']
        ax.bar(x, np.array(table_in['nuclear_fix']), bottom=bot, color='brown'); bot = bot + table_in['nuclear_fix']
        ax.bar(x, np.array(table_in['storage_fix']), bottom=bot, color='indigo'); bot = bot + table_in['storage_fix']
    
    ax1 = plt.subplot(211)
    make_plot(table_1e24_00x_out, 1, ax1)
    make_plot(table_1e24_02x_out, 2, ax1)
    make_plot(table_1e24_20x_out, 3, ax1)
    ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
    make_plot(table_0_00x_out, 1, ax2)
    make_plot(table_0_02x_out, 2, ax2)
    make_plot(table_0_20x_out, 3, ax2)
    ax1.set_xlim(0, 4)
    ax1.set_xticks([1, 2, 3])
    ax1.set_ylim(0, 0.1)
    # plt.savefig('fig111.ps')
    plt.show()
    plt.clf()
    # """

    # %%
