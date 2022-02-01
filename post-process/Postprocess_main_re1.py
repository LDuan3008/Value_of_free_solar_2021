
import numpy as np
import pickle, os
from Postprocess_func import Get_Table
from Postprocess_func import get_case_dispatch
from Postprocess_func import for_fun
from scipy import stats
from Shared_fun import update_series
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import signal, stats

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



def fig_capchg(table, cap_list, idx):
    stack_list = []
    color_list = []
    cap_nuclear = np.array(table['nuclear_fix']).astype(float)   ; stack_list.append(cap_nuclear)   ;   color_list.append('brown')
    cap_wind = np.array(table['wind_fix']).astype(float)         ; stack_list.append(cap_wind)      ;   color_list.append('skyblue')
    cap_storage = np.array(table['storage_fix']).astype(float)   ; stack_list.append(cap_storage)   ;   color_list.append('indigo')
    cap_natgas = np.array(table['natgas_tot']).astype(float)     ; stack_list.append(cap_natgas)    ;   color_list.append('black')
    cap_ccs = np.array(table['natgas_ccs_tot']).astype(float)    ; stack_list.append(cap_ccs)       ;   color_list.append('grey')
    cap_pgp = np.array(table['to_PGP_fix']).astype(float) + np.array(table['PGP_storage_fix']).astype(float) + np.array(table['from_PGP_fix']).astype(float); stack_list.append(cap_pgp); color_list.append('green')
    ax1 = plt.subplot(111)
    ax1.stackplot(np.array(cap_list), stack_list, colors=color_list)
    ax1.set_xticks(np.arange(0, 4.4, 0.4))
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 0.12)
    plt.show()
    # plt.savefig(f'inves{idx}.ps')
    plt.clf()


def highpass_filter(var, approach, highpass):
    fs = 1
    if approach == 'butter':
        npt = 10
        cutoff_period = highpass
        cutoff_freque = (1/cutoff_period) / (fs/2)
        z, p, k = signal.butter(npt, cutoff_freque, 'highpass', output="zpk")
        smoothed0 = signal.zpk2sos(z, p, k)
        output_highpass = signal.sosfiltfilt(smoothed0, var)
    if approach == 'fft':
        length = len(var)
        cutoff_period = highpass
        cutoff_freque = (1/cutoff_period)
        input_fft = np.fft.fft(var)
        input_fre = np.fft.fftfreq(length)
        input_fft[np.abs(input_fre)<=cutoff_freque] = 0.
        output_highpass = np.fft.ifft(input_fft).real      
    return output_highpass

def lowpass_filter(var, approach, lowpass):
    fs = 1
    if approach == 'butter':
        npt = 10
        cutoff_period = lowpass
        cutoff_freque = (1/cutoff_period) / (fs/2)
        z, p, k = signal.butter(npt, cutoff_freque, 'lowpass', output="zpk")
        smoothed0 = signal.zpk2sos(z, p, k)
        output_lowpass = signal.sosfiltfilt(smoothed0, var)
    if approach == 'fft':
        length = len(var)
        cutoff_period = lowpass
        cutoff_freque = (1/cutoff_period)
        input_fft = np.fft.fft(var)
        input_fre = np.fft.fftfreq(length)
        input_fft[np.abs(input_fre)>=cutoff_freque] = 0.
        output_lowpass = np.fft.ifft(input_fft).real      
    return output_lowpass





if __name__ == "__main__":
    


    w_input_name = 'US_capacity_wind_25pctTop_unnormalilzed_19800101_20200930.csv'
    w1, ws1 = create_table(2019, w_input_name, -1)
    print (w1)
    print (ws1[:120])

    stop 






    # %%
    """
    ### Do calculations here
    s_input_name = 'US_capacity_solar_25pctTop_unnormalilzed_19800101_20200930.csv'
    w_input_name = 'US_capacity_wind_25pctTop_unnormalilzed_19800101_20200930.csv'
    d_input_name = 'US_demand_unnormalized.csv'

    year = 2019
    filter_type = 'butter'
    frequency = 120
    s1, ss1 = create_table(year, s_input_name, -1)
    w1, ws1 = create_table(year, w_input_name, -1)
    d1, ds1 = create_table(year, d_input_name, 1)
    ss1_highpass = highpass_filter(ss1, filter_type, frequency)
    ws1_highpass = highpass_filter(ws1, filter_type, frequency)
    ds1_highpass = highpass_filter(ds1, filter_type, frequency)
    ss1_lowpass = lowpass_filter(ss1, filter_type, frequency)
    ws1_lowpass = lowpass_filter(ws1, filter_type, frequency)
    ds1_lowpass = lowpass_filter(ds1, filter_type, frequency)
    print (stats.pearsonr(ss1, ds1)[0])
    print (stats.pearsonr(ss1_highpass, ds1_highpass)[0])
    print (stats.pearsonr(ss1_lowpass, ds1_lowpass)[0])
    print ()
    print ()
    # print (stats.pearsonr(ws1, ds1)[0])
    # print (stats.pearsonr(ws1_highpass, ds1_highpass)[0])
    # print (stats.pearsonr(ws1_lowpass, ds1_lowpass)[0])
    # """





    """
    ### Compare annual mean capacity factors
    s_input_name = 'US_capacity_solar_25pctTop_unnormalilzed_19800101_20200930.csv'
    w_input_name = 'US_capacity_wind_25pctTop_unnormalilzed_19800101_20200930.csv'

    # list_s, list_w = [], []
    # for year in np.arange(1980, 2020, 1):
    #     print (year)
    #     s1, tmp = create_table(year, 'US_capacity_solar_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
    #     w1, tmp = create_table(year, 'US_capacity_wind_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
    #     list_s.append(s1)
    #     list_w.append(w1)
    # print (list_s)
    # print (list_w)

    list_s = [0.28074680944548064, 0.27426163462840636, 0.2674324453033425, 0.2675381278776369, 0.27317141768466896, 0.27952307124960046, 
              0.2738441444439384, 0.2750744073887329, 0.2809903994310856, 0.2864107800298973, 0.2795040735528996, 0.2744742778930251, 
              0.2700504508008916, 0.27875294501364156, 0.2786320544501838, 0.27926861374695205, 0.28320025063236304, 0.27805411672727165, 
              0.2797507295114498, 0.2849600484142923, 0.2807979224701359, 0.2787003001391815, 0.2841065710250342, 0.2793688648348516, 
              0.2776154238850057, 0.2774090900059931, 0.2787318480840982, 0.2807770307596986, 0.2860458384752169, 0.27911880530060507, 
              0.2797767009233596, 0.28539314758469175, 0.2822160756566667, 0.28050717050500007, 0.2758929186784189, 0.2684607247093835, 
              0.2772249208382865, 0.2767645890773573, 0.2755078354169669, 0.27245239700984014]
    list_w = [0.43468140915662107, 0.4262623533130137, 0.42486625622180363, 0.41274052012579904, 0.4351435693265982, 0.43579005693333334, 
              0.42189097877111875, 0.4108799488321918, 0.45855043039178084, 0.42630644078481733, 0.45512630369235163, 0.4370021843125571, 
              0.4020245736247717, 0.41490300934235164, 0.4183795412320777, 0.41930761805251143, 0.43477227317579914, 0.4198923383635844, 
              0.4015081787787671, 0.45524627235068493, 0.4338548068665525, 0.425953671498516, 0.4416274125292237, 0.4262108015803653, 
              0.4245752429699772, 0.42868870993938357, 0.4475560837041096, 0.43624402916849314, 0.4526572244833334, 0.422203588161758, 
              0.41869472037226024, 0.44723303377751145, 0.44457393319794514, 0.42746753454885844, 0.4475853392487443, 0.42097776279897264, 
              0.4408858294555936, 0.43450643240867587, 0.40790864109908676, 0.41862413468961185]
    ax1 = plt.subplot(211)
    ax1.bar(np.arange(len(list_s)), np.array(list_s), color='red')
    ax1.set_xlim(-1, 40)
    ax1.set_xticks([])
    ax1.set_ylim(0, 0.3)
    ax2 = plt.subplot(212)
    ax2.bar(np.arange(len(list_w)), np.array(list_w), color='blue')
    ax2.set_xlim(-1, 40)
    ax2.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax2.set_xticklabels([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015])
    ax2.set_ylim(0, 0.5)
    # plt.show()
    plt.savefig('mean_cfs.ps')
    plt.clf()

    # """





    """
    ### Senstivity test
    data_find = '/Volumes/WorkingDisk/MEM_AdvNuc/Solar_Power_Night_2/'
    capacity_specified = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 
                          0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78,
                          0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18,
                          1.20, 1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34, 1.36, 1.38, 1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52, 1.54, 1.56, 1.58,
                          1.60, 1.62, 1.64, 1.66, 1.68, 1.70, 1.72, 1.74, 1.76, 1.78, 1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98,
                          2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.00, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90, 4.00]
    co2_constraints = np.array([1e24, 50.0, 20.0, 10.0, 1.0, 0.1, 0.0])

    def plot_subplot(table, table_op, cap_op, tech_cost, co2_cons, cap_list, year, MSS, fix_cost, name): 
        color_list = ['#723C70', '#5C4D7D', '#2E6F95', '#34A0A4', '#76C893', '#99D98C', '#D9ED92']
        ax1 = plt.subplot(111)
        for idx in range(len(co2_cons)):
            system_cost = np.array(table[idx]['system_cost']).astype(float)
            techno_cost = np.array(table[idx][tech_cost]).astype(float)
            differ_cost = system_cost - techno_cost
            optmized_solar_cap = table_op[idx][cap_op][0] * MSS
            optmized_diff_cost = table_op[idx]['system_cost'][0] - table_op[idx][tech_cost][0]
            ax1.plot(np.array(cap_list), differ_cost, color=color_list[idx])
            ax1.scatter(optmized_solar_cap, optmized_diff_cost, marker='*', s=25, color='firebrick')
        ax1.plot(np.r_[0, 2], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
        ax1.set_xticks(np.arange(0, 4.4, 0.4))
        ax1.set_xlim(0, 2)
        ax1.set_ylim(0, 0.12)
        ax1.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12])
        # plt.show() 
        plt.savefig(f'solar_{name}_{year}_1.ps')
        plt.clf()
        ax2 = plt.subplot(111)
        for idx in range(len(co2_cons)):
            system_cost = np.array(table[idx]['system_cost']).astype(float)
            techno_cost = np.array(table[idx][tech_cost]).astype(float)
            differ_cost = system_cost - techno_cost
            marginal_cost = (differ_cost[:-1] - differ_cost[1:]) / (cap_list[1:]-cap_list[:-1]) 
            ax2.plot(np.array(cap_list[:-1]), marginal_cost, color=color_list[idx])
            ax2.plot(np.array(cap_list[:-1]), np.ones(len(cap_list[:-1])) * fix_cost / MSS, color='black', linestyle='--')
        ax2.plot(np.r_[0, 2], np.r_[0, 0], color='black', linewidth=0.5, linestyle='--')
        ax2.set_xticks(np.arange(0, 4.4, 0.4))
        ax2.set_xlim(0, 2)
        ax2.set_ylim(0, 0.2)
        ax2.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20])
        # plt.show() 
        plt.savefig(f'solar_{name}_{year}_2.ps')
        plt.clf()

    
    year = 2019
    with open(f'save_ValueOfSolar_{str(year)}_1127_Sensitivity1.pickle', 'rb') as handle:
        TableS_ERA5, TableS_Mtd1, TableS_Future, TableS_PGP = pickle.load(handle) 
    with open(f'save_ValueOfSolar_{str(year)}_1127_Sensitivity2.pickle', 'rb') as handle:
        TableOpt_ERA5, TableOpt_Mtd1, TableOpt_Future, TableOpt_PGP = pickle.load(handle) 
    with open(f'save_ValueOfSolar_2019_1127_Sensitivity5.pickle', 'rb') as handle:
        Table_ERA5_mtd3, Table_NgCcsSoNu, Table_NgCcsSoStNu = pickle.load(handle) 
    with open(f'save_ValueOfSolar_2019_1127_Sensitivity6.pickle', 'rb') as handle:
        TableOpt_ERA5_mtd3, TableOpt_NgCcsSoNu, TableOpt_NgCcsSoStNu = pickle.load(handle) 
    
    MSS_MERRA2_25tp, SS = create_table(year, 'US_capacity_solar_25pctTop_unnormalilzed_19800101_20200930.csv', -1)
    MSS_MERRA2_mtd1, SS = create_table(year, 'US_capacity_solar_CONUS_unnormalized_19800101_20200930.csv',     -1)
    MSS_ERA5_25tp,   SS = create_table(year, 'ERA5_20210921_US_mthd1_1950-2020_solar.csv',                     -1)
    MSS_ERA5_25tp_t, SS = create_table(year, '20210921_US_mthd3_1950-2020_solar.csv',                          -1)
    fix_cost_now = 0.014772123
    fix_cost_future = 0.007824859

    # print (  len(TableS_PGP)  )
    # print (TableS_PGP[4].keys())
    # print (TableS_PGP[4]['to_PGP_fix'])
    # print (TableS_PGP[4]['to_PGP_cap'])∏

    # for idx in range(len(TableS_PGP)):
    #     fig_capchg(TableS_PGP[idx], capacity_specified, idx)

    # plot_subplot(TableS_ERA5,   TableOpt_ERA5,   'solar_cap', 'solar_fix', co2_constraints, np.array(capacity_specified), year, MSS_ERA5_25tp,   fix_cost_now,     'ERA5')
    # plot_subplot(TableS_Mtd1,   TableOpt_Mtd1,   'solar_cap', 'solar_fix', co2_constraints, np.array(capacity_specified), year, MSS_MERRA2_mtd1, fix_cost_now,     'Mtd1')
    # plot_subplot(TableS_Future, TableOpt_Future, 'solar_cap', 'solar_fix', co2_constraints, np.array(capacity_specified), year, MSS_MERRA2_25tp,   fix_cost_future,  'Future')
    # plot_subplot(TableS_PGP,    TableOpt_PGP,    'solar_cap', 'solar_fix', co2_constraints, np.array(capacity_specified), year, MSS_MERRA2_25tp,   fix_cost_now,     'PGP')

    # plot_subplot(Table_ERA5_mtd3,   TableOpt_ERA5_mtd3,   'solar_cap', 'solar_fix', co2_constraints, np.array(capacity_specified), year, MSS_ERA5_25tp_t, fix_cost_now, 'ERA5_t')
    # plot_subplot(Table_NgCcsSoNu,   TableOpt_NgCcsSoNu,   'solar_cap', 'solar_fix', co2_constraints, np.array(capacity_specified), year, MSS_MERRA2_25tp, fix_cost_now, 'NgCcsSoNu')
    # plot_subplot(Table_NgCcsSoStNu, TableOpt_NgCcsSoStNu, 'solar_cap', 'solar_fix', co2_constraints, np.array(capacity_specified), year, MSS_MERRA2_25tp, fix_cost_now, 'NgCcsSoStNu')
    # """



    """
    with open(f'save_ValueOfSolar_1127_Sensitivity3.pickle', 'rb') as handle:
        Table_1000, Table_1020, Table_1e2400, Table_1e2420 = pickle.load(handle) 
    cost_1000, cost_1020, cost_1e2400, cost_1e2420 = [], [], [], []
    for i in range(40):
        cost_1000.append(  float(np.array(Table_1000[i]['system_cost']).astype(float)   - np.array(Table_1000[i]['solar_fix']).astype(float))  )
        cost_1020.append(  float(np.array(Table_1020[i]['system_cost']).astype(float)   - np.array(Table_1020[i]['solar_fix']).astype(float))  )
        cost_1e2400.append(  float(np.array(Table_1e2400[i]['system_cost']).astype(float) - np.array(Table_1e2400[i]['solar_fix']).astype(float))  )
        cost_1e2420.append(  float(np.array(Table_1e2420[i]['system_cost']).astype(float) - np.array(Table_1e2420[i]['solar_fix']).astype(float))  )
    plt.scatter(np.zeros(40)+4, cost_1000, s=5, c='firebrick')
    plt.scatter(np.zeros(40)+3, cost_1020, s=5, c='firebrick')
    plt.scatter(np.zeros(40)+2, cost_1e2400, s=5, c='firebrick')
    plt.scatter(np.zeros(40)+1, cost_1e2420, s=5, c='firebrick')
    plt.xlim(0, 5)
    plt.ylim(0, 0.1)
    plt.show()
    # plt.savefig('testtest.ps')
    plt.clf()
    # """



    """
    scale_factor_St = [0.1,        0.1023293,  0.10471285, 0.10715193, 0.10964782, 0.11220185,
                   0.11481536, 0.11748976, 0.12022644, 0.12302688, 0.12589254, 0.12882496,
                   0.13182567, 0.13489629, 0.13803843, 0.14125375, 0.14454398, 0.14791084,
                   0.15135612, 0.15488166, 0.15848932, 0.16218101, 0.16595869, 0.16982437,
                   0.17378008, 0.17782794, 0.18197009, 0.18620871, 0.19054607, 0.19498446,
                   0.19952623, 0.20417379, 0.20892961, 0.21379621, 0.21877616, 0.22387211,
                   0.22908677, 0.23442288, 0.23988329, 0.24547089, 0.25118864, 0.25703958,
                   0.2630268 , 0.26915348, 0.27542287, 0.28183829, 0.28840315, 0.29512092,
                   0.30199517, 0.30902954, 0.31622777, 0.32359366, 0.33113112, 0.33884416,
                   0.34673685, 0.35481339, 0.36307805, 0.37153523, 0.3801894 , 0.38904514,
                   0.39810717, 0.40738028, 0.41686938, 0.42657952, 0.43651583, 0.44668359,
                   0.45708819, 0.46773514, 0.47863009, 0.48977882, 0.50118723, 0.51286138,
                   0.52480746, 0.5370318 , 0.54954087, 0.56234133, 0.57543994, 0.58884366,
                   0.60255959, 0.616595  , 0.63095734, 0.64565423, 0.66069345, 0.67608298,
                   0.69183097, 0.70794578, 0.72443596, 0.74131024, 0.75857758, 0.77624712,
                   0.79432823, 0.81283052, 0.83176377, 0.85113804, 0.87096359, 0.89125094,
                   0.91201084, 0.9332543 , 0.95499259, 0.97723722, 1.        ]
    log_scale_xaxisSt = np.log10(scale_factor_St)
    scale_factor_Wi = [ 1.        ,  1.02329299,  1.04712855,  1.07151931,  1.0964782 ,  1.12201845,
                    1.14815362,  1.17489755,  1.20226443,  1.23026877,  1.25892541,  1.28824955,
                    1.31825674,  1.34896288,  1.38038426,  1.41253754,  1.44543977,  1.47910839,
                    1.51356125,  1.54881662,  1.58489319,  1.6218101 ,  1.65958691,  1.69824365,
                    1.73780083,  1.77827941,  1.81970086,  1.86208714,  1.90546072,  1.9498446 ,
                    1.99526231,  2.04173794,  2.08929613,  2.13796209,  2.18776162,  2.23872114,
                    2.29086765,  2.34422882,  2.39883292,  2.45470892,  2.51188643,  2.57039578,
                    2.63026799,  2.6915348 ,  2.7542287 ,  2.81838293,  2.8840315 ,  2.95120923,
                    3.01995172,  3.09029543,  3.16227766,  3.23593657,  3.31131121,  3.38844156,
                    3.4673685 ,  3.54813389,  3.63078055,  3.71535229,  3.80189396,  3.89045145,
                    3.98107171,  4.07380278,  4.16869383,  4.26579519,  4.36515832,  4.46683592,
                    4.5708819 ,  4.67735141,  4.78630092,  4.89778819,  5.01187234,  5.12861384,
                    5.2480746 ,  5.37031796,  5.49540874,  5.62341325,  5.75439937,  5.88843655,
                    6.02559586,  6.16595002,  6.30957344,  6.45654229,  6.60693448,  6.76082975,
                    6.91830971,  7.07945784,  7.2443596 ,  7.41310241,  7.58577575,  7.76247117,
                    7.94328235,  8.12830516,  8.31763771,  8.51138038,  8.7096359 ,  8.91250938,
                    9.12010839,  9.33254301,  9.54992586,  9.77237221,  10. ]
    log_scale_xaxisWi = np.log10(scale_factor_Wi)

    with open(f'save_ValueOfSolar_1127_Sensitivity4.pickle', 'rb') as handle:
        Table_StCost, Table_WiCost = pickle.load(handle)  # 0 with nuclaer, 1 without nuclear


    system_cost11 = np.array(Table_StCost[0]['system_cost']).astype(float) - np.array(Table_StCost[0]['solar_fix']).astype(float) 
    system_cost12 = np.array(Table_StCost[1]['system_cost']).astype(float) - np.array(Table_StCost[1]['solar_fix']).astype(float) 
    system_cost21 = np.array(Table_WiCost[0]['system_cost']).astype(float) - np.array(Table_WiCost[0]['solar_fix']).astype(float)
    system_cost22 = np.array(Table_WiCost[1]['system_cost']).astype(float) - np.array(Table_WiCost[1]['solar_fix']).astype(float) 
    wc11 = np.array(Table_StCost[0]['wind_cap']).astype(float)
    wc12 = np.array(Table_StCost[1]['wind_cap']).astype(float)
    wc21 = np.array(Table_WiCost[0]['wind_cap']).astype(float)
    wc22 = np.array(Table_WiCost[1]['wind_cap']).astype(float)

    ### 
    plt.plot(scale_factor_St, wc11, color='black', linestyle='solid')
    plt.plot(scale_factor_St, wc12, color='black', linestyle='dashed')
    plt.xlim(0.3, 1.0)
    plt.ylim(0, 2.5)
    # plt.savefig('test_StCost.ps')
    plt.show()
    plt.clf()

    ### Linear interpolation starts:
    wind_cap_interval = np.arange(0.2, 2.32, 0.02)
    from_storage  = np.array( np.interp(wind_cap_interval, wc11, scale_factor_St) )
    from_wind_wNu = np.array( np.interp(wind_cap_interval, wc21[::-1], scale_factor_Wi[::-1]) )
    from_wind_nNu = np.array( np.interp(wind_cap_interval, wc22[::-1], scale_factor_Wi[::-1]) )
    plt.plot(from_storage, from_wind_wNu, color='black')
    plt.plot(from_storage, from_wind_nNu, color='brown')
    plt.xlim(0.3, 1.0)
    plt.ylim(1.0, 3.0)
    # plt.show()
    plt.savefig('valuestwi.ps')
    plt.clf()
    # """




# %%
