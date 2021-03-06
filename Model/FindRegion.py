"""
Lei updated on Feb 8, 2020;
Used to find the corresponding demand/solar/wind input data given year and region name; 
Also re-calculate the total days of that year.
"""


from Preprocess_Input import read_csv_dated_data_file
import numpy as np
import datetime

from scipy import signal
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

def GetNameList():
    TwoLettersCode = ['DZ','AR','AU','BR','CA','CL','CN',
                      'CO','EG','FR','DE','GH','IN','ID',
                      'IR','IT','JP','LY','MY','MX','MA',
                      'MZ','NZ','NG','PY','PE','PL','RU',
                      'SA','ZA','KR','ES','SD','SE','TH',
                      'TN','TR','UA','GB','US','VE','VN']
    return TwoLettersCode


def GetCFsName(region_name):
    FullDemandList = {'DZ': 'demand_series_Dan_normalized_to_1_mean_Algeria.csv', 'AR': 'demand_series_Dan_normalized_to_1_mean_Argentina.csv',
                      'AU': 'demand_series_Dan_normalized_to_1_mean_Australia.csv', 'BR': 'demand_series_Dan_normalized_to_1_mean_Brazil.csv',
                      'CA': 'demand_series_Dan_normalized_to_1_mean_Canada.csv', 'CL': 'demand_series_Dan_normalized_to_1_mean_Chile.csv',
                      'CN': 'demand_series_Dan_normalized_to_1_mean_China.csv', 'CO': 'demand_series_Dan_normalized_to_1_mean_Colombia.csv',
                      'EG': 'demand_series_Dan_normalized_to_1_mean_Egypt.csv', 'FR': 'demand_series_Dan_normalized_to_1_mean_France.csv',
                      'DE': 'demand_series_Dan_normalized_to_1_mean_Germany.csv', 'GH': 'demand_series_Dan_normalized_to_1_mean_Ghana.csv',
                      'IN': 'demand_series_Dan_normalized_to_1_mean_India.csv', 'ID': 'demand_series_Dan_normalized_to_1_mean_Indonesia.csv',
                      'IR': 'demand_series_Dan_normalized_to_1_mean_Iran.csv', 'IT': 'demand_series_Dan_normalized_to_1_mean_Italy.csv',
                      'JP': 'demand_series_Dan_normalized_to_1_mean_Japan.csv', 'LY': 'demand_series_Dan_normalized_to_1_mean_Libya.csv',
                      'MY': 'demand_series_Dan_normalized_to_1_mean_Malaysia.csv', 'MX': 'demand_series_Dan_normalized_to_1_mean_Mexico.csv',
                      'MA': 'demand_series_Dan_normalized_to_1_mean_Morocco.csv', 'MZ': 'demand_series_Dan_normalized_to_1_mean_Mozambique.csv',
                      'NZ': 'demand_series_Dan_normalized_to_1_mean_New Zealand.csv', 'NG': 'demand_series_Dan_normalized_to_1_mean_Nigeria.csv',
                      'PY': 'demand_series_Dan_normalized_to_1_mean_Paraguay.csv', 'PE': 'demand_series_Dan_normalized_to_1_mean_Peru.csv',
                      'PL': 'demand_series_Dan_normalized_to_1_mean_Poland.csv', 'RU': 'demand_series_Dan_normalized_to_1_mean_Russia.csv',
                      'SA': 'demand_series_Dan_normalized_to_1_mean_Saudi Arabia.csv', 'ZA': 'demand_series_Dan_normalized_to_1_mean_South Africa.csv',
                      'KR': 'demand_series_Dan_normalized_to_1_mean_South Korea.csv', 'ES': 'demand_series_Dan_normalized_to_1_mean_Spain.csv',
                      'SD': 'demand_series_Dan_normalized_to_1_mean_Sudan.csv', 'SE': 'demand_series_Dan_normalized_to_1_mean_Sweden.csv',
                      'TH': 'demand_series_Dan_normalized_to_1_mean_Thailand.csv', 'TN': 'demand_series_Dan_normalized_to_1_mean_Tunisia.csv',
                      'TR': 'demand_series_Dan_normalized_to_1_mean_Turkey.csv', 'UA': 'demand_series_Dan_normalized_to_1_mean_Ukraine.csv',
                      'GB': 'demand_series_Dan_normalized_to_1_mean_United Kingdom.csv', 'US': 'demand_series_Dan_normalized_to_1_mean_United States.csv',
                      'VE': 'demand_series_Dan_normalized_to_1_mean_Venezuela.csv', 'VN': 'demand_series_Dan_normalized_to_1_mean_Vietnam.csv'}
    DemandName = FullDemandList[region_name]
    SolarCFsName = '20201218_' + str(region_name) + '_mthd3_solar.csv'
    WindCFsName = '20201218_' + str(region_name) + '_mthd3_wind.csv'
    return DemandName, SolarCFsName, WindCFsName


def update_series(case_dic, tech_dic, delta_t=-1):
    series = read_csv_dated_data_file(case_dic['year_start'],
                                      case_dic['month_start'],
                                      case_dic['day_start'],
                                      case_dic['hour_start'],
                                      case_dic['year_end'],
                                      case_dic['month_end'],
                                      case_dic['day_end'],
                                      case_dic['hour_end'],
                                      case_dic['data_path'],
                                      tech_dic['series_file'])
    # Remove Feb 29 if exists
    if len(series) != 8760:
        series = np.r_[series[:1416], series[1440:]]
        if len(series) != 8760:
            print ('error removing Feb 29')
            sys.exit()
    
    if delta_t != -1:
        if delta_t != -2:
            mean_series = np.average(series)
            series = lowpass_filter(series, 'fft', delta_t)
            # if np.average(series) < 1:
            #     series = series + mean_series
        elif delta_t == -2:
            series = np.ones(8760) * np.mean(series)

    # Normalization the demand curve
    if 'normalization' in tech_dic:
        if tech_dic['normalization'] >= 0.0:
            series = series * tech_dic['normalization']/np.average(series)

    tech_dic['series'] = series
    mean_series = np.mean(series)
    return mean_series

def update_timenum(case_dic):
    num_time_periods = (24 * (
            datetime.date(case_dic['year_end'],case_dic['month_end'],case_dic['day_end']) - 
            datetime.date(case_dic['year_start'],case_dic['month_start'],case_dic['day_start'])
            ).days +
            (case_dic['hour_end'] - case_dic['hour_start'] ) + 1)
    return num_time_periods