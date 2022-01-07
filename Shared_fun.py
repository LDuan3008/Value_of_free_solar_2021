import csv 
import numpy as np
import sys

def read_csv_dated_data_file(start_year,start_month,start_day,start_hour,
                             end_year,end_month,end_day,end_hour,
                             data_path, data_filename):
    
    # turn dates into yyyymmddhh format for comparison.
    # Assumes all datasets are on the same time step and are not missing any data.
    start_hour = start_hour + 100 * (start_day + 100 * (start_month + 100* start_year)) 
    end_hour = end_hour + 100 * (end_day + 100 * (end_month + 100* end_year)) 
      
    path_filename = data_path + '/' + data_filename
    
    data = []
    with open(path_filename) as fin:
        # read to keyword 'BEGIN_DATA' and then one more line (header line)
        data_reader = csv.reader(fin)
        
        #Throw away all lines up to and include the line that has 'BEGIN_GLOBAL_DATA' in the first cell of the line
        while True:
            line = next(data_reader)
            if line[0] == 'BEGIN_DATA':
                break
        # Now take the header row
        line = next(data_reader)
        
        # Now take all non-blank lines
        data = []
        while True:
            try:
                line = next(data_reader)
                if any(field.strip() for field in line):
                    data.append([int(line[0]),int(line[1]),int(line[2]),int(line[3]),float(line[4])])
                    # the above if clause was from: https://stackoverflow.com/questions/4521426/delete-blank-rows-from-csv
            except:
                break
            
    data_array = np.array(data) # make into a numpy object
    
    hour_num = data_array[:,3] + 100 * (data_array[:,2] + 100 * (data_array[:,1] + 100* data_array[:,0]))   
    

    series = [item[1] for item in zip(hour_num,data_array[:,4]) if item[0]>= start_hour and item[0] <= end_hour]
    
    return np.array(series).flatten() # return flatten series


def update_series(case_dic, tech_dic):
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
    # Normalization the demand curve
    if 'normalization' in tech_dic:
        if tech_dic['normalization'] >= 0.0:
            series = series * tech_dic['normalization']/np.average(series)
    tech_dic['series'] = series
    mean_series = np.mean(series)
    return mean_series