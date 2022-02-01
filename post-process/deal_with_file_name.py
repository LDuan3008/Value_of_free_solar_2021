# %%

import os    

import shutil

data_path1 = '/Volumes/My Passport/MEM_AdvNuc/testIN/'

data_path2 = '/Volumes/My Passport/MEM_AdvNuc/testOUT/'

file_list = os.listdir(data_path1)

pre_fix = 'zzz_Solar100x_Year2019_'

len_pre_fix = len(pre_fix)

for file in file_list:

    if file[:(int(len_pre_fix))] == pre_fix:

        rest_string = file[(int(len_pre_fix)):]

        for sub_idx in range(len(rest_string)):
            if rest_string[sub_idx:sub_idx+2] == 'LS': LS_start_idx = sub_idx
            if rest_string[sub_idx:sub_idx+2] == 'ST': ST_start_idx = sub_idx

        LS_value = rest_string[LS_start_idx+2:ST_start_idx]
        ST_value = rest_string[ST_start_idx+2:]

        if len(LS_value) > 4: 
            LS_value_new = str(float(LS_value[:4]))
        else:
            LS_value_new = str(float(LS_value))

        if len(ST_value) > 4: 
            ST_value_new = str(float(ST_value[:4]))
        else:
            ST_value_new = str(float(ST_value))

        new_file_name = f'{pre_fix}LS{LS_value_new}ST{ST_value_new}'
        shutil.copytree(f'{data_path1}{file}', f'{data_path2}{new_file_name}')

print ('done')

# %%