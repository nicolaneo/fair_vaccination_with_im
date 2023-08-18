import pandas as pd
from tqdm import tqdm
import os
import numpy as np

# ### combine weekly files ###
def filter_and_combine_weekly_files(path_to_week, week_string):
    new_week_base_path = "/home/mila/n/nicola.neophytou/scratch/experiments/vaccination/weekly_patterns/IL_IN_WI_PA_NJ_DE_MD_NY/"
    all_week_files = sorted([filename for filename in os.listdir(path_to_week)])
    pois_to_keep = pd.DataFrame()
    states_to_keep = ['IL','IN','WI','PA','NJ','DE','MD','NY'] # for Phildelphia, New York, Chicago MSAs
    
    for weekly_file in tqdm(all_week_files):
        chunk = pd.read_csv(path_to_week + weekly_file, compression='gzip', chunksize=1000)
        df = pd.concat(chunk)
        del chunk
        df = df[df['region'].isin(states_to_keep)]
        pois_to_keep = pd.concat([pois_to_keep, df])
        del df
    
    # write out separate weekly_files in chunks, read them in and combine later
    for idx, chunk in tqdm(enumerate(np.array_split(pois_to_keep, 25))):
        chunk.to_csv(new_week_base_path + week_string + f'-weekly-patterns-{idx}.csv.gz', index=False, compression='gzip', chunksize=1000)

### filter monthly files ###
def just_filter_monthly_files(old_path_to_month, new_path_to_month):
    states_to_keep = ['IL','IN','WI','PA','NJ','DE','MD','NY']
    all_month_files = sorted([filename for filename in os.listdir(old_path_to_month) if (filename.startswith('core_poi-geometry-patterns-part') and filename.endswith('.csv.gz'))])    

    for monthly_file in tqdm(all_month_files):
        chunk = pd.read_csv(old_path_to_month + monthly_file, chunksize=1000, compression='gzip')
        df = pd.concat(chunk)
        del chunk
        df = df[df['region'].isin(states_to_keep)]
        df.to_csv(new_path_to_month + monthly_file, index=False, compression='gzip', chunksize=1000)
        del df

if __name__ == "__main__":    
    # filter and combine weekly files
    weekly_base = "/home/mila/n/nicola.neophytou/scratch/experiments/vaccination/weekly_patterns/main_files/"
    filter_and_combine_weekly_files(path_to_week=weekly_base + "march02/", week_string='2020-03-02')
    filter_and_combine_weekly_files(path_to_week=weekly_base + "march09/", week_string='2020-03-09')
    filter_and_combine_weekly_files(path_to_week=weekly_base + "march16/", week_string='2020-03-16')
    filter_and_combine_weekly_files(path_to_week=weekly_base + "march23/", week_string='2020-03-23')
    filter_and_combine_weekly_files(path_to_week=weekly_base + "march30/", week_string='2020-03-30')
    
    # filter monthly files
    monthly_base = "/home/mila/n/nicola.neophytou/scratch/experiments/vaccination/monthly_patterns/"
    just_filter_monthly_files(old_path_to_month=monthly_base + "2020_01_old/", new_path_to_month=monthly_base + "2020_01/")
    just_filter_monthly_files(old_path_to_month=monthly_base + "2020_02_old/", new_path_to_month=monthly_base + "2020_02/")