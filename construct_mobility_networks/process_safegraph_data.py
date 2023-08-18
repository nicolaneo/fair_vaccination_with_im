'''
Chang et al. 2020
(amendments made throughout by Neophytou et al. 2023)
'''

import helper_methods as helper
from covid_constants_and_utils import *
import pandas as pd
import os
import json
import re 

def count_visitors(old_dict):
    if "US" in old_dict.keys():
        return old_dict["US"]


JUST_TESTING = False

# make sure we're not appending to existing h5 files
assert not os.path.exists(os.path.join(helper.ANNOTATED_H5_DATA_DIR, helper.CHUNK_FILENAME)) 

# Write out dataframe of Census data for use in subsequent analysis. 
helper.write_out_acs_5_year_data() 
print("DONE WRITING OUT ACS 5 YEARS DATA.")

# read in individual dataframes for monthly and weekly data [raw SafeGraph data].

all_monthly_dfs = []
all_weekly_dfs = []   

for week_string in helper.ALL_WEEKLY_STRINGS:
    all_weekly_dfs.append(helper.load_patterns_data(week_string=week_string, just_testing=JUST_TESTING))

for month, year in [(1, 2020),(2, 2020)][::-1]:
    # Note ::-1: we load most recent files first so we will take their places info if it is available.
    print("Processing month " + str(month) + " and year " + str(year))
    all_monthly_dfs.append(helper.load_patterns_data(month=month, year=year, just_testing=JUST_TESTING))

# Merge monthly DFs into a single dataframe. Each row is one POI. 
base = all_monthly_dfs[0]
core = all_monthly_dfs[1].columns.intersection(base.columns).to_list() # need to have more than one month going on for this
for i, df in enumerate(all_monthly_dfs[1:]):
    print(i)
    # merge all new places into base so that core info is not nan for new sgids
    new_places = df.loc[df.index.difference(base.index)][core]
    base = pd.concat([base, new_places], join='outer', sort=False)
    # can now left merge in the df because all sgids will be in base
    cols_to_use = df.columns.difference(base.columns).to_list()
    base =  pd.merge(base, df[cols_to_use], left_index=True, right_index=True, how='left')

# Merge in weekly dataframes. Just merge on SafeGraph ID, left merge. 
# This means that our final POI set is those that have both monthly and weekly data. 
# at the end of this cell we will have a single dataframe. 

for i, weekly_df in enumerate(all_weekly_dfs):
    print("\n\n********Weekly dataframe %i/%i" % (i + 1, len(all_weekly_dfs)))
    assert len(base.columns.intersection(weekly_df.columns)) == 0
    
    ids_in_weekly_but_not_monthly = set(weekly_df.index) - set(base.index)
    print("Warning: %i/%i POIs in weekly but not monthly data; dropping these" % (len(ids_in_weekly_but_not_monthly), 
                                                                  len(df)))
    base = pd.merge(base, weekly_df, how='left', left_index=True, right_index=True, validate='one_to_one')
    print("Missing data for weekly columns")
    print(pd.isnull(base[weekly_df.columns]).mean())

# sanity check: how much do weekly visits change if we drop parent IDs. 
parent_ids = set(base['parent_placekey'].dropna())
first_week_of_march_cols = ['hourly_visits_2020.3.%i.%i' % (i, j) for i in range(2, 8) for j in range(24)] # changed from range(1, 8) to range(2, 8), missing first day of march because it's in Febs last weekly data
total_daily_child_visits = base.loc[~pd.isnull(base['parent_placekey']), first_week_of_march_cols].dropna().values.sum()
total_daily_parent_visits = base.loc[base.index.map(lambda x:x in parent_ids), first_week_of_march_cols].dropna().values.sum()
total_daily_visits = base[first_week_of_march_cols].dropna().values.sum()
print("Total daily child visits are fraction %2.3f of total visits; parent visits are %2.3f; dropping parent visits" 
      % (total_daily_child_visits/total_daily_visits, total_daily_parent_visits/total_daily_visits))

# Drop parents to avoid double-counting visits. 
base = base.loc[base.index.map(lambda x:x not in parent_ids)]

# annotate with demographic info and save dataframe. Dataframe is saved in h5py format, separated into chunks. 
annotated = base.sample(frac=1) # shuffle so rows are in random order [in case we want to prototype on subset].
annotated = helper.annotate_with_demographic_info_and_write_out_in_chunks(annotated, just_testing=JUST_TESTING) # WRITES OUT IN CHUNKS, H5PY FILES.

# Stratify by MSA and write out outfiles.  
just_in_msas = annotated.loc[annotated['poi_lat_lon_Metropolitan/Micropolitan Statistical Area'] == 'Metropolitan Statistical Area']
assert pd.isnull(just_in_msas['poi_lat_lon_CBSA Title']).sum() == 0  # POIs in MSAs must have CBSA title
print("%i/%i POIs are in MSAs (%i MSAs total)" % (len(just_in_msas), 
                                                  len(annotated), 
                                                  len(set(just_in_msas['poi_lat_lon_CBSA Title']))))
grouped_by_msa = just_in_msas.groupby('poi_lat_lon_CBSA Title')
total_written_out = 0
for msa_name, small_d in grouped_by_msa:
    small_d = small_d.copy().sample(frac=1) # make sure rows in random order. 
    small_d.index = range(len(small_d))
    name_without_spaces = re.sub('[^0-9a-zA-Z]+', '_', msa_name)
    filename = os.path.join(helper.STRATIFIED_BY_AREA_DIR, '%s.csv' % name_without_spaces) # WRITES OUT IN CHUNKS STRATIFIED BY AREA.
    for k in ['aggregated_cbg_population_adjusted_visitor_home_cbgs', 'aggregated_visitor_home_cbgs']:
        small_d[k] = small_d[k].map(lambda x:json.dumps(dict(x))) # cast to json so properly saved in CSV. 
    
    print(small_d.columns)
    small_d.to_csv(filename)
    print("Wrote out dataframe with %i POIs to %s" % (len(small_d), '%s.csv' % name_without_spaces))
    total_written_out += 1
print("Total written out: %i" % total_written_out)
