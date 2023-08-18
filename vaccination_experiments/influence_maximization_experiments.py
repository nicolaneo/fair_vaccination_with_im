'''
Chang et al. 2020
(amendments made throughout by anonymous authors 2023)
'''

import os
import re
import time
import copy
import json
import math
import pickle
import datetime
import argparse
import numpy as np
import pandas as pd 
from collections import Counter 
from scipy.stats import pearsonr, scoreatpercentile
from vaccination_methods import setup_vaccine_experiment
#from new_parser import experiments_array

BASE_DIR = # anonymised path 
PATH_TO_OVERALL_HOME_PANEL_SUMMARY = os.path.join(BASE_DIR, 'monthly_patterns/2020_02/visit_panel_summary.csv') 
ANNOTATED_H5_DATA_DIR = os.path.join(BASE_DIR, 'all_aggregate_data/chunks_with_demographic_annotations/') 
CHUNK_FILENAME = 'chunk_1.2017-3.2020_c2.h5' 
STRATIFIED_BY_AREA_DIR_old = os.path.join(BASE_DIR, 'all_aggregate_data/chunks_with_demographic_annotations_stratified_by_area_old/') 
STRATIFIED_BY_AREA_DIR = os.path.join(BASE_DIR, 'all_aggregate_data/chunks_with_demographic_annotations_stratified_by_area/') 
PATH_TO_SAFEGRAPH_AREAS = os.path.join(BASE_DIR, 'places_geometry/safegraph_areas_sq_feet.csv') 
PATH_TO_IPF_OUTPUT = os.path.join(BASE_DIR, 'all_aggregate_data/ipf_output/') 
PATH_TO_WEEKLY_PATTERNS = os.path.join(BASE_DIR, 'weekly_patterns/main_files/')
PATH_TO_MONTHLY_PATTERNS = os.path.join(BASE_DIR, 'monthly_patterns/')
PATH_TO_HOME_PANEL_SUMMARY = os.path.join(BASE_DIR, 'weekly_patterns/home_panel_summary/') 
PATH_TO_NEIGHBORHOOD_PATTERNS = os.path.join(BASE_DIR, 'neighborhood_patterns/')
FITTED_MODEL_DIR = os.path.join(BASE_DIR, 'all_aggregate_data/fitted_models/') 
PATH_TO_ACS_1YR_DATA = os.path.join(BASE_DIR, "census/1_year_2018/nhgis0005_ds239_20185_blck_grp_E.csv") 
PATH_TO_ACS_5YR_DATA = os.path.join(BASE_DIR, 'census/5_year_acs_data/2017_five_year_acs_data.csv') 
PATH_TO_CENSUS_BLOCK_GROUP_DATA = os.path.join(BASE_DIR, 'census/ACS_5_year_2013_to_2017_joined_to_blockgroup_shapefiles/') 
PATH_FOR_CBG_MAPPER = os.path.join(BASE_DIR, 'census/new_census_data/') 
PATH_FOR_CBG_MAPPER_BY_STATE = os.path.join(BASE_DIR, 'census/census_block_group_shapefiles_by_state/') 
PATH_TO_COUNTY_TO_MSA_MAPPING = os.path.join(BASE_DIR, 'county_to_msa/list1.csv') 
PATH_TO_NYT_DATA = os.path.join(BASE_DIR, 'ny_times/us-counties-march27-to-april04.csv')
PATH_TO_CBG_OUT_PROPORTIONS = # anonymised path
ALL_WEEKLY_STRINGS = ['2020-03-02','2020-03-09','2020-03-16','2020-03-23','2020-03-30']

AREA_CLIPPING_BELOW = 5
AREA_CLIPPING_ABOVE = 95
DWELL_TIME_CLIPPING_ABOVE = 90
HOURLY_VISITS_CLIPPING_ABOVE = 95
SUBCATEGORY_CLIPPING_THRESH = 100
TOPCATEGORY_CLIPPING_THRESH = 50

FIPS_CODES_FOR_50_STATES_PLUS_DC = { # https://gist.github.com/wavded/1250983/bf7c1c08f7b1596ca10822baeb8049d7350b0a4b
    "10": "Delaware",
    "11": "Washington, D.C.",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    }

codes_to_states = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AS": "American Samoa",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "Washington, D.C.",
    "FM": "Federated States Of Micronesia",
    "FL": "Florida",
    "GA": "Georgia",
    "GU": "Guam",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MH": "Marshall Islands",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "MP": "Northern Mariana Islands",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PW": "Palau",
    "PA": "Pennsylvania",
    "PR": "Puerto Rico",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VI": "Virgin Islands",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming"
}


JUST_50_STATES_PLUS_DC = {'Alabama',
                         'Alaska',
                         'Arizona',
                         'Arkansas',
                         'California',
                         'Colorado',
                         'Connecticut',
                         'Delaware',
                         'Florida',
                         'Georgia',
                         'Hawaii',
                         'Idaho',
                         'Illinois',
                         'Indiana',
                         'Iowa',
                         'Kansas',
                         'Kentucky',
                         'Louisiana',
                         'Maine',
                         'Maryland',
                         'Massachusetts',
                         'Michigan',
                         'Minnesota',
                         'Mississippi',
                         'Missouri',
                         'Montana',
                         'Nebraska',
                         'Nevada',
                         'New Hampshire',
                         'New Jersey',
                         'New Mexico',
                         'New York',
                         'North Carolina',
                         'North Dakota',
                         'Ohio',
                         'Oklahoma',
                         'Oregon',
                         'Pennsylvania',
                         'Rhode Island',
                         'South Carolina',
                         'South Dakota',
                         'Tennessee',
                         'Texas',
                         'Utah',
                         'Vermont',
                         'Virginia',
                         'Washington',
                         'Washington, D.C.',
                         'West Virginia',
                         'Wisconsin',
                         'Wyoming'}

def compute_cbg_day_prop_out(cbgs_of_interest=None):
    # open saved cbg out proportions #
    chunk = pd.read_csv(PATH_TO_CBG_OUT_PROPORTIONS + 'prop_df_IL_IN_WI_PA_NJ_DE_MD_NY.csv', chunksize=1000)
    prop_df = pd.concat(chunk)

    dates = [date for date in list(prop_df.columns) if date.startswith('2020')]
    T = len(dates)
    prop_df[prop_df[dates] > 1.0] = 1.0 # make sure proportions are smaller than 1.0

    prop_df = prop_df.loc[prop_df['census_block_group'].isin(cbgs_of_interest)]
    # if missing CBGs proportions out then fill with median
    if len(prop_df) < len(cbgs_of_interest):
        missing_cbgs = set(cbgs_of_interest) - set(cbgs_of_interest).intersection(prop_df.census_block_group)
        print('Filling %d CBGs with median props' % len(missing_cbgs))
        print('Percentage missing: ' + str(len(missing_cbgs)*100.0/len(cbgs_of_interest)))
        #median_prop = np.median(out, axis=0)
        median_prop = prop_df[dates[17]].median() # get a median proportion
        missing_props = np.broadcast_to(median_prop, (len(missing_cbgs), T))
        missing_props_df = pd.DataFrame(missing_props, columns=dates)
        missing_props_df['census_block_group'] = list(missing_cbgs)
        prop_df = pd.concat((prop_df, missing_props_df))

    return prop_df

def list_datetimes_in_range(min_day, max_day):
    """
    Return a list of datetimes in a range from min_day to max_day, inclusive. Increment is one day. 
    """
    assert(min_day <= max_day)
    days = []
    while min_day <= max_day:
        days.append(min_day)
        min_day = min_day + datetime.timedelta(days=1)
    return days 

def list_hours_in_range(min_hour, max_hour):
    """
    Return a list of datetimes in a range from min_hour to max_hour, inclusive. Increment is one hour. 
    """
    assert(min_hour <= max_hour)
    hours = []
    while min_hour <= max_hour:
        hours.append(min_hour)
        min_hour = min_hour + datetime.timedelta(hours=1)
    return hours

def get_fips_codes_from_state_and_county_fp(state_vec, county_vec):
    fips_codes = []
    for state, county in zip(state_vec, county_vec):
        state = str(state)
        if len(state) == 1:
            state = '0' + state
        county = str(county)
        if len(county) == 1:
            county = '00' + county
        elif len(county) == 2:
            county = '0' + county
        fips_codes.append(np.int64(state + county))
    return fips_codes

def cast_keys_to_ints(old_dict):
    new_dict = {}
    for k in old_dict:
        new_dict[int(k)] = old_dict[k]
    return new_dict

def list_hours_in_range(min_hour, max_hour):
    """
    Return a list of datetimes in a range from min_hour to max_hour, inclusive. Increment is one hour. 
    """
    assert(min_hour <= max_hour)
    hours = []
    while min_hour <= max_hour:
        hours.append(min_hour)
        min_hour = min_hour + datetime.timedelta(hours=1)
    return hours

def get_nyt_outcomes_over_counties(counties=None):
    outcomes = pd.read_csv(PATH_TO_NYT_DATA)
    if counties is not None:
        outcomes = outcomes[outcomes['fips'].isin(counties)]
    return outcomes

def load_dataframe_for_individual_msa(MSA_name, nrows=None):
    """
    This loads all the POI info for a single MSA.
    """
    t0 = time.time()
    filename = os.path.join(STRATIFIED_BY_AREA_DIR, '%s.csv' % MSA_name)
    d = pd.read_csv(filename, nrows=nrows)
    for k in (['aggregated_cbg_population_adjusted_visitor_home_cbgs', 'aggregated_visitor_home_cbgs']):
        d[k] = d[k].map(lambda x:cast_keys_to_ints(json.loads(x)))
    for k in ['%s.visitor_home_cbgs' % a for a in ALL_WEEKLY_STRINGS]:
        d[k] = d[k].fillna('{}')
        d[k] = d[k].map(lambda x:cast_keys_to_ints(json.loads(x)))
    print("Loaded %i rows for %s in %2.3f seconds" % (len(d), MSA_name, time.time() - t0))
    print(d.columns)
    return d

def get_variables_for_evaluating_msa_model(msa_name, verbose=True):
    acs_data = pd.read_csv(PATH_TO_ACS_5YR_DATA)
    
    msa_data = acs_data.copy()
    msa_data['id_to_match_to_safegraph_data'] = msa_data['GEOID'].map(lambda x:x.split("US")[1]).astype(int)
    msa_cbgs = msa_data['id_to_match_to_safegraph_data'].values
    msa_data['fips'] = get_fips_codes_from_state_and_county_fp(msa_data.STATEFP, msa_data.COUNTYFP)

    ################################
    if msa_name == 'Chicago_Naperville_Elgin_IL_IN_WI':
        msa_counties = [17063, 17093, 17111, 17197, 18073, 18111, 18127] # counties for which we have > 80% coverage
    elif msa_name == 'New_York_Newark_Jersey_City_NY_NJ_PA':
        msa_counties = [36005, 36059, 36079, 36103, 36119, 42103]
    elif msa_name == 'Philadelphia_Camden_Wilmington_PA_NJ_DE_MD':
        msa_counties = [34005, 34007, 34015, 42017, 42029, 42045, 42091, 42101]
    else: msa_counties = []
    ###############################

    nyt_outcomes = get_nyt_outcomes_over_counties(msa_counties)
    nyt_counties = set(nyt_outcomes.fips.unique())
    nyt_cbgs = msa_data[msa_data['fips'].isin(nyt_counties)]['id_to_match_to_safegraph_data'].values
    print("nyt cbgs: " + str(len(nyt_cbgs)))
    if verbose:
        print('Found NYT data matching %d counties and %d CBGs' % (len(nyt_counties), len(nyt_cbgs)))
    return nyt_outcomes, nyt_counties, nyt_cbgs, msa_counties, msa_cbgs

def get_ipf_filename(msa_name, min_datetime, max_datetime, clip_visits, correct_visits=True):
    """
    Get the filename matching these parameters of IPF.
    """
    fn = '%s_%s_to_%s_clip_visits_%s' % (msa_name,
                                min_datetime.strftime('%Y-%m-%d'),
                                max_datetime.strftime('%Y-%m-%d'),
                                clip_visits)
    if correct_visits:
        fn += '_correct_visits_True'
    filename = os.path.join(PATH_TO_IPF_OUTPUT, '%s.pkl' % fn)
    return filename

def get_datetime_hour_as_string(datetime_hour):
    return '%i.%i.%i.%i' % (datetime_hour.year, datetime_hour.month,
                            datetime_hour.day, datetime_hour.hour)

def correct_visit_vector(v, median_dwell_in_minutes):
    """
    Given an original hourly visit vector v and a dwell time in minutes,
    return a new hourly visit vector which accounts for spillover.
    """
    v = np.array(v)
    d = median_dwell_in_minutes/60.
    new_v = v.copy().astype(float)
    max_shift = math.floor(d + 1) # maximum hours we can spill over to.
    for i in range(1, max_shift + 1):
        if i < max_shift:
            new_v[i:] += v[:-i] # this hour is fully occupied
        else:
            new_v[i:] += (d - math.floor(d)) * v[:-i] # this hour only gets part of the visits.
    return new_v

def load_dataframe_to_correct_for_population_size(just_load_census_data=False):
    """
    Load in a dataframe with rows for the 2018 ACS Census population code in each CBG
    and the SafeGraph population count in each CBG (from home-panel-summary.csv). 
    The correlation is not actually that good, likely because individual CBG counts are noisy. 
    Definition of
    num_devices_residing: Number of distinct devices observed with a primary nighttime location in the specified census block group.
    """
    print("READING 1 YR ACS FILE")
    acs_data = pd.read_csv(PATH_TO_ACS_1YR_DATA,
                          encoding='cp1252',
                       usecols=['STATEA', 'COUNTYA', 'TRACTA', 'BLKGRPA','AJWME001'], # was AJWBE001
                       dtype={'STATEA':str,
                              'COUNTYA':str,
                              'BLKGRPA':str,
                             'TRACTA':str})
    # https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html
    # FULL BLOCK GROUP CODE = STATE+COUNTY+TRACT+BLOCK GROUP

    acs_data['STATEA'] = acs_data['STATEA'].str.zfill(2)
    acs_data['COUNTYA'] = acs_data['COUNTYA'].str.zfill(3)
    acs_data['TRACTA'] = acs_data['TRACTA'].str.zfill(6)
    acs_data['BLKGRPA'] = acs_data['BLKGRPA'].str.zfill(1)

    assert (acs_data['STATEA'].map(len) == 2).all()
    assert (acs_data['COUNTYA'].map(len) == 3).all()
    assert (acs_data['TRACTA'].map(len) == 6).all()
    assert (acs_data['BLKGRPA'].map(len) == 1).all()
    acs_data['census_block_group'] = (acs_data['STATEA'] +
                                    acs_data['COUNTYA'] +
                                    acs_data['TRACTA'] +
                                    acs_data['BLKGRPA'])
    acs_data['census_block_group'] = acs_data['census_block_group'].astype(int)
    assert len(set(acs_data['census_block_group'])) == len(acs_data)
    acs_data['county_code'] = (acs_data['STATEA'] + acs_data['COUNTYA']).astype(int)
    acs_data = acs_data[['census_block_group', 'AJWME001', 'STATEA', 'county_code']] # was AJWBE001
    acs_data = acs_data.rename(mapper={'AJWME001':'total_cbg_population', # was AJWBE001
                                       'STATEA':'state_code'}, axis=1)
    print("%i rows of 2018 1-year ACS data read" % len(acs_data))
    if just_load_census_data:
        return acs_data
    combined_data = acs_data

    # now read in safegraph data to use as normalizer. Months and years first.
    all_filenames = []
    all_date_strings = []
    for month, year in [(1, 2020), (2, 2020)]:
        month_and_year_string = '%i_%02d' % (year, month) # see example in constants and utils
        filename = os.path.join(PATH_TO_MONTHLY_PATTERNS, month_and_year_string, 'home_panel_summary.csv')
        all_filenames.append(filename)
        all_date_strings.append('%i.%i' % (year, month))

    # now weeks
    for date_string in ALL_WEEKLY_STRINGS:
        all_filenames.append(os.path.join(PATH_TO_HOME_PANEL_SUMMARY, date_string, 'home_panel_summary.csv'))
        all_date_strings.append(date_string)

    cbgs_with_ratio_above_one = np.array([False for a in range(len(acs_data))])

    for filename_idx, filename in enumerate(all_filenames):
        date_string = all_date_strings[filename_idx]
        print("\n*************")
        safegraph_counts = pd.read_csv(filename, dtype={'census_block_group':str})
        print("%s: %i devices read from %i rows" % (
            date_string, safegraph_counts['number_devices_residing'].sum(), len(safegraph_counts)))
        
        safegraph_counts = safegraph_counts[safegraph_counts['iso_country_code'] == 'US'] # filter out Canada data
        print(safegraph_counts['census_block_group'])

        safegraph_counts = safegraph_counts[['census_block_group', 'number_devices_residing']]
        col_name = 'number_devices_residing_%s' % date_string
        safegraph_counts.columns = ['census_block_group', col_name]
        
        safegraph_counts['census_block_group'] = safegraph_counts['census_block_group'].map(int)
        assert len(safegraph_counts['census_block_group'].dropna()) == len(safegraph_counts)
        print("Number of unique Census blocks: %i; unique blocks %i: WARNING: DROPPING NON-UNIQUE ROWS" %
              (len(safegraph_counts['census_block_group'].drop_duplicates(keep=False)), len(safegraph_counts)))
        safegraph_counts = safegraph_counts.drop_duplicates(subset=['census_block_group'], keep=False)

        combined_data = pd.merge(combined_data,
                                 safegraph_counts,
                                 how='left',
                                 validate='one_to_one',
                                 on='census_block_group')
        missing_data_idxs = pd.isnull(combined_data[col_name])
        print("Missing data for %i rows; filling with zeros" % missing_data_idxs.sum())
        combined_data.loc[missing_data_idxs, col_name] = 0

        r, p = pearsonr(combined_data['total_cbg_population'], combined_data[col_name])
        combined_data['ratio'] = combined_data[col_name]/combined_data['total_cbg_population']
        cbgs_with_ratio_above_one = cbgs_with_ratio_above_one | (combined_data['ratio'].values > 1)
        combined_data.loc[combined_data['total_cbg_population'] == 0, 'ratio'] = None
        print("Ratio of SafeGraph count to Census count")
        print(combined_data['ratio'].describe(percentiles=[.25, .5, .75, .9, .99, .999]))
        print("Correlation between SafeGraph and Census counts: %2.3f" % (r))
    print("Warning: %i CBGs with a ratio greater than 1 in at least one month" % cbgs_with_ratio_above_one.sum())
    del combined_data['ratio']
    combined_data.index = range(len(combined_data))
    assert len(combined_data.dropna()) == len(combined_data)
    return combined_data

def load_age_data():
    """
    Load in a dataframe with rows for the 2018 ACS Census median age of each CBG.
    """
    PATH_TO_ACS_1YR_DATA = # anonymised path
    acs_data = pd.read_csv(PATH_TO_ACS_1YR_DATA,
                          encoding='cp1252',
                       usecols=['STATEA', 'COUNTYA', 'TRACTA', 'BLKGRPA','ALT1E001'], # median age of cbg
                       dtype={'STATEA':str,
                              'COUNTYA':str,
                              'BLKGRPA':str,
                             'TRACTA':str})
    
    acs_data['STATEA'] = acs_data['STATEA'].str.zfill(2)
    acs_data['COUNTYA'] = acs_data['COUNTYA'].str.zfill(3)
    acs_data['TRACTA'] = acs_data['TRACTA'].str.zfill(6)
    acs_data['BLKGRPA'] = acs_data['BLKGRPA'].str.zfill(1)
    acs_data = acs_data.iloc[1: , :]# there's another header, remove
    acs_data["ALT1E001"] = pd.to_numeric(acs_data["ALT1E001"])

    assert (acs_data['STATEA'].map(len) == 2).all()
    assert (acs_data['COUNTYA'].map(len) == 3).all()
    assert (acs_data['TRACTA'].map(len) == 6).all()
    assert (acs_data['BLKGRPA'].map(len) == 1).all()

    acs_data['census_block_group'] = (acs_data['STATEA'] +
                                    acs_data['COUNTYA'] +
                                    acs_data['TRACTA'] +
                                    acs_data['BLKGRPA'])
    acs_data['census_block_group'] = acs_data['census_block_group'].astype(int)
    assert len(set(acs_data['census_block_group'])) == len(acs_data)
    acs_data['county_code'] = (acs_data['STATEA'] + acs_data['COUNTYA']).astype(int)
    acs_data = acs_data[['census_block_group', 'ALT1E001', 'STATEA', 'county_code']] 
    acs_data = acs_data.rename(mapper={'ALT1E001':'median_age_2017_5YR', 
                                       'STATEA':'state_code'}, axis=1)

    print("%i rows of 2018 1-year ACS age data read" % len(acs_data))
    acs_data = acs_data[['census_block_group', 'median_age_2017_5YR']]

    return acs_data

def load_and_reconcile_multiple_acs_data():
    """
    Because we use Census data from two data sources, load a single dataframe that combines both. 
    """
    acs_1_year_d = load_dataframe_to_correct_for_population_size(just_load_census_data=True)
    column_rename = {'total_cbg_population':'total_cbg_population_2018_1YR'}
    acs_1_year_d = acs_1_year_d.rename(mapper=column_rename, axis=1)
    acs_1_year_d['state_name'] = acs_1_year_d['state_code'].map(lambda x:FIPS_CODES_FOR_50_STATES_PLUS_DC[str(x)] if str(x) in FIPS_CODES_FOR_50_STATES_PLUS_DC else np.nan)

    acs_5_year_d = pd.read_csv(PATH_TO_ACS_5YR_DATA)
    print('%i rows of 2017 5-year ACS data read' % len(acs_5_year_d))
    acs_5_year_d['census_block_group'] = acs_5_year_d['GEOID'].map(lambda x:x.split("US")[1]).astype(int)
    # rename dynamic attributes to indicate that they are from ACS 2017 5-year
    dynamic_attributes = ['p_black', 'p_white', 'p_asian', 'median_household_income',
                          'block_group_area_in_square_miles', 'people_per_mile']
    column_rename = {attr:'%s_2017_5YR' % attr for attr in dynamic_attributes}
    acs_5_year_d = acs_5_year_d.rename(mapper=column_rename, axis=1)
    # repetitive with 'state_code' and 'county_code' column from acs_1_year_d
    acs_5_year_d = acs_5_year_d.drop(['Unnamed: 0', 'STATEFP', 'COUNTYFP'], axis=1)
    combined_d = pd.merge(acs_1_year_d, acs_5_year_d, on='census_block_group', how='outer', validate='one_to_one')
    combined_d['people_per_mile_hybrid'] = combined_d['total_cbg_population_2018_1YR'] / combined_d['block_group_area_in_square_miles_2017_5YR']

    acs_age_d = load_age_data()
    combined_d = pd.merge(combined_d, acs_age_d, on='census_block_group', how='outer', validate='one_to_one')

    return combined_d

def fit_and_save_one_model(timestring,
                           model_kwargs,
                           data_kwargs,
                           vax_kwargs,
                           d=None,
                           experiment_to_run=None,
                           filter_for_cbgs_in_msa=False):
    
    assert all([k in model_kwargs for k in ['min_datetime', 'max_datetime', 'exogenous_model_kwargs',
                                            'poi_attributes_to_clip']])
    assert 'MSA_name' in data_kwargs
    t0 = time.time()
    return_without_saving = False
    if timestring is None:
        print("Fitting single model. Timestring is none so not saving model and just returning fitted model.")
        return_without_saving = True
    else:
        print("Fitting single model. Results will be saved using timestring %s" % timestring)
    if d is None:  # load data
        d = load_dataframe_for_individual_msa(**data_kwargs)
    nyt_outcomes, nyt_counties, nyt_cbgs, msa_counties, msa_cbgs = get_variables_for_evaluating_msa_model(data_kwargs['MSA_name'])
    if 'counties_to_track' not in model_kwargs:
        model_kwargs['counties_to_track'] = msa_counties
    cbg_groups_to_track = {}
    cbg_groups_to_track['nyt'] = nyt_cbgs
    if filter_for_cbgs_in_msa:
        print("Filtering for %i CBGs within MSA %s" % (len(msa_cbgs), data_kwargs['MSA_name']))
        cbgs_to_filter_for = set(msa_cbgs) # filter for CBGs within MSA
    else:
        cbgs_to_filter_for = None

    correct_visits = model_kwargs['correct_visits'] if 'correct_visits' in model_kwargs else True  # default to True
    if experiment_to_run == 'just_save_ipf_output':
        # If we're saving IPF output, don't try to reload file.
        preload_poi_visits_list_filename = None
    elif 'poi_cbg_visits_list' in model_kwargs:
        print('Passing in poi_cbg_visits_list, will not load from file')
        preload_poi_visits_list_filename = None
    else:
        # Otherwise, default to attempting to load file.
        preload_poi_visits_list_filename = get_ipf_filename(msa_name=data_kwargs['MSA_name'],
            min_datetime=model_kwargs['min_datetime'],
            max_datetime=model_kwargs['max_datetime'],
            clip_visits=model_kwargs['poi_attributes_to_clip']['clip_visits'],
            correct_visits=correct_visits)
        if not os.path.exists(preload_poi_visits_list_filename):
            print("Warning: path %s does not exist; regenerating POI visits" % preload_poi_visits_list_filename)
            preload_poi_visits_list_filename = None
        else:
            print("Reloading POI visits from %s" % preload_poi_visits_list_filename)
    model_kwargs['preload_poi_visits_list_filename'] = preload_poi_visits_list_filename

    fitted_model = fit_disease_model_on_real_data(
        d,
        vax_kwargs=vax_kwargs,
        cbg_groups_to_track=cbg_groups_to_track,
        cbgs_to_filter_for=cbgs_to_filter_for,
        **model_kwargs)
    
def fit_disease_model_on_real_data(d,
                                   vax_kwargs,
                                   min_datetime,
                                   max_datetime,
                                   exogenous_model_kwargs,
                                   poi_attributes_to_clip,
                                   preload_poi_visits_list_filename=None,
                                   poi_cbg_visits_list=None,
                                   correct_poi_visits=True,
                                   multiply_poi_visit_counts_by_census_ratio=True,
                                   aggregate_col_to_use='aggregated_cbg_population_adjusted_visitor_home_cbgs',
                                   cbg_count_cutoff=10,
                                   cbgs_to_filter_for=None,
                                   cbg_groups_to_track=None,
                                   counties_to_track=None,
                                   include_cbg_prop_out=False,
                                   model_init_kwargs=None,
                                   simulation_kwargs=None,
                                   counterfactual_poi_opening_experiment_kwargs=None,
                                   counterfactual_retrospective_experiment_kwargs=None,
                                   return_model_without_fitting=False,
                                   return_model_and_data_without_fitting=False,
                                   model_quality_dict=None,
                                   verbose=True):
    """
    Function to prepare data as input for the disease model, and to run the disease simulation on formatted data.
    d: pandas DataFrame; POI data from SafeGraph
    min_datetime, max_datetime: DateTime objects; the first and last hour to simulate
    exogenous_model_kwargs: dict; extra arguments for Model.init_exogenous_variables()
        required keys: p_sick_at_t0, poi_psi, and home_beta
    poi_attributes_to_clip: dict; which POI attributes to clip
        required keys: clip_areas, clip_dwell_times, clip_visits
    preload_poi_visits_list_filename: str; name of file from which to load precomputed hourly networks
    poi_cbg_visits_list: list of sparse matrices; precomputed hourly networks
    correct_poi_visits: bool; whether to correct hourly visit counts with dwell time
    multiply_poi_visit_counts_by_census_ratio: bool; whether to upscale visit counts by a constant factor
        derived using Census data to try to get real visit volumes
    aggregate_col_to_use: str; the field that holds the aggregated CBG proportions for each POI
    cbg_count_cutoff: int; the minimum number of POIs a CBG must visit to be included in the model
    cbgs_to_filter_for: list; only model CBGs in this list
    cbg_groups_to_track: dict; maps group name to CBGs, will track their disease trajectories during simulation
    counties_to_track: list; names of counties, will track their disease trajectories during simulation
    include_cbg_prop_out: bool; whether to adjust the POI-CBG network based on Social Distancing Metrics (SDM);
        should only be used if precomputed poi_cbg_visits_list is not in use
    model_init_kwargs: dict; extra arguments for initializing Model
    simulation_kwargs: dict; extra arguments for Model.simulate_disease_spread()
    counterfactual_poi_opening_experiment_kwargs: dict; arguments for POI category reopening experiments
    counterfactual_retrospective_experiment_kwargs: dict; arguments for counterfactual mobility reduction experiment
    """
    assert min_datetime <= max_datetime
    assert all([k in exogenous_model_kwargs for k in ['poi_psi', 'home_beta', 'p_sick_at_t0']])
    assert all([k in poi_attributes_to_clip for k in ['clip_areas', 'clip_dwell_times', 'clip_visits']])
    assert aggregate_col_to_use in ['aggregated_cbg_population_adjusted_visitor_home_cbgs',
                                    'aggregated_visitor_home_cbgs']
    if cbg_groups_to_track is None:
        cbg_groups_to_track = {}
    if model_init_kwargs is None:
        model_init_kwargs = {}
    if simulation_kwargs is None:
        simulation_kwargs = {}
    assert not (return_model_without_fitting and return_model_and_data_without_fitting)

    # pre-loading IPF output
    if preload_poi_visits_list_filename is not None:
        f = open(preload_poi_visits_list_filename, 'rb')
        poi_cbg_visits_list = pickle.load(f)
        f.close()

    t0 = time.time()
    print('1. Processing SafeGraph data...')
    # get hour column strings
    all_hours = list_hours_in_range(min_datetime, max_datetime)
    if poi_cbg_visits_list is not None:
        assert len(poi_cbg_visits_list) == len(all_hours)
    hour_cols = ['hourly_visits_%s' % get_datetime_hour_as_string(dt) for dt in all_hours]

    assert(all([col in d.columns for col in hour_cols]))
    print("Found %d hours in all (%s to %s) -> %d hourly visits" % (len(all_hours),
         get_datetime_hour_as_string(min_datetime),
         get_datetime_hour_as_string(max_datetime),
         np.nansum(d[hour_cols].values)))
    all_states = sorted(list(set(d['region'].dropna())))

    # aggregate median_dwell time over weeks
    weekly_median_dwell_pattern = re.compile('2020-\d\d-\d\d.median_dwell')
    median_dwell_cols = [col for col in d.columns if re.match(weekly_median_dwell_pattern, col)]
    print('Aggregating median_dwell from %s to %s' % (median_dwell_cols[0], median_dwell_cols[-1]))
    # note: this may trigger "RuntimeWarning: All-NaN slice encountered" if a POI has all nans for median_dwell;
    # this is not a problem and will be addressed in apply_percentile_based_clipping_to_msa_df
    avg_dwell_times = d[median_dwell_cols].median(axis=1).values
    d['avg_median_dwell'] = avg_dwell_times

    # clip before dropping data so we have more POIs as basis for percentiles
    # this will also drop POIs whose sub and top categories are too small for clipping
    poi_attributes_to_clip = poi_attributes_to_clip.copy()  # copy in case we need to modify
    if poi_cbg_visits_list is not None:
        poi_attributes_to_clip['clip_visits'] = False
        print('Precomputed POI-CBG networks (IPF output) were passed in; will NOT be clipping hourly visits in dataframe')
    if poi_attributes_to_clip['clip_areas'] or poi_attributes_to_clip['clip_dwell_times'] or poi_attributes_to_clip['clip_visits']:
        d, categories_to_clip, cols_to_clip, thresholds, medians = clip_poi_attributes_in_msa_df(
            d, min_datetime, max_datetime, **poi_attributes_to_clip)
        print('After clipping, %i POIs' % len(d))

    # remove POIs with missing data
    d = d.dropna(subset=hour_cols)
    if verbose: print("After dropping for missing hourly visits, %i POIs" % len(d))
    d = d.loc[d[aggregate_col_to_use].map(lambda x:len(x.keys()) > 0)]
    if verbose: print("After dropping for missing CBG home data, %i POIs" % len(d))
    d = d.dropna(subset=['avg_median_dwell'])
    if verbose: print("After dropping for missing avg_median_dwell, %i POIs" % len(d))

    # reindex CBGs
    poi_cbg_proportions = d[aggregate_col_to_use].values  # an array of dicts; each dict represents CBG distribution for POI
    all_cbgs = [a for b in poi_cbg_proportions for a in b.keys()]
    cbg_counts = Counter(all_cbgs).most_common()
    # only keep CBGs that have visited at least this many POIs
    all_unique_cbgs = [cbg for cbg, count in cbg_counts if count >= cbg_count_cutoff]

    if cbgs_to_filter_for is not None:
        print("Prior to filtering for CBGs in MSA, %i CBGs" % len(all_unique_cbgs))
        all_unique_cbgs = [a for a in all_unique_cbgs if a in cbgs_to_filter_for]
        print("After filtering for CBGs in MSA, %i CBGs" % len(all_unique_cbgs))

    # order CBGs lexicographically
    all_unique_cbgs = sorted(all_unique_cbgs)
    N = len(all_unique_cbgs)

    if verbose: print("After dropping CBGs that appear in < %i POIs, %i CBGs (%2.1f%%)" %
          (cbg_count_cutoff, N, 100.*N/len(cbg_counts)))
    cbgs_to_idxs = dict(zip(all_unique_cbgs, range(N)))

    # convert data structures with CBG names to CBG indices
    poi_cbg_proportions_int_keys = []
    kept_poi_idxs = []
    E = 0   # number of connected POI-CBG pairs
    for poi_idx, old_dict in enumerate(poi_cbg_proportions):
        new_dict = {}
        for string_key in old_dict:
            if string_key in cbgs_to_idxs:
                int_key = cbgs_to_idxs[string_key]
                new_dict[int_key] = old_dict[string_key]
                E += 1
        if len(new_dict) > 0:
            poi_cbg_proportions_int_keys.append(new_dict)
            kept_poi_idxs.append(poi_idx)
    M = len(kept_poi_idxs)
    if verbose:
        print('Dropped %d POIs whose visitors all come from dropped CBGs' %
              (len(poi_cbg_proportions) - M))
    print('FINAL: number of CBGs (N) = %d, number of POIs (M) = %d' % (N, M))
    print('Num connected POI-CBG pairs (E) = %d, network density (E/N) = %.3f' %
          (E, E / N))  # avg num POIs per CBG
    if poi_cbg_visits_list is not None:
        expected_M, expected_N = poi_cbg_visits_list[0].shape
        assert M == expected_M
        assert N == expected_N

    # for getting mobility information
    all_visits_pre_lockdown = np.empty([1, N])
    all_visits_in_lockdown = np.empty([1, N])
    for matrix in poi_cbg_visits_list[:481]:
        all_visits_pre_lockdown = np.add(all_visits_pre_lockdown, matrix.sum(axis=0))
    pre_lockdown_visits = all_visits_pre_lockdown.tolist()[0]
    for matrix in poi_cbg_visits_list[481:]:
        all_visits_in_lockdown = np.add(all_visits_in_lockdown, matrix.sum(axis=0))
    in_lockdown_visits = all_visits_in_lockdown.tolist()[0]

    cbg_idx_groups_to_track = {}
    for group in cbg_groups_to_track:
        cbg_idx_groups_to_track[group] = [
            cbgs_to_idxs[a] for a in cbg_groups_to_track[group] if a in cbgs_to_idxs]
        if verbose: print(f'{len(cbg_groups_to_track[group])} CBGs in {group} -> matched {len(cbg_idx_groups_to_track[group])} ({(len(cbg_idx_groups_to_track[group]) / len(cbg_groups_to_track[group])):.3f})')

    # get POI-related variables
    d = d.iloc[kept_poi_idxs]
    poi_subcategory_types = d['sub_category'].values
    poi_areas = d['safegraph_computed_area_in_square_feet'].values
    poi_dwell_times = d['avg_median_dwell'].values
    poi_dwell_time_correction_factors = (poi_dwell_times / (poi_dwell_times+60)) ** 2
    print('Dwell time correction factors: mean = %.2f, min = %.2f, max = %.2f' %
          (np.mean(poi_dwell_time_correction_factors), min(poi_dwell_time_correction_factors), max(poi_dwell_time_correction_factors)))
    poi_time_counts = d[hour_cols].values
    if correct_poi_visits:
        if poi_cbg_visits_list is not None:
            print('Precomputed POI-CBG networks (IPF output) were passed in; will NOT be applying correction to hourly visits in dataframe')
        else:
            print('Correcting POI hourly visit vectors...')
            new_poi_time_counts = []
            for i, (visit_vector, dwell_time) in enumerate(list(zip(poi_time_counts, poi_dwell_times))):
                new_poi_time_counts.append(correct_visit_vector(visit_vector, dwell_time))
            poi_time_counts = np.array(new_poi_time_counts)
            d[hour_cols] = poi_time_counts
            new_hourly_visit_count = np.sum(poi_time_counts)
            print('After correcting, %.2f hourly visits' % new_hourly_visit_count)

    # get CBG-related variables from census data
    print('2. Processing ACS data...')
    acs_d = load_and_reconcile_multiple_acs_data()
    cbgs_to_census_pops = dict(zip(acs_d['census_block_group'].values,
                                   acs_d['total_cbg_population_2018_1YR'].values))  # use most recent population data
    cbg_sizes = np.array([cbgs_to_census_pops[a] for a in all_unique_cbgs])

    assert np.sum(np.isnan(cbg_sizes)) == 0
    if verbose:
        print('CBGs: median population size = %d, sum of population sizes = %d' %
          (np.median(cbg_sizes), np.sum(cbg_sizes)))

    if multiply_poi_visit_counts_by_census_ratio:
        # Get overall undersampling factor.
        # Basically we take ratio of ACS US population to SafeGraph population in Feb 2020.
        # SafeGraph thinks this is reasonable.
        # https://safegraphcovid19.slack.com/archives/C0109NPA543/p1586801883190800?thread_ts=1585770817.335800&cid=C0109NPA543
        total_us_population_in_50_states_plus_dc = acs_d.loc[acs_d['state_code'].map(lambda x:x in FIPS_CODES_FOR_50_STATES_PLUS_DC), 'total_cbg_population_2018_1YR'].sum()
        safegraph_visitor_count_df = pd.read_csv(PATH_TO_OVERALL_HOME_PANEL_SUMMARY)
        safegraph_visitor_count = safegraph_visitor_count_df.loc[safegraph_visitor_count_df['region'] == 'ALL', 'num_unique_visitors'].iloc[0]

        # remove a few safegraph visitors from non-US states.
        two_letter_codes_for_states = set([a.lower() for a in codes_to_states if codes_to_states[a] in JUST_50_STATES_PLUS_DC])
        safegraph_visitor_count_to_non_states = safegraph_visitor_count_df.loc[safegraph_visitor_count_df['region'].map(lambda x:x not in two_letter_codes_for_states and x != 'ALL'), 'num_unique_visitors'].sum()
        if verbose:
            print("Removing %2.3f%% of people from SafeGraph count who are not in 50 states or DC" %
                (100. * safegraph_visitor_count_to_non_states/safegraph_visitor_count))
        safegraph_visitor_count = safegraph_visitor_count - safegraph_visitor_count_to_non_states
        correction_factor = 1. * total_us_population_in_50_states_plus_dc / safegraph_visitor_count
        if verbose:
            print("Total US population from ACS: %i; total safegraph visitor count: %i; correction factor for POI visits is %2.3f" %
                (total_us_population_in_50_states_plus_dc,
                safegraph_visitor_count,
                correction_factor))
        poi_time_counts = poi_time_counts * correction_factor

    if counties_to_track is not None:
        print('Found %d counties to track...' % len(counties_to_track))
        county2cbgs = {}
        high_coverage_counties = []
        for county in counties_to_track:
            print("county id: " + str(county))
            county_cbgs = acs_d[acs_d['county_code'] == county]['census_block_group'].values
            print("county cbgs before matching: " + str(len(county_cbgs)))
            orig_len = len(county_cbgs)
            county_cbgs = sorted(set(county_cbgs).intersection(set(all_unique_cbgs)))
            print("county cbgs after matching with all_unique_cbgs: " + str(len(county_cbgs)))
            if orig_len > 0:
                coverage = len(county_cbgs) / orig_len
                if coverage < 0.8:
                    print('Low coverage warning: only modeling %d/%d (%.1f%%) of the CBGs in %s' %
                          (len(county_cbgs), orig_len, 100. * coverage, county))
                else:
                    high_coverage_counties.append(county)

            if len(county_cbgs) > 0:
                county_cbg_idx = np.array([cbgs_to_idxs[a] for a in county_cbgs])
                county2cbgs[county] = (county_cbgs, county_cbg_idx)
        print('Modeling CBGs from %d of the counties' % len(county2cbgs))
    else:
        county2cbgs = None

    # turn off warnings temporarily so that using > or <= on np.nan does not cause warnings
    np.warnings.filterwarnings('ignore')
    cbg_idx_to_track = set(range(N))  # include all CBGs

    cbg_demographics = {}
    cbg_demographics['pre_lockdown_mobility'] = pre_lockdown_visits
    cbg_demographics['in_lockdown_mobility'] = in_lockdown_visits

    for attribute in ['p_black', 'p_white', 'p_asian', 'median_household_income', 'median_age']:
        attr_col_name = '%s_2017_5YR' % attribute  # using 5-year ACS data for attributes bc less noisy
        assert attr_col_name in acs_d.columns
        mapper_d = dict(zip(acs_d['census_block_group'].values, acs_d[attr_col_name].values))

        attribute_vals = np.array([mapper_d[a] if a in mapper_d and cbgs_to_idxs[a] in cbg_idx_to_track else np.nan for a in all_unique_cbgs])
        cbg_demographics[attribute] = attribute_vals
        non_nan_vals = attribute_vals[~np.isnan(attribute_vals)]
        median_cutoff = np.median(non_nan_vals)
        if verbose:
            print("Attribute %s: was able to compute for %2.1f%% out of %i CBGs, median is %2.3f" %
                (attribute, 100. * len(non_nan_vals) / len(cbg_idx_to_track),
                 len(cbg_idx_to_track), median_cutoff))

        cbg_idx_groups_to_track[f'{attribute}_above_median'] = list(set(np.where(attribute_vals > median_cutoff)[0]).intersection(cbg_idx_to_track))
        cbg_idx_groups_to_track[f'{attribute}_below_median'] = list(set(np.where(attribute_vals <= median_cutoff)[0]).intersection(cbg_idx_to_track))

        top_decile = scoreatpercentile(non_nan_vals, 90)
        bottom_decile = scoreatpercentile(non_nan_vals, 10)
        cbg_idx_groups_to_track[f'{attribute}_top_decile'] = list(set(np.where(attribute_vals >= top_decile)[0]).intersection(cbg_idx_to_track))
        cbg_idx_groups_to_track[f'{attribute}_bottom_decile'] = list(set(np.where(attribute_vals <= bottom_decile)[0]).intersection(cbg_idx_to_track))

        if county2cbgs is not None:
            above_median_in_county = []
            below_median_in_county = []
            for county in county2cbgs:
                county_cbgs, cbg_idx = county2cbgs[county]
                attribute_vals = np.array([mapper_d[a] if a in mapper_d and cbgs_to_idxs[a] in cbg_idx_to_track else np.nan for a in county_cbgs])
                non_nan_vals = attribute_vals[~np.isnan(attribute_vals)]
                median_cutoff = np.median(non_nan_vals)
                above_median_idx = cbg_idx[np.where(attribute_vals > median_cutoff)[0]]
                above_median_idx = list(set(above_median_idx).intersection(cbg_idx_to_track))
                above_median_in_county.extend(above_median_idx)
                below_median_idx = cbg_idx[np.where(attribute_vals <= median_cutoff)[0]]
                below_median_idx = list(set(below_median_idx).intersection(cbg_idx_to_track))
                below_median_in_county.extend(below_median_idx)
            cbg_idx_groups_to_track[f'{attribute}_above_median_in_own_county'] = above_median_in_county
            cbg_idx_groups_to_track[f'{attribute}_below_median_in_own_county'] = below_median_in_county
    np.warnings.resetwarnings()

    if include_cbg_prop_out:
        model_days = list_datetimes_in_range(min_datetime, max_datetime)
        cols_to_keep = ['%s.%s.%s' % (dt.year, dt.month, dt.day) for dt in model_days]
        print('Giving model prop out for %s to %s' % (cols_to_keep[0], cols_to_keep[-1]))
        assert((len(cols_to_keep) * 24) == len(hour_cols))
        print('Loading Social Distancing Metrics and computing CBG prop out per day: warning, this could take a while...')

        cbg_day_prop_out = compute_cbg_day_prop_out(all_unique_cbgs)
        assert(len(cbg_day_prop_out) == len(all_unique_cbgs))
        # sort lexicographically, like all_unique_cbgs
        cbg_day_prop_out = cbg_day_prop_out.sort_values(by='census_block_group')
        assert list(cbg_day_prop_out['census_block_group'].values) == all_unique_cbgs
        cbg_day_prop_out = cbg_day_prop_out[cols_to_keep].values
    else:
        cbg_day_prop_out = None

    print('Total time to prep data: %.3fs' % (time.time() - t0))

    # here, we run different vaccination experiments.    
    if vax_kwargs['vax_experiment'] != 'none':
        print('Running vaccination experiment for ' + str(msa_name))
    
    # store all kwargs so not passing loads of params
    all_model_kwargs = {}
    all_model_kwargs['vax_kwargs'] = vax_kwargs
    all_model_kwargs['vax_kwargs']['msa_name'] = msa_name
    all_model_kwargs['model_init_kwargs'] = model_init_kwargs
    all_model_kwargs['exog_model_kwargs'] = exogenous_model_kwargs
    all_model_kwargs['sim_model_kwargs'] = simulation_kwargs
    all_model_kwargs['extra_kwargs'] = {}
    all_model_kwargs['extra_kwargs']['poi_cbg_proportions_int_keys'] = poi_cbg_proportions_int_keys
    all_model_kwargs['extra_kwargs']['poi_time_counts'] = poi_time_counts
    all_model_kwargs['extra_kwargs']['poi_areas'] = poi_areas
    all_model_kwargs['extra_kwargs']['poi_dwell_time_correction_factors'] = poi_dwell_time_correction_factors
    all_model_kwargs['extra_kwargs']['cbg_sizes'] = cbg_sizes
    all_model_kwargs['extra_kwargs']['all_unique_cbgs'] = all_unique_cbgs
    all_model_kwargs['extra_kwargs']['cbgs_to_idxs'] = cbgs_to_idxs
    all_model_kwargs['extra_kwargs']['all_states'] = all_states
    all_model_kwargs['extra_kwargs']['poi_cbg_visits_list'] = poi_cbg_visits_list
    all_model_kwargs['extra_kwargs']['all_hours'] = all_hours
    all_model_kwargs['extra_kwargs']['cbg_idx_groups_to_track'] = cbg_idx_groups_to_track
    all_model_kwargs['extra_kwargs']['cbg_day_prop_out'] = cbg_day_prop_out
    all_model_kwargs['extra_kwargs']['poi_subcategory_types'] = poi_subcategory_types
    all_model_kwargs['extra_kwargs']['cbg_demographics'] = cbg_demographics

    # start vaccination experiment
    setup_vaccine_experiment(all_model_kwargs=all_model_kwargs)

def clip_poi_attributes_in_msa_df(d, min_datetime, max_datetime,
                                  clip_areas, clip_dwell_times, clip_visits,
                                  area_below=AREA_CLIPPING_BELOW,
                                  area_above=AREA_CLIPPING_ABOVE,
                                  dwell_time_above=DWELL_TIME_CLIPPING_ABOVE,
                                  visits_above=HOURLY_VISITS_CLIPPING_ABOVE,
                                  subcat_cutoff=SUBCATEGORY_CLIPPING_THRESH,
                                  topcat_cutoff=TOPCATEGORY_CLIPPING_THRESH):
    '''
    Deal with POI outliers by clipping their hourly visits, dwell times, and physical areas
    to some percentile of the corresponding distribution for each POI category.
    '''
    attr_cols = []
    if clip_areas:
        attr_cols.append('safegraph_computed_area_in_square_feet')
    if clip_dwell_times:
        attr_cols.append('avg_median_dwell')
    all_hours = list_hours_in_range(min_datetime, max_datetime)
    hour_cols = ['hourly_visits_%s' % get_datetime_hour_as_string(dt) for dt in all_hours]
    if clip_visits:
        attr_cols.extend(hour_cols)

    assert all([col in d.columns for col in attr_cols])
    print('Clipping areas: %s (below=%d, above=%d), clipping dwell times: %s (above=%d), clipping visits: %s (above=%d)' %
          (clip_areas, area_below, area_above, clip_dwell_times, dwell_time_above, clip_visits, visits_above))

    subcats = []
    left_out_subcats = []
    indices_covered = []
    subcategory2idx = d.groupby('sub_category').indices
    for cat, idx in subcategory2idx.items():
        if len(idx) >= subcat_cutoff:
            subcats.append(cat)
            indices_covered.extend(idx)
        else:
            left_out_subcats.append(cat)
    num_subcat_pois = len(indices_covered)

    # group by top_category for POIs whose sub_category's are too small
    topcats = []
    topcategory2idx = d.groupby('top_category').indices
    remaining_pois = d[d['sub_category'].isin(left_out_subcats)]
    necessary_topcats = set(remaining_pois.top_category.unique())  # only necessary to process top_category's that have at least one remaining POI
    for cat, idx in topcategory2idx.items():
        if cat in necessary_topcats and len(idx) >= topcat_cutoff:
            topcats.append(cat)
            new_idx = np.array(list(set(idx) - set(indices_covered)))  # POIs that are not covered by sub_category clipping
            assert len(new_idx) > 0
            topcategory2idx[cat] = (idx, new_idx)
            indices_covered.extend(new_idx)

    print('Found %d sub-categories with >= %d POIs and %d top categories with >= %d POIs -> covers %d POIs' %
          (len(subcats), subcat_cutoff, len(topcats), topcat_cutoff, len(indices_covered)))
    kept_visits = np.nansum(d.iloc[indices_covered][hour_cols].values)
    all_visits = np.nansum(d[hour_cols].values)
    lost_visits = all_visits - kept_visits
    lost_pois = len(d) - len(indices_covered)
    print('Could not cover %d/%d POIs (%.1f%% POIs, %.1f%% hourly visits) -> dropping these POIs' %
          (lost_pois, len(d), 100. * lost_pois/len(d), 100 * lost_visits / all_visits))

    # commenting this out so that it removes POIs if it thinks the visit count is too high
    # if lost_pois / len(d) > .03:
    #     raise Exception('Dropping too many POIs during clipping phase')

    all_cats = topcats + subcats  # process top categories first so sub categories will compute percentiles on raw data
    new_data = np.array(d[attr_cols].copy().values)  # n_pois x n_cols_to_clip
    thresholds = np.zeros((len(all_cats), len(attr_cols)+1))  # clipping thresholds for category x attribute
    medians = np.zeros((len(all_cats), len(attr_cols)))  # medians for category x attribute
    indices_processed = []
    for i, cat in enumerate(all_cats):
        if i < len(topcats):
            cat_idx, new_idx = topcategory2idx[cat]
        else:
            cat_idx = subcategory2idx[cat]
            new_idx = cat_idx
        indices_processed.extend(new_idx)
        first_col_idx = 0  # index of first column for this attribute

        if clip_areas:
            cat_areas = new_data[cat_idx, first_col_idx]  # compute percentiles on entire category
            min_area = np.nanpercentile(cat_areas, area_below)
            max_area = np.nanpercentile(cat_areas, area_above)
            median_area = np.nanmedian(cat_areas)
            thresholds[i][first_col_idx] = min_area
            thresholds[i][first_col_idx+1] = max_area
            medians[i][first_col_idx] = median_area
            new_data[new_idx, first_col_idx] = np.clip(new_data[new_idx, first_col_idx], min_area, max_area)
            first_col_idx += 1

        if clip_dwell_times:
            cat_dwell_times = new_data[cat_idx, first_col_idx]
            max_dwell_time = np.nanpercentile(cat_dwell_times, dwell_time_above)
            median_dwell_time = np.nanmedian(cat_dwell_times)
            thresholds[i][first_col_idx+1] = max_dwell_time
            medians[i][first_col_idx] = median_dwell_time
            new_data[new_idx, first_col_idx] = np.clip(new_data[new_idx, first_col_idx], None, max_dwell_time)
            first_col_idx += 1

        if clip_visits:
            col_idx = np.arange(first_col_idx, first_col_idx+len(hour_cols))
            assert col_idx[-1] == (len(attr_cols)-1)
            orig_visits = new_data[cat_idx][:, col_idx].copy()  # need to copy bc will modify
            orig_visits[orig_visits == 0] = np.nan  # want percentile over positive visits
            # can't take percentile of col if it is all 0's or all nan's
            cols_to_process = col_idx[np.sum(~np.isnan(orig_visits), axis=0) > 0]
            max_visits_per_hour = np.nanpercentile(orig_visits[:, cols_to_process-first_col_idx], visits_above, axis=0)
            assert np.sum(np.isnan(max_visits_per_hour)) == 0
            thresholds[i][cols_to_process + 1] = max_visits_per_hour
            medians[i][cols_to_process] = np.nanmedian(orig_visits[:, cols_to_process-first_col_idx], axis=0)

            orig_visit_sum = np.nansum(new_data[new_idx][:, col_idx])
            orig_attributes = new_data[new_idx]  # return to un-modified version
            orig_attributes[:, cols_to_process] = np.clip(orig_attributes[:, cols_to_process], None, max_visits_per_hour)
            new_data[new_idx] = orig_attributes
            new_visit_sum = np.nansum(new_data[new_idx][:, col_idx])
            print('%s -> has %d POIs, processed %d POIs, %d visits before clipping, %d visits after clipping' %
              (cat, len(cat_idx), len(new_idx), orig_visit_sum, new_visit_sum))
        else:
            print('%s -> has %d POIs, processed %d POIs' % (cat, len(cat_idx), len(new_idx)))

    assert len(indices_processed) == len(set(indices_processed))  # double check that we only processed each POI once
    assert set(indices_processed) == set(indices_covered)  # double check that we processed the POIs we expected to process
    new_d = d.iloc[indices_covered].copy()
    new_d[attr_cols] = new_data[indices_covered]
    return new_d, all_cats, attr_cols, thresholds, medians

def evaluate_all_fitted_models_for_msa(msa_name, min_timestring=None,
                                        max_timestring=None,
                                        timestrings=None,
                                       required_properties=None,
                                       required_model_kwargs=None,
                                       recompute_losses=False,
                                       key_to_sort_by=None,
                                       old_directory=False):

    """
    required_properties refers to params that are defined in data_and_model_kwargs, outside of ‘model_kwargs’ and ‘data_kwargs`
    """

    if required_model_kwargs is None:
        required_model_kwargs = {}
    if required_properties is None:
        required_properties = {}

    if timestrings is None:
        timestrings = filter_timestrings_for_properties(
            required_properties=required_properties,
            required_model_kwargs=required_model_kwargs,
            required_data_kwargs={'MSA_name':msa_name},
            min_timestring=min_timestring,
            max_timestring=max_timestring,
            old_directory=old_directory)
        print('Found %d fitted models for %s' % (len(timestrings), msa_name))
    else:
        # sometimes we may wish to pass in a list of timestrings to evaluate models
        # so we don't have to call filter_timestrings_for_properties a lot.
        assert min_timestring is None
        assert max_timestring is None
        assert required_model_kwargs == {}

    if recompute_losses:
        nyt_outcomes, _, _, _, _ = get_variables_for_evaluating_msa_model(msa_name)

    results = []
    start_time = time.time()
    for ts in timestrings:
        _, kwargs, _, model_results, fast_to_load_results = load_model_and_data_from_timestring(ts,
            verbose=False,
            load_fast_results_only=(not recompute_losses), old_directory=old_directory)
        model_kwargs = kwargs['model_kwargs']
        exo_kwargs = model_kwargs['exogenous_model_kwargs']
        data_kwargs = kwargs['data_kwargs']
        experiment_to_run = kwargs['experiment_to_run']
        assert data_kwargs['MSA_name'] == msa_name

        results_for_ts = {'timestring':ts,
                         'data_kwargs':data_kwargs,
                         'model_kwargs':model_kwargs,
                         'results':model_results,
                         'experiment_to_run':experiment_to_run}

        if 'final infected fraction' in fast_to_load_results:
            results_for_ts['final infected fraction'] = fast_to_load_results['final infected fraction']

        for result_type in ['loss_dict', 'train_loss_dict', 'test_loss_dict', 'ses_race_summary_results', 'estimated_R0', 'clipping_monitor']:
            if (result_type in fast_to_load_results) and (fast_to_load_results[result_type] is not None):
                for k in fast_to_load_results[result_type]:
                    full_key = result_type + '_' + k
                    assert full_key not in results_for_ts
                    results_for_ts[full_key] = fast_to_load_results[result_type][k]

        for k in exo_kwargs:
            assert k not in results_for_ts
            results_for_ts[k] = exo_kwargs[k]
        for k in model_kwargs:
            if k == 'exogenous_model_kwargs':
                continue
            else:
                assert k not in results_for_ts
                results_for_ts[k] = model_kwargs[k]
        results.append(results_for_ts)

    end_time = time.time()
    print('Time to load and score all models: %.3fs -> %.3fs per model' %
          (end_time-start_time, (end_time-start_time)/len(timestrings)))
    results = pd.DataFrame(results)

    if key_to_sort_by is not None:
        results = results.sort_values(by=key_to_sort_by)
    return results

def filter_timestrings_for_properties(required_properties=None,
                                      required_model_kwargs=None,
                                      required_data_kwargs=None,
                                      min_timestring=None,
                                      max_timestring=None,
                                      return_msa_names=False,
                                      old_directory=False):
    """
    required_properties refers to params that are defined in data_and_model_kwargs, outside of ‘model_kwargs’ and ‘data_kwargs
    """
    if required_properties is None:
        required_properties = {}
    if required_model_kwargs is None:
        required_model_kwargs = {}
    if required_data_kwargs is None:
        required_data_kwargs = {}
    if max_timestring is None:
        max_timestring = str(datetime.datetime.now()).replace(' ', '_').replace('-', '_').replace('.', '_').replace(':', '_')
    print("Loading models with timestrings between %s and %s" % (str(min_timestring), max_timestring))
    config_dir = os.path.join(FITTED_MODEL_DIR, 'data_and_model_configs')
    matched_timestrings = []
    msa_names = []
    configs_to_evaluate = os.listdir(config_dir)
    print("%i files in directory %s" % (len(configs_to_evaluate), config_dir))
    for fn in configs_to_evaluate:
        if fn.startswith('config_'):
            timestring = fn.lstrip('config_').rstrip('.pkl')
            if (timestring < max_timestring) and (min_timestring is None or timestring >= min_timestring):
                f = open(os.path.join(config_dir, fn), 'rb')
                data_and_model_kwargs = pickle.load(f)
                f.close()
                if test_if_kwargs_match(required_properties,
                                        required_data_kwargs,
                                        required_model_kwargs,
                                        data_and_model_kwargs):
                    matched_timestrings.append(timestring)
                    msa_names.append(data_and_model_kwargs['data_kwargs']['MSA_name'])
    if not return_msa_names:
        return matched_timestrings
    else:
        return matched_timestrings, msa_names
    
def test_if_kwargs_match(req_properties, req_data_kwargs,
                         req_model_kwargs, test_data_and_model_kwargs):
    # check whether direct properties in test_data_and_model_kwargs match
    prop_match = all([req_properties[key] == test_data_and_model_kwargs[key] for key in req_properties if key not in ['data_kwargs', 'model_kwargs']])
    if not prop_match:
        return False

def load_model_and_data_from_timestring(timestring, verbose=False, load_original_data=False,
                                        load_full_model=False, load_fast_results_only=True,
                                        load_filtered_data_model_was_fitted_on=False,
                                        old_directory=False):

    if verbose:
        print("Loading model from timestring %s" % timestring)
    model_dir = FITTED_MODEL_DIR
    f = open(os.path.join(model_dir, 'data_and_model_configs', 'config_%s.pkl' % timestring), 'rb')
    data_and_model_kwargs = pickle.load(f)
    f.close()
    model = None
    model_results = None
    f = open(os.path.join(model_dir, 'fast_to_load_results_only', 'fast_to_load_results_%s.pkl' % timestring), 'rb')
    fast_to_load_results = pickle.load(f)
    f.close()

    if not load_fast_results_only:
        if load_full_model:
            f = open(os.path.join(model_dir, 'full_models', 'fitted_model_%s.pkl' % timestring), 'rb')
            model = pickle.load(f)
            f.close()

    if load_original_data:
        if verbose:
            print("Loading original data as well...warning, this may take a while")
        d = load_dataframe_for_individual_msa(**data_and_model_kwargs['data_kwargs'])
    else:
        d = None

    return model, data_and_model_kwargs, d, model_results, fast_to_load_results

def load_date_col_as_date(x):
    # we allow this to return None because sometimes we want to filter for cols which are dates.
    try:
        year, month, day = x.split('.')  # e.g., '2020.3.1'
        return datetime.datetime(int(year), int(month), int(day))             
    except:
        return None

def load_chunk(chunk, load_backup=False):
    """
    Load a single 100k chunk from the h5 file; chunks are randomized and so should be reasonably representative. 
    """
    filepath = get_h5_filepath(load_backup=load_backup)
    print("Reading chunk %i from %s" % (chunk, filepath))

    d = pd.read_hdf(filepath, key=f'chunk_{chunk}')
    date_cols = [load_date_col_as_date(a) for a in d.columns]
    date_cols = [a for a in date_cols if a is not None]
    print("Dates range from %s to %s" % (min(date_cols), max(date_cols)))
    return d

def get_h5_filepath(load_backup):
    backup_string = 'BACKUP_' if load_backup else ''
    filepath = os.path.join(ANNOTATED_H5_DATA_DIR, backup_string + CHUNK_FILENAME)
    return filepath

def generate_data_and_model_configs(experiment_to_run='rerun_best_models_and_save_cases_per_poi',
                                    how_to_select_best_grid_search_models=None,
                                    msa_to_use=None,
                                    acceptable_loss_tolerance=1.2):
    """
    Here we actually just want to generate data and model configs for just 1 experiment, 1 msa.
    But still put it in a list. 
    So don't need to grab everything for all 3 msas.
    """
    short_name_to_official_name = {'Philadelphia':'Philadelphia_Camden_Wilmington_PA_NJ_DE_MD',
                                   'NY':'New_York_Newark_Jersey_City_NY_NJ_PA',
                                   'Chicago':'Chicago_Naperville_Elgin_IL_IN_WI'}

    config_generation_start_time = time.time()
    previously_fitted_data_and_model_kwargs = []
    # Helper dataframe to check current status of data
    d = load_chunk(1, load_backup=False)

    # Data kwargs
    data_kwargs = []
    msa_name = msa_to_use
    name_without_spaces = re.sub('[^0-9a-zA-Z]+', '_', short_name_to_official_name[msa_name])
    print(name_without_spaces)
    data_kwargs.append({'MSA_name':name_without_spaces, 'nrows':None})

    # Now generate model kwargs.
    min_datetime = datetime.datetime(2020, 3, 2, 0)  # changed this to March 2nd
    date_cols = [load_date_col_as_date(a) for a in d.columns]
    date_cols = [a for a in date_cols if a is not None]
    max_date = max(date_cols)  # HERE IS WHERE WE SET THE TIME PERIOD FOR EVALUATION. But it's only as long as what columns exist in the dataframe already
    max_datetime = datetime.datetime(max_date.year, max_date.month, max_date.day, 23)  # latest hour
    print('Min datetime: %s. Max datetime: %s.' % (min_datetime, max_datetime))

    # Generate model kwargs. How exactly we do this depends on which experiments we're running.
    list_of_data_and_model_kwargs = []
    key_to_sort_by = 'loss_dict_daily_cases_RMSE'

    for row in data_kwargs:
        msa_t0 = time.time()
        msa_name = row['MSA_name']
        msa_to_best_model_timestring = {'Philadelphia':'2023_04_11_19_40_34_380732', # after deciding on best models
                                        'NY':'2023_04_11_17_50_25_554454', 
                                        'Chicago':'2023_04_11_19_10_57_037858'} 
        
        # here just set the timestring for the related msa
        timestrings_for_msa = []
        timestrings_for_msa.append(msa_to_best_model_timestring[msa_to_use]) # make sure the name format fits
        print("Evaluating %i timestrings for %s" % (len(timestrings_for_msa), msa_name))
        best_msa_models = evaluate_all_fitted_models_for_msa(msa_name, timestrings=timestrings_for_msa)

        best_loss = float(best_msa_models.iloc[0][key_to_sort_by])

        for i in range(len(best_msa_models)):
            loss_ratio = best_msa_models.iloc[i][key_to_sort_by]/best_loss
            assert loss_ratio >= 1 and loss_ratio <= acceptable_loss_tolerance
            model_quality_dict = {'model_fit_rank_for_msa':i,
                                      'how_to_select_best_grid_search_models':how_to_select_best_grid_search_models,
                                      'ratio_of_%s_to_that_of_best_fitting_model' % key_to_sort_by:loss_ratio,
                                      'model_timestring':best_msa_models.iloc[i]['timestring']}
            _, kwargs_i, _, _, _ = load_model_and_data_from_timestring(best_msa_models.iloc[i]['timestring'], load_fast_results_only=True)
            kwargs_i['experiment_to_run'] = experiment_to_run
            del kwargs_i['model_kwargs']['counties_to_track']

            # Rerun best fit models so that we can track the infection contribution of each POI,
            # overall and for each income decile.
            kwarg_copy = copy.deepcopy(kwargs_i) # don't modify by mistake in pass-by-reference.
            simulation_kwargs = {
                        'groups_to_track_num_cases_per_poi':['all',
                            'median_household_income_bottom_decile',
                            'median_household_income_top_decile']}
            kwarg_copy['model_kwargs']['simulation_kwargs'] = simulation_kwargs
            kwarg_copy['model_kwargs']['model_quality_dict'] = model_quality_dict.copy()

            list_of_data_and_model_kwargs.append(kwarg_copy)
        print("In total, it took %2.3f seconds to generate configs for MSA" % (time.time() - msa_t0))

    # sanity check to make sure nothing strange - number of parameters we expect.
    expt_params = []
    for row in list_of_data_and_model_kwargs:
        expt_params.append(
                {'home_beta':row['model_kwargs']['exogenous_model_kwargs']['home_beta'],
                 'poi_psi':row['model_kwargs']['exogenous_model_kwargs']['poi_psi'],
                 'p_sick_at_t0':row['model_kwargs']['exogenous_model_kwargs']['p_sick_at_t0'],
                 'MSA_name':row['data_kwargs']['MSA_name']})
    expt_params = pd.DataFrame(expt_params)

    n_experiments_per_param_setting = expt_params.groupby(['home_beta',
                                                  'poi_psi',
                                                  'p_sick_at_t0',
                                                 'MSA_name']).size()

    #print(n_experiments_per_param_setting)
    assert (n_experiments_per_param_setting.values == 1).all()

    print("Total time to generate configs: %2.3f seconds" % (time.time() - config_generation_start_time))
    return list_of_data_and_model_kwargs

if __name__ == "__main__":
    # task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    # args = experiments_array[int(task_id)]
    # print(args)

    parser = argparse.ArgumentParser()
    parser.add_argument('msa_name', choices=['Philadelphia','NY', 'Chicago'])
    parser.add_argument('vax_experiment', choices=['no_vax', 'random_vax', 'vax_oldest', 'just_im', 'im_eq_treatment', 
                                                   'im_with_age', 'im_with_income', 'imi_ima','imr_ima'])
    args = parser.parse_args()
   
    msa_name = args.msa_name
    timestring = str(datetime.datetime.now()).replace(' ', '_').replace('-', '_').replace('.', '_').replace(':', '_')
    data_and_model_config = generate_data_and_model_configs(experiment_to_run='rerun_best_models_and_save_cases_per_poi',
                how_to_select_best_grid_search_models='daily_cases_rmse',
                msa_to_use = msa_name)[0] # just took first element of the list of data and model kwargs
    
    train_test_partition = None

    # set kwargs for vaccination
    data_and_model_config['vax_kwargs'] = {}
    data_and_model_config['vax_kwargs']['vax_experiment'] = args.vax_experiment

    fit_and_save_one_model(timestring,
        model_kwargs=data_and_model_config['model_kwargs'],
        data_kwargs=data_and_model_config['data_kwargs'],
        experiment_to_run=data_and_model_config['experiment_to_run'],
        vax_kwargs=data_and_model_config['vax_kwargs'])

