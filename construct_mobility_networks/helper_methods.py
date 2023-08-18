'''
Chang et al. 2020
(amendments made throughout by anonymous authors 2023)
'''

from covid_constants_and_utils import *
import json
from collections import Counter
from scipy.stats import pearsonr
import fiona
import geopandas
#import csv
import os
from geopandas.tools import sjoin
import time

def load_date_col_as_date(x):
    # we allow this to return None because sometimes we want to filter for cols which are dates.
    try:
        year, month, day = x.split('.')  # e.g., '2020.3.1'
        return datetime.datetime(int(year), int(month), int(day))             
    except:
        return None

def load_patterns_data(month=None, year=None, week_string=None, extra_cols=[], just_testing=False):
    """
    Load in Patterns data for a single month and year, or for a single week. (These options are mutually exclusive). 
    Use extra_cols to define non-default columns to load.
    just_testing is a flag to allow quicker prototyping; it will load only a subset of the data. 
    """
    change_by_date = ['visitor_home_cbgs', 'visitor_country_of_origin',
                      'distance_from_home', 'median_dwell', 'bucketed_dwell_times']  # fields that are time-varying

    if month is not None and year is not None:
        month_and_year = True
        assert week_string is None
        assert month in range(1, 13)
        assert year in [2017, 2018, 2019, 2020]

        month_and_year_string = '%i_%02d' % (year, month)
        # we need to put the 'core_poi-geometry-patterns-part' files into folders of month and year like below
        month_year_old = month_and_year_string + '_old'
        base_dir = os.path.join(PATH_TO_MONTHLY_PATTERNS, '%s' % month_year_old)
        print("Loading all files from %s" % base_dir)

        filenames = [a for a in os.listdir(base_dir) if
                     (a.startswith('core_poi-geometry-patterns-part') and a.endswith('.csv.gz'))]

        if just_testing:
            filenames = filenames[:2]
        print("Number of files to load: %i" % len(filenames))
        full_paths = [os.path.join(base_dir, a) for a in filenames]
        # changed use_dask to False 
        x = load_csv_possibly_with_dask(full_paths, use_dask=False, usecols=['placekey', # was safegraph_place_id before 
                                                                            'parent_placekey',
                                                                            'location_name',
                                                                            'latitude',
                                                                            'longitude',
                                                                            'city',
                                                                            'region',
                                                                            'postal_code',
                                                                            'top_category',
                                                                            'sub_category',
                                                                            'naics_code',
                                                                            "polygon_wkt",
                                                                            "polygon_class",
                                                                            'visits_by_day',
                                                                            'visitor_home_cbgs',
                                                                            'visitor_country_of_origin',
                                                                            'distance_from_home',
                                                                            'median_dwell',
                                                                            'bucketed_dwell_times'] +
                                                                            extra_cols,
                                                                            dtype={'naics_code': 'float64'})

        print("Fraction %2.3f of NAICS codes are missing" % pd.isnull(x['naics_code']).mean())
        x = x.rename(columns={k: f'{year}.{month}.{k}' for k in change_by_date})
    else:
        # weekly patterns data. 
        month_and_year = False
        assert month is None and year is None
        assert week_string in ALL_WEEKLY_STRINGS

        ######### need to recombine weekly files into one df here ########
        weekstring_to_directory = {'2020-03-02':'march02/','2020-03-09':'march09/','2020-03-16':'march16/','2020-03-23':'march23/','2020-03-30':'march30/'}
        x = pd.DataFrame()
        weekly_filenames = [a for a in os.listdir(PATH_TO_WEEKLY_PATTERNS + weekstring_to_directory[week_string])]
        for filename in tqdm(weekly_filenames):
            filepath = os.path.join(PATH_TO_WEEKLY_PATTERNS + weekstring_to_directory[week_string], filename)
            # Filename is misleading - it is really a zipped file.
            weekly_df = pd.read_csv(filepath, escapechar='\\', compression='gzip', nrows=10000 if just_testing else None, usecols=['placekey',
            'visits_by_day',
            'visitor_home_cbgs',
            'visitor_country_of_origin',
            'distance_from_home',
            'median_dwell',
            'bucketed_dwell_times',
            'date_range_start',
            'visits_by_each_hour'])
            x = pd.concat([x, weekly_df])
            del weekly_df
        x = x.reset_index(drop=True)

        percent_nan_start_times = x['date_range_start'].isna().sum()*100.0/len(x)
        print("Missing start times for %2.3f%% percent of rows" % percent_nan_start_times)
        x.date_range_start = x.date_range_start.fillna(week_string + 'T00:00:00-05:00')
        x['date_range_start'] = x['date_range_start'].astype(str) # convert to strings.

        x['offset_from_gmt'] = x['date_range_start'].map(lambda x:x.split('-')[-1])
        print(x['date_range_start'])
        print(x['offset_from_gmt'])
        print(week_string)
        assert x['date_range_start'].map(lambda x:x.startswith(week_string + 'T' + '00:00:00')).all() # make sure date range starts where we expect for all rows. 
        print("Offset from GMT value counts")
        print(x['offset_from_gmt'].value_counts())
        del x['date_range_start']
        x = x.rename(columns={k: f'{week_string}.{k}' for k in change_by_date})

    print("Prior to dropping rows with no visits by day, %i rows" % len(x))
    x = x.dropna(subset=['visits_by_day'])
    x['visits_by_day'] = x['visits_by_day'].map(json.loads) # convert string lists to lists.

    if month_and_year:
        days = pd.DataFrame(x['visits_by_day'].values.tolist(),
                     columns=[f'{year}.{month}.{day}'
                              for day in range(1, len(x.iloc[0]['visits_by_day']) + 1)])
    else:
        year = int(week_string.split('-')[0])
        month = int(week_string.split('-')[1])
        start_day = int(week_string.split('-')[2])
        start_datetime = datetime.datetime(year, month, start_day)
        all_datetimes = [start_datetime + datetime.timedelta(days=i) for i in range(7)]
        days = pd.DataFrame(x['visits_by_day'].values.tolist(),
                     columns=['%i.%i.%i' % (dt.year, dt.month, dt.day) for dt in all_datetimes])

        # Load hourly data as well.
        # Per SafeGraph documentation:
        # Start time for measurement period in ISO 8601 format of YYYY-MM-DDTHH:mm:SS±hh:mm
        # (local time with offset from GMT). The start time will be 12 a.m. Sunday in local time.
        x['visits_by_each_hour'] = x['visits_by_each_hour'].map(json.loads) # convert string lists to lists.
        assert all_datetimes[0].strftime('%A') == 'Monday' # was Sunday before, but safegraph now says Monday.
        hours = pd.DataFrame(x['visits_by_each_hour'].values.tolist(),
                     columns=[f'hourly_visits_%i.%i.%i.%i' % (dt.year, dt.month, dt.day, hour)
                              for dt in all_datetimes
                              for hour in range(0, 24)])

    # reset index
    x = x.reset_index(drop=True)
    days.index = x.index
    x = pd.concat([x, days], axis=1)
    if not month_and_year:
        print(len(x))
        assert list(x.index) == list(range(len(x)))
        assert (hours.index.values == x.index.values).all()
        hours.index = x.index
        old_len = len(x)
        x = pd.concat([x, hours], axis=1)
        assert len(x) == old_len
        x = x.drop(columns=['visits_by_each_hour'])

        # The hourly data has some spurious spikes
        # related to the GMT-day boundary which we have to correct for.
        date_cols = [load_date_col_as_date(a) for a in x.columns]
        date_cols = [a for a in date_cols if a is not None]
        assert len(date_cols) == 7

        if week_string >= '2020-03-15': # think this is because of DST. Basically, these are the timezone strings we look for and correct; they shift at DST. 
            hourly_offsets = [4, 5, 6, 7]
        else:
            hourly_offsets = [4, 5, 6, 7, 8]
        hourly_offset_strings = ['0%i:00' % hourly_offset for hourly_offset in hourly_offsets]

        percent_rows_being_corrected = (x['offset_from_gmt'].map(lambda a:a in hourly_offset_strings).mean() * 100)
        print("%2.3f%% of rows have timezones that we spike-correct for." % percent_rows_being_corrected) 
        assert percent_rows_being_corrected > 99 # make sure we're correcting almost all rows

        # have to correct for each timezone separately.
        for hourly_offset in hourly_offsets:
            idxs = x['offset_from_gmt'] == ('0%i:00' % hourly_offset)
            for date_col in date_cols: # loop over days.
                date_string = '%i.%i.%i' % (date_col.year, date_col.month, date_col.day)
                # not totally clear which hours are messed up - it's mainly one hour, but the surrounding ones look weird too -
                # or what the best way to interpolate is, but this yields plots which look reasonable.

                for hour_to_correct in [24 - hourly_offset - 1,
                                        24 - hourly_offset,
                                        24 - hourly_offset + 1]:

                    # interpolate using hours fairly far from hour_to_correct to avoid pollution.
                    if hour_to_correct < 21:
                        cols_to_use = ['hourly_visits_%s.%i' % (date_string, a) for a in [hour_to_correct - 3, hour_to_correct + 3]]
                    else:
                        # Use smaller offset so we don't have hours >= 24. This technically overlaps with earlier hours, 
                        # but I think it should be okay because they will already have been corrected. 
                        cols_to_use = ['hourly_visits_%s.%i' % (date_string, a) for a in [hour_to_correct - 2, hour_to_correct + 2]]
                    assert all([col in x.columns for col in cols_to_use])
                    x.loc[idxs, 'hourly_visits_%s.%i' % (date_string, hour_to_correct)] = x.loc[idxs, cols_to_use].mean(axis=1)
        del x['offset_from_gmt']
    x = x.set_index('placekey')
    x = x.drop(columns=['visits_by_day'])

    if month_and_year:
        print("%i rows loaded for month and year %s" % (len(x), month_and_year_string))
    else:
        print("%i rows loaded for week %s" % (len(x), week_string))
    return x


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
                       usecols=['STATEA', 'COUNTYA', 'TRACTA', 'BLKGRPA','AJWME001'], # was AJWBE001 previously
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
        
        safegraph_counts = safegraph_counts[safegraph_counts['iso_country_code'] == 'US'] # filter out Canada data on Dewey
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


def annotate_with_demographic_info_and_write_out_in_chunks(full_df, just_testing=False):
    """
    Annotate the Safegraph POI data with Census data and other useful POI data. 
    """
    full_df['placekey'] = full_df.index
    full_df.index = range(len(full_df))

    ############## merge with safegraph areas ##############
    chunk = pd.read_csv(PATH_TO_SAFEGRAPH_AREAS, chunksize=1000, usecols=['safegraph_place_id', 'safegraph_computed_area_in_square_feet'])
    safegraph_areas = pd.concat(chunk)
    safegraph_areas = safegraph_areas.rename(mapper={'safegraph_place_id':'placekey'}, axis=1)
    print("Prior to merging with safegraph areas, %i rows" % len(full_df))
    full_df = pd.merge(full_df, safegraph_areas, how='inner', on='safegraph_place_id', validate='one_to_one')
    print("After merging with areas, %i rows" % len(full_df))
    ########################################################

    # map to demo info. The basic class we use here is CensusBlockGroups, which processes the Census data. 
    print("Mapping SafeGraph POIs to demographic info, including race and income.")
    gdb_files = ['ACS_2017_5YR_BG_51_VIRGINIA.gdb'] if just_testing else None
    cbg_mapper = CensusBlockGroups(base_directory=PATH_TO_CENSUS_BLOCK_GROUP_DATA, gdb_files=gdb_files) # was PATH_FOR_CBG_MAPPER before
    pop_df = load_dataframe_to_correct_for_population_size()
    chunksize = 100000

    annotated_df = []
    for chunk_number in range(len(full_df) // chunksize + 1):
        print("******************Annotating chunk %i" % chunk_number)
        start, end = chunk_number * chunksize, min((chunk_number + 1) * chunksize, len(full_df))
        d = full_df.iloc[start:end].copy()

        # Now annotate each POI on the basis of its location.
        mapped_pois = cbg_mapper.get_demographic_stats_of_points(d['latitude'].values,
                                          d['longitude'].values,
                                          desired_cols=['p_white', 'p_asian', 'p_black', 'median_household_income', 'people_per_mile'])
        mapped_pois['county_fips_code'] = mapped_pois['county_fips_code'].map(lambda x:int(x) if x is not None else x)
        mapped_pois.columns = ['poi_lat_lon_%s' % a for a in mapped_pois.columns]
        for c in mapped_pois.columns:
            d[c] = mapped_pois[c].values

        # Then annotate with demographic data based on where visitors come from (visitor_home_cbgs).
        d = aggregate_visitor_home_cbgs_over_months(d, population_df=pop_df)
        block_group_d = cbg_mapper.block_group_d.copy()
        block_group_d['id_to_match_to_safegraph_data'] = block_group_d['GEOID'].map(lambda x:x.split("US")[1]).astype(int)
        block_group_d = block_group_d[['id_to_match_to_safegraph_data', 'p_black', 'p_white', 'p_asian', 'median_household_income']]
        block_group_d = block_group_d.dropna()

        for col in block_group_d:
            if col == 'id_to_match_to_safegraph_data':
                continue
            cbg_dict = dict(zip(block_group_d['id_to_match_to_safegraph_data'].values, block_group_d[col].values))
            d['cbg_visitor_weighted_%s' % col] = d['aggregated_cbg_population_adjusted_visitor_home_cbgs'].map(lambda x:compute_weighted_mean_of_cbg_visitors(x, cbg_dict))

        # see how well we did.
        for c in [a for a in d.columns if 'poi_lat_lon_' in a or 'cbg_visitor_weighted' in a]:
            print("Have data for %s for fraction %2.3f of people" % (c, 1 - pd.isnull(d[c]).mean()))
        d.to_hdf(os.path.join(ANNOTATED_H5_DATA_DIR, CHUNK_FILENAME) ,f'chunk_{chunk_number}', mode='a', complevel=2) # writes out in chunks # mode = a APPENDs to existing one
        annotated_df.append(d)
    annotated_df = pd.concat(annotated_df)
    annotated_df.index = range(len(annotated_df))
    return annotated_df


def compute_weighted_mean_of_cbg_visitors(cbg_visitor_fracs, cbg_values):
    """
    Given a dictionary cbg_visitor_fracs which gives the fraction of people from a CBG which visit a POI
    and a dictionary cbg_values which maps CBGs to values, compute the weighted mean for the POI.
    """
    if len(cbg_visitor_fracs) == 0:
        return None
    else:
        numerator = 0.
        denominator = 0.
        for cbg in cbg_visitor_fracs:
            if cbg not in cbg_values:
                continue
            numerator += cbg_visitor_fracs[cbg] * cbg_values[cbg]
            denominator += cbg_visitor_fracs[cbg]
        if denominator == 0:
            return None
        return numerator/denominator

def remove_Canadian_visits_and_cast_keys_to_ints(old_dict):
    new_dict = dict([(int(x), y) for x, y in old_dict.items() if not x.startswith("CA:")])
    return new_dict

def cast_keys_to_ints(old_dict):
    new_dict = {}
    for k in old_dict:
        new_dict[int(k)] = old_dict[k]
    return new_dict

def aggregate_visitor_home_cbgs_over_months(d, cutoff_year=2019, population_df=None, periods_to_include=None):
    """
    Aggregate visitor_home_cbgs across months and produce a normalized aggregate field.
    Usage: d = aggregate_visitor_home_cbgs_over_months(d).
    cutoff = the earliest time (could be year or year.month) to aggregate data from
    population_df = the DataFrame loaded by load_dataframe_to_correct_for_population_size
    """
    t0 = time.time()
    if periods_to_include is not None:
        cols = ['%s.visitor_home_cbgs' % period for period in periods_to_include]
        assert cutoff_year is None
    else:
        # Not using CBG data from weekly files for now because of concerns that it's inconsistently
        # processed - they change how they do the privacy filtering.
        assert cutoff_year is not None
        weekly_cols_to_exclude = ['%s.visitor_home_cbgs' % a for a in ALL_WEEKLY_STRINGS] 
        cols = [a for a in d.columns if (a.endswith('.visitor_home_cbgs') and (a >= str(cutoff_year)) and (a not in weekly_cols_to_exclude))]
    print('Aggregating data from: %s' % cols)
    assert all([a in d.columns for a in cols])

    # Helper variables to use if visitor_home_cbgs counts need adjusting for differential sampling across CBGs. 
    adjusted_cols = []
    if population_df is not None:
        int_cbgs = [int(cbg) for cbg in population_df.census_block_group]

    for k in cols:
        if type(d.iloc[0][k]) != Counter: # maybe this is not getting called for all cols if they're already a Counter?
            print('Filling %s with Counter objects' % k)
            d[k] = d[k].fillna('{}').map(lambda x:Counter(remove_Canadian_visits_and_cast_keys_to_ints(json.loads(x))))  # map strings to counters. # remove Canadian visits.

        if population_df is not None:
            sub_t0 = time.time()
            new_col = '%s_adjusted' % k
            assert new_col not in d.columns
            total_population = population_df.total_cbg_population.to_numpy()
            time_period = k.strip('.visitor_home_cbgs')
            population_col = 'number_devices_residing_%s' % time_period
            assert(population_col in population_df.columns)
            num_devices = population_df[population_col].to_numpy()
            assert np.isnan(num_devices).sum() == 0
            assert np.isnan(total_population).sum() == 0
            cbg_coverage = num_devices / total_population # getting divide by zero errors here
            median_coverage = np.nanmedian(cbg_coverage)
            cbg_coverage = dict(zip(int_cbgs, cbg_coverage))
            assert ~np.isnan(median_coverage)
            assert ~np.isinf(median_coverage)
            assert median_coverage > 0.001 
            # want to make sure we aren't missing data for too many CBGs, so a small hack - have
            # adjust_home_cbg_counts_for_coverage return two arguments, where the second argument
            # tells us if we had to clip or fill in the missing coverage number.
            d[new_col] = d[k].map(lambda x:adjust_home_cbg_counts_for_coverage(x, cbg_coverage, median_coverage=median_coverage))
            print('Finished adjusting home CBG counts for %s [time=%.3fs] had to fill in or clip coverage for %2.6f%% of rows; in those cases used median coverage %2.3f' %
                  (time_period, time.time() - sub_t0, 100 * d[new_col].map(lambda x:x[1]).mean(), median_coverage))
            d[new_col] = d[new_col].map(lambda x:x[0]) # remove the second argument of adjust_home_cbg_counts_for_coverage, we don't need it anymore.
            adjusted_cols.append(new_col)

            # make sure there are no NAs anywhere. 
            assert d[k].map(lambda x:len([a for a in x.values() if np.isnan(a)])).sum() == 0
            assert d[new_col].map(lambda x:len([a for a in x.values() if np.isnan(a)])).sum() == 0

    # add counters together across months.
    d['aggregated_visitor_home_cbgs'] = d[cols].aggregate(func=sum, axis=1)
    # normalize each counter so its values sum to 1.
    d['aggregated_visitor_home_cbgs'] = d['aggregated_visitor_home_cbgs'].map(normalize_dict_values_to_sum_to_one_and_cast_keys_to_ints)

    if len(adjusted_cols) > 0:
        d['aggregated_cbg_population_adjusted_visitor_home_cbgs'] = d[adjusted_cols].aggregate(func=sum, axis=1)
        d['aggregated_cbg_population_adjusted_visitor_home_cbgs'] = d['aggregated_cbg_population_adjusted_visitor_home_cbgs'].map(normalize_dict_values_to_sum_to_one_and_cast_keys_to_ints)
        d = d.drop(columns=adjusted_cols)

    for k in ['aggregated_cbg_population_adjusted_visitor_home_cbgs', 
          'aggregated_visitor_home_cbgs']:
        y = d.loc[d[k].map(lambda x:len(x) > 0), k]
        y = y.map(lambda x:sum(x.values()))
        assert np.allclose(y, 1)

    print("Aggregating CBG visitors over %i time periods took %2.3f seconds" % (len(cols), time.time() - t0))
    print("Fraction %2.3f of POIs have CBG visitor data" % (d['aggregated_visitor_home_cbgs'].map(lambda x:len(x) != 0).mean()))
    return d


def normalize_dict_values_to_sum_to_one_and_cast_keys_to_ints(old_dict):
    """
    Self-explanatory; used by aggregate_visitor_home_cbgs_over_months.
    """
    new_dict = {}
    value_sum = 1.*sum(old_dict.values())
    if len(old_dict) > 0:
        assert value_sum > 0
    for k in old_dict:
        new_dict[int(k)] = old_dict[k] / value_sum
    return new_dict

def adjust_home_cbg_counts_for_coverage(cbg_counter, cbg_coverage, median_coverage, max_upweighting_factor=100):
    """
    Adjusts the POI-CBG counts from SafeGraph to estimate the true count, based on the
    coverage that SafeGraph has for this CBG.
    cbg_counter: a Counter object mapping CBG to the original count
    cbg_coverage: a dictionary where keys are CBGs and each data point represents SafeGraph's coverage: num_devices / total_population
    This should be between 0 and 1 for the vast majority of cases, although for some weird CBGs it may not be.
    Returns the adjusted dictionary and a Bool flag had_to_guess_coverage_value which tells us whether we had to adjust the coverage value.
    """
    had_to_guess_coverage_value = False
    if len(cbg_counter) == 0:
        return cbg_counter, had_to_guess_coverage_value
    new_counter = Counter()
    for cbg in cbg_counter:
        # cover some special cases which should happen very rarely. 
        if cbg not in cbg_coverage:
            upweighting_factor = 1 / median_coverage
            had_to_guess_coverage_value = True
        elif np.isnan(cbg_coverage[cbg]): # not sure this case ever actually happens, but just in case. 
            upweighting_factor = 1 / median_coverage
            had_to_guess_coverage_value = True
        else: 
            assert cbg_coverage[cbg] >= 0
            upweighting_factor = 1 / cbg_coverage[cbg]  # need to invert coverage # getting divide by zero errors here
            if upweighting_factor > max_upweighting_factor:
                upweighting_factor = 1 / median_coverage
                had_to_guess_coverage_value = True
        new_counter[cbg] = cbg_counter[cbg] * upweighting_factor
    return new_counter, had_to_guess_coverage_value

def write_out_acs_5_year_data():
    cbg_mapper = CensusBlockGroups(base_directory=PATH_TO_CENSUS_BLOCK_GROUP_DATA, gdb_files=None) # was PATH_FOR_CBG_MAPPER

    geometry_cols = ['STATEFP',
              'COUNTYFP',
              'TRACTCE',
              'Metropolitan/Micropolitan Statistical Area',
              'CBSA Title',
              'State Name']
    block_group_cols = ['GEOID',
                              'p_black',
                              'p_white',
                              'p_asian',
                              'median_household_income',
                             'block_group_area_in_square_miles',
                             'people_per_mile']
    for k in geometry_cols:
        cbg_mapper.block_group_d[k] = cbg_mapper.geometry_d[k].values
    df_to_write_out = cbg_mapper.block_group_d[block_group_cols + geometry_cols]
    print("Total rows: %i" % len(df_to_write_out))
    print(pd.isnull(df_to_write_out).mean())
    df_to_write_out.to_csv(PATH_TO_ACS_5YR_DATA) 


class CensusBlockGroups: # takes a while to read in 5 year file
    """
    A class for loading geographic and demographic data from the ACS.
    A census block group is a relatively small area.
    Less good than houses but still pretty granular. https://en.wikipedia.org/wiki/Census_block_group
    Data was downloaded from https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-data.html
    We use the most recent ACS 5-year estimates: 2013-2017, eg:
    wget https://www2.census.gov/geo/tiger/TIGER_DP/2017ACS/ACS_2017_5YR_BG.gdb.zip
    These files are convenient because they combine both geographic boundaries + demographic data, leading to a cleaner join.
    The main method for data access is get_demographic_stats_of_point. Sample usage:
    x = CensusBlockGroups(gdb_files=['ACS_2017_5YR_BG_51_VIRGINIA.gdb'])
    x.get_demographic_stats_of_points(latitudes=[38.8816], longitudes=[-77.0910], desired_cols=['p_black', 'p_white', 'mean_household_income'])
    """
    def __init__(self, base_directory=PATH_TO_CENSUS_BLOCK_GROUP_DATA,
        gdb_files=None,
        county_to_msa_mapping_filepath=PATH_TO_COUNTY_TO_MSA_MAPPING):
        self.base_directory = base_directory
        if gdb_files is None:
            self.gdb_files = ['ACS_2017_5YR_BG.gdb']
        else:
            self.gdb_files = gdb_files
        self.crs_to_use = WGS_84_CRS # https://epsg.io/4326, WGS84 - World Geodetic System 1984, used in GPS.
        self.county_to_msa_mapping_filepath = county_to_msa_mapping_filepath
        self.load_raw_dataframes() # Load in raw geometry and demographic dataframes.

        # annotate demographic data with more useful columns.
        self.annotate_with_race()
        self.annotate_with_income()
        self.annotate_with_counties_to_msa_mapping()
        self.annotate_with_area_and_pop_density()

    def annotate_with_area_and_pop_density(self):
        # https://gis.stackexchange.com/questions/218450/getting-polygon-areas-using-geopandas. 
        # See comments about using cea projection. 
        gdf = self.geometry_d[['geometry']].copy().to_crs({'proj':'cea'})
        area_in_square_meters = gdf['geometry'].area.values
        self.block_group_d['block_group_area_in_square_miles'] = area_in_square_meters / (1609.34 ** 2)
        self.block_group_d['people_per_mile'] = (self.block_group_d['B03002e1'] /
                                               self.block_group_d['block_group_area_in_square_miles'])
        print(self.block_group_d[['block_group_area_in_square_miles', 'people_per_mile']].describe())


    def annotate_with_race(self):
        """
        Analysis focuses on black and non-white population groups. Also annotate with p_asian because of possible anti-Asian discrimination. 
        B03002e1  HISPANIC OR LATINO ORIGIN BY RACE: Total: Total population -- (Estimate)
        B03002e3  HISPANIC OR LATINO ORIGIN BY RACE: Not Hispanic or Latino: White alone: Total population -- (Estimate)
        B03002e4  HISPANIC OR LATINO ORIGIN BY RACE: Not Hispanic or Latino: Black or African American alone: Total population -- (Estimate)
        B03002e6  HISPANIC OR LATINO ORIGIN BY RACE: Not Hispanic or Latino: Asian alone: Total population -- (Estimate)
        """
        print("annotating with race")
        self.block_group_d['p_black'] = self.block_group_d['B03002e4'] / self.block_group_d['B03002e1']
        self.block_group_d['p_white'] = self.block_group_d['B03002e3'] / self.block_group_d['B03002e1']
        self.block_group_d['p_asian'] = self.block_group_d['B03002e6'] / self.block_group_d['B03002e1']
        print(self.block_group_d[['p_black', 'p_white', 'p_asian']].describe())

    def load_raw_dataframes(self): # first demographic layer takes time
        """
        Read in the original demographic + geographic data.
        """
        self.block_group_d = None
        self.geometry_d = None
        demographic_layer_names = ['X25_HOUSING_CHARACTERISTICS', 'X01_AGE_AND_SEX', 'X03_HISPANIC_OR_LATINO_ORIGIN', 'X19_INCOME'] # do we need housing characteristics?

        for file in self.gdb_files:
            # https://www.reddit.com/r/gis/comments/775imb/accessing_a_gdb_without_esri_arcgis/doj9zza
            full_path = os.path.join(self.base_directory, file)
            layer_list = fiona.listlayers(full_path)
            print(file)
            print(layer_list)
            geographic_layer_name = [a for a in layer_list if a[:15] == 'ACS_2017_5YR_BG'] 
            assert len(geographic_layer_name) == 1
            geographic_layer_name = geographic_layer_name[0]

            # geographic data
            geographic_data = geopandas.read_file(full_path, layer=geographic_layer_name).to_crs(self.crs_to_use) # 220,333 rows
            # by default when you use the read file command, the column containing spatial objects is named "geometry", and will be set as the active column.
            print(geographic_data.columns)
            print(len(geographic_data))
            geographic_data = geographic_data.sort_values(by='GEOID_Data')[['GEOID_Data', 'geometry', 'STATEFP', 'COUNTYFP', 'TRACTCE']]

            # demographic data
            for demographic_idx, demographic_layer_name in enumerate(demographic_layer_names):
                assert demographic_layer_name in layer_list
                if demographic_idx == 0:

                    # take slices of 50k rows at a time, total 220,333 rows
                    for i in range(5):
                        slice_end = int((i+1)*50000)
                        slice_start = int(slice_end - 50000)
                        if i == 0:
                            demographic_data = geopandas.read_file(full_path, layer=demographic_layer_name, rows=slice(slice_start, slice_end))
                            print(len(demographic_data))
                        else:
                            df2 = geopandas.read_file(full_path, layer=demographic_layer_name, rows=slice(slice_start, slice_end))
                            demographic_data = pd.concat([demographic_data, df2], ignore_index=True)
                            print(len(demographic_data))

                else:
                    old_len = len(demographic_data)
                    new_df = geopandas.read_file(full_path, layer=demographic_layer_name)
                    assert sorted(new_df['GEOID']) == sorted(demographic_data['GEOID'])
                    demographic_data = demographic_data.merge(new_df, on='GEOID', how='inner')
                    assert old_len == len(demographic_data)
            demographic_data = demographic_data.sort_values(by='GEOID')

            shared_geoids = set(demographic_data['GEOID'].values).intersection(set(geographic_data['GEOID_Data'].values))
            print("Length of demographic data: %i; geographic data %i; %i GEOIDs in both" % (len(demographic_data), len(geographic_data), len(shared_geoids)))

            demographic_data = demographic_data.loc[demographic_data['GEOID'].map(lambda x:x in shared_geoids)]
            geographic_data = geographic_data.loc[geographic_data['GEOID_Data'].map(lambda x:x in shared_geoids)]

            demographic_data.index = range(len(demographic_data))
            geographic_data.index = range(len(geographic_data))

            assert (geographic_data['GEOID_Data'] == demographic_data['GEOID']).all()
            assert len(geographic_data) == len(set(geographic_data['GEOID_Data']))


            if self.block_group_d is None:
                self.block_group_d = demographic_data
            else:
                self.block_group_d = pd.concat([self.block_group_d, demographic_data])

            if self.geometry_d is None:
                self.geometry_d = geographic_data
            else:
                self.geometry_d = pd.concat([self.geometry_d, geographic_data])

        assert pd.isnull(self.geometry_d['STATEFP']).sum() == 0
        good_idxs = self.geometry_d['STATEFP'].map(lambda x:x in FIPS_CODES_FOR_50_STATES_PLUS_DC).values
        print("Warning: the following State FIPS codes are being filtered out")
        print(self.geometry_d.loc[~good_idxs, 'STATEFP'].value_counts())
        print("%i/%i Census Block Groups in total removed" % ((~good_idxs).sum(), len(good_idxs)))
        self.geometry_d = self.geometry_d.loc[good_idxs]
        self.block_group_d = self.block_group_d.loc[good_idxs]
        self.geometry_d.index = self.geometry_d['GEOID_Data'].values
        self.block_group_d.index = self.block_group_d['GEOID'].values

    def annotate_with_income(self):
        """
        We want a single income number for each block group. This method computes that.
        """
        print("Computing household income")
        # copy-pasted column definitions right out of the codebook.
        codebook_string = """
        B19001e2    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): Less than $10,000: Households -- (Estimate)
        B19001e3    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $10,000 to $14,999: Households -- (Estimate)
        B19001e4    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $15,000 to $19,999: Households -- (Estimate)
        B19001e5    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $20,000 to $24,999: Households -- (Estimate)
        B19001e6    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $25,000 to $29,999: Households -- (Estimate)
        B19001e7    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $30,000 to $34,999: Households -- (Estimate)
        B19001e8    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $35,000 to $39,999: Households -- (Estimate)
        B19001e9    HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $40,000 to $44,999: Households -- (Estimate)
        B19001e10   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $45,000 to $49,999: Households -- (Estimate)
        B19001e11   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $50,000 to $59,999: Households -- (Estimate)
        B19001e12   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $60,000 to $74,999: Households -- (Estimate)
        B19001e13   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $75,000 to $99,999: Households -- (Estimate)
        B19001e14   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $100,000 to $124,999: Households -- (Estimate)
        B19001e15   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $125,000 to $149,999: Households -- (Estimate)
        B19001e16   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $150,000 to $199,999: Households -- (Estimate)
        B19001e17   HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): $200,000 or more: Households -- (Estimate)
        """
        self.income_bin_edges = [0] + list(range(10000, 50000, 5000)) + [50000, 60000, 75000, 100000, 125000, 150000, 200000]

        income_column_names_to_vals = {}
        column_codes = codebook_string.split('\n')
        for f in column_codes:
            if len(f.strip()) == 0:
                continue
            col_name = f.split('HOUSEHOLD INCOME')[0].strip()
            if col_name == 'B19001e2':
                val = 10000
            elif col_name == 'B19001e17':
                val = 200000
            else:
                lower_bound = float(f.split('$')[1].split()[0].replace(',', ''))
                upper_bound = float(f.split('$')[2].split(':')[0].replace(',', ''))
                val = (lower_bound + upper_bound) / 2
            income_column_names_to_vals[col_name] = val
            print("The value for column %s is %2.1f" % (col_name, val))

        # each column gives the count of households with that income. So we need to take a weighted sum to compute the average income.
        self.block_group_d['total_household_income'] = 0.
        self.block_group_d['total_households'] = 0.
        for col in income_column_names_to_vals:
            self.block_group_d['total_household_income'] += self.block_group_d[col] * income_column_names_to_vals[col]
            self.block_group_d['total_households'] += self.block_group_d[col]
        self.block_group_d['mean_household_income'] = 1.*self.block_group_d['total_household_income'] / self.block_group_d['total_households']
        self.block_group_d['median_household_income'] = self.block_group_d['B19013e1'] # MEDIAN HOUSEHOLD INCOME IN THE PAST 12 MONTHS (IN 2017 INFLATION-ADJUSTED DOLLARS): Median household income in the past 12 months (in 2017 inflation-adjusted dollars): Households -- (Estimate)
        assert (self.block_group_d['total_households'] == self.block_group_d['B19001e1']).all() # sanity check: our count should agree with theirs.
        assert (pd.isnull(self.block_group_d['mean_household_income']) == (self.block_group_d['B19001e1'] == 0)).all()
        print("Warning: missing income data for %2.1f%% of census blocks with 0 households" % (pd.isnull(self.block_group_d['mean_household_income']).mean() * 100))
        self.income_column_names_to_vals = income_column_names_to_vals
        assert len(self.income_bin_edges) == len(self.income_column_names_to_vals)
        print(self.block_group_d[['mean_household_income', 'total_households']].describe())

    def annotate_with_counties_to_msa_mapping(self):
        """
        Annotate with metropolitan area info for consistency with Experienced Segregation paper.
        # https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2017/delineation-files/list1.xls
        """
        print("Loading county to MSA mapping")
        self.counties_to_msa_df = pd.read_csv(self.county_to_msa_mapping_filepath, dtype={'FIPS State Code':str, 'FIPS County Code':str}) # skiprows was 2 here but got rid of it
        print("%i rows read" % len(self.counties_to_msa_df))
        self.counties_to_msa_df = self.counties_to_msa_df[['CBSA Title',
                                                           'Metropolitan/Micropolitan Statistical Area',
                                                           'State Name',
                                                           'FIPS State Code',
                                                           'FIPS County Code']]

        self.counties_to_msa_df.columns = ['CBSA Title',
                                           'Metropolitan/Micropolitan Statistical Area',
                                           'State Name',
                                           'STATEFP',
                                           'COUNTYFP']

        self.counties_to_msa_df = self.counties_to_msa_df.dropna(how='all') # remove a couple blank rows.
        assert self.counties_to_msa_df['Metropolitan/Micropolitan Statistical Area'].map(lambda x:x in ['Metropolitan Statistical Area', 'Micropolitan Statistical Area']).all()
        print("Number of unique Metropolitan statistical areas: %i" %
            len(set(self.counties_to_msa_df.loc[self.counties_to_msa_df['Metropolitan/Micropolitan Statistical Area'] == 'Metropolitan Statistical Area', 'CBSA Title'])))
        print("Number of unique Micropolitan statistical areas: %i" %
            len(set(self.counties_to_msa_df.loc[self.counties_to_msa_df['Metropolitan/Micropolitan Statistical Area'] == 'Micropolitan Statistical Area', 'CBSA Title'])))
        old_len = len(self.geometry_d)
        assert len(self.counties_to_msa_df.drop_duplicates(['STATEFP', 'COUNTYFP'])) == len(self.counties_to_msa_df)


        self.geometry_d = self.geometry_d.merge(self.counties_to_msa_df,
                                                on=['STATEFP', 'COUNTYFP'],
                                                how='left')
        # For some reason the index gets reset here. Annoying, not sure why.
        self.geometry_d.index = self.geometry_d['GEOID_Data'].values

        assert len(self.geometry_d) == old_len
        assert (self.geometry_d.index == self.block_group_d.index).all()

    def get_demographic_stats_of_points(self, latitudes, longitudes, desired_cols):
        """
        Given a list or array of latitudes and longitudes, matches to Census Block Group.
        Returns a dictionary which includes the state and county FIPS code, along with any columns in desired_cols.
        This method assumes the latitudes and longitudes are in https://epsg.io/4326, which is what I think is used for Android/iOS -> SafeGraph coordinates.
        """
        def dtype_pandas_series(obj):
            return str(type(obj)) == "<class 'pandas.core.series.Series'>"
        assert not dtype_pandas_series(latitudes)
        assert not  dtype_pandas_series(longitudes)
        assert len(latitudes) == len(longitudes)

        t0 = time.time()

        # we have to match stuff a million rows at a time because otherwise we get weird memory warnings.
        start_idx = 0
        end_idx = start_idx + int(1e6)
        merged = []
        while start_idx < len(longitudes):
            print("Doing spatial join on points with indices from %i-%i" % (start_idx, min(end_idx, len(longitudes))))

            points = geopandas.GeoDataFrame(pd.DataFrame({'placeholder':np.array(range(start_idx, min(end_idx, len(longitudes))))}), # this column doesn't matter. We just have to create a geo data frame.
                geometry=geopandas.points_from_xy(longitudes[start_idx:end_idx], latitudes[start_idx:end_idx]),
                crs=self.crs_to_use)
            # see eg gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude)). http://geopandas.org/gallery/create_geopandas_from_pandas.html
            merged.append(sjoin(points, self.geometry_d[['geometry']], how='left', op='within'))
            assert len(merged[-1]) == len(points)
            start_idx += int(1e6)
            end_idx += int(1e6)
        merged = pd.concat(merged)
        merged.index = range(len(merged))
        assert list(merged.index) == list(merged['placeholder'])

        could_not_match = pd.isnull(merged['index_right']).values
        print("Cannot match to a CBG for a fraction %2.3f of points" % could_not_match.mean())

        results = {}
        for k in desired_cols + ['state_fips_code', 'county_fips_code', 'Metropolitan/Micropolitan Statistical Area', 'CBSA Title', 'GEOID_Data', 'TRACTCE']:
            results[k] = [None] * len(latitudes)
        results = pd.DataFrame(results)
        matched_geoids = merged['index_right'].values[~could_not_match]
        for c in desired_cols:
            results.loc[~could_not_match, c] = self.block_group_d.loc[matched_geoids, c].values
            if c in ['p_white', 'p_black', 'mean_household_income', 'median_household_income', 'new_census_monthly_rent_to_annual_income_multiplier', 'new_census_median_monthly_rent_to_annual_income_multiplier']:
                results[c] = results[c].astype('float')

        results.loc[~could_not_match, 'state_fips_code'] = self.geometry_d.loc[matched_geoids, 'STATEFP'].values
        results.loc[~could_not_match, 'county_fips_code'] = self.geometry_d.loc[matched_geoids, 'COUNTYFP'].values
        results.loc[~could_not_match, 'Metropolitan/Micropolitan Statistical Area'] = self.geometry_d.loc[matched_geoids,'Metropolitan/Micropolitan Statistical Area'].values
        results.loc[~could_not_match, 'CBSA Title'] = self.geometry_d.loc[matched_geoids, 'CBSA Title'].values
        results.loc[~could_not_match, 'GEOID_Data'] = self.geometry_d.loc[matched_geoids, 'GEOID_Data'].values
        results.loc[~could_not_match, 'TRACTCE'] = self.geometry_d.loc[matched_geoids, 'TRACTCE'].values

        print("Total query time is %2.3f" % (time.time() - t0))
        return results

def compute_cbg_day_prop_out(cbgs_of_interest=None):
    
    ########## open saved cbg out proportions ##########
    chunk = pd.read_csv(PATH_TO_CBG_OUT_PROPORTIONS + 'prop_df_IL_IN_WI_PA_NJ_DE_MD_NY.csv', chunksize=1000)
    prop_df = pd.concat(chunk)

    dates = [date for date in list(prop_df.columns) if date.startswith('2020')]
    T = len(dates)
    prop_df[prop_df[dates] > 1.0] = 1.0 # make sure proportions are <= 1.0

    prop_df = prop_df.loc[prop_df['census_block_group'].isin(cbgs_of_interest)]
    # if missing CBGs proportions out then fill with median
    if len(prop_df) < len(cbgs_of_interest):
        missing_cbgs = set(cbgs_of_interest) - set(cbgs_of_interest).intersection(prop_df.census_block_group)
        print('Filling %d CBGs with median props' % len(missing_cbgs))
        print('Percentage missing: ' + str(len(missing_cbgs)*100.0/len(cbgs_of_interest)))
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

def load_age_data():
    """
    Load in a dataframe with rows for the 2018 ACS Census median age of each CBG.
    """
    PATH_TO_ACS_1YR_DATA = "/home/mila/n/nicola.neophytou/scratch/experiments/vaccination/census/1_year_2018/nhgis0008_ds244_20195_blck_grp_E.csv"
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
    return d

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
    print("READING A CHUNK")
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

def load_multiple_chunks(chunks, load_backup=False, cols=None):
    """
    Loads multiple chunks from the h5 file. Currently quite slow; quicker if only a subset of columns are kept.
    Use the parameters cols to specify which columns to keep; if None then all are kept.
    """
    dfs = []
    for i in chunks:
        t0 = time.time()
        chunk = load_chunk(i, load_backup=load_backup)
        print("Loaded chunk %i in %2.3f seconds" % (i, time.time() - t0))
        if cols is not None:
            chunk = chunk[cols]
        dfs.append(chunk)
    t0 = time.time()
    df = pd.concat(dfs)
    print("Concatenated %d chunks in %2.3f seconds" % (len(chunks), time.time() - t0))
    return df

