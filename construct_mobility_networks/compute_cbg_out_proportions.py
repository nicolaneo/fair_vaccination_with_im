import pandas as pd
import json
from tqdm import tqdm
import datetime
import random
from collections import Counter

NEIGHBOURHOOD_PATH = # anonymised path
PATH_TO_CBG_OUT_PROPORTIONS = # anonymised path

MIN_DATETIME = datetime.datetime(2020, 3, 2, 0)
MAX_DATETIME = datetime.datetime(2020, 4, 5, 23) # 5 weeks

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

def compute_cbg_day_prop_out():

    def divide_stops_by_g(daily_stops, g):
        return [int(stops/g) for stops in daily_stops]
    
    def separate_cbgs_and_counts(cbg_counts_dict):
        return [list(cbg_counts_dict.keys()),list(cbg_counts_dict.values())]
    
    def convert_daily_list_to_counter(list_of_days):
        return [Counter(x) for x in list_of_days]
    
    def get_counts_for_given_day(list_of_counters, cbg_of_row, day):
        day_counter = list_of_counters[day]
        try: # remove self-visits
            del day_counter[cbg_of_row]
        except KeyError:
            print("No self-visits detected for cbg " + cbg_of_row)
        return day_counter
    
    def remove_day_from_list(list_of_counters):
        return list_of_counters[1:]

    def sample_from_cbg_distribution(daily_devices, weekday_dist, weekend_dist):
        sampled_cbgs = [] # a list for each day containing the sampled cbgs
        # assign each date its day of the week
        for i in range(len(daily_devices)):
            assert type(i%7) is int
            if i%7 == 5 or i%7 == 6: # we don't have to worry about this for april because days we want are only in range 0 to 4.
                # weekend
                try:
                    sample = random.choices(weekend_dist[0], weekend_dist[1], k=daily_devices[i])
                except:
                    sample = []
            else: # weekday
                try:
                    sample = random.choices(weekday_dist[0], weekday_dist[1], k=daily_devices[i])
                except:
                    sample = []
            sampled_cbgs.append(sample)
        return sampled_cbgs

    def process_neighborhood_patterns(neighborhood_month_filename, n_days):
        # read in neighborhood patterns in chunks
        chunk = pd.read_csv(neighborhood_month_filename, chunksize=1000, compression='gzip', usecols=["area","device_home_areas","day_counts","raw_stop_counts","raw_device_counts","stops_by_day","weekday_device_home_areas","weekend_device_home_areas"])
        df = pd.concat(chunk)
        #print(df)

        df["area"] = df["area"].astype(str)
        df = df[df["area"].str.startswith(("17", "18", "55", "42", "34", "10", "24", "36"))] # IL, IN, WI, PA, NJ, DE, MD, NY
        df = df.reset_index(drop=True)

        # get factor to convert stop counts to device counts
        df["g_factor"] = df["raw_stop_counts"] / df["raw_device_counts"]

        # convert daily stops to daily device counts
        df['stops_by_day'] = df['stops_by_day'].map(json.loads)
        df["devices_by_day"] = df.apply(lambda x: divide_stops_by_g(x["stops_by_day"], x["g_factor"]), axis=1)
        df.drop(["stops_by_day", "g_factor"], axis=1)

        # convert weekday and weekend device home areas to a distribution of CBGs per CBG
        df["weekday_device_home_areas"] = df["weekday_device_home_areas"].map(json.loads)
        df["weekend_device_home_areas"] = df["weekend_device_home_areas"].map(json.loads)
        df["weekday_cbg_distribution"] = df["weekday_device_home_areas"].apply(separate_cbgs_and_counts)
        df["weekend_cbg_distribution"] = df["weekend_device_home_areas"].apply(separate_cbgs_and_counts)

        # sample for all the days available
        df["daily_cbg_samples"] = df.apply(lambda x: sample_from_cbg_distribution(x["devices_by_day"], x["weekday_cbg_distribution"], x["weekend_cbg_distribution"]), axis=1)
        df.drop(['devices_by_day','weekday_cbg_distribution','weekend_cbg_distribution'], axis=1)
        # convert each day to a dictionary of cbg counts
        df["daily_cbg_device_counts"] = df["daily_cbg_samples"].apply(convert_daily_list_to_counter)
        df.drop(['daily_cbg_samples'], axis=1)
        
        # make a column for each day, and fill with each corresponding dictionary already made
        for i in tqdm(range(n_days)):
            df["day_" + str(i)] = df.apply(lambda x: get_counts_for_given_day(x["daily_cbg_device_counts"], x['area'], 0), axis=1)
            df['daily_cbg_device_counts'] = df["daily_cbg_device_counts"].apply(remove_day_from_list) # remove the day from list to save space
        df.drop(['daily_cbg_device_counts'], axis=1) # for space
        return df
    
    # get number of days of interest
    dates = list_datetimes_in_range(MIN_DATETIME, MAX_DATETIME)
    dates = [x.strftime('%m-%d-%Y') for x in dates]
    print(dates)
    n_days = len(dates) + 1 # 36, because we will process 1st March but don't include it later

    # process neighborhood patterns for march and april
    df = process_neighborhood_patterns(anonymised_path, n_days=31)
    april_df = process_neighborhood_patterns(anonymised_path, n_days=5)
    april_df = april_df.rename(columns={'day_0': 'day_31', 'day_1': 'day_32', 'day_2':'day_33', 'day_3':'day_34', 'day_4':'day_35'})
    df_len_before = len(df)
    df = pd.merge(df, april_df, on=['area'])
    df_len_after = len(df)
    print("number of cbgs: " + str(df_len_after))
    print("loss in cbgs after merging with april: " + str((df_len_before-df_len_after)*100.0/df_len_before))
    del april_df

    # now aggregate counts per day, watch out for empty dicts
    # make a dataframe for each day, and merge with existing
    daily_out_counts = pd.DataFrame()
    daily_out_counts['area'] = df['area'] # this is how we make sure we're only counting counts of cbgs in the states we care about. 
    for i in range(n_days)[1:]: # we don't want to start from day_0, because this is 1st March
        one_day_out_counts = pd.DataFrame()
        print("day " + str(i))
        out_counts = Counter()
        for index, row in tqdm(df.iterrows()):
            out_counts = out_counts + row["day_" + str(i)]
        one_day_out_counts = pd.DataFrame.from_dict(out_counts, orient='index').reset_index()
        one_day_out_counts = one_day_out_counts.rename(columns={'index':'area', 0:"day_" + str(i)})
        daily_out_counts = pd.merge(daily_out_counts, one_day_out_counts, on=['area'])
    print("number of cbgs in daily out counts: " + str(len(daily_out_counts)))
    daily_out_counts = daily_out_counts.rename(columns={'area':'census_block_group'})
    daily_out_counts = daily_out_counts.astype({"census_block_group": 'int64'}) # must change to ints other wise it won't merge!

    # can get total residing from the panel summary
    panel = pd.read_csv(NEIGHBOURHOOD_PATH + 'neighborhood_home_panel_summary.csv', chunksize=1000)
    summary = pd.concat(panel)
    df = summary[summary["iso_country_code"] == "US"]
    df = df[["census_block_group","number_devices_residing"]]
    print("NUMBER OF US CBGs IN HOME PANEL SUMMARY: " + str(len(df)))
    str_df = df[df.census_block_group.apply(type) == str]
    df = df[df.census_block_group.apply(type) == int]
    str_df = str_df.astype({"census_block_group": 'int64'})
    df = pd.concat([df, str_df])
    df = df.sort_values(by="census_block_group")

    # merge with out counts
    out_counts_len_before = len(daily_out_counts)
    out = pd.merge(daily_out_counts, df, on='census_block_group')
    out_counts_len_after = len(out)
    print("loss after merging out counts with cbgs in home panel summary: " + str((out_counts_len_before-out_counts_len_after)*100.0/out_counts_len_before))
    out = out[out.number_devices_residing > 0]
    
    # get dates formatted correctly and make prop_df
    prop_df = pd.DataFrame()
    for i in range(n_days)[1:]: # again, start from 2nd March
        prop_df[dates[i-1]] = out['day_' + str(i)] / out['number_devices_residing'] # get fraction
    N, T = prop_df.shape # T = no. of days in dates
    prop_df["census_block_group"] = out["census_block_group"]
    
    print(prop_df.columns)
    prop_df.to_csv(PATH_TO_CBG_OUT_PROPORTIONS + 'prop_df_IL_IN_WI_PA_NJ_DE_MD_NY.csv', index=False, chunksize=1000)

if __name__ == "__main__":
    compute_cbg_day_prop_out()


