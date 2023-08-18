
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from disease_model import Model

def calc_budget_used(V, cbg_sizes):
    '''
    Takes V, the current vaccinated set of CBGs, and calculates how many people
    are in it, so how many vaccines have been used. 
    '''
    person_count = np.sum([cbg_sizes[i] for i in V])
    return person_count

def add_to_demographic_stats(all_model_kwargs, cbg_sizes):

    cbg_demographics = all_model_kwargs['extra_kwargs']['cbg_demographics']
    p_white = cbg_demographics['p_white']
    p_black = cbg_demographics['p_black']
    p_asian = cbg_demographics['p_asian']

    # fill in nans
    nan_idx = [i for (i,v) in enumerate(p_white) if np.isnan(v)]
    for idx in nan_idx:
        p_white[idx] = np.nanmean(p_white)
    all_model_kwargs['extra_kwargs']['cbg_demographics']['p_white'] = p_white
    
    nan_idx = [i for (i,v) in enumerate(p_black) if np.isnan(v)]
    for idx in nan_idx:
        p_black[idx] = np.nanmean(p_black)
    all_model_kwargs['extra_kwargs']['cbg_demographics']['p_black'] = p_black

    nan_idx = [i for (i,v) in enumerate(p_asian) if np.isnan(v)]
    for idx in nan_idx:
        p_asian[idx] = np.nanmean(p_asian)
    all_model_kwargs['extra_kwargs']['cbg_demographics']['p_asian'] = p_asian

    races = np.sum([p_white, p_black, p_asian], axis=0)
    p_other = np.ones(races.shape) - races

    all_model_kwargs['extra_kwargs']['cbg_demographics']['p_other'] = p_other # percentage of other races in each cbg
    all_model_kwargs['extra_kwargs']['cbg_demographics']['total_p_white'] = np.sum(np.multiply(p_white, cbg_sizes))/np.sum(cbg_sizes)
    all_model_kwargs['extra_kwargs']['cbg_demographics']['total_p_black'] = np.sum(np.multiply(p_black, cbg_sizes))/np.sum(cbg_sizes)
    all_model_kwargs['extra_kwargs']['cbg_demographics']['total_p_asian'] = np.sum(np.multiply(p_asian, cbg_sizes))/np.sum(cbg_sizes)
    all_model_kwargs['extra_kwargs']['cbg_demographics']['total_p_other'] = np.sum(np.multiply(p_other, cbg_sizes))/np.sum(cbg_sizes)

    return all_model_kwargs

def calc_race_infections(infections_per_cbg, cbg_demographics):
    p_white = cbg_demographics['p_white']
    p_black = cbg_demographics['p_black']
    p_asian = cbg_demographics['p_asian']
    p_other = cbg_demographics['p_other']

    infections_per_race = {}
    infections_per_race['white'] = np.sum(np.multiply(infections_per_cbg, p_white))
    infections_per_race['black'] = np.sum(np.multiply(infections_per_cbg, p_black))
    infections_per_race['asian'] = np.sum(np.multiply(infections_per_cbg, p_asian))
    infections_per_race['other'] = np.sum(np.multiply(infections_per_cbg, p_other))

    return infections_per_race

def calculate_gain_infections(start_cbgs, spread, all_model_kwargs, risk_weights=None):
    print("Calculating gain.")
    model = get_prior_model_from_cbgs(start_cbgs=start_cbgs, 
                                      all_model_kwargs=all_model_kwargs)
    final_gain = model.total_infected - spread

    # if using risk weights by age
    if risk_weights is not None: 
        return np.sum(np.multiply(model.all_infected, risk_weights)) - spread
    else:
        return final_gain

def get_prior_model_from_cbgs(start_cbgs, all_model_kwargs):
    '''
    start_cbgs = cbgs to begin disease from.
    '''
    if start_cbgs == []:
        all_model_kwargs['exog_model_kwargs']['cbg_idx_to_seed_in'] = None # this means let all CBGs start with disease
    else:
        all_model_kwargs['exog_model_kwargs']['cbg_idx_to_seed_in'] = np.array(start_cbgs)
    all_model_kwargs['exog_model_kwargs']['cbg_idx_to_vaccinate'] = None

    all_model_kwargs['model_init_kwargs']['num_seeds'] = 30 # number of randomized runs
    init_kwargs = all_model_kwargs['model_init_kwargs']
    exog_kwargs = all_model_kwargs['exog_model_kwargs']
    sim_kwargs = all_model_kwargs['sim_model_kwargs']
    extra_kwargs = all_model_kwargs['extra_kwargs']
    sim_kwargs['simulation_type'] = 'prior'

    model = Model(**init_kwargs)
    model.init_exogenous_variables(poi_cbg_proportions=extra_kwargs['poi_cbg_proportions_int_keys'],
                                   poi_time_counts = extra_kwargs['poi_time_counts'],
                                   poi_areas=extra_kwargs['poi_areas'],
                                   poi_dwell_time_correction_factors=extra_kwargs['poi_dwell_time_correction_factors'],
                                   cbg_sizes=extra_kwargs['cbg_sizes'],
                                   all_unique_cbgs=extra_kwargs['all_unique_cbgs'],
                                   cbgs_to_idxs=extra_kwargs['cbgs_to_idxs'],
                                   all_states=extra_kwargs['all_states'],
                                   poi_cbg_visits_list=extra_kwargs['poi_cbg_visits_list'],
                                   all_hours=extra_kwargs['all_hours'],
                                   cbg_idx_groups_to_track=extra_kwargs['cbg_idx_groups_to_track'],
                                   cbg_day_prop_out=extra_kwargs['cbg_day_prop_out'],
                                   intervention_cost=None,
                                   poi_subcategory_types=extra_kwargs['poi_subcategory_types'],
                                   **exog_kwargs)
    
    model.init_endogenous_variables(simulation_type='prior')
    model.simulate_disease_spread(**sim_kwargs)
    
    return model

def get_posterior_model_from_cbgs(vaxed_cbgs, all_model_kwargs): # only used for evaluation of final set V
    '''
    vaxed_cbgs: set of cbgs to vaccinate. All other cbgs are susceptible to infection.
    '''
    # by setting this to None, all cbgs are susceptible
    all_model_kwargs['exog_model_kwargs']['cbg_idx_to_seed_in'] = None
    if all_model_kwargs['vax_kwargs']['vax_experiment'] == 'no_vax' or vaxed_cbgs == []:
        all_model_kwargs['exog_model_kwargs']['cbg_idx_to_vaccinate'] = None
    else: # then set these cbgs as vaccinated
        all_model_kwargs['exog_model_kwargs']['cbg_idx_to_vaccinate'] = np.array(vaxed_cbgs)

    all_model_kwargs['model_init_kwargs']['num_seeds'] = 30 # number of randomized runs
    init_kwargs = all_model_kwargs['model_init_kwargs']
    exog_kwargs = all_model_kwargs['exog_model_kwargs']
    sim_kwargs = all_model_kwargs['sim_model_kwargs']
    extra_kwargs = all_model_kwargs['extra_kwargs']
    sim_kwargs['simulation_type'] = 'eval'

    model = Model(**init_kwargs)
    model.init_exogenous_variables(poi_cbg_proportions=extra_kwargs['poi_cbg_proportions_int_keys'],
                                   poi_time_counts = extra_kwargs['poi_time_counts'],
                                   poi_areas=extra_kwargs['poi_areas'],
                                   poi_dwell_time_correction_factors=extra_kwargs['poi_dwell_time_correction_factors'],
                                   cbg_sizes=extra_kwargs['cbg_sizes'],
                                   all_unique_cbgs=extra_kwargs['all_unique_cbgs'],
                                   cbgs_to_idxs=extra_kwargs['cbgs_to_idxs'],
                                   all_states=extra_kwargs['all_states'],
                                   poi_cbg_visits_list=extra_kwargs['poi_cbg_visits_list'],
                                   all_hours=extra_kwargs['all_hours'],
                                   cbg_idx_groups_to_track=extra_kwargs['cbg_idx_groups_to_track'],
                                   cbg_day_prop_out=extra_kwargs['cbg_day_prop_out'],
                                   intervention_cost=None,
                                   poi_subcategory_types=extra_kwargs['poi_subcategory_types'],
                                   **exog_kwargs)
    model.init_endogenous_variables(simulation_type='eval')
    model.simulate_disease_spread(**sim_kwargs)

    return model

def evaluate(V, all_model_kwargs):
    print("Evaluating final selection of vaccinated cbgs.")
    lines_to_output = []
    cbg_sizes = all_model_kwargs['extra_kwargs']['cbg_sizes']
    cbg_demographics = all_model_kwargs['extra_kwargs']['cbg_demographics']
    n_seeds = all_model_kwargs['model_init_kwargs']['num_seeds']
    cbg_list = np.arange(1, len(cbg_sizes))

    lines_to_output.append("selected CBG set V: " + str(V))

    # get posterior outcomes
    posterior_model = get_posterior_model_from_cbgs(vaxed_cbgs=V, all_model_kwargs=all_model_kwargs)
    lines_to_output.append("posterior total infected: " + str(posterior_model.total_infected))
    lines_to_output.append("total infected error: " + str(posterior_model.total_infected_std/np.sqrt(n_seeds)))

    # get spread weighted by risk weights (by age)
    median_age = cbg_demographics['median_age']
    mean_median_age = np.nanmean(median_age)
    nan_idx = np.where(np.isnan(median_age))
    median_age[nan_idx] = mean_median_age

    # assign risk weights based on age bracket
    risk_weights = [1.0] * len(cbg_sizes)
    for i, v in enumerate(cbg_sizes): # add id of the cbg to the dictionary
        if median_age[i] < 40.0 and median_age[i] >= 30.0:
            risk_weights[i] *= 3.5
        elif median_age[i] < 50.0 and median_age[i] >= 40.0:
            risk_weights[i] *= 10.0
        elif median_age[i] < 65.0 and median_age[i] >= 50.0:
            risk_weights[i] *= 25.0
        elif median_age[i] < 75.0 and median_age[i] >= 65.0:
            risk_weights[i] *= 60.0
        elif median_age[i] < 85.0 and median_age[i] >= 75.0:
            risk_weights[i] *= 140.0
        elif median_age[i] >= 85.0:
            risk_weights[i] *= 350.0
    
    final_infection_rate_per_cbg = posterior_model.all_infected
    final_spread_with_risk_weights = np.sum(np.multiply(final_infection_rate_per_cbg, risk_weights))
    lines_to_output.append("posterior spread with risk weights: " + str(final_spread_with_risk_weights))
    # propagate error
    std_risk_weight_spread = np.square(np.multiply(posterior_model.all_infected_std, risk_weights))
    final_std_risk_weight_spread = np.sqrt(np.sum(std_risk_weight_spread))
    lines_to_output.append("posterior spread with risk weights error: " + str(final_std_risk_weight_spread/np.sqrt(n_seeds)))

    # get number of infections in each racial group
    full_cbg_history = posterior_model.full_history_for_all_CBGs 
    final_E = full_cbg_history['latent'][-1] # [-1] for the final hour of simulation
    final_I = full_cbg_history['infected'][-1]
    final_R = full_cbg_history['removed'][-1]
    EIR = np.sum([final_E, final_I, final_R], axis=0)
    p_white = cbg_demographics['p_white'] 
    p_black = cbg_demographics['p_black'] 
    p_asian = cbg_demographics['p_asian']
    races = np.sum([p_white, p_black, p_asian], axis=0)
    p_other = np.ones(races.shape) - races

    df = pd.DataFrame() # put results together per cbg
    df['p_white'] = p_white
    df['p_black'] = p_black
    df['p_asian'] = p_asian
    df['p_other'] = p_other
    df['population'] = cbg_sizes
    df['EIR'] = list(posterior_model.all_infected_fraction) # infection rate per cbg
    df['eir_std'] = list(posterior_model.all_infected_fraction_std) 
    df['eir_error'] = df['eir_std'] / np.sqrt(n_seeds)
    df['square_error'] = df['eir_error'] * df['eir_error']
    df['n_EIR'] = df.EIR * df.population

    for race_fraction in ['p_white', 'p_black', 'p_asian', 'p_other']:
        df['n_eir_' + race_fraction] = df[race_fraction] * df['n_EIR']
        lines_to_output.append('number of infections in race ' + race_fraction + ': ' + str(np.sum(df['n_eir_' + race_fraction])))

        # error propagation
        df['constants'] = df.population * df[race_fraction]
        df['sq_constants'] = df['constants'] * df['constants']
        df['sq_error'] = df['sq_constants'] * df['square_error']
        lines_to_output.append('number of infections in race ' + race_fraction + ' error: ' + str(np.sqrt(np.sum(df['sq_error']))))

    # median income
    median_income = all_model_kwargs['extra_kwargs']['cbg_demographics']['median_household_income']
    mean_median_income = np.nanmean(median_income)
    inds = np.where(np.isnan(median_income))
    median_income[inds] = mean_median_income

    quartiles = list(np.quantile(median_income, [0, .25, .5, .75, 1.]))
    quartile_lookup = {}
    # loop over income quartiles
    for i in range(len(quartiles)-1):
        # get list of cbgs whose income is in that quartile
        cbg_list_quartile = [j for [j, v] in enumerate(median_income) if v > quartiles[i] and v <= quartiles[i+1]]
        for cbg in cbg_list_quartile:
            quartile_lookup[cbg] = str(i)
        n_eir_quartile = np.sum([np.multiply(EIR, cbg_sizes)[cbg] for cbg in cbg_list_quartile])
        lines_to_output.append('number of infections in income quartile ' + str(i+1) + ', :' + str(n_eir_quartile))

        # error
        eir_std = posterior_model.all_infected_fraction_std
        error = eir_std / np.sqrt(n_seeds)
        sq_error = np.multiply(error, error)
        sq_constant = np.multiply(cbg_sizes, cbg_sizes)
        total_error = [np.multiply(sq_error, sq_constant)[cbg] for cbg in cbg_list_quartile] # only take cbgs in this income quartile
        lines_to_output.append('number infected in income quartile ' + str(i+1) + ' error: ' + str(np.sqrt(np.sum(total_error))))

    for cbg in cbg_list:
        if median_income[cbg] == quartiles[0]:
            quartile_lookup[cbg] = '0'

    # get demographic makeup of selected cbgs to vaccinate
    attributes = [p_white, p_black, p_asian, p_other, median_age, median_income]
    for attr in attributes:
        V_attr = attr[V]
        V_size = cbg_sizes[V] 
        weighted_sum = np.sum(np.multiply(V_attr, V_size)) / np.sum(V_size)
        lines_to_output.append("weighted average of attribute in vaccinated cbgs: " + str(weighted_sum))
    
    for i in range(len(quartiles)-1):
        quartile_population_in_V = np.sum([cbg_sizes[cbg] for cbg in V if quartile_lookup[cbg] == str(i)])
        quartile_fraction_in_V = quartile_population_in_V / np.sum(V_size)
        lines_to_output.append("income quartile fraction in V: " + str(quartile_fraction_in_V))

    # demographic distribution of whole network
    lines_to_output.append("total percentage white population: " + str(np.sum(np.multiply(p_white, cbg_sizes)) / np.sum(cbg_sizes)))
    lines_to_output.append("total percentage black population: " + str(np.sum(np.multiply(p_black, cbg_sizes)) / np.sum(cbg_sizes)))
    lines_to_output.append("total percentage asian population: " + str(np.sum(np.multiply(p_asian, cbg_sizes)) / np.sum(cbg_sizes)))
    lines_to_output.append("total percentage other population: " + str(np.sum(np.multiply(p_other, cbg_sizes)) / np.sum(cbg_sizes)))
    
    #### mobility information on V ####
    pre_lockdown_mobility = all_model_kwargs['extra_kwargs']['cbg_demographics']['pre_lockdown_mobility']
    pre_mobility = [visits/size for visits,size in zip(pre_lockdown_mobility, cbg_sizes)]
    in_lockdown_mobility = all_model_kwargs['extra_kwargs']['cbg_demographics']['in_lockdown_mobility']
    in_mobility = [visits/size for visits,size in zip(in_lockdown_mobility, cbg_sizes)]
    lines_to_output.append('whole network pre lockdown mobility: ' + str(np.sum(pre_mobility)/len(pre_mobility)))
    lines_to_output.append('whole network in lockdown mobility: ' + str(np.sum(in_mobility)/len(in_mobility)))

    pre_mobility_V = [visits for i, visits in enumerate(pre_mobility) if i in V]
    lines_to_output.append('pre lockdown mobility for V: ' + str(np.sum(pre_mobility_V)/len(pre_mobility_V)))
    in_mobility_V = [visits for i, visits in enumerate(in_mobility) if i in V]
    lines_to_output.append('in lockdown mobility for V: ' + str(np.sum(in_mobility_V)/len(in_mobility_V)))

    # output all results to file
    msa_name = all_model_kwargs['vax_kwargs']['msa_name']
    vax_experiment = all_model_kwargs['vax_kwargs']['vax_experiment']
    date_and_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = # anonymised path

    with open(path + 'results_' + str(msa_name) + '_' + str(vax_experiment) + '_' + str(date_and_time) + '.txt', 'w') as f:
        for line in lines_to_output:
            f.write(line)
            f.write('\n')

def no_vax(all_model_kwargs):
    evaluate(V=[], all_model_kwargs=all_model_kwargs)

def update_race_budgets_used(race_budgets_used, new_cbg, cbg_demographics, cbg_sizes):
    p_white = cbg_demographics['p_white']
    p_black = cbg_demographics['p_black']
    p_asian = cbg_demographics['p_asian']
    p_other = cbg_demographics['p_other']

    race_budgets_used['white'] += cbg_sizes[new_cbg] * p_white[new_cbg]
    race_budgets_used['black'] += cbg_sizes[new_cbg] * p_black[new_cbg]
    race_budgets_used['asian'] += cbg_sizes[new_cbg] * p_asian[new_cbg]
    race_budgets_used['other'] += cbg_sizes[new_cbg] * p_other[new_cbg]
    return race_budgets_used

def update_income_budgets_used(income_budgets_used, new_cbg, quartile_lookup, cbg_sizes):
    quartile = quartile_lookup[new_cbg]
    income_budgets_used[quartile] += cbg_sizes[new_cbg]
    return income_budgets_used

def keep_valid_cbgs_within_budget(candidate_cbgs, cbg_sizes, budget_used, budget):
    valid_cbgs = [x for x in candidate_cbgs if (budget_used + cbg_sizes[x[0]]) <= budget]
    return valid_cbgs

def keep_valid_cbgs_within_income_budgets(candidate_cbgs, cbg_sizes, income_budgets_used, income_budgets, quartile_lookup):
    candidate_cbgs = [x for x in candidate_cbgs if (income_budgets_used[quartile_lookup[x[0]]] + cbg_sizes[x[0]]) <= income_budgets[quartile_lookup[x[0]]]]
    return candidate_cbgs

def keep_valid_cbgs_within_race_budgets(candidate_cbgs, cbg_sizes, race_budgets_used, race_budgets, cbg_demographics):
    p_white = cbg_demographics['p_white']
    p_black = cbg_demographics['p_black']
    p_asian = cbg_demographics['p_asian']
    p_other = cbg_demographics['p_other']

    candidate_cbgs = [x for x in candidate_cbgs if (race_budgets_used['white'] + (cbg_sizes[x[0]] * p_white[x[0]])) <= race_budgets['p_white']]
    candidate_cbgs = [x for x in candidate_cbgs if (race_budgets_used['black'] + (cbg_sizes[x[0]] * p_black[x[0]])) <= race_budgets['p_black']]
    candidate_cbgs = [x for x in candidate_cbgs if (race_budgets_used['asian'] + (cbg_sizes[x[0]] * p_asian[x[0]])) <= race_budgets['p_asian']]
    candidate_cbgs = [x for x in candidate_cbgs if (race_budgets_used['other'] + (cbg_sizes[x[0]] * p_other[x[0]])) <= race_budgets['p_other']]
    return candidate_cbgs

def initialize_race_budgets():
    race_budgets_used = {}
    for race in ['white','black','asian','other']:
        race_budgets_used[race] = 0

    return race_budgets_used

def initialize_income_budgets(income_budgets):
    income_budgets_used={}
    for income in income_budgets.keys():
        income_budgets_used[income] = 0

    return income_budgets_used

def vax_with_influence_maximization(all_model_kwargs, 
                                    cbg_list, 
                                    budget,
                                    risk_weights=None,
                                    race_budgets=None,
                                    income_budgets=None,
                                    quartile_lookup=None):
    budget_used, spread_without_cost = 0, 0
    candidate_cbgs = []
    cbg_sizes = all_model_kwargs['extra_kwargs']['cbg_sizes']
    cbg_demographics = all_model_kwargs['extra_kwargs']['cbg_demographics']

    if income_budgets is not None:
        income_budgets_used = initialize_income_budgets(income_budgets)
    if race_budgets is not None:
        race_budgets_used = initialize_race_budgets()

    # for testing: make cbg_list smaller
    # cbg_list = cbg_list[:10]

    # get values for individual gain for each cbg
    for cbg in tqdm(cbg_list):
        gain_without_cost = calculate_gain_infections([cbg], 
                                            spread_without_cost,
                                            all_model_kwargs,
                                            risk_weights=risk_weights)
        print("gain without cost: " + str(gain_without_cost))
        gain_with_cost = gain_without_cost / cbg_sizes[cbg]
        candidate_cbgs.append([cbg, gain_with_cost])

    # sort individual gains in descending order
    candidate_cbgs.sort(key=lambda x:x[1], reverse=True)
    print(candidate_cbgs)
    V = [candidate_cbgs[0][0]]
    spread_without_cost += (candidate_cbgs[0][1] * cbg_sizes[candidate_cbgs[0][0]]) # add gain without cost
    candidate_cbgs = candidate_cbgs[1:]
    budget_used = cbg_sizes[V[-1]]

    if race_budgets is not None: # doing equal treatment
        race_budgets_used = update_race_budgets_used(race_budgets_used, candidate_cbgs[0][0],
                                                     cbg_demographics, cbg_sizes)
    if income_budgets is not None:
        income_budgets_used = update_income_budgets_used(income_budgets_used, candidate_cbgs[0][0],
                                                         quartile_lookup, cbg_sizes)

    # add cbgs to vax set while some budget remains
    while len(candidate_cbgs) > 0:
        if risk_weights is None: # only do the checks for experiments without age weights
            matched = False # bool to check if the top cbg is still the same as prev. round
            while not matched:
                print("checking for a match")
                current_best = candidate_cbgs[0][0]
                print(V + [current_best])
                # re-calculate its marginal gain with the new set V            
                gain_without_cost = calculate_gain_infections(V + [current_best],
                                                  spread_without_cost,
                                                  all_model_kwargs,
                                                  risk_weights=risk_weights)
            
                gain_with_cost = gain_without_cost / cbg_sizes[current_best]
                candidate_cbgs[0][1] = gain_with_cost
                candidate_cbgs.sort(key=lambda x:x[1], reverse=True)
                matched=(candidate_cbgs[0][0] == current_best)
                print("check passed, adding node to seed set")
        # with the new best candidate
        spread_without_cost += gain_without_cost
        V.append(candidate_cbgs[0][0]) # add best node to vaccinated set
        budget_used += cbg_sizes[V[-1]] # update budget used

        if race_budgets is not None:
            race_budgets_used = update_race_budgets_used(race_budgets_used, candidate_cbgs[0][0],
                                                         cbg_demographics, cbg_sizes)
        if income_budgets is not None:
            income_budgets_used = update_income_budgets_used(income_budgets_used, candidate_cbgs[0][0],
                                                         quartile_lookup, cbg_sizes)
            
        candidate_cbgs = candidate_cbgs[1:] # remove it from the set of candidates
        candidate_cbgs = keep_valid_cbgs_within_budget(candidate_cbgs, cbg_sizes, budget_used, budget)
        if race_budgets is not None:
            candidate_cbgs = keep_valid_cbgs_within_race_budgets(candidate_cbgs, cbg_sizes,
                                                                 race_budgets_used, race_budgets, cbg_demographics)
        if income_budgets is not None:
            candidate_cbgs = keep_valid_cbgs_within_income_budgets(candidate_cbgs, cbg_sizes,
                                                                   income_budgets_used, income_budgets, quartile_lookup)

    return V
    
def equal_treatment(all_model_kwargs, cbg_list, budget):
    cbg_sizes = all_model_kwargs['extra_kwargs']['cbg_sizes']
    cbg_demographics = all_model_kwargs['extra_kwargs']['cbg_demographics']

    race_budgets = {}
    for race in ['p_white', 'p_black', 'p_asian', 'p_other']:
        p_race = cbg_demographics[race]
        # calculate budget per race
        total_p_race = np.sum(np.multiply(p_race, cbg_sizes)) / np.sum(cbg_sizes)
        race_budgets[race] = int(budget*total_p_race)

    V = vax_with_influence_maximization(all_model_kwargs,
                                  cbg_list,
                                  budget,
                                  risk_weights=None,
                                  race_budgets=race_budgets)
    
    evaluate(V, all_model_kwargs)

def random_vax(all_model_kwargs, cbg_list, cbg_sizes, budget):
    budget_used, V = 0, []
    candidate_cbgs = cbg_list.copy()
    
    while len(candidate_cbgs) > 0:
        cbg_selected = np.random.choice(candidate_cbgs)
        V.append(cbg_selected)
        budget_used += cbg_sizes[cbg_selected]
        candidate_cbgs = np.delete(candidate_cbgs, np.where(candidate_cbgs == cbg_selected))
        candidate_cbgs = [x for x in candidate_cbgs if (budget_used + cbg_sizes[x]) <= budget]
    
    evaluate(V, all_model_kwargs)

def vax_by_age(all_model_kwargs, cbg_sizes, budget):
    '''
    Vaccinate cbgs with the highest median age first.
    '''
    median_age = all_model_kwargs['extra_kwargs']['cbg_demographics']['median_age']
    mean_med_age = np.nanmean(median_age)
    inds = np.where(np.isnan(median_age))
    median_age[inds] = mean_med_age
    
    idx_and_age = [[i,v] for i, v in enumerate(median_age)]
    idx_and_age.sort(key = lambda x:x[1], reverse=True)

    budget_used, V = 0, []
    while len(idx_and_age) > 0:
        V.append(idx_and_age[0][0])
        budget_used += cbg_sizes[idx_and_age[0][0]]
        idx_and_age = idx_and_age[1:]
        idx_and_age = keep_valid_cbgs_within_budget(idx_and_age, cbg_sizes, budget_used, budget)
    
    evaluate(V, all_model_kwargs)

def im_with_risk_weights(all_model_kwargs, cbg_sizes, cbg_list, budget):
    median_age = all_model_kwargs['extra_kwargs']['cbg_demographics']['median_age']
    mean_med_age = np.nanmean(median_age)
    inds = np.where(np.isnan(median_age))
    median_age[inds] = mean_med_age

    # assign risk weights based on age bracket
    risk_weights = [1.0] * len(cbg_sizes)
    for i, v in enumerate(cbg_sizes): # add id of the cbg to the dictionary
        if median_age[i] < 40.0 and median_age[i] >= 30.0:
            risk_weights[i] *= 3.5
        elif median_age[i] < 50.0 and median_age[i] >= 40.0:
            risk_weights[i] *= 10.0
        elif median_age[i] < 65.0 and median_age[i] >= 50.0:
            risk_weights[i] *= 25.0
        elif median_age[i] < 75.0 and median_age[i] >= 65.0:
            risk_weights[i] *= 60.0
        elif median_age[i] < 85.0 and median_age[i] >= 75.0:
            risk_weights[i] *= 140.0
        elif median_age[i] >= 85.0:
            risk_weights[i] *= 350.0

    V = vax_with_influence_maximization(all_model_kwargs,
                                        cbg_list=cbg_list,
                                        budget=budget,
                                        risk_weights=risk_weights)
    
    evaluate(V, all_model_kwargs)

def im_with_income(all_model_kwargs, cbg_sizes, cbg_list, budget):
    # median household income, give nans the mean
    median_income = all_model_kwargs['extra_kwargs']['cbg_demographics']['median_household_income']
    mean_median_income = np.nanmean(median_income)
    inds = np.where(np.isnan(median_income))
    median_income[inds] = mean_median_income

    quartile_budgets={}
    quartile_lookup={}

    # income quantiles 
    quartiles = list(np.quantile(median_income, [0, .25, .5, .75, 1.]))
    for i in range(len(quartiles)-1):
        # get list of cbgs whose income is in that quartile
        quartile_cbg_list = [j for [j, v] in enumerate(median_income) if v > quartiles[i] and v <= quartiles[i+1]]
        # get population in that quartile
        quartile_population_fraction = np.sum([cbg_sizes[x] for x in quartile_cbg_list]) / np.sum(cbg_sizes)
        quartile_budget = int(quartile_population_fraction*budget)
        quartile_budgets[str(i)] = quartile_budget
        
        for cbg in quartile_cbg_list:
            quartile_lookup[cbg] = str(i)
    
    for cbg in cbg_list:
        if median_income[cbg] == quartiles[0]:
            quartile_lookup[cbg] = '0'

    V = vax_with_influence_maximization(all_model_kwargs,
                                  cbg_list,
                                  budget,
                                  risk_weights=None,
                                  race_budgets=None,
                                  income_budgets=quartile_budgets,
                                  quartile_lookup=quartile_lookup)
    
    evaluate(V, all_model_kwargs)

def combine_imi_ima(all_model_kwargs, cbg_sizes, cbg_list, budget):
    # median household income, give nans the mean
    median_income = all_model_kwargs['extra_kwargs']['cbg_demographics']['median_household_income']
    mean_median_income = np.nanmean(median_income)
    inds = np.where(np.isnan(median_income))
    median_income[inds] = mean_median_income

    quartile_budgets={}
    quartile_lookup={}
    # income quantiles 
    quartiles = list(np.quantile(median_income, [0, .25, .5, .75, 1.]))
    for i in range(len(quartiles)-1):
        # get list of cbgs whose income is in that quartile
        quartile_cbg_list = [j for [j, v] in enumerate(median_income) if v > quartiles[i] and v <= quartiles[i+1]]
        # get population in that quartile
        quartile_population_fraction = np.sum([cbg_sizes[x] for x in quartile_cbg_list]) / np.sum(cbg_sizes)
        quartile_budget = int(quartile_population_fraction*budget)
        quartile_budgets[str(i)] = quartile_budget
        
        for cbg in quartile_cbg_list:
            quartile_lookup[cbg] = str(i)

    for cbg in cbg_list:
        if median_income[cbg] == quartiles[0]:
            quartile_lookup[cbg] = '0'

    # calculate risk weights from median age
    median_age = all_model_kwargs['extra_kwargs']['cbg_demographics']['median_age']
    mean_med_age = np.nanmean(median_age)
    inds = np.where(np.isnan(median_age))
    median_age[inds] = mean_med_age

    # assign risk weights based on age bracket
    risk_weights = [1.0] * len(cbg_sizes)
    for i, v in enumerate(cbg_sizes): # add id of the cbg to the dictionary
        if median_age[i] < 40.0 and median_age[i] >= 30.0:
            risk_weights[i] *= 3.5
        elif median_age[i] < 50.0 and median_age[i] >= 40.0:
            risk_weights[i] *= 10.0
        elif median_age[i] < 65.0 and median_age[i] >= 50.0:
            risk_weights[i] *= 25.0
        elif median_age[i] < 75.0 and median_age[i] >= 65.0:
            risk_weights[i] *= 60.0
        elif median_age[i] < 85.0 and median_age[i] >= 75.0:
            risk_weights[i] *= 140.0
        elif median_age[i] >= 85.0:
            risk_weights[i] *= 350.0

    V = vax_with_influence_maximization(all_model_kwargs,
                                  cbg_list,
                                  budget,
                                  risk_weights=risk_weights,
                                  race_budgets=None,
                                  income_budgets=quartile_budgets,
                                  quartile_lookup=quartile_lookup)
    
    evaluate(V, all_model_kwargs)

def combine_imr_ima(all_model_kwargs, cbg_sizes, cbg_list, budget):
    cbg_demographics = all_model_kwargs['extra_kwargs']['cbg_demographics']
    race_budgets = {}
    for race in ['p_white', 'p_black', 'p_asian', 'p_other']:
        p_race = cbg_demographics[race]
        # calculate budget per race
        total_p_race = np.sum(np.multiply(p_race, cbg_sizes)) / np.sum(cbg_sizes)
        race_budgets[race] = int(budget*total_p_race)

    # calculate risk weights from median age
    median_age = all_model_kwargs['extra_kwargs']['cbg_demographics']['median_age']
    mean_med_age = np.nanmean(median_age)
    inds = np.where(np.isnan(median_age))
    median_age[inds] = mean_med_age

    # assign risk weights based on age bracket
    risk_weights = [1.0] * len(cbg_sizes)
    for i, v in enumerate(cbg_sizes): # add id of the cbg to the dictionary
        if median_age[i] < 40.0 and median_age[i] >= 30.0:
            risk_weights[i] *= 3.5
        elif median_age[i] < 50.0 and median_age[i] >= 40.0:
            risk_weights[i] *= 10.0
        elif median_age[i] < 65.0 and median_age[i] >= 50.0:
            risk_weights[i] *= 25.0
        elif median_age[i] < 75.0 and median_age[i] >= 65.0:
            risk_weights[i] *= 60.0
        elif median_age[i] < 85.0 and median_age[i] >= 75.0:
            risk_weights[i] *= 140.0
        elif median_age[i] >= 85.0:
            risk_weights[i] *= 350.0

    V = vax_with_influence_maximization(all_model_kwargs,
                                  cbg_list,
                                  budget,
                                  risk_weights=risk_weights,
                                  race_budgets=race_budgets)
    
    evaluate(V, all_model_kwargs)

def setup_vaccine_experiment(all_model_kwargs):
    '''
    Distribute appropriate vaccination experiment.
    '''
    cbg_sizes = all_model_kwargs['extra_kwargs']['cbg_sizes']
    cbg_list = np.arange(1, len(cbg_sizes))
    budget = int(np.sum(cbg_sizes)*0.05)

    # add more statistics to cbg_demographics
    all_model_kwargs = add_to_demographic_stats(all_model_kwargs, cbg_sizes)

    # run corresponding experiment
    if all_model_kwargs['vax_kwargs']['vax_experiment'] == 'no_vax':
        no_vax(all_model_kwargs=all_model_kwargs)
    elif all_model_kwargs['vax_kwargs']['vax_experiment'] == 'random_vax':
        random_vax(all_model_kwargs=all_model_kwargs, cbg_list=cbg_list, cbg_sizes=cbg_sizes, budget=budget)
    elif all_model_kwargs['vax_kwargs']['vax_experiment'] == 'vax_oldest':
        vax_by_age(all_model_kwargs=all_model_kwargs, cbg_sizes=cbg_sizes, budget=budget)
    elif all_model_kwargs['vax_kwargs']['vax_experiment'] == 'just_im':
        V = vax_with_influence_maximization(all_model_kwargs=all_model_kwargs,
                                            cbg_list=cbg_list,
                                            budget=budget,
                                            risk_weights=None)
        evaluate(V, all_model_kwargs=all_model_kwargs)
    elif all_model_kwargs['vax_kwargs']['vax_experiment'] == 'im_eq_treatment':
        equal_treatment(all_model_kwargs=all_model_kwargs, cbg_list=cbg_list, budget=budget)
    elif all_model_kwargs['vax_kwargs']['vax_experiment'] == 'im_with_age':
        im_with_risk_weights(all_model_kwargs=all_model_kwargs, 
                             cbg_sizes=cbg_sizes, cbg_list=cbg_list, budget=budget)
    elif all_model_kwargs['vax_kwargs']['vax_experiment'] == 'im_with_income':
        im_with_income(all_model_kwargs=all_model_kwargs,
                            cbg_sizes=cbg_sizes, cbg_list=cbg_list, budget=budget)
    elif all_model_kwargs['vax_kwargs']['vax_experiment'] == 'imi_ima':
        combine_imi_ima(all_model_kwargs=all_model_kwargs,
                        cbg_sizes=cbg_sizes, cbg_list=cbg_list, budget=budget) 
    elif all_model_kwargs['vax_kwargs']['vax_experiment'] == 'imr_ima':
        combine_imr_ima(all_model_kwargs=all_model_kwargs,
                        cbg_sizes=cbg_sizes, cbg_list=cbg_list, budget=budget)

