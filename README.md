# Fair Vaccination Strategies with Influence Maximization
Code to run experiments in our 2023 paper: Promoting Fair Vaccination Strategies through Influence Maximization: A Case Study on COVID-19 Spread

## Data to download
### Safegraph data from Dewey
Safegraph now provides their mobility data through the Dewey platform. The data is available for research groups through a paid subscription.
We use the following mobility patterns data for the 5 week period from 2nd March to 5th April 2023:
* **weekly patterns** - weekly data of visits between census block groups (CBGs) and points of interest (POIs). This has sparsity issues so is supplemented with monthly patterns.
* **monthly patterns** - aggregated monthly data of visits between census block groups (CBGs) and points of interest (POIs)
* **neighborhood patterns** - we use this data on visits to CBGs to work out the proportion of a CBG staying at home. This is to replace the social distancing data which Safegraph no longer provides ([documentation found here](https://docs.safegraph.com/docs/neighborhood-patterns)).
### Census data
* We supplement this with data from [NHGIS IPUMS](https://data2.nhgis.org/main) to collect the 1-year 2018 estimates of the populations of each CBG.
* We also use the same NHGIS platform to collect the median age of CBGs.
* The ACS 5 year 2013-2017 file and the mapping from counties to MSAs is available on Stanford's [covid-mobility repository](https://github.com/snap-stanford/covid-mobility). 
### NY Times data
* We use values of real case counts from NY Times from 2020, [available here](https://github.com/nytimes/covid-19-data).

## Environment
We use a conda environment which can be repliated using the command `conda env create -f vax_env.yml`. This environment should be activated before running any code.

## Constructing mobility networks
In the `construct_mobility_networks` folder, we include the code from [Chang et al. 2020](https://github.com/snap-stanford/covid-mobility) to construct the networks of visits from CBGs to POIs (there is currenly no license for on code). They use an iterative proportional fitting procedure (IPFP) to construct the hourly edge weights. We include our modifications due to changes in what data is made available through Safegraph. Most notably, the social distancing data is no longer available, which was previously used to estimate the fraction of a CBG who were not at home. Instead, we provide code for how to estimate this using the neighborhood-patterns data from Safegraph. Additionally, we are only creating mobility networks for three MSAs - Philadelphia, New York and Chicago.

1. Run `filter_patterns_data.py` - this includes some pre-processing steps to deal with changes after Safegraph data migrated to the Dewey platform. We also filter only for the three MSAs we are interested in.
2. Run `compute_cbg_out_proportions.py`- this is where we use the neighborhood patterns data to replace the social distancing metrics estimations of the proportion of CBGs not at home.
3. Run `process_safegraph_data.py` - from here onwards, we follow the procedure from Chang et al.
4. Run python `model_experiments.py run_many_models_in_parallel just_save_ipf_output`. This will start one job for each MSA which generates the hourly visit matrices through the iterative proportional fitting procedure (IPFP).
5. Determine plausible ranges for model parameters over which to conduct grid search. Run `python model_experiments.py run_many_models_in_parallel calibrate_r0`.
6. Conduct grid search to find models which best fit case counts. Run `python model_experiments.py run_many_models_in_parallel normal_grid_search`.

## Running influence maximization experiments
In the `vaccination_experiments` folder, we setup and conduct our method for vaccinating with influence maximization. Our main contribution is in `vaccination methods.py`.
Run `influence_maximization_experiments.py` followed by your choice of MSA and vaccination experiment. E.g. `influence_maximization_experiments.py Philadelphia just_im`. The choice of experiments are: 
* `no_vax`: free-spreading Covid-19 with no vaccination strategy.
* `random_vax`: randomly selecting CBGs to vaccinate up to the vaccine budget.
* `vax_oldest`: this is our proxy for the current strategy, which selects CBGs with the highest median age up to the budget.
* `just_im`: selecting CBGs to vaccinate using influence maximization, with no additional fairness constraints.
* `im_eq_treatment`: influence maximization with equal treatment for racial groups.
* `im_with_income`: influence maximization with equal treatment for income groups.
* `im_with_age`: influence maximization with age-associated risk weights using the CBG median age.
* `imr_ima`: the combination of equal treatment for racial groups and using age-associated risk weights.
* `imi_ima`: the combination of equal treatment for income groups and using age-associated risk weights.
