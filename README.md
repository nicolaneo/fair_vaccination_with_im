# Fair Vaccination Strategies with Influence Maximization
Code to run experiments in our 2023 paper: Promoting Fair Vaccination Strategies through Influence Maximization: A Case Study on COVID-19 Spread

## Data
### Safegraph data from Dewey
### Census data
### NY times data

## Constructing mobility networks
In the `construct_mobility_networks` folder, we include the code used from Chang et al. 2020 to construct the networks of visits from CBGs to POIs (there is currenly no license for their code). We include our modifications due to changes in what data is made available through Safegraph. Most notably, the social distancing data is no longer available, which was previously used to estimate the fraction of a CBG who were not at home. Instead, we provide code for how to estimate this using the neighborhood-patterns data from Safegraph. Additionally, we are only creating mobility networks for 3 MSAs - Philadelphia, New York and Chicago.
The following steps are from [Chang et al.'s repository](https://github.com/snap-stanford/covid-mobility). 

1. Run `process_safegraph_data.py`
2. To select only the data for our 3 desired MSAs, we filter the Safegraph files. We also restructure them to make them consistent with the previous work. `read_cluster_patterns.py`
3. Generate the hourly visit matrices by running IPFP. Run python `model_experiments.py run_many_models_in_parallel just_save_ipf_output`. This will start one job for each MSA which generates the hourly visit matrices through the iterative proportional fitting procedure (IPFP).
4. Determine plausible ranges for model parameters over which to conduct grid search. Run `python model_experiments.py run_many_models_in_parallel calibrate_r0`.
5. Conduct grid search to find models which best fit case counts. Run `python model_experiments.py run_many_models_in_parallel normal_grid_search`.


## Running influence maximization experiments
1. Run `influence_maximization_experiments.py` followed by your choice of MSA and vaccination experiment. E.g. `influence_maximization_experiments.py Philadelphia just_im`.
