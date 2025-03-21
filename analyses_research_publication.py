import pathlib
import pandas as pd
from config.data_paths import data_paths
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from library.clustering_functions import compute_pca_and_kmeans
from library.table_one import MakeTableOne
from library.regression_models_class import ComputeRegression
from library.depreciated.hypothesis_testing import HypothesisTesting
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from scipy.stats import spearmanr
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.multitest import multipletests
from tabulate import tabulate
from library.visualizations import plot_variable, plot_mixed_correlation_heatmap, summarize_data, plot_symptom_severity
from library.stat_tests import stats_test_binary_symptoms, stats_test_ordinal_symptoms, correct_pvalues, run_regression_models


"""
ISI: 0-7 = not clinically significant; 8-14 = subthreshold insomnia; 15-21 = moderate insomnia; 22-28 = severe insomnia
ESS: 0-7 = unlikely that you are abnormally sleepy, 8-9= average amount of daytime sleepiness, 10-15: excessively sleepy, 16-24: excessively sleepy + seek medical attention
RLS: unlikely, unlikely (possibly in past), possible, likely - recode to binary variable where unlikely, unlikely (possibly in the past), possible are 0 (no RLS) and likely is 1 (yes)

Are there certain sleep  phenotypes that we could see in this long covid patients dataset

Biplot for 2, 4 
K-means of 3, use only parasomnia and not all of them 
s

"""

# Mapper of the columns used in the study
col_mapper = {
    'age': 'Age',
    'anx_depression': 'Anxiety/Depression',
    'bmi': 'BMI',
    'brain_fog': 'Brain Fog',
    'breathing_symptoms': 'Breathing Symptoms',
    'cir_0700_bin': 'Extreme Circadian',
    'covid_vaccine': 'Vaccine Status',
    'cough': 'Cough',
    'days_with_covid': 'Duration',
    'ess_0900': 'ESS Score',
    'ess_0900_bin': 'Excessive Daytime Sleepiness ESS',
    'fatigue': 'Fatigue',
    'fosq_1100': 'FOSQ Score',
    'gi_symptoms': 'GI Symptoms',
    'gender': 'Gender',
    'headaches': 'Headaches',
    'hospitalized': 'Hospitalized',
    'insomnia': 'Insomnia',
    'isi_score_bin': 'Insomnia ISI Score',
    'isi_score': 'ISI Score',
    'lethargy': 'Lethargy',
    'mdhx_5700': 'Hypertension',
    'mdhx_5800': 'Asthma',
    'mdhx_5710': 'Chronic Heart Failure',
    'mdhx_6300': 'High Cholesterol',
    'mdhx_6310': 'Diabetes Type 2',
    'nasal_congestion': 'Nasal Congestion',
    'parasomnia': 'Parasomnia',
    'post_exercial_malaise': 'Post-Exertional Malaise',
    'race': 'Race',
    'rls_binary': 'Restless Legs',
    'shortness_breath': 'Shortness of Breath',
    'unrefreshing_sleep': 'Unrefreshing Sleep',
    'mdhx_sleep_problem_0':'No sleep problem',
     'mdhx_sleep_problem_1':'Snoring',
     'mdhx_sleep_problem_2':'My breathing stops at nighttime',
     'mdhx_sleep_problem_3':'Sleepiness during the day',
     'mdhx_sleep_problem_4':'Unrefreshing sleep',
     'mdhx_sleep_problem_5':'Difficulty falling asleep',
     'mdhx_sleep_problem_6':'Difficulty staying asleep',
     'mdhx_sleep_problem_7':'Difficulty keeping a normal sleep schedule',
     'mdhx_sleep_problem_8':'Talk, walk, and/or other behavior in sleep',
     'mdhx_sleep_problem_9':'Nightmares',
     'mdhx_sleep_problem_10':'Act out dreams',
     'mdhx_sleep_problem_11':'Restless legs or unpleasant sensations in legs',
     'mdhx_sleep_problem_12':'Weakness in muscles when surprised',
     'mdhx_sleep_problem_13':'Other',
     'mdhx_sleep_problem_14':'Bruxism',
}




if __name__ == '__main__':
    # Step 1: Load the dataset
    df_data = pd.read_csv(data_paths.get('pp_data').get('asq_covid_ehr_cortisol'))

    sleep_scales = df_data[['ess_0900', 'isi_score', 'rls_probability']]
    parasomnia_columns = ['par_0205', 'par_0305', 'par_0505', 'par_0605', 'par_1005']
    sleep_breathing_columns = ['map_0100', 'map_0300', 'map_0600']

    columns_all = [*sleep_scales.columns]  + sleep_breathing_columns # +  parasomnia_columns

    # Display a summary of the recoded variables
    df_data[['ess_0900', 'isi_score', 'rls_binary', 'parasomnia', 'breathing_symptoms']].describe()

    columns_all = columns_all + ['parasomnia', 'breathing_symptoms']

    folder_results = data_paths.get('results').joinpath('publication_results')

    # %% Create general tables one
    col_maper_tab_one = {
        'bmi': 'BMI',
        'age': 'Age',
        'race': 'Race',
        'gender': 'Gender',
        'mdhx_5800': 'Asthma',
        'mdhx_6300': 'High cholesterol',
        'mdhx_6310': 'Diabetes Type 2',
        'fosq_1100':'FOSQ Score',
        'mdhx_5700': 'Hypertension',
        'mdhx_5710': 'Chronic heart failure',

    }
    continuous_var = ['bmi',  # BMI
                      'age',  # Age
                      'ess_0900',
                      'isi_score',
                      'map_score',
                      'fosq_1100']
    var_cate = [
            'gender',  # Gender
            'insomnia',
            'mdhx_5800',
            'mdhx_6300',
            'mdhx_6310',
            'mdhx_5700',
            'mdhx_5710',
            'race',
            'lightheadedness',
            'headaches',
            'unrefreshing_sleep',
            'gi_symptoms',
            'shortness_breath',
            'fatigue',
            'parasomnia',
            'change_in_taste',
            'rls_binary',
            'change_in_smell',
            'prevalence_sleep_symptom',
            'covid_vaccine',
            'anx_depression',
            'lethargy',
            'nasal_congestion',
            'brain_fog',
            'breathing_symptoms',
            'hospitalized',
            'cough',
            'UNREFRESHING SLEEP',
            'DIFFICULTY SLEEPING',
            'DAYTIME SLEEPINESS',
                ]
    var_cate_sorted = [(var, df_data[var].nunique()) for var in var_cate]
    var_cate_sorted = sorted(var_cate_sorted, key=lambda x: x[1])
    var_cate = [var[0] for var in var_cate_sorted]
    tab_one_all = MakeTableOne(categorical_var=var_cate,
                                  continuous_var=continuous_var,
                                  strata=None,
                                  df=df_data)

    df_tab_one = tab_one_all.create_table()
    df_tab_one_edited = tab_one_all.group_variables_table(df_tab_one)
    df_tab_one_edited['variable'] = df_tab_one_edited['variable'].replace(col_maper_tab_one)


    # %% 0 Correlation of the features
    for col in df_data[columns_all]:
        print(f'{col}: {df_data[col].unique()}')


    for col in df_data[columns_all]:
        print(f'{col}: {df_data[col].unique()}')

    binary_cols = ['rls_binary',
                   'parasomnia',
                   'breathing_symptoms',
                   ]
    cont_cols = ['ess_0900',
                 'isi_score',
                 'map_0100',
                 'map_0300',
                 'map_0600']

    plot_mixed_correlation_heatmap(data=df_data[binary_cols+cont_cols].dropna(),
                                   binary_cols=binary_cols,
                                   cont_cols=cont_cols,
                                   output_path=folder_results.joinpath('correlation_heatmap.png')
                                   )

    # %% Severity plot of selected symptoms
    col_symptoms = {
        # 'DIFFICULTY SLEEPING': 'Difficulty\nSleeping',  # DROP

        'insomnia': 'Insomnia',
        'DAYTIME SLEEPINESS': 'Daytime\nSleepiness',
        'unrefreshing_sleep': 'Unrefreshing\nSleep',

        'brain_fog': 'Brain\nFog',
        'fatigue': 'Fatigue',
        'anx_depression': 'Anxiety\nDepression',

        'lightheadedness': 'Light\nheadedness',
        'headaches': 'Headaches',
        'gi_symptoms': 'GI\nSymptoms',
        'change_in_taste': 'Change\nin Taste',

        'cough': 'Cough',
        'change_in_smell': 'Change\nin Smell',
        'nasal_congestion': 'Nasal\nCongestion',
        'shortness_breath': 'Shortness\nof Breath',

        # 'lethargy': 'Lethargy', # DROP
    }

    df_stack = df_data[col_symptoms.keys()].fillna(0).copy()
    # Ensure columns are ordinal and count occurrences by severity
    severity_counts = {}
    for symptom_ in col_symptoms.keys():
        df_stack[symptom_] = df_stack[symptom_].astype(int)
        counts = df_stack[symptom_].value_counts().sort_index()
        severity_counts[symptom_] = counts

    # Create a DataFrame with severity levels as rows and symptoms as columns
    severity_df = pd.DataFrame(severity_counts) #.fillna(0)

    # Reorder severity levels if needed
    severity_df = severity_df.sort_index()
    severity_df.rename(columns=col_symptoms, inplace=True)
    # severity_df = severity_df[sorted(severity_df.columns)]  # sort alphabeticallly
    # convert to percentages
    severity_df = severity_df.applymap(lambda x: x*100/df_data.shape[0])

    plot_symptom_severity(severity_df=severity_df,
                          df_data=df_data)


    # %% Table 2: Prevlance main 6 sleep disoders statistical test
    # We investigated the prevalence of six categories of sleep complaints:
    # - insomnia,
    # - excessive daytime sleepiness,
    # - sleep-related breathing symptoms,
    # - restless legs,
    # - parasomnia activity,
    # - extreme circadian phenotype.

    col_test = {
        # 'insomnia_bin': 'Insomnia',
        'isi_score_bin': 'Insomnia ISI Score',
        # 'DAYTIME SLEEPINESS': 'Daytime\nSleepiness',
        'ess_0900_bin': 'Excessive Daytime\nSleepiness',
        'breathing_symptoms': 'Breath\nSymptoms',
        'parasomnia': 'Parasomnia',
        'rls_binary': 'Restless legs',
        'cir_0700_bin': 'Extreme Circadian',
    }

    df_describe = df_data[[*col_test.keys()]].describe().loc[['min','max']].T
    print(tabulate(df_describe,
                       headers=df_describe.columns,
                       tablefmt='grid'))

    # Running the corrected function with simulated data
    df_strata_symptoms_by_gender = stats_test_binary_symptoms(data=df_data,
                                                             symptoms_mapping=col_test,
                                                             gender_col='gender',
                                                              binary_cut_off=3)

    df_strata_symptoms_by_gender_corrected = correct_pvalues(df=df_strata_symptoms_by_gender,
                    pvalue_columns=['P-value (Fisher)'])

    df_strata_symptoms_ordinal_by_gender = stats_test_ordinal_symptoms(data=df_data,
                                                              symptoms_mapping=col_test,
                                                              gender_col='gender')

    df_strata_symptoms_ordinal_by_gender_corrected = correct_pvalues(df=df_strata_symptoms_ordinal_by_gender,
                    pvalue_columns=['P-value (Mann-Whitney U)'])
    #%% MCF
    def group_mcf(df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies subjects based on specific symptom severity thresholds.
        For each symptom in the provided conditions, a new boolean column is added to the DataFrame,
        indicating whether the subject meets or exceeds the threshold, thus being classified as MCF (true).

        Parameters:
        df (pd.DataFrame): A DataFrame containing symptom columns.

        Returns:
        pd.DataFrame: The original DataFrame with additional boolean columns indicating MCF classification.

        Example:
        Conditions:
            - ('fatigue', 4): MCF if 'fatigue' >= 4
            - ('brain_fog', 1): MCF if 'brain_fog' >= 1
            - ('lightheadedness', 1): MCF if 'lightheadedness' >= 1
            - ('unrefreshing_sleep', 1): MCF if 'unrefreshing_sleep' >= 1

        The function appends columns named '<symptom>_mcf' with True/False values.
        """
        # List of tuples with symptom column names and threshold values
        conditions = [
            ('fatigue', 4),
            ('brain_fog', 1),
            ('lightheadedness', 1),
            ('unrefreshing_sleep', 1)
        ]

        # Apply condition to classify as MCF (True if value >= threshold)
        for column, threshold in conditions:
            if column in df.columns:
                df[f'{column}_mcf'] = df[column] >= threshold
            else:
                print(f"Warning: Column '{column}' not found in DataFrame.")

        return df


    df_data = pd.DataFrame(df_data)
    # %% Prevalence of Specific Long COVID Sleep Complaints
    # Calculate percentage of  mayor sleep complains who ansered yes
    # Check if any of the three columns have a value > 1
    subjects_with_gt_1 = (df_data[['DIFFICULTY SLEEPING',
                                   'DAYTIME SLEEPINESS',
                                   'UNREFRESHING SLEEP']] > 1).any(axis=1)
    df_data['any_sleep_compaint'] = subjects_with_gt_1.astype(int)
    percentage = (subjects_with_gt_1.sum() / len(df_data)) * 100


    col_test = ['parasomnia',  'rls_binary',
                'breathing_symptoms', 'ess_0900_bin',
                'isi_score_bin', 'insomnia_bin',
                'cir_0700_bin', 'any_sleep_compaint']
    # Calculate counts and percentages for each column where the value == 1
    counts = {}
    for col in col_test:
        df_counts = df_data.groupby(by="gender")[col].value_counts().reset_index('gender').reset_index(col)
        # Count only those who answered the response, exclude non responders for the count
        df_counts = df_counts.loc[(df_counts[col] == 1) & (df_counts['gender'].isin({0, 1})), :]
        n_total =  df_data.loc[~df_data[col].isna()].shape[0]
        df_counts['percentage'] = (df_counts['count'] * 100) / n_total
        # not grouping for the total
        df_counts['total_count'] =  df_data.loc[df_data[col] == 1].shape[0]
        df_counts['total_percentage'] =  (df_data.loc[df_data[col] == 1].shape[0] * 100) / n_total
        counts[col] = df_counts
        print(f'\n{col}\n{df_counts}')


    # Perform the appropriate test for each binary response column
    final_results = {}
    small_frequency_threshold = 5
    for col in col_test:
        contingency_table = pd.crosstab(df_data['gender'], df_data[col])
        if contingency_table.shape == (2, 2) and contingency_table.values.min() < small_frequency_threshold:
            # Fisher's Exact Test
            odds_ratio, p_value = fisher_exact(contingency_table, alternative='two-sided')
            final_results[col] = f"Fisher's Exact: p={round(p_value, 4)}"
        else:
            # Chi-Square Test
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            final_results[col] = f"Chi-Square: p={round(p_value, 4)}"

    # Convert to DataFrame for better visualization
    final_results_df = pd.DataFrame(list(final_results.items()), columns=['binary_response', 'test_result'])


    # %% 1 What is the prevalence of sleep disorders in the Long COVID Clinic, measured by validated scales?
    summary_df = summarize_data(df_data, columns=['ess_0900',
                                                  'isi_score',
                                                  'rls_binary',
                                                  'parasomnia',
                                                  'breathing_symptoms'])

    for cont_ in ['ess_0900', 'isi_score']:
        plot_variable(data=df_data,
                      x=cont_,
                      variable_type='continuous',
                      hue=None,
                      title=cont_.capitalize().replace('_', ' ')
                      )

    for bin_ in ['rls_binary', 'parasomnia', 'breathing_symptoms']:
        plot_variable(data=df_data,
                      x=bin_,
                      variable_type='binary',
                      hue=None,
                      title=cont_.capitalize().replace('_', ' '),
                      )

    # %% Rename columns before analysis
    # Rename DataFrame columns using the formalized names
    df_data.rename(columns=col_mapper, inplace=True)

    # %% 2.	Research Question: Are sleep disturbance symptoms connected with other symptoms? Do they commonly occur with
    # certain other symptoms?
    vars_of_interest = ['headaches',
                        'nasal_congestion',
                        'fatigue',
                        'brain_fog',
                        'unrefreshing_sleep',
                        'insomnia',
                        'lethargy',
                        'post_exercial_malaise',
                        'anx_depression',
                        'cough',
                        'shortness_breath',
                        'gi_symptoms']

    vars_of_interest = [col_mapper.get(var) for var in vars_of_interest]

    vars_of_interest = [var for var in vars_of_interest if var in df_data.columns]

    df_data[vars_of_interest] = df_data[vars_of_interest].apply(pd.to_numeric, errors='coerce')
    df_nan_count = (df_data[vars_of_interest].isna().sum() / df_data.shape[0]) * 100
    df_nan_count = df_nan_count.reset_index()
    df_nan_count.columns = ['Variable', 'NaN Percentage']
    print(df_nan_count)

    df_for_pca = df_data[vars_of_interest].dropna()

    # run pca and kmeans
    (q2_pca_components,
     q2_loadings,
     q2_cumulative_variance,
     q2_explained_variance,
     q2_series_clusters) = compute_pca_and_kmeans(df=df_for_pca,
                           n_clusters_kmeans=2,
                           n_clusters_biplot=4,
                           figsize_biplot=(12,8),
                            figsize_pca=(6,6),
                            figsize_kmeans=(6,6)
                           )
    # make a table one of the groups identified in the PCA and Kmeans
    df_q2_data = pd.concat([df_for_pca[vars_of_interest], q2_series_clusters,], axis=1)
    q2_cluster_tab = MakeTableOne(categorical_var=vars_of_interest,
                                  continuous_var=None,
                                  strata='Cluster',
                                  df=df_q2_data)

    df_q2_cluster_tab = q2_cluster_tab.create_table()
    df_q2_cluster_tab = q2_cluster_tab.group_variables_table(df_q2_cluster_tab)

    # Perform a hypothesis testing in the data between the identified groups
    tester = HypothesisTesting(
        df=df_q2_data,
        continuous_vars=None,
        strata='Cluster',
        discrete_vars=vars_of_interest,
        binary_vars=None,
        correction_type="bonferroni"
    )

    # Run tests
    df_q2_hypothesis_test_results = tester.run_tests()


    # %% 3.	Research Question: Do specific sleep disturbances tend to co-occur?
    vars_of_interest = ['gender',
                        'ess_0900',
                        'isi_score',
                        'rls_binary',
                        'parasomnia',
                        'breathing_symptoms']
    vars_of_interest = [col_mapper.get(var) for var in vars_of_interest]

    vars_of_interest = [var for var in vars_of_interest if var in df_data.columns]

    df_data_for_pca = df_data[vars_of_interest].dropna()
    df_data[vars_of_interest] = df_data[vars_of_interest].apply(pd.to_numeric, errors='coerce')
    df_nan_count = (df_data[vars_of_interest].isna().sum() / df_data.shape[0]) * 100
    df_nan_count = df_nan_count.reset_index()

    # run pca and kmeans
    (q3_pca_components,
    q3_loadings,
    q3_cumulative_variance,
    q3_explained_variance,
    q3_series_clusters) = compute_pca_and_kmeans(df=df_data_for_pca,
                           n_clusters_kmeans=2,
                           n_clusters_biplot=4,
                           figsize_biplot=(12,6)
                           )
    # make a table one of the groups identified in the PCA and Kmeans
    df_q3_data = pd.concat([df_data_for_pca[vars_of_interest], q3_series_clusters,], axis=1)
    continuous_var = [var for var in vars_of_interest if df_data_for_pca[var].nunique() > 7]
    categorical_var = [var for var in vars_of_interest if not var in continuous_var]
    q3_cluster_tab = MakeTableOne(categorical_var=categorical_var,
                                  continuous_var=continuous_var,
                                  strata='Cluster',
                                  df=df_q3_data)
    df_q3_cluster_tab = q3_cluster_tab.create_table()
    df_q3_cluster_tab = q3_cluster_tab.group_variables_table(df_q3_cluster_tab)

    # Perform a hypothesis testing in the data between the identified groups
    binary_vars = [var for var in vars_of_interest if set(df_q3_data[var].unique()).issubset({0, 1})]
    tester = HypothesisTesting(
        df=df_q3_data,
        continuous_vars=continuous_var,
        strata='Cluster',
        discrete_vars=categorical_var,
        binary_vars=binary_vars,
        correction_type="bonferroni"
    )
    df_q3_hypothesis_test_results = tester.run_tests()



    # %% 4.	Research Question: Do specific sleep disturbances tend to co-occur?
    vars_of_interest = [new_col for col, new_col in col_mapper.items() if col.startswith('mdhx_sleep_problem')]


    vars_of_interest = [var for var in vars_of_interest if var in df_data.columns]


    df_data_for_pca = df_data[vars_of_interest].dropna()

    # run pca and kmeans
    (q4_pca_components,
     q4_loadings,
     q4_cumulative_variance,
     q4_explained_variance,
     q4_series_clusters) = compute_pca_and_kmeans(df=df_data_for_pca,
                           n_clusters_kmeans=2,
                           n_clusters_biplot=4,
                           figsize_biplot=(20,8),
                            figsize_pca=(6,6),
                            figsize_kmeans=(6,6)
                           )
    # make a table one of the groups identified in the PCA and Kmeans
    df_q4_data = pd.concat([df_data_for_pca[vars_of_interest], q4_series_clusters, ], axis=1)
    # continuous_var = [var for var in vars_of_interest if df_data_for_pca[var].nunique() > 7]
    # categorical_var = [var for var in vars_of_interest if not var in continuous_var]
    q4_cluster_tab = MakeTableOne(categorical_var=vars_of_interest,
                                  continuous_var=None,
                                  strata='Cluster',
                                  df=df_q4_data)
    df_q4_cluster_tab = q4_cluster_tab.create_table()
    df_q4_cluster_tab = q4_cluster_tab.group_variables_table(df_q4_cluster_tab)

    # Perform a hypothesis testing in the data between the identified groups
    tester = HypothesisTesting(
        df=df_q4_data,
        continuous_vars=None,
        strata='Cluster',
        discrete_vars=None,
        binary_vars=vars_of_interest,
        correction_type="bonferroni"
    )
    df_q4_hypothesis_test_results = tester.run_tests()

    # %% 5.	Research Question: What are the risk factors for long COVID sleep disturbances?
    vars_of_interest = ['gender',  # gender
                        'bmi',  # BMI
                        'race',  # race
                        'hospitalized',
                        'covid_vaccine',
                        'parasomnia',
                        # 'ess_0900',  # asq
                        # 'isi_score',  # asq
                        # 'rls_probability',  # asq
                        # 'breathing_symptoms'  # asq
                        # 'map_0100', # highly correlated with breathing symptoms
                        # 'map_0300', # highly correlated with breathing symptoms
                        # 'map_0600' # highly correlated with breathing symptoms
                        ]

    vars_of_outcome = ['ess_0900',  # continuous
                       'isi_score',  # continuous
                       'rls_binary'  # binary
                       'breathing_symptoms'
                       ]

    vars_of_outcome_dict = {'ess_0900': 'linear',
                       'isi_score': 'linear',
                       'rls_binary': 'logistic',
                        'breathing_symptoms': 'logistic',
                       }
    df_summary_all = pd.DataFrame()
    for outcome, type in vars_of_outcome_dict.items():
        model_reg = ComputeRegression(df=df_data,
                                        covariates=vars_of_interest,
                                        regression_type=type,
                                        target_variable=outcome,
                                          )
        df_summary = model_reg.compute_regression()
        df_summary.to_csv(folder_results.joinpath(f'research_question_five_reg_{outcome}.csv'), index=False)
        df_summary_all = pd.concat([df_summary_all, df_summary])


    # %% 6.	Research Question: Are certain demographic characteristics associated with certain sleep phenotypes?
    categorical_var = ['anx_depression',
                         'brain_fog',
                         'breathing_symptoms',
                         'cough',
                         # 'race',  # race
                        'gender',  # gender
                         'fatigue',
                         'gi_symptoms',
                         'headaches',
                         'hospitalized',
                         'insomnia',
                         'lethargy',
                         'nasal_congestion',
                         'parasomnia',
                         'post_exercial_malaise',
                         'rls_binary',
                         'shortness_breath',
                         'unrefreshing_sleep']

    continuous_var = ['bmi',  # BMI
                      'age',  # Age
                      'ess_0900',
                      'isi_score' ]

    table_one_race = MakeTableOne(df=df_data,
                 categorical_var=categorical_var,
                 continuous_var=continuous_var,
                 # strata='race'
                                  )

    df_table_one_race = table_one_race.create_table()
    df_table_one_race = table_one_race.group_variables_table(df_table_one_race)


    # %% 7.	Research Question: Does the duration of the Long COVID symptoms influence sleep disturbance severity and type?
    # it's a constant column, not meaningful to implement
    # vars_of_interest = [col for col in df_data if col.startswith('mdhx_sleep_problem')]
    # df_data['prior sleep disturbance'] = df_data[vars_of_interest].apply(lambda row: 1 if any(row != 0) else 0, axis=1)

    var_outcome = 'days_with_covid'

    var_interest = ['isi_score',
                    'ess_0900',
                    'rls_binary',
                    'parasomnia',
                    'map_0100',
                    'map_0300',
                    'map_0600'
                    ]

    var_adjust = ['age',  # Age
                  'bmi',  # BMI
                  'race',  # race
                  'gender',  # gender
                  ]
    vars_all = var_interest + var_adjust + [var_outcome]

    df_q7_data = df_data.loc[(~df_data['days_with_covid'].isna()) &
                                (df_data['days_with_covid'] > 2) , vars_all].copy()

    # normalize the outcome with a Z-score normalization
    def z_score_norm(df: pd.DataFrame, col: str) -> pd.Series:
        return (df[col] - df[col].mean()) / df[col].std()

    columns_to_normalize = [
        'days_with_covid',
        'isi_score',
        'ess_0900',
        'age',
        'bmi'
    ]
    # Apply the normalization to each specified column
    for col in columns_to_normalize:
        df_q7_data[col] = z_score_norm(df_q7_data, col)

    # plt.figure(figsize=(10, 6))
    # sns.kdeplot(df_q7_data['days_with_covid'], shade=True, color="blue")
    # plt.title("KDE Plot for Days with COVID")
    # plt.xlabel("Days with COVID")
    # plt.ylabel("Density")
    # plt.grid(True)
    # plt.show()


    # Plotting
    fig, axes = plt.subplots(len(columns_to_normalize), 1, figsize=(10, 15), sharex=True)
    fig.tight_layout(pad=5.0)
    for i, col in enumerate(columns_to_normalize):
        sns.histplot(data=df_q7_data,
                     x=col,
                     kde=True, ax
                     =axes[i],
                     bins=20)
        axes[i].set_title(f'Z-Score Normalized: {col} Z-score')
        axes[i].set_ylabel('Z-Score')
        axes[i].set_xlabel('Z-score')
    plt.tight_layout()
    plt.show()

    model_reg_un_adjusted = ComputeRegression(
                                    df=df_q7_data,
                                    covariates=var_interest,
                                    regression_type='linear',
                                    target_variable=var_outcome,
                                      )
    df_summary_unadjusted = model_reg_un_adjusted.compute_regression()

    model_reg_adjusted = ComputeRegression(
                                    df=df_q7_data,
                                    covariates=var_interest + var_adjust,
                                    regression_type='linear',
                                    target_variable=var_outcome,
                                      )
    df_summary_adjusted = model_reg_adjusted.compute_regression()

    # %% regression models
    vars_of_interest = ['age',
                        'gender',
                        'bmi',
                        'race',
                        'hospitalized',
                        'covid_vaccine',
                        'days_with_covid',
                        'parasomnia',
                        'cir_0700_bin'
                        ]

    targets_of_interest = ['isi_score_bin',  # target 1,
                        'ess_0900_bin',  # target 2
                        'rls_binary',  # target 3
                        'breathing_symptoms',  # target 4
                        # 'cir_0700_bin',  # target 5
                        ]

    vars_of_interest = [col_mapper.get(var) for var in
                        vars_of_interest + targets_of_interest]
    targets_of_interest = [col_mapper.get(var) for var in targets_of_interest]
    vars_of_interest = [var for var in vars_of_interest if var in df_data.columns]
    df_model = df_data[vars_of_interest].copy()
    # low reponses for race 5 and 3, so we merge into other category
    df_model['Race'] = df_model['Race'].replace({3: 5})

    # multicolinearty
    cont_cols = ['Age', 'BMI', 'Duration', 'Race']
    plot_mixed_correlation_heatmap(data=df_model,
                                   binary_cols=[col for col in vars_of_interest if not col in cont_cols],
                                   cont_cols=cont_cols,
                                   output_path=None
                                   )


    df_reg_results = run_regression_models(df=df_model.copy(), \
                                           targets=targets_of_interest)
    # format the columns and names
    # df_reg_results.Variable.unique()
    mapping_variables_regression = {
        "Intercept": "Intercept",
        "C(Gender, Treatment(reference=0))[T.1]": "Male",
        "C(Hospitalized, Treatment(reference=0))[T.1]": "Hospitalized (Yes)",
        "C(Q('Vaccine Status'), Treatment(reference=0))[T.1]": "Vaccine (Yes)",
        "C(Race, Treatment(reference=0))[T.2]": "Race 2",
        "C(Race, Treatment(reference=0))[T.4]": "Race 4",
        "C(Race, Treatment(reference=0))[T.5]": "Race 5",
        "C(Race, Treatment(reference=0))[T.6]": "Race 6",
        "Age": "Age",
        "Duration": "Duration",
        "BMI": "BMI",
        "C(Q('Extreme Circadian'), Treatment(reference=0))[T.1]": 'Extreme Circadian (Yes)',
        "C(Parasomnia, Treatment(reference=0))[T.1]": 'Parasomnia (Yes)',
    }
    df_reg_results['Variable'] = df_reg_results.Variable.map(mapping_variables_regression)
    df_reg_results['p-value'] = df_reg_results['p-value'].apply(
        lambda p: f"{float(p):.4}" if float(p) >= 0.001 else "p < 0.001"
    )
    df_reg_results['OR (95% CI)'] = df_reg_results['OR (95% CI)'].apply(
        lambda s: s.replace(" ", "\n", 1)
    )
