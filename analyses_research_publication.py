import pathlib
import pandas as pd
from config.data_paths import data_paths
from typing import Any, Dict, List, Optional, Tuple, Union
import seaborn as sns
import matplotlib.pyplot as plt
from library.clustering_functions import compute_pca_and_kmeans
from library.table_one import MakeTableOne
from library.regression_models_class import ComputeRegression
from library.hypothesis_testing import HypothesisTesting

def plot_variable(data, x: str,
                  variable_type: str = "binary",
                  hue: Optional[str] = None,
                  title: Optional[str] = None,
                  xlabel: Optional[str] = None,
                  figsize: Tuple[int, int] = (8, 6)):
    """
    Plots binary or continuous variables with custom styles using Seaborn.

    Parameters:
    - data: DataFrame containing the data to plot.
    - x: str, column name for the x-axis variable.
    - variable_type: str, "binary" or "continuous", type of variable for custom styling.
    - hue: Optional[str], column name for an optional hue variable.
    - title: Optional[str], title for the plot.
    - xlabel: Optional[str], custom label for the x-axis.
    - figsize: Tuple[int, int], figure size for the plot.
    """
    sns.set_theme(style="whitegrid")  # Apply a pleasant Seaborn theme

    # Set up the figure
    plt.figure(figsize=figsize)

    if variable_type == "binary":
        # Use countplot with unique colors for each bar by setting hue to x if hue is None
        palette = sns.color_palette("pastel", n_colors=len(data[x].unique()))
        ax = sns.countplot(data=data, x=x, hue=hue if hue else x, palette=palette, dodge=False)

        # Remove the legend if we set hue to x for coloring
        if hue is None:
            ax.legend_.remove()

        # Set tick labels for binary data if values are 0 and 1
        unique_values = data[x].unique()
        if set(unique_values).issubset({0, 1}):
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['No', 'Yes'])

        # Add annotations for each bar
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
        ax.set_ylabel("Count")
    elif variable_type == "continuous":
        ax = sns.histplot(data=data, x=x, hue=hue, kde=True, stat="density")
        ax.set_ylabel("Density")

    # Set x-axis label to provided `xlabel` or default to the column name
    ax.set_xlabel(xlabel if xlabel else x)

    # Set title if provided
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Optimize layout
    plt.tight_layout()
    plt.grid(alpha=0.1)  # Light grid for better readability
    plt.show()


def summarize_data(df, columns):
    summary = {}
    for col in columns:
        if set(df[col].dropna().unique()).issubset({0, 1}):  # Check if binary (0 or 1 values)
            summary[col] = df[col].value_counts().to_dict()
        else:
            summary[col] = df[col].describe().to_dict()
    return pd.DataFrame(summary)


"""
ISI: 0-7 = not clinically significant; 8-14 = subthreshold insomnia; 15-21 = moderate insomnia; 22-28 = severe insomnia
ESS: 0-7 = unlikely that you are abnormally sleepy, 8-9= average amount of daytime sleepiness, 10-15: excessively sleepy, 16-24: excessively sleepy + seek medical attention
RLS: unlikely, unlikely (possibly in past), possible, likely - recode to binary variable where unlikely, unlikely (possibly in the past), possible are 0 (no RLS) and likely is 1 (yes)

Are there certain sleep  phenotypes that we could see in this long covid patients dataset

Biplot for 2, 4 
K-means of 3, use only parasomnia and not all of them 
s

"""


def plot_mixed_correlation_heatmap(
        data: pd.DataFrame,
        binary_cols: List[str],
        cont_cols: List[str],
        title: str = "Correlation Heatmap for Mixed Data Types",
        output_path: pathlib.Path = None
) -> None:
    """
    Correlation heat map for when we have data types of type binary and continuous.
    :param data: dataframe containing the data to plot.
    :param binary_cols: list of columns of type continuous
    :param cont_cols: list of columns of type binary
    :param title: title of the heatmap
    :return:
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pointbiserialr, pearsonr
    from sklearn.metrics import matthews_corrcoef
    import seaborn as sns
    import numpy as np

    # Ensure binary columns are strictly binary
    for col in binary_cols:
        unique_vals = data[col].unique()
        if not np.array_equal(np.unique(unique_vals), [0, 1]):
            raise ValueError(f"Column {col} contains non-binary values: {unique_vals}")

    # Initialize a correlation matrix with NaN
    columns = binary_cols + cont_cols
    corr_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)

    # Fill the correlation matrix
    for i in columns:
        for j in data.columns:
            if i == j:
                # Correlation with itself is always 1
                corr_matrix.loc[i, j] = 1.0
            elif i in binary_cols and j in binary_cols:
                # Use Matthews correlation for binary-binary pairs
                corr_matrix.loc[i, j] = matthews_corrcoef(data[i], data[j])
            elif i in cont_cols and j in cont_cols:
                # Use Pearson correlation for continuous-continuous pairs
                corr_matrix.loc[i, j] = pearsonr(data[i], data[j])[0]
            elif (i in binary_cols and j in cont_cols) or (i in cont_cols and j in binary_cols):
                # Use Point-Biserial correlation for binary-continuous pairs
                corr_matrix.loc[i, j] = pointbiserialr(data[i], data[j])[0] if i in binary_cols else \
                    pointbiserialr(data[j], data[i])[0]

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        mask=corr_matrix.isna(),
        cbar_kws={'label': 'Correlation'}
    )
    plt.title(title)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    # Step 1: Load the dataset
    df_data = pd.read_csv(data_paths.get('pp_data').get('asq_covid_ehr_cortisol'))
    # Step 2: Load the data dictionary
    # df_data['dem_0500'].replace({2:1}, inplace=True)
    # Preliminary analysis and variable preparation

    # Extract and clean relevant columns for sleep scales and recoding
    sleep_scales = df_data[['ess_0900', 'isi_score', 'rls_probability']]
    parasomnia_columns = ['par_0205', 'par_0305', 'par_0505', 'par_0605', 'par_1005']
    sleep_breathing_columns = ['map_0100', 'map_0300', 'map_0600']

    columns_all = [*sleep_scales.columns]  + sleep_breathing_columns # +  parasomnia_columns

    col_par = [col for col in df_data.columns if col.startswith('par')]
    # Recode RLS_probability as binary (1 if likely, 0 otherwise)
    df_data['rls_binary'] = df_data['rls_probability'].apply(lambda x: 0 if 'Unlikely' in x else 1)

    # Create parasomnia variable (1 if any parasomnia activity reported (>0), else 0)
    df_data['parasomnia'] = df_data[parasomnia_columns].apply(lambda row: 1 if (row > 0).any() else 0, axis=1)

    # Create sleep-related breathing disorder variable (1 if symptoms present, 0 otherwise)
    df_data['breathing_symptoms'] = df_data[sleep_breathing_columns].apply(lambda row: 1 if (row > 0).any() else 0, axis=1)
    df_data[sleep_breathing_columns] = df_data[sleep_breathing_columns].clip(lower=0)

    # TODO: extreme circadian
    # Display a summary of the recoded variables
    df_data[['ess_0900', 'isi_score', 'rls_binary', 'parasomnia', 'breathing_symptoms']].describe()

    columns_all = columns_all + ['parasomnia', 'breathing_symptoms']

    folder_results = data_paths.get('results').joinpath('publication_results')

    # %% Create general tables one
    col_maper_tab_one = {
        'dem_0800': 'BMI',
        'dem_0110': 'Age',
        'dem_1000': 'Race',
        'dem_0500': 'Gender',
        'mdhx_5800': 'Asthma',
        'mdhx_6300': 'High cholesterol',
        'mdhx_6310': 'Diabetes Type 2',
        'fosq_1100':'FOSQ Score',
        'mdhx_5700': 'Hypertension',
        'mdhx_5710': 'Chronic heart failure',

    }
    continuous_var = ['dem_0800',  # BMI
                      'dem_0110',  # Age
                      'ess_0900',
                      'isi_score',
                      'map_score',
                      'fosq_1100']
    var_cate = [
            'dem_0500',  # Gender
            'insomnia',
            'mdhx_5800',
            'mdhx_6300',
            'mdhx_6310',
            'mdhx_5700',
            'mdhx_5710',
            'dem_1000',
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

    # %% Statistical test of simple complains

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
    vars_of_interest = [var for var in vars_of_interest if var in df_data.columns]

    df_data[vars_of_interest] = df_data[vars_of_interest].apply(pd.to_numeric, errors='coerce')
    df_nan_count = (df_data[vars_of_interest].isna().sum() / df_data.shape[0]) * 100
    df_nan_count = df_nan_count.reset_index()
    df_nan_count.columns = ['Variable', 'NaN Percentage']

    df_for_pca = df_data[vars_of_interest].dropna()

    # run pca and kmeans
    (q2_pca_components,
     q2_loadings,
     q2_cumulative_variance,
     q2_explained_variance,
     q2_series_clusters) = compute_pca_and_kmeans(df=df_for_pca,
                           n_clusters_kmeans=2,
                           n_clusters_biplot=4,
                           figsize_biplot=(16,8)
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
    vars_of_interest = ['dem_0500',
                        'ess_0900',
                        'isi_score',
                        'rls_binary',
                        'parasomnia',
                        'breathing_symptoms']
    vars_of_interest = vars_of_interest # + parasomnia_columns

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

    # Search clinical association by the correlation of the features and drawing a network
    categorical_var = [var for var in categorical_var if not var in binary_vars]
    # q3_network_results = correlation_and_network_analysis_by_type(data=df_q3_data,
    #                                  continuous_vars=continuous_var,
    #                                  ordinal_vars=categorical_var,
    #                                  binary_vars=binary_vars,
    #                                 correlation_threshold=0.3
    #                                  )

    # Access the results
    print("Degree Centrality:")
    print(q3_network_results["degree_centrality"])

    print("Betweenness Centrality:")
    print(q3_network_results["betweenness_centrality"])


    # %% 4.	Research Question: Do specific sleep disturbances tend to co-occur?
    vars_of_interest = [col for col in df_data if col.startswith('mdhx_sleep_problem')]

    df_data_for_pca = df_data[vars_of_interest].dropna()

    # run pca and kmeans
    (q4_pca_components,
     q4_loadings,
     q4_cumulative_variance,
     q4_explained_variance,
     q4_series_clusters) = compute_pca_and_kmeans(df=df_data_for_pca,
                           n_clusters_kmeans=2,
                           n_clusters_biplot=5,
                           figsize_biplot=(16,10)
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
    vars_of_interest = ['dem_0500',  # gender
                        'dem_0800',  # BMI
                        'dem_1000',  # race
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
                         # 'dem_1000',  # race
                        'dem_0500',  # gender
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

    continuous_var = ['dem_0800',  # BMI
                      'dem_0110',  # Age
                      'ess_0900',
                      'isi_score' ]

    table_one_race = MakeTableOne(df=df_data,
                 categorical_var=categorical_var,
                 continuous_var=continuous_var,
                 # strata='dem_1000'
                                  )

    df_table_one_race = table_one_race.create_table()
    df_table_one_race = table_one_race.group_variables_table(df_table_one_race)


    # %% 7.	Research Question: Does the duration of the Long COVID symptoms influence sleep disturbance severity and type?
    df_data['days_with_covid'] = pd.to_datetime(df_data['date_admin_clinic']) - pd.to_datetime(df_data['date_incidence_covid'])
    df_data['days_with_covid'] = df_data['days_with_covid'].astype(str).apply(lambda x: x.split(' ')[0])
    df_data['days_with_covid'] = df_data['days_with_covid'].replace('NaT', None)
    df_data['days_with_covid'] = pd.to_numeric(df_data['days_with_covid'], errors='coerce').fillna(0).astype(int)

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

    var_adjust = ['dem_0110',  # Age
                  'dem_0800',  # BMI
                  'dem_1000',  # race
                  'dem_0500',  # gender
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
        'dem_0110',
        'dem_0800'
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

    # %% Better clustering method













