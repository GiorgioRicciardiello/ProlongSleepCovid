import pathlib

import pandas as pd
import numpy as np
from typing import Union
from config.data_paths import data_paths, multi_response_col
from library.utils import NameDateProcessor, FuzzySearch, compute_sparse_encoding
from config.columns_use import columns_interest, col_ehr

def pre_processing(df: pd.DataFrame,
                   columns_interest: list) -> pd.DataFrame:
    """
    Pre-processing for the columns of interest in the sleep covid study
    :param df:
    :return:
    """

    def check_columns(df: pd.DataFrame, columns: list[str]) -> list:
        col_missing = []
        for col in columns:
            if col in df.columns:
                continue
            else:
                col_missing.append(col)
        return col_missing

    col_missing = check_columns(df=df, columns=columns_interest)
    if len(col_missing):
        raise ValueError(f'Columns not found: {col_missing}')

    ### MAP
    col_map = [col for col in columns_interest if 'map_' in col]
    # Replace -55 with 0 for the selected columns
    df[col_map] = df[col_map].replace(-55, 0)
    df[col_map] = df[col_map].astype(int)
    ### PAR
    col_par = [col for col in columns_interest if 'par_' in col]
    # Replace -88 with 0 for the selected columns
    for col in col_par:
        df[col] = df[col].replace({-88: 0,
                                   -55: 0,
                                   -66: 0
                                   })
        if ~pd.isna(df[col]).any():
            df[col] = df[col].astype(int)

    # par_0505
    df['par_0505'] = df['par_0505'].apply(lambda x: int(np.ceil(x)))

    ### RLS
    # 0,1,2 - > no , Unlikely, Unlikely (possibly in past)
    rls_coding = {'Unlikely': 0,
                  'Unlikely (possibly in past)': 1,
                  'Possible': 2,
                  'Likely': 3
                  }

    df['rls_probability'] = df['rls_probability'].map(rls_coding)

    # correct the rest data types
    for col in columns_interest:
        if ~pd.isna(df[col]).any():
            df[col] = df[col].astype(int)

    return df

def process_time_column(df:pd.DataFrame,
                        col:str)->pd.DataFrame:
    """
    creates the sine and cosine transformations of a time column and computes their arctan2 to
    generate an angle series:
    :param df:
    :param col:
    :return:
    """
    df = df.copy()

    # Convert the column to datetime type & then convert to hour of day
    df[col + '_angle'] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce').dt.hour

    # Normalize the time between 0 and 1
    df[col + '_angle'] = df[col + '_angle'].divide(24)

    # Applying sine and cosine transformation
    df[col + '_sin'] = np.sin(2 * np.pi * df[col + '_angle'])
    df[col + '_cos'] = np.cos(2 * np.pi * df[col + '_angle'])

    # Calculate angle
    df[col + '_angle'] = np.arctan2(df[col + '_sin'], df[col + '_cos'])
    # df[['Time Cortisol Collected', 'Time Cortisol Collected_angle']]

    df.drop(columns=[col + '_sin', col + '_cos', col],
            inplace=True)
    df.rename(columns={col + '_angle': col}, inplace=True)
    return df


def categorize_ess(ess_score):
    """categorize ESS scores"""
    mapper_ess = {
        0: "Unlikely abnormally sleepy",
        1: "Average daytime sleepiness",
        2: "Excessively sleepy",
        3: "Excessively sleepy + seek medical attention",
        4: "Invalid ESS score",
    }
    if ess_score <= 7:
        return 0
    elif 8 <= ess_score <= 9:
        return 1
    elif 10 <= ess_score <= 15:
        return 2
    elif 16 <= ess_score <= 24:
        return 3
    else:
        return 4

def categorize_isi(isi_score):
    """categorize ISI scores"""
    mapper_isi = {
        0: "Not clinically significant",
        1: "Subthreshold insomnia",
        2: "Moderate insomnia",
        3: "Severe insomnia",
        4: "Invalid ISI score",
    }
    if isi_score <= 7:
        return 0
    elif 8 <= isi_score <= 14:
        return 1
    elif 15 <= isi_score <= 21:
        return 2
    elif 22 <= isi_score <= 28:
        return 3
    else:
        return 4

def categorize_circadian(cir_0900):
    """categorize circadian RMEQ scores"""
    mapper_circadian = {
        0: "Definitely evening tendency",
        1: "Moderately evening type",
        2: "Neither type",
        3: "Moderately morning type",
        4: "Definitely morning type",
        5: "Invalid circadian score"
    }
    if 4 <= cir_0900 <= 7:
        return 0
    elif 8 <= cir_0900 <= 11:
        return 1
    elif 12 <= cir_0900 <= 17:
        return 2
    elif 18 <= cir_0900 <= 21:
        return 3
    elif 22 <= cir_0900 <= 25:
        return 4
    else:
        return 5


if __name__ == "__main__":
    # df_raw_data = pd.read_excel(data_paths.get('raw_data').get('results_pull'))
    df_asq = pd.read_csv(data_paths.get('raw_data').get('asq'))
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df_asq.columns:
        df_asq.drop(columns='Unnamed: 0', inplace=True)

    name_mapper = {asq_db: ehr_db for asq_db, ehr_db in zip(df_asq['name'], df_asq['subject_name']) }
    # %% clean the ASQ dataset
    # only completed records
    df_asq = df_asq.loc[df_asq['next_module'] == 'complete', :]
    # assign 1=male, 0=females
    df_asq.loc[~df_asq['dem_0500'].isin({0, 1}), 'dem_0500'] = 0
    # df_asq = df_asq.loc[df_asq['dem_0500'].isin({0, 1}), :]
    print(df_asq['dem_0500'].value_counts())
    df_asq['dem_0500'] = 1 - df_asq['dem_0500']
    print(df_asq['dem_0500'].value_counts())

    # subjects that completed more than one ASQ. Select the last one they completed
    df_unique_subjects = df_asq[['name', 'date_of_birth']].copy()
    df_unique_subjects = df_unique_subjects.drop_duplicates(subset=list(df_unique_subjects.columns),
                                                            inplace=False)
    df_unique_subjects.reset_index(inplace=True, drop=True)

    df_asq_selected_records = pd.DataFrame()
    # fuzzy search to search matches
    for index, row in df_unique_subjects.iterrows():
        # index = 1
        # row = df_unique_subjects.loc[index, :]
        # Create a DataFrame for the current row
        row_frame = pd.DataFrame(row).transpose()

        # search how many matches of the same subject we have in the dataset
        fuzzy_search = FuzzySearch(asq_df=df_asq,
                                   subjects_df=row_frame,
                                   )
        matches = fuzzy_search.search_by_name_dob_matches(method='fuzzy', fuzzy_filter=93)

        if matches.shape[0] == 1:
            matches = df_asq.loc[df_asq['survey_id'] == matches.asq_survey_id.values[0]]
            df_asq_selected_records = pd.concat([df_asq_selected_records, matches], axis=0)
            # one single match
            continue

        # get the most recent ASQ
        df_subsets = df_asq.loc[df_asq['survey_id'].isin(list(matches.asq_survey_id.values))]
        df_subsets = df_subsets.sort_values(by='start_time', ascending=False)
        most_recent_row = df_subsets.iloc[[0]]
        df_asq_selected_records = pd.concat([df_asq_selected_records, most_recent_row], axis=0)

    # because two names are searched we must remove the pair it found
    df_asq_selected_records = df_asq_selected_records.drop_duplicates()

    df_asq_selected_records.reset_index(inplace=True,
                                        drop=True)

    # apply sparse encoding
    df_asq_selected_sparse = compute_sparse_encoding(multi_response_col=multi_response_col,
                                                     df=df_asq_selected_records)


    # rename dem questions
    df_asq_selected_sparse.rename(columns={'dem_0110': 'age',
                                        'dem_0500': 'gender',
                                       'dem_0800': 'bmi',
                                       'dem_1000': 'race',
                                           },
                                  inplace=True)
    # implement pre-processing
    df_asq_selected_sparse = pre_processing(df=df_asq_selected_sparse,
                                            columns_interest=list(columns_interest.keys()))
    all_asq_idx = df_asq_selected_sparse.survey_id


    # %% Match the EHR dataset with the ASQ
    df_ehr = pd.read_excel(data_paths.get('raw_data').get('ehr_admission'))
    df_ehr['name'] = df_ehr['first_name'] + ' ' + df_ehr['last_name']
    df_ehr['name'] = df_ehr['name'].str.strip().str.replace(r'\s{2,}', ' ', regex=True)
    df_ehr['name'] = df_ehr['name'].replace('-', '')
    df_ehr.rename(columns={'Date of Birth':'dob',
                           'MRN': 'mrn'}, inplace=True)

    df_asq_selected_sparse['name'] = df_asq_selected_sparse['name'].replace(name_mapper)

    fuzzy_search = FuzzySearch(asq_df=df_asq_selected_sparse,
                               subjects_df=df_ehr,
                               )
    df_matches = fuzzy_search.assign_mrn_by_name_dob_matches(fuzzy_filter=92)

    df_ehr_missing = df_ehr.loc[~df_ehr['mrn'].isin(df_matches.subject_mrn), :]

    # df_ehr_missing[['name', 'date_of_birth', 'mrn']]

    df_asq_selected_sparse.drop(columns=['asq_survey_id'], inplace=True)
    df_matches.rename(columns={'asq_survey_id': 'survey_id'}, inplace=True)

    # first we merge the mrn with the matches and the ASQ
    df_asq_ehr = pd.merge(
        left=df_asq_selected_sparse,
        right=df_matches[['survey_id', 'subject_mrn']],
        on='survey_id',
        how='left'
    )
    # drop the mrn column from the ASQ
    df_asq_ehr.drop(columns=['mrn'], inplace=True)
    df_asq_ehr.rename(columns={'subject_mrn': 'mrn'}, inplace=True)
    col_head = ['mrn', 'name', 'date_of_birth', 'survey_id']
    remaining_cols = [col for col in df_asq_ehr.columns if col not in col_head]
    df_asq_ehr = df_asq_ehr[col_head + remaining_cols]
    # using the mrn from the matches we select from the ehr data
    df_asq_ehr = pd.merge(left=df_asq_ehr,
                      right=df_ehr,
                      on='mrn')
    # %% Inlucde the covid sleep questions
    df_covid = pd.read_csv(data_paths.get('raw_data').get('covid_clinic'))
    df_covid.columns = df_covid.columns.str.strip()

    col_covid = ['UNREFRESHING SLEEP', 'DIFFICULTY SLEEPING', 'DAYTIME SLEEPINESS']
    df_asq_ehr_covid = pd.merge(left=df_asq_ehr,
                                right=df_covid[['MRN'] + col_covid],
                                left_on='mrn',
                                right_on='MRN',
                                how='left'
                                )

    df_asq_ehr_covid[col_covid] = df_asq_ehr_covid[col_covid].replace({'No Answer': np.nan})

    # select only males and females
    df_asq_ehr_covid = df_asq_ehr_covid.loc[df_asq_ehr_covid.gender.isin({0, 1}), :]
    # %% Pre-process columns and feature enginerring 
    df_asq_ehr_covid = df_asq_ehr_covid.replace('.', np.nan)
    # Extract and clean relevant columns for sleep scales and recoding
    sleep_scales = df_asq_ehr_covid[['ess_0900', 'isi_score', 'rls_probability']]
    parasomnia_columns = ['par_0205', 'par_0305', 'par_0505', 'par_0605', 'par_1005']
    sleep_breathing_columns = ['map_0100', 'map_0300', 'map_0600']

    columns_all = [*sleep_scales.columns]  + sleep_breathing_columns # +  parasomnia_columns

    col_par = [col for col in df_asq_ehr_covid.columns if col.startswith('par')]
    # Recode RLS_probability as binary (1 if likely (3 see pre-processing pipeline), 0 otherwise)
    df_asq_ehr_covid['rls_binary'] = df_asq_ehr_covid['rls_probability'].map({3: 1}).fillna(0)

    # Create parasomnia variable (1 if any parasomnia activity reported (>0), else 0)
    df_asq_ehr_covid['parasomnia'] = df_asq_ehr_covid[parasomnia_columns].apply(lambda row: 1 if (row > 0).any() else 0, axis=1)

    # Create sleep-related breathing disorder variable (1 if symptoms present, 0 otherwise)
    df_asq_ehr_covid['breathing_symptoms'] = df_asq_ehr_covid[sleep_breathing_columns].apply(lambda row: 1 if (row > 0).any() else 0, axis=1)
    df_asq_ehr_covid[sleep_breathing_columns] = df_asq_ehr_covid[sleep_breathing_columns].clip(lower=0)

    # definition of excessive day time sleepiness patients
    df_asq_ehr_covid["ess_0900_cat"] = df_asq_ehr_covid["ess_0900"].apply(categorize_ess)
    df_asq_ehr_covid["ess_0900_bin"] = df_asq_ehr_covid["ess_0900"].apply(lambda row: 1 if row >= 10 else 0)
    # df_asq_ehr_covid[["ess_0900_bin", "ess_0900"]]

    # definition of insomnia patients
    df_asq_ehr_covid["isi_score_cat"] = df_asq_ehr_covid["isi_score"].apply(categorize_isi)
    df_asq_ehr_covid["isi_score_bin"] = df_asq_ehr_covid["isi_score"].apply(lambda row: 1 if row >= 15 else 0)
    df_asq_ehr_covid['insomnia_bin'] = df_asq_ehr_covid["insomnia"].apply(lambda row: 1 if row >= 3 else 0)

    # Extreme circadian
    # we will use the column cir_0700 as the extreme circadian
    df_asq_ehr_covid['cir_0700_cat'] = df_asq_ehr_covid["cir_0700"].apply(categorize_circadian)
    df_asq_ehr_covid['cir_0700_bin'] = df_asq_ehr_covid['cir_0700'].apply(lambda row: 1 if (3 <= row <= 7 or row > 22) else 0)

    # sleep complains
    col_sleep = [col for col in df_asq_ehr_covid.columns if 'mdhx_sleep' in col]
    df_asq_ehr_covid['sleep_problems_bin'] = df_asq_ehr_covid[col_sleep].apply(lambda row: 1 if (row > 0).any() else 0, axis=1)


    # days since covid infection
    df_asq_ehr_covid['days_with_covid'] = pd.to_datetime(df_asq_ehr_covid['date_admin_clinic']) - pd.to_datetime(df_asq_ehr_covid['date_incidence_covid'])
    df_asq_ehr_covid['days_with_covid'] = df_asq_ehr_covid['days_with_covid'].astype(str).apply(lambda x: x.split(' ')[0])
    df_asq_ehr_covid['days_with_covid'] = df_asq_ehr_covid['days_with_covid'].replace('NaT', None)
    df_asq_ehr_covid['days_with_covid'] = pd.to_numeric(df_asq_ehr_covid['days_with_covid'], errors='coerce').fillna(0).astype(int)


    # Display a summary of the recoded variables
    df_asq_ehr_covid[['ess_0900', 'isi_score', 'rls_binary', 'parasomnia', 'breathing_symptoms']].describe()

    columns_all = columns_all + ['parasomnia', 'breathing_symptoms']


    # %% save the pre-processing result
    df_asq_ehr_covid.to_csv(data_paths.get('pp_data').get('asq_covid_ehr_cortisol'), index=False)
    print(f'Dimensions of pre-process dataset: {df_asq_ehr.shape}')


