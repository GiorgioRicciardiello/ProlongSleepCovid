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

def convert_to_float_or_nan(value):
    try:
        return float(value)
    except ValueError:
        return np.nan

def process_time_column(df:pd.DataFrame, col:str)->pd.DataFrame:
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

if __name__ == "__main__":
    df_raw_data = pd.read_excel(data_paths.get('raw_data').joinpath('results_data_pull.xlsx'))
    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df_raw_data.columns:
        df_raw_data.drop(columns='Unnamed: 0', inplace=True)

    # %% Select correct patients
    # only completed records
    df_raw_data = df_raw_data.loc[df_raw_data['next_module'] == 'complete', :]
    # subjects that completed more than one ASQ. Select the last one they completed
    df_unique_subjects = df_raw_data[['name', 'date_of_birth']].copy()
    df_unique_subjects = df_unique_subjects.drop_duplicates(subset=list(df_unique_subjects.columns),
                                                            inplace=False)
    df_unique_subjects.reset_index(inplace=True, drop=True)

    df_selected_records = pd.DataFrame()
    # fuzzy search to search matches
    for index, row in df_unique_subjects.iterrows():
        # index = 1
        # row = df_unique_subjects.loc[index, :]
        # Create a DataFrame for the current row
        row_frame = pd.DataFrame(row).transpose()

        # search how many matches of the same subject we have in the dataset
        fuzzy_search = FuzzySearch(asq_df=df_raw_data,
                                   subjects_df=row_frame,
                                   )
        matches = fuzzy_search.search_by_name_dob_matches(method='fuzzy', fuzzy_filter=93)

        if matches.shape[0] == 1:
            matches = df_raw_data.loc[df_raw_data['survey_id'] == matches.asq_survey_id.values[0]]
            df_selected_records = pd.concat([df_selected_records,matches], axis=0)
            # one single match
            continue

        # get the most recent ASQ
        df_subsets = df_raw_data.loc[df_raw_data['survey_id'].isin(list(matches.asq_survey_id.values))]
        df_subsets = df_subsets.sort_values(by='start_time', ascending=False)
        most_recent_row = df_subsets.iloc[[0]]
        df_selected_records = pd.concat([df_selected_records, most_recent_row], axis=0)

    # because two names are searched we must remove the pair it found
    df_selected_records = df_selected_records.drop_duplicates()

    df_selected_records.reset_index(inplace=True,
                                    drop=True)

    # %% apply sparse encoding
    df_selected_sparse = compute_sparse_encoding(multi_response_col=multi_response_col,
                                                 df=df_selected_records)


    # %% rename dem questions
    df_selected_sparse.rename(columns={'dem_0110': 'age',
                                        'dem_0500': 'gender',
                                       'dem_0800': 'bmi',
                                       'dem_1000': 'race',
                                       },
                               inplace=True)
    # %% implement pre-processing
    df_selected_sparse = pre_processing(df=df_selected_sparse,
                                        columns_interest=list(columns_interest.keys()))
    all_asq_idx = df_selected_sparse.survey_id
    # %% pre-process the EHR
    df_ehr = pd.read_excel(data_paths.get('raw_data').joinpath('ehr_dataset.xlsx'))
    df_ehr.columns = df_ehr.columns.map(lambda x: x.rstrip())

    df_ehr['Preexisting sleep symptoms - binary'] = df_ehr['Preexisting sleep symptoms - binary'].map({'yes': 1,
                                               'no': 0})

    df_ehr = process_time_column(df_ehr, col='Time Cortisol Collected')


    df_ehr['Cortisol Levels'] = df_ehr['Cortisol Levels'].apply(lambda x: convert_to_float_or_nan(x))
    # Convert the column to float
    df_ehr['Cortisol Levels'] = df_ehr['Cortisol Levels'].astype(float)

    # df_ehr[['Time Cortisol Collected', 'Time Cortisol Collected_angle']]
    # %% combine the EHR with the ASQ
    name_processor = NameDateProcessor()
    subjects_name_df = name_processor.encode_names(df_ehr) #.sort_index(axis=1)
    subjects_name_date_df = name_processor.encode_date_columns(frame=df_ehr,
                                                          dob_column='Date of Birth')

    fuzzy_search = FuzzySearch(asq_df=df_selected_sparse,
                               subjects_df=df_ehr,
                               )
    matches = fuzzy_search.search_by_name_dob_matches(method='fuzzy', fuzzy_filter=95)

    df_ehr.reset_index(drop=False,
                       names='subject_idx',
                       inplace=True)

    df_ehr_asq_id = pd.merge(left=df_ehr,
                         right=matches[['asq_survey_id', 'subject_idx']],
                         left_on='subject_idx',
                         right_on='subject_idx',
                         how='inner')
    col_ehr_list = list(col_ehr.keys())
    col_ehr_list.append('asq_survey_id')

    # keep only the columns of interest for the ehr
    df_ehr_asq_id = df_ehr_asq_id[col_ehr_list]
    df_asq_ehr = pd.merge(left=df_selected_sparse,
                             right=df_ehr_asq_id,
                             left_on='survey_id',
                             right_on='asq_survey_id',
                             how='inner')
    df_asq_ehr.drop(columns='asq_survey_id',
                    inplace=True)

    # not all those show answered the ASQ has an EHR
    asqs_no_ehr = [id_ for id_ in all_asq_idx if not id_ in df_asq_ehr.survey_id.to_list()]

    pdf_asq_ehr = pd.concat([df_asq_ehr, df_selected_sparse.loc[df_selected_sparse['survey_id'].isin(asqs_no_ehr), :]])

    # %% save the pre-processing result
    pdf_asq_ehr.to_csv(data_paths.get('pp_data').joinpath('pp_data.csv'), index=False)
    print(f'Dimensions of pre-process dataset: {pdf_asq_ehr.shape}')


