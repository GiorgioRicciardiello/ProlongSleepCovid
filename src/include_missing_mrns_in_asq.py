"""
When pulling the ASQ dataset using the patient list file, some patients did not had the MRN.
We will include them here to later merge and create the unique dataset with
1. Cortisol levels
2. The three sleep questions of the covid clinic
"""

import pandas as pd
import pathlib
from config.data_paths import data_paths
import ast
if __name__ == '__main__':
    df_asq = pd.read_csv(data_paths.get('raw_data').get('asq'))
    df_cortisol = pd.read_excel(data_paths.get('raw_data').get('cortisol'))

    # %% clean cortisol dataset
    df_cortisol['name'] = df_cortisol['Patient First Name '] + ' ' +df_cortisol['Patient Last Name']
    df_cortisol = df_cortisol.loc[df_cortisol['MRN'] != 'Completed', :]

    # df_asq.loc[df_asq['MRN'].isna(), ['subject_name', 'date_of_birth']]
    # we have 4 subjects with missing MRN
    for index, row in df_asq.loc[df_asq['MRN'].isna()].iterrows():
        subject_name = row['subject_name']
        # Print for debug
        print(f"Processing subject: {subject_name}")

        # Find corresponding MRN in df_cortisol
        matched_mrn = df_cortisol.loc[df_cortisol['name'] == subject_name, 'MRN']

        if not matched_mrn.empty:
            # Assign the found MRN to the correct row in df_asq
            df_asq.at[index, 'MRN'] = int(matched_mrn.values[0])
            print(f"\tAssigned MRN {matched_mrn.values[0]} to subject {subject_name}")
        else:
            print(f"MRN not found for subject {subject_name}")

    # %% The missing one
    # df_cortisol.iloc[118, :][['name', 'Date of Birth ', 'MRN']]
    df_asq.at[110, 'MRN'] = int(df_cortisol.at[118, 'MRN'])

    # %% asq old mrn, drop
    id_columns = df_asq.filter(like='_id').columns
    id_columns = [col for col in id_columns if col != 'survey_id']
    to_drop = list(id_columns) + ['mrn']
    df_asq.drop(columns=to_drop, inplace=True)
    df_asq.rename(columns={'MRN': 'mrn'}, inplace=True)
    # %% re-order columns
    col_head = ['survey_id', 'mrn']
    col_rest = [col for col in df_asq.columns if not col in col_head]
    columns = col_head + col_rest
    df_asq = df_asq[columns]
    assert df_asq['mrn'].isna().any() == False
    # %% Cortisol dataset
    # 3 subjects with missing MRN
    # df_cortisol.loc[df_cortisol['MRN'].isna()]
    for index, row in df_cortisol.loc[df_cortisol['MRN'].isna()].iterrows():
        subject_name = row['name']
        # Print for debug
        print(f"Processing subject: {subject_name}")

        # Find corresponding MRN in df_cortisol
        matched_mrn = df_asq.loc[df_asq['subject_name'] == subject_name, 'mrn']

        if not matched_mrn.empty:
            # Assign the found MRN to the correct row in df_asq
            df_cortisol.loc[index, 'MRN'] = matched_mrn.values[0]
            print(f"\tAssigned MRN {matched_mrn.values[0]} to subject {subject_name}")
        else:
            print(f"\tMRN not found for subject {subject_name}")

    df_cortisol['MRN'] = df_cortisol['MRN'].astype(float)
    df_cortisol.rename(columns=
                       {'MRN':'mrn',
                        'Patient First Name ': 'name_first',
                        'Patient Last Name': 'name_last',
                        'Date of Birth ': 'date_of_birth',
                       'Date of admission / Clinic date': 'admission_date',
                       'Completed?': 'completed',
                       'Cortisol Levels (nan= incomplete info)': 'cortisol_level',
                       'Time Cortisol Collected (nan= incomplete info)': 'cortisol_time',
                       'Cortisol in Time Period':'cortisol_time_period',
                       'Notes': 'notes',
                        }, inplace=True)
    df_cortisol['name'] = df_cortisol['name_first'] + ' ' + df_cortisol['name_last']
    df_cortisol.drop(columns=['name_first', 'name_last', 'notes', 'completed'], inplace=True)
    col_head = ['name', 'date_of_birth', 'mrn', 'admission_date', 'cortisol_level']
    col_rest = [col for col in df_cortisol.columns if not col in col_head]
    columns = col_head + col_rest
    df_cortisol = df_cortisol[columns]
    df_cortisol = df_cortisol.loc[~df_cortisol['cortisol_level'].isna(), :]
    df_cortisol.reset_index(inplace=True, drop=True)
    # %% check and save
    assert df_asq.loc[df_asq['mrn'].isna(), ['subject_name', 'date_of_birth']].shape[0] == 0
    print('\nAll MRN are completed and filled, saving pre-processed ASQ data')
    df_asq.to_csv(data_paths.get('pp_data').get('asq'), index=False)
    df_cortisol.to_csv(data_paths.get('pp_data').get('cortisol'), index=False)

    # %% Check and clean the MRN columns
    # if done before it the str.replace('`', '') creates nans in the MRNs we got from the other dataset
    df_asq = pd.read_csv((data_paths.get('pp_data').get('asq')))
    df_asq['mrn'] = df_asq['mrn'].str.replace('`', '')
    df_asq['mrn'] = df_asq['mrn'].astype(int)
    assert df_asq['mrn'].isna().any() == False
    df_asq.to_csv(data_paths.get('pp_data').get('asq'), index=False)

    df_cortisol = pd.read_csv(data_paths.get('pp_data').get('cortisol'))


