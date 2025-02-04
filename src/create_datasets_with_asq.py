"""
Create the different datasets that will be used for the project

1. asq + cortisol dataset
2. asq + covid clinic dataset
3. covid clinic mathing asq dates
4. asq + covid + cortisol patients

"""
import pandas as pd
import pathlib
from config.data_paths import data_paths, multi_response_col
from typing import Optional
import ast
from library.utils import compute_sparse_encoding

col_mapper = {
    'first_name': 'first_name',
    'last_name': 'last_name',
    'Date of Birth': 'dob',
    'Date of admission / Clinic date': 'date_admin_clinic',
    'MRN': 'mrn',
    'Date of initial COVID-19 infection': 'date_incidence_covid',
    'Days the Person Has Had Long COVID': 'days_long_covid',
    'Hospitalized (no = 0, yes = 1)': 'hospitalized',
    'COVID-19 vaccination status (no = 0, yes = 1), (2 doses of the vaccine)': 'covid_vaccine',
    'Preexisting sleep symptoms  (no = 0, yes = 1)': 'prevalence_sleep_symptom',
    'Notes on sleep symptoms': 'note_sleep_symptom',
    'Date of ASQ Questionnaire ': 'date_asq_question',
    'Headaches': 'headaches',
    'Nasal Congestion': 'nasal_congestion',
    'Fatigue': 'fatigue',
    'Brain fog': 'brain_fog',
    'Unrefreshing sleep': 'unrefreshing_sleep',
    'Insomnia': 'insomnia',
    'Lethargy': 'lethargy',
    'Post-exertional Malaise': 'post_exercial_malaise',
    'Change in Smell': 'change_in_smell',
    'Change in Taste': 'change_in_taste',
    'Anxiety/Depression': 'anx_depression',
    'Cough': 'cough',
    'Shortness of Breath': 'shortness_breath',
    'Lightheadedness': 'lightheadedness',
    'GI Symptoms': 'gi_symptoms',
    'Functional Satus': 'functional_status',
    'Number of COVID-19 symptoms': 'number_of_covid_symptoms',
    'Any other notes?': 'notes',
}

if __name__ == '__main__':
    df_asq = pd.read_csv(data_paths.get('pp_data').get('asq'))
    df_asq.columns = df_asq.columns.str.strip()
    df_asq = df_asq.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df_asq.reset_index(inplace=True, drop=True)
    df_asq = compute_sparse_encoding(multi_response_col=multi_response_col, df=df_asq)
    # %% 1. Cortisol Dataset
    df_cortisol = pd.read_csv(data_paths.get('pp_data').get('cortisol'))
    df_cortisol.columns = df_cortisol.columns.str.strip()
    df_cortisol = df_cortisol.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # 1.2 Merge the ASQ and Cortisol datasets, keeping all rows from df_asq (outer join)
    df_cortisol_asq = pd.merge(left=df_asq,
                               right=df_cortisol[['mrn', 'cortisol_level', 'cortisol_time']],
                               on='mrn',
                               how='left')  # Use 'left' join to keep all rows from df_asq
    # we get one more subject, but it's the same, it's a duplicate row
    df_duplicate_survey_ids = df_cortisol_asq[df_cortisol_asq.duplicated(subset='survey_id', keep=False)]
    df_cortisol_asq = df_cortisol_asq.drop_duplicates(subset='survey_id', keep='first')
    # keep only those subjects with ASQ responses
    df_cortisol_asq_clean = df_cortisol_asq.loc[~df_cortisol_asq['cortisol_level'].isna(), :]
    df_cortisol_asq_clean.drop(columns=['name', 'next_module', 'origin', 'survey_type'], inplace=True)
    df_cortisol_asq_clean.to_csv(data_paths.get('pp_data').get('asq_cortisol'), index=False)

    # %% 2. Covid Clinic Datasets
    df_covid = pd.read_csv(data_paths.get('raw_data').get('covid_clinic'))
    df_covid.columns = df_covid.columns.str.strip()
    df_covid = df_covid.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    df_covid.rename(columns={'MRN': 'mrn'}, inplace=True)
    covid_sleep_questions = [col for col in df_covid.columns if 'SLEEP' in col]

    df_covid[covid_sleep_questions] = df_covid[covid_sleep_questions].replace({'No Answer': 0}, regex=True)
    for col in covid_sleep_questions:
        df_covid[col] = df_covid[col].astype(int)

    # To create this dataset we will sample subjects that completed the survey the same time the
    # asq was completed
    df_covid['Date'] = pd.to_datetime(df_covid['Date']).dt.date
    df_asq['completed'] = pd.to_datetime(df_asq['completed']).dt.date

    matching_rows = df_covid[df_covid['Date'].isin(df_asq['completed'])]

    sampled_rows = matching_rows.sample(n=matching_rows.shape[0])  # Adjust n to your required sample size
    # dataset with the covid clinic patients that were admitted the same date an ASQ was submitted
    sampled_rows.columns = sampled_rows.columns.str.strip()
    sampled_rows = sampled_rows.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    sampled_rows.to_csv(data_paths.get('pp_data').get('covid_clinic_same_asq_dates'), index=False)

    # match by MRN
    df_covid_asq = pd.merge(left=df_asq,
                            right=df_covid[['mrn', 'Date'] + covid_sleep_questions],
                            on='mrn',
                            how='left')
    df_covid_asq.rename(columns={'Date': 'date_covid_sleep_questions'}, inplace=True)
    df_covid_asq.drop(columns='survey_type', inplace=True)
    df_covid_asq_clean = df_covid_asq.dropna(subset=covid_sleep_questions)
    df_covid_asq_clean.to_csv(data_paths.get('pp_data').get('asq_cov_clinic'), index=False)

    # %% Create a single dataset with all the data
    df_covid_cortisol_asq = pd.merge(left=df_cortisol_asq,
                            right=df_covid[['mrn', 'Date'] + covid_sleep_questions],
                            on='mrn',
                            how='left')
    df_covid_cortisol_asq.rename(columns={'Date': 'date_covid_sleep_questions'}, inplace=True)
    df_covid_cortisol_asq.to_csv(data_paths.get('pp_data').get('asq_covid_cortisol'), index=False)

    # %% Create a sparse version of the ASQ multi-response questions
    # df_covid_asq_clean.reset_index(inplace=True, drop=True)
    # df_expanded = compute_sparse_encoding(multi_response_col=multi_response_col,
    #                                        df=df_covid_asq_clean)
    df_covid_asq_clean.to_csv(data_paths.get('pp_data').get('asq_cov_clinic_sparse'), index=False)

    # %% 4. EHR dataset with the covid and cortisol dataset
    df_ehr = pd.read_excel(data_paths.get('raw_data').get('ehr_admission'))
    df_ehr = df_ehr.rename(columns=col_mapper)
    df_ehr = df_ehr.loc[df_ehr['notes'].isna()]
    df_ehr = df_ehr.dropna(axis=1, how='any', inplace=False)
    df_ehr['date_admin_clinic'] = pd.to_datetime(df_ehr['date_admin_clinic']).dt.date
    df_ehr['date_incidence_covid'] = pd.to_datetime(df_ehr['date_incidence_covid']).dt.date


    df_ehr.drop(columns=['first_name',
                         'last_name',
                         'dob',
                         ], inplace=True)


    df_covid_cortisol_ehr_asq = pd.merge(left=df_covid_cortisol_asq,
                               right=df_ehr,
                               on='mrn',
                               how='left')

    df_covid_cortisol_ehr_asq.to_csv(data_paths.get('pp_data').get('asq_covid_ehr_cortisol'), index=False)














