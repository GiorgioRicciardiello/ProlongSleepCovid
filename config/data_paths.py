"""
Configuration file to keep the folder directories
"""
import pathlib
from pathlib import Path
import json

import pandas as pd

root_path = pathlib.Path(__file__).resolve().parents[1]
raw_path = root_path.joinpath(pathlib.Path('data/raw_data'))
pp_path = root_path.joinpath(pathlib.Path('data/pp_data'))
results_path = root_path.joinpath(pathlib.Path('results'))
patient_list_path = root_path.joinpath(pathlib.Path('patients_list'))
ehr_dataset_path = root_path.joinpath(pathlib.Path('ehr_dataset'))
# data_paths = {
#     'raw_data': raw_path,
#     'pp_data': pp_path,
#     'results': results_path,
# }

data_paths = {
    'root': root_path,
    'raw_data': {
        'covid_clinic': raw_path.joinpath('covid_clinic_three_questions_questionnaire.csv'),
        'cortisol': raw_path.joinpath('cortisol_database.xlsx'),
        'asq': raw_path.joinpath('asq_for_study.csv'),
        'ehr_admission': raw_path.joinpath('ehr_dataset.xlsx'),
        'cortisol_filtered': raw_path.joinpath('cortisol_filtered.xlsx'),
        'results_pull': raw_path.joinpath('results_data_pull.xlsx'),
    },
    'pp_data': {
        'root': pp_path,
        'asq': pp_path.joinpath('asq_for_study.csv'),  # cleaned asq dataset
        'cortisol': pp_path.joinpath('cortisol.csv'),  # cleaned cortisol dataset
        'asq_cortisol': pp_path.joinpath('asq_cortisol.csv'),  # asq + cortisol dataset
        'asq_cov_clinic': pp_path.joinpath('asq_covid.csv'),  # asq + covid clinic dataset
        'asq_cov_clinic_sparse': pp_path.joinpath('asq_covid_sparse.csv'),  # asq + covid clinic dataset
        'covid_clinic_same_asq_dates': pp_path.joinpath('covid_date_match.csv'),  # covid clinic mathing asq dates
        'asq_covid_cortisol': pp_path.joinpath('asq_covid_cortisol.csv'),  # asq + covid + cortisol patients (join)
        'asq_covid_ehr_cortisol': pp_path.joinpath('asq_covid_ehr_cortisol.csv')  # asq + covid + cortisol patients + ehr(join)
    },
    'patient_list': {
        'data_to_pull': patient_list_path.joinpath('Data Pull Request of Completed ASQs (Updated 7_3).xlsx'),
    },
    'results': results_path,
    'asq_dictionary': root_path.joinpath('docs/asq_dictionary_description.json'),
}

#
# def load_column_dict(json_path):
#     with open(json_path, 'r') as json_file:
#         return json.load(json_file)
#
#
#
# df_temp = pd.read_excel(r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\sleep_covid\docs\asq_dictionary_v4.xlsx')
#
# df_temp = df_temp.loc[df_temp['Allow Multiple Responses'] == 'Yes', ['Table Name', 'Column Name', 'Question Name (Abbreviated)']]
#
#
# # multi_resp_cols = {tab_name: col_name for tab_name, col_name in zip(df_temp['Table Name'], df_temp['Column Name'])}
#

multi_response_col = {
 'mdhx_sleep_problem': 'problem',
 'mdhx_sleep_diagnosis': 'mdhx_0120',
 'mdhx_sleep_treatment': 'treatment',
 'mdhx_pap_problem': 'problem',
 'mdhx_pap_improvement': 'improvement',
 'mdhx_cardio_problem': 'problem',
 'mdhx_cardio_surgery': 'surgery',
 'mdhx_pulmonary_problem': 'problem',
 'mdhx_ent_surgery': 'surgery',
 'mdhx_ent_problem': 'problem',
 'mdhx_dental_work': 'procedure',
 'mdhx_orthodontics': 'procedure',
 'mdhx_gi_problem': 'problem',
 'mdhx_neurology_problem': 'problem',
 'mdhx_metabolic_endocrine_problem': 'problem',
 'mdhx_urinary_kidney_problem': 'problem',
 'mdhx_pain_fatigue_problem': 'problem',
 'mdhx_headache_problem': 'problem',
 'mdhx_psych_problem': 'problem',
 'mdhx_anxiety_problem': 'problem',
 'mdhx_eating_disorder': 'disorder',
 'mdhx_other_problem': 'problem',
 'mdhx_cancer': 'cancer',
 'mdhx_autoimmune_disease': 'disease',
 'mdhx_hematological_disease': 'disease',
 'famhx_insomnia': 'relation',
 'famhx_sleep_apnea': 'relation',
 'famhx_narcolepsy': 'relation',
 'famhx_rls': 'relation',
 'famhx_other_sleep_disorder': 'relation',
 'famhx_sleepwalk': 'relation',
 'famhx_fibromyalgia': 'relation',
 'famhx_depression': 'relation',
 'famhx_anxiety': 'relation',
 'famhx_psych_illness': 'relation',
 'famhx_psych_treatment': 'relation',
 'famhx_sleep_death': 'relation',
 'bthbts_sleep_disruption': 'disruption',
 'bthbts_employment': 'employment',
 'sched_rotating_shift': 'shift',
}