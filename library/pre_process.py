import pathlib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from config.data_paths import data_paths
from library.utils import merge_racial_categories
from tabulate import tabulate


class PreProcess:
    def __init__(self, csv_path: Path):
        """
        Initializes the PreProcess class with the provided CSV file path.

        :param csv_path: Path to the input CSV file
        """
        self.csv_path = csv_path
        self.df_raw = pd.read_csv(csv_path)
        self.asq_columns = pd.read_csv(data_paths.get('pp_data').get('asq')).columns

    def process_and_save(self, plot: Optional[bool] = False) -> pd.DataFrame:
        """
        Processes the data and overwrites the CSV file with the cleaned version.

        :param plot: If True, generates a plot of negative responses
        """
        self._merge_race_categories()
        df_processed = self._pre_processing_for_model(plot=plot)
        df_processed = self._convert_columns_to_int_if_no_decimal(df_processed)
        df_processed = self._remove_constant_columns(df_processed)

        # Replace negative values with NaN, fill with median, and remove unnecessary columns
        # df_processed = df_processed.applymap(lambda x: np.nan if x < 0 else x)
        # df_processed = df_processed.apply(lambda col: col.fillna(col.median()), axis=0)
        df_processed.drop(
            columns=['dem_0100', 'dem_0600', 'dem_0610', 'dem_0620', 'dem_0630', 'dem_0650', 'dem_0700', 'dem_0750'],
            inplace=True)
        # Save the processed DataFrame back to the CSV file, overwriting it
        df_processed.to_csv(self.csv_path, index=False)
        print(f'Data saved to {self.csv_path}')
        return df_processed

    def _merge_race_categories(self):
        self.df_raw = merge_racial_categories(self.df_raw, 'dem_1000', 'dem_1010')

    def _pre_processing_for_model(self, plot: Optional[bool] = False) -> pd.DataFrame:
        """
        Pre-processes the input data for model training.

        :param plot: If True, plots the negative and positive response percentage
        :return: Processed DataFrame ready for modeling
        """
        # Initial dimensions of the data
        print(f'*Initial data dimensions: {self.df_raw.shape}')

        # Remove columns with more than 10% NaN values
        cap = 0.1  # The threshold for removing columns with more than 10% NaN values
        print(f'*Removing nan columns that have more than {cap * 100}% of nan responses')
        # Calculate the threshold based on the number of rows
        threshold = int(cap * self.df_raw.shape[0])
        # Drop columns where the number of non-NA values is less than the threshold
        # Create a DataFrame to record NaN counts for each variable
        df_nan_record = pd.DataFrame(self.df_raw.columns, columns=['variables'])
        df_nan_record['nan_count'] = df_nan_record['variables'].apply(lambda col: self.df_raw[col].isna().sum())
        # ASQ column
        df_nan_record['ASQ'] = df_nan_record['variables'].apply(lambda col: col in self.asq_columns)
        # Filter for columns that have at least one NaN
        df_nan_record = df_nan_record.loc[df_nan_record['nan_count'] > 0]
        # Add a column to indicate if the column was removed (i.e., if its NaN count exceeds the threshold)
        df_nan_record['removed'] = df_nan_record.apply(
            lambda row: row['nan_count'] > (self.df_raw.shape[0] - threshold) and row['ASQ'], axis=1
        )
        df_nan_record.sort_values(by='nan_count', ascending=True, inplace=True)
        print(tabulate(df_nan_record, headers='keys', tablefmt='grid'))

        col_to_remove = df_nan_record.loc[df_nan_record['removed'] == True, 'variables'].tolist()

        df = self.df_raw.drop(columns=col_to_remove, inplace=False)
        print(f'\tNew dimension after nan removal: {df.shape}')
        print(f'\tRemoved columns: {set(df.columns) ^ set(self.df_raw)}')

        # Mapping categorical values to numeric in 'rls_probability'
        df['rls_probability'] = df['rls_probability'].map({
            'Unlikely': 0,
            'Unlikely (possibly in past)': 1,
            'Possible': 2,
            'Likely': 3
        })

        # Map race to numeric values
        race_mapper = {race: count for count, race in enumerate(np.sort(df['race'].unique()))}
        df['race'] = df['race'].map(race_mapper)

        # Remove PHI and open-ended questions
        print(f'*Removing PHI and open-ended questions')
        col_open_question = df.select_dtypes(include=['object']).columns.tolist() + ['mrn']
        col_open_question = [col for col in col_open_question if not 'sched_' in col]
        col_open_question = ['survey_id',
                             'start_time',
                             'completed',
                             'created_at',
                             'next_module',
                             'name',
                             'subject_name',
                             'date_of_birth',
                             'origin',
                             'bthbts_0210',
                             'dem_0200',
                             'dem_0300',
                             'dem_0920',
                             'dem_1020',
                             'dem_1110',
                             'diet_0210',
                             'diet_0800',
                             'diet_0810',
                             'diet_0820',
                             'diet_0830',
                             'diet_0840',
                             'end_time',
                             'famhx_0520',
                             'famhx_1020',
                             'mdhx_0110',
                             'mdhx_0130',
                             'mdhx_0310',
                             'mdhx_0820',
                             'mdhx_0850',
                             'mdhx_0870',
                             'mdhx_5510',
                             'mdhx_5740',
                             'mdhx_5760',
                             'mdhx_5840',
                             'mdhx_5940',
                             'mdhx_5970',
                             'mdhx_6040',
                             'mdhx_6120',
                             'mdhx_6130',
                             'mdhx_6220',
                             'mdhx_6230',
                             'mdhx_6340',
                             'mdhx_6350',
                             'mdhx_6430',
                             'mdhx_6520',
                             'mdhx_6540',
                             'mdhx_6550',
                             'mdhx_6650',
                             'mdhx_6660',
                             'mdhx_6730',
                             'mdhx_6750',
                             'mdhx_6770',
                             'mdhx_6780',
                             'mdhx_6790',
                             'mdhx_6800',
                             'mdhx_6810',
                             'med_0200',
                             'med_0200_x',
                             'med_0400',
                             'med_0500',
                             'med_0500_x',
                             'med_0501',
                             'narc_1800',
                             # 'sched_0700',
                             # 'sched_0800',
                             'sched_0855',
                             # 'sched_0900',
                             # 'sched_1000',
                             # 'sched_1900',
                             # 'sched_2000',
                             'sched_2110',
                             "survey_type",
                             # 'soclhx_0850',
                             # 'soclhx_1000',
                             # 'soclhx_1450',
                             # 'soclhx_1590',
                             # 'cortisol_time',
                             # 'date_covid_sleep_questions',
                             # 'date_admin_clinic',
                             # 'date_incidence_covid',
                             # 'date_asq_question',
                             # 'anx_depression',
                             'mrn']
        col_nan = [col for col in df.columns if col.endswith('_nan')]
        col_med = [col for col in df.columns if col.startswith('med_')]
        col_open_question = col_open_question + col_nan + col_med + ['survey_type']
        df.drop(col_open_question, axis=1, inplace=True)
        df.replace('.', np.nan, inplace=True)
        print(
            f'\tColumns removed: {len(col_open_question)}\n\tNew dimensions: {df.shape}\n\tColumns:{col_open_question}')

        # Handle columns with negative values and positive response percentages
        numeric_df = df.select_dtypes(include='number')
        df_negatives = (numeric_df.lt(0).sum() / len(numeric_df) * 100).to_frame(name='negative')
        df_negatives.reset_index(inplace=True)
        df_negatives.rename(columns={'index': 'column'}, inplace=True)
        df_negatives['group'] = df_negatives['column'].apply(lambda x: x.split('_')[0])
        df_negatives['positive_response_percent'] = df_negatives['column'].apply(
            lambda x: np.round(((df[x] > 0).sum() / df.shape[0]) * 100, 2)
        )

        df_negatives_plot = df_negatives.loc[df_negatives['negative'] > 0, :].sort_values(by='negative',
                                                                                          ascending=False)
        df_negatives_plot.reset_index(inplace=True)

        if plot:
            self._plot_negative_responses(df_negatives_plot)

        # Remove columns based on filtering criteria
        df_negatives_selected = df_negatives_plot.loc[
                                (df_negatives_plot['negative'] < 70) & (
                                        df_negatives_plot['positive_response_percent'] > 70), :]
        df_negatives_remove = df_negatives_plot.loc[
            ~df_negatives_plot['column'].isin(df_negatives_selected['column']), 'column']
        print(f'*Removing negative and non-positive responses using criteria\n\t '
              f'Columns to drop: \n{df_negatives_remove}')
        df.drop(columns=df_negatives_remove, inplace=True)
        print(f'Final data dimensions: {df.shape}')
        return df

    def _convert_columns_to_int_if_no_decimal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts columns to int type if all values in the column have no decimal part (e.g., 1.0 -> 1).

        :param df: DataFrame to be processed
        :return: DataFrame with appropriate columns converted to int
        """
        for col in df.columns:
            if df[col].apply(lambda x: x.is_integer() if isinstance(x, float) else True).all():
                df[col] = df[col].astype(int)
        return df

    def _remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns from the DataFrame that contain constant values (i.e., all values are the same).

        :param df: DataFrame to be processed
        :return: DataFrame with constant columns removed
        """
        return df.loc[:, (df != df.iloc[0]).any()]

    def _plot_negative_responses(self, df_negatives_plot: pd.DataFrame) -> None:
        """
        Plots negative and positive response percentages by feature.

        :param df_negatives_plot: DataFrame containing negative responses data
        """
        fig, ax = plt.subplots(figsize=(12, 6), nrows=2)

        sns.barplot(data=df_negatives_plot, y='negative', x='column', palette="Blues_d", ax=ax[0])
        ax[0].set_xticks(ticks=range(0, df_negatives_plot.shape[0], 50),
                         labels=range(0, df_negatives_plot.shape[0], 50))
        ax[0].set_xlabel("")
        ax[0].set_ylabel("Negative Count", fontsize=12)
        ax[0].grid(alpha=0.9)

        sns.barplot(data=df_negatives_plot, y='positive_response_percent', x='column', palette="Greens_d", ax=ax[1])
        ax[1].set_xticks(ticks=range(0, df_negatives_plot.shape[0], 50),
                         labels=range(0, df_negatives_plot.shape[0], 50))
        ax[1].set_ylabel("Positive Response %", fontsize=12)
        ax[1].grid(alpha=0.9)

        plt.xlabel("Features", fontsize=12)
        plt.suptitle("Negative Responses and Positive Response Percentage by Feature", fontsize=16)
        plt.tight_layout()
        plt.show()


class RecodeResponses:
    def __init__(self, data: pd.DataFrame,
                 output_path: Optional[pathlib.Path] = None):
        """
        Follow the same recording of the variables as done in the MSc Thesis.
        :param data:
        """
        self.df_data = data.copy()
        self.output_path = output_path

    def process_data(self):
        self.df_data.rename(columns={'dem_0500': 'gender'}, inplace=True)
        # get only define genders
        df = self.df_data.loc[self.df_data['gender'].isin([0, 1]), :]
        try:
            # Recode ESS, ISI, and other variables
            df['ess_cat'] = self._recode_ess(df['ess_0900'])
        except Exception:
            pass  # Ignore error if recoding ESS fails

        try:
            df['isi_cat'] = self._recode_isi(df['isi_score'])
        except Exception:
            pass  # Ignore error if recoding ISI fails

        try:
            df['rls_probability'] = self._recode_rls(df['rls_probability'])
        except Exception:
            pass  # Ignore error if recoding RLS fails

        try:
            # Recode Parasomnia variables
            df = self._recode_parasomnia(df)
        except Exception:
            pass  # Ignore error if recoding Parasomnia fails

        try:
            # Recode FOSQ
            df = self._recode_fosq(df)
        except Exception:
            pass  # Ignore error if recoding FOSQ fails

        # Combine responses for parasomnias
        df = self._total_responses(data=df,
                                   columns=['sleepwalking', 'sleepeating', 'actingout', 'violentbehavior'],
                                   new_column='totals_response')

        # Create binary response for sleep-related breathing disorders
        df = self._recode_breathing_disorders(df)
        df = self._total_responses(data=df,
                                   columns=['loudsnore', 'snorting', 'breathingstops'],
                                   new_column='total_response_sd')

        df = self._binary_response(data=df,
                                   total_column='total_response_sd',
                                   new_column='sleeprelated',
                                   threshold=2)

        if self.output_path:
            df.to_csv(self.output_path, index=False)
            print(f'Data saved to {self.output_path}')

        return df

    @staticmethod
    def _recode_ess(ess_column):
        # Recode ESS
        conditions = [
            (ess_column.between(0, 7)),
            (ess_column.between(8, 9)),
            (ess_column.between(10, 15)),
            (ess_column.between(16, 24))
        ]
        choices = [0, 1, 2, 3]
        return np.select(conditions, choices, default=np.nan)

    @staticmethod
    def _recode_isi(isi_column):
        # Recode ISI
        conditions = [
            (isi_column.between(0, 7)),
            (isi_column.between(8, 14)),
            (isi_column.between(15, 21)),
            (isi_column.between(22, 28))
        ]
        choices = [0, 1, 2, 3]
        return np.select(conditions, choices, default=np.nan)

    @staticmethod
    def _recode_rls(rls_column: pd.Series) -> pd.Series:
        categories = {0: "unlikely",
                      1: "unlikely, possibly in past",
                      2: "possible",
                      3: "likely"}
        return rls_column.map(categories)

    @staticmethod
    def _recode_parasomnia(data):
        """
        Re code to keep only answer that are 9
        :param data:
        :return:
        """
        # Example for recoding parasomnia variables
        data['sleepwalking'] = np.where(data['par_0205'] > 0, 1, 0)
        data['sleepeating'] = np.where(data['par_0305'] == 2, 1, 0)
        data['actingout'] = np.where(data['par_0505'].between(1, 28), 1, 0)
        data['violentbehavior'] = np.where(data['par_0605'].between(1, 28), 1, 0)
        data['sexwithnomem'] = np.where(data['par_1005'].between(1, 28), 1, 0)
        return data

    @staticmethod
    def _recode_breathing_disorders(data):
        # Example for recoding breathing disorders variables
        data['loudsnore'] = np.where(data['map_0100'].between(3, 4), 1, 0)
        data['snorting'] = np.where(data['map_0300'].between(3, 4), 1, 0)
        data['breathingstops'] = np.where(data['map_0600'].between(3, 4), 1, 0)
        return data

    @staticmethod
    def _recode_fosq(df):
        # Generate binary 'fosq' based on fosq_1100 values
        df['fosq'] = np.where(df['fosq_1100'] > 17.9, 0, 1)

        # Handle missing values (-33 as missing in Stata code)
        df['fosq'] = np.where(df['fosq_1100'] == -33, np.nan, df['fosq'])

        # Label 'fosq' with "no" and "yes"
        df['fosq_label'] = df['fosq'].map({0: "no", 1: "yes"})

        return df

    @staticmethod
    def _total_responses(data, columns, new_column):
        data[new_column] = data[columns].sum(axis=1)
        return data

    @staticmethod
    def _binary_response(data, total_column, new_column, threshold):
        """Function for combining multiple variables into a total score"""
        data[new_column] = np.where(data[total_column] >= threshold, 1, 0)
        return data

    @staticmethod
    def total_responses(data, columns, new_column):
        """Function for combining multiple variables into a total score"""
        data[new_column] = data[columns].sum(axis=1)
        return data


if __name__ == '__main__':
    # Initialize the PreProcess class with the path to the CSV file
    csv_file_path = data_paths.get('pp_data').get('asq_covid_ehr_cortisol')
    processor = PreProcess(csv_file_path)
    pp_data = processor.process_and_save(plot=False)

    # csv_recoded = RecodeResponses(data=pp_data,
    #                               output_path=csv_file_path)
    # csv_recoded.process_data()
