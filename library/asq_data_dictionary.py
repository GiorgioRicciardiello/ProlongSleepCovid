import pathlib
import re
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import json
from config.data_paths import data_paths
import numpy as np
import pickle
import warnings

class AsqDataDictionary:
    def __init__(self,
                 data_dict_path: pathlib.Path,
                 output_path: Optional[pathlib.Path] = None,
                 overwrite: Optional[bool] = False
                 ):
        """
        From the asq_dictionary.xlsx we will extract the column questions, data types and numerical scoring in a format
        to easily map the response of the dataset into their description. This is mainly useful for when we want to
        create an informative Table One of the full ASQ and need to place the proper labels to the questions and
        responses.

        The class generates a dictionary
        :param data_dict_path: pathlib.Path, path to the asq_dictionary.xlsx file
        :param output_path: pathlib.Path, path to the output file we will generate
        """
        self.output_path = output_path
        self.overwrite = overwrite
        self.data_dict = self._read_data_dict(data_dict_path=data_dict_path)

    def run(self) -> Dict[str, Any]:
        """
        Runs the process to generate the data dictionary.
        If the output file exists and overwrite is False, reads and returns the existing file.
        """
        # Check if output file exists and load it if overwrite is False
        if self.output_path and self.output_path.exists() and not self.overwrite:
            return self._load_existing_dict()

        # Process to generate the new data dictionary
        self._clean_data_dictionary()
        column_dict = self._generate_numeric_scoring_dict()
        self.print_column_dict(column_dict)
        new_dict = self._modify_multiple_response_codes(column_dict=column_dict)
        self.print_column_dict(new_dict)
        self._save_dict(data=new_dict)
        return new_dict

    @staticmethod
    def print_column_dict(column_dict: Dict[str, Any]):
        """
        Nicely prints the column dictionary with indentation and labels, including dtype.
        """
        print("Column Dictionary:")
        for col_name, details in column_dict.items():
            print(f"\nColumn Name: {col_name}")
            print(f"  Description: {details['description']}")
            print("  Numeric Scoring:")
            for score_key, score_value in details['numeric_scoring'].items():
                print(f"    {score_key}: {score_value}")
            print(f"  Multiple Responses: {'Yes' if details['multiple_responses'] else 'No'}")
            print(f"  dtype: {details['dtype']}")


    @staticmethod
    def merge_dictionaries(new_entries: Dict[str, Any], column_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merges column_dict into new_entries. Raises an error if a key already exists in new_entries.
        """
        for key, value in column_dict.items():
            if key in new_entries:
                # Raise an error if the key already exists in new_entries
                raise KeyError(f"Key '{key}' already exists in new_entries.")
            else:
                # If the key does not exist in new_entries, add it directly
                new_entries[key] = value

        return new_entries

    def _read_data_dict(self, data_dict_path: pathlib.Path) -> pd.DataFrame:
        return pd.read_excel(data_dict_path)

    def _clean_data_dictionary(self):
        def parse_multiple_responses(value) -> bool:
            if pd.isnull(value) or pd.isna(value):
                return False
            if isinstance(value, str):
                value_lower = value.lower().strip()
                if 'y' in value_lower:
                    return True
                elif 'n' in value_lower:
                    return False
            return False  # Keep as NaN if it doesn't match

        # Clean up column names by stripping any leading/trailing spaces
        self.data_dict.columns = self.data_dict.columns.str.strip()

        # Remove old questions
        self.data_dict = self.data_dict[
            ~(
                    self.data_dict['key notes'].str.contains('Removed|Deprecated|Archived|Replaced', case=False,
                                                             na=False) &
                    ~self.data_dict['key notes'].str.contains('was deprecated', case=False, na=False)
            )
        ]

        # Remove open questions
        self.data_dict = self.data_dict[
            ~self.data_dict['Question Name (Abbreviated)'].str.contains('Other Description|Description', case=False,
                                                                        na=False)]

        # Remove specific tables
        self.data_dict = self.data_dict[~self.data_dict['Table Name'].isin(['login', 'survey', 'phi', 'psf', 'irb'])]


        # Identify rows where there are both <string>_1 and <string>_2, and keep only the ones ending in '_2'
        # Extract rows ending in '_1' and '_2'
        ends_with_1 = self.data_dict['Column Name'].str.endswith('_1', na=False)
        ends_with_2 = self.data_dict['Column Name'].str.endswith('_2', na=False)
        # self.data_dict.loc[ends_with_1, 'Column Name']

        # Create a set of unique base names that have both _1 and _2 versions
        base_names_with_2 = set(self.data_dict.loc[ends_with_2, 'Column Name'].str[:-2])
        self.data_dict = self.data_dict[~(ends_with_1 & self.data_dict['Column Name'].str[:-2].isin(base_names_with_2))]

        # Standardize column names by removing '_2' suffix if it exists
        self.data_dict['Column Name'] = self.data_dict['Column Name'].apply(
            lambda x: x[:-2] if isinstance(x, str) and x.endswith('_2') else x)

        # # Remove rows where 'Numeric Scoring Code' is not NaN
        # self.data_dict = self.data_dict.dropna(subset=['Numeric Scoring Code'])
        self.data_dict = self.data_dict.loc[~self.data_dict['Column Name'].str.contains('end_time|start_time', case=False,
                                                             na=False)]
        # do not consider the columns that are for tables ID's
        self.data_dict = self.data_dict.loc[~self.data_dict['Column Name'].str.endswith('_id')]

        # Apply this parsing function to the 'Allow Multiple Responses' column
        self.data_dict['Allow Multiple Responses'] = self.data_dict['Allow Multiple Responses'].apply(
            parse_multiple_responses)

        # Remove duplicate rows based on certain columns
        columns_to_check = ["Table Name", "Column Name", "Question Name (Abbreviated)", "Numeric Scoring Code"]
        self.data_dict = self.data_dict.drop_duplicates(subset=columns_to_check)

    def _generate_numeric_scoring_dict(self) -> Dict[str, Any]:
        # multiple responses question cann't use the Column Name as key because of duplicates
        # if it finds any of these duplicates, it will use the Table Name
        unwanted_column_names = ['problem', 'treatment', 'improvement', 'disorder', 'cancer',
       'disease', 'relation', 'disruption', 'employment', 'shift',
       'unused', 'use', 'drug', 'score', 'diet', 'surgery', 'procedure']
        table_sections = self.data_dict.loc[~self.data_dict['Table Name'].str.contains('_'), 'Table Name'].unique()
        if len(table_sections) != 21:
            raise ValueError("There should be 21 tables in the dataframe source.")

        multiple_responses_list = []  # List to track missing multiple response columns
        column_dict = {}
        multiple_responses_count = 0
        # for debug, select a row given the column name
        # idx = self.data_dict.loc[self.data_dict['Column Name'] == 'surgery', :].index
        # row = [val for val in [*self.data_dict.iterrows()] if val[0] ==idx][0][1]

        for _, row in self.data_dict.iterrows():
            col_name = row['Column Name'].lower()  # Ensure case-insensitivity for Column Name

            if col_name in unwanted_column_names and not row['Table Name'] in table_sections:
                col_name = row['Table Name']
                multiple_responses_list.append(col_name)

            # Parse the numeric scoring
            numeric_scoring = self._improved_parse_numeric_scoring(row['Numeric Scoring Code'])

            # Determine multiple_responses based on conditions
            if row['Allow Multiple Responses'] and len(numeric_scoring) > 2:
                multiple_responses = True
                multiple_responses_count += 1
                if not row['Table Name'] in table_sections:
                    col_name = row['Table Name']
                    multiple_responses_list.append(col_name)
            else:
                multiple_responses = False

            # Determine dtype based on column name and numeric scoring content
            if 'score' in row['Question Name (Abbreviated)'].lower():
                dtype = 'continuous'
            elif numeric_scoring == {}:
                dtype = 'continuous'
            elif all(self._is_binary_key(key) for key in numeric_scoring.keys()):
                dtype = 'binary'
            elif multiple_responses:
                dtype = 'categorical'
            else:
                dtype = 'ordinal'

            if col_name in column_dict.keys():
                raise ValueError(f"Duplicate column name: {col_name} "
                                 f"Multi response {multiple_responses} - {row['Column Name'].lower()}")

            # Add the entry to column_dict
            column_dict[col_name] = {
                'description': row['Question Name (Abbreviated)'],
                'numeric_scoring': numeric_scoring,
                'multiple_responses': multiple_responses,
                'dtype': dtype,
                # 'table_name': row['Table Name'],
            }
        multiple_responses_list = list(set(multiple_responses_list))
        print(f'Total multiple response counts {multiple_responses_count}')
        if multiple_responses_count != self.data_dict['Allow Multiple Responses'].sum():
            warnings.warn(
                f"Not all multiple responses were captured. Found {multiple_responses_count} "
                f"but expected {self.data_dict['Allow Multiple Responses'].sum()}",
                UserWarning
            )
        return column_dict

    @staticmethod
    def _improved_parse_numeric_scoring(scoring_str: str) -> Dict[str, str]:
        """
        Parse a scoring string with the format 'key=value' pairs, handling cases where line breaks or inconsistent formatting
        may interrupt the expected structure. Returns an empty dictionary if parsing fails.
        """
        try:
            # Split the scoring string by newlines or commas and clean up whitespace
            parts = scoring_str.replace('\n', ',').split(',')
        except AttributeError:
            # Return an empty dictionary if scoring_str is not a string
            return {}

        parsed_dict = {}
        current_key = None

        for part in parts:
            # Attempt to find and parse each key-value pair
            if '=' in part:
                key, value = part.split('=', 1)
                current_key = key.strip()
                parsed_dict[current_key] = value.strip()
            elif current_key is not None:
                # For incomplete values that continue from previous items
                parsed_dict[current_key] += f", {part.strip()}"

        return parsed_dict

    @staticmethod
    def _is_binary_key(key: str) -> bool:
        """
        Helper function to determine if a key should be considered as part of binary scoring.
        - Returns True if the key is '0', '1', or a negative integer.
        """
        try:
            int_key = int(key)
            return int_key in [0, 1] or int_key < 0
        except ValueError:
            return False  # Non-integer keys are not considered binary

    def _modify_multiple_response_codes(self, column_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modifies column_dict to handle multiple response questions by expanding them into separate binary items.
        For each item with multiple responses, creates individual entries for each numeric scoring value.
        """
        new_entries = {}

        for col_name, details in column_dict.items():
            # col_name = 'cir_0200'
            # details = column_dict.get(col_name)
            if details['multiple_responses']:
                # For each numeric scoring item, create a new binary entry
                for score_key, score_value in details['numeric_scoring'].items():
                    # Construct the new key as "col_name_score_key" (e.g., shift_0)
                    new_key = f"{col_name}_{score_key}"
                    new_entries[new_key] = {
                        'description': details['description'] + ' ' + score_value,
                        'numeric_scoring': {1: 'Yes',  # score_value,
                                            0: 'No'},  # Just the single scoring for binary
                        'multiple_responses': False,  # Individual items are now binary, not multiple
                        'dtype': 'binary'
                    }
            else:
                # Retain non-multiple response items as they are
                new_entries[col_name] = details

        # self.print_column_dict(new_entries)

        return new_entries

    def _load_existing_dict(self) -> Dict[str, Any]:
        """
        Loads the dictionary from the existing file based on the file extension (.json or .pkl).
        """
        file_extension = self.output_path.suffix.lower()

        if file_extension == '.json':
            with open(self.output_path, "r") as json_file:
                data = json.load(json_file)
            print(f"Loaded data from JSON at {self.output_path}")

        elif file_extension == '.pkl':
            with open(self.output_path, "rb") as pickle_file:
                data = pickle.load(pickle_file)
            print(f"Loaded data from pickle at {self.output_path}")

        else:
            raise ValueError("Unsupported file extension. Please use '.json' or '.pkl'.")

        return data
    def _save_dict(self, data: Dict) -> None:
        """
        Saves a dictionary to a file in either JSON or pickle format based on the output path extension.

        Parameters:
        - data: Dictionary to save.
        """
        if self.output_path is None:
            print('Unable to save ASQ Dictionary as the output path is None')
            return None

        # Determine the file format based on the file extension
        file_extension = self.output_path.suffix.lower()

        if file_extension == '.json':
            with open(self.output_path, "w") as json_file:
                json.dump(data, json_file, indent=4)
            print(f"Data saved as JSON at {self.output_path}")

        elif file_extension == '.pkl':
            with open(self.output_path, "wb") as pickle_file:
                pickle.dump(data, pickle_file)
            print(f"Data saved as pickle at {self.output_path}")

        else:
            raise ValueError("Unsupported file extension. Please use '.json' or '.pkl'.")

    @staticmethod
    def get_columns_dtypes(asq_dict: Dict, df_asq: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Uses a combination of the asq dict definition and the structure of how variables are stored in the ASQ
        dataset to properly label the data type of the questions.

        Classify columns in a DataFrame as 'categorical', 'ordinal', 'continuous', or 'time' based on their values.
        Categorical variables are integers strictly within the set {0,1,2,3,4,5,6,7}.
        Time variables contain time-formatted values or unique values with '-' or ':' (excluding negative values).



        :param asq_dict: Dictionary containing metadata for each variable
        :param df_asq: DataFrame containing the ASQ data
        :return: A dictionary with data types as keys and lists of column names as value
        """
        data_types = {'categorical': [], 'ordinal': [], 'continuous': [], 'time': []}
        categorical_set = {0, 1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}

        for col in df_asq.columns:
            dtype_dict = asq_dict.get(col)
            if not dtype_dict is None:
                dtype_dict = dtype_dict.get('dtype')
            if any((':' in str(val) or ('-' in str(val) and not str(val).startswith('-'))) for val in
                   df_asq[col].dropna().unique()):
                data_types['time'].append(col)
                continue

            unique_values = set(df_asq[col].dropna().unique())
            if dtype_dict == 'continuous':
                data_types['continuous'].append(col)

            elif (unique_values.issubset(categorical_set) or
                  len(unique_values) < 10 and
                  dtype_dict != 'continuous'):
                data_types['categorical'].append(col)

            else:
                data_types['continuous'].append(col)

        return data_types

if __name__ == '__main__':
    asq_tab_one = AsqDataDictionary(output_path=data_paths.get('asq_dictionary'),
                                    data_dict_path=data_paths.get('root').joinpath(r'docs/asq_dictionary_v4.xlsx'),
                                    overwrite=True)

    asq_dict = asq_tab_one.run()

