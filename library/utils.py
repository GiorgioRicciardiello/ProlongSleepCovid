"""
Helper functions that we can use across scripts
"""
import pathlib
from pathlib import Path
import ast
import pandas as pd
from typing import Optional, Tuple, Union, Any, Dict
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from tqdm import tqdm
import re
import numpy as np
import matplotlib.pyplot as plt


mapper_columns = {
        'dem_0110': 'Age',
        'dem_0500': 'Gender',
        'dem_0800': 'BMI',
        'map_score': 'MAP score',
        'ess_0900': 'ESS score',
        'isi_score': 'ISI score',
        'cir_0700': 'rMEQ Total Score',
        'fosq_1100': 'FOSQ Score',
}

def compute_sparse_encoding(multi_response_col: list[str],
                            df: pd.DataFrame,
                            nan_int: Optional[int] = -200) -> pd.DataFrame:
    """
    Compute sparse encoding to the multiple response columns
    :para, df: pd.Dataframe, dataset
    :param multi_response_col: list[str], columns that are of multiple response type in the dataset
    :para nan_int: Optional[int], integer to mark the nan when doing the exploded dataframe
    :return:
    """

    def make_list(x, nan_replace: Optional[str] = '[-200]'):
        """
        To do the explode all values most be list format. Because they are saved as astrings in the .csv we must
        make all the cells as strings and nans are not recognized by the ast.literal_eval method.
        :param x:
        :param nan_replace: str, optional, string to replace the nan
        :return:
        """
        if pd.isna(x):
            return nan_replace
        elif isinstance(x, list) or isinstance(x, int):
            return str(x)
        else:
            return x

    def sparse_encoding(df_exploded: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Multiple responses can have more than one category, this the index in the df_exploded will have duplicates
        for the asq responses that have more than one response in a cell. Therefore, the traditional
        one-hot-encoding algorithm will result in more rows than it should.

        This function takes this duplicates into consideration and applies the categorical encoding. Be aware, that
        this is not one-hot because in a row we can have more than one category.

        :param df_exploded: pd.Dataframe, exploded dataframe of the column we will one hot encode
        :param column_name: str, column name of the exposed dataframe we are using
        :return:
            pd.Dataframe with the categorical encoding
        """
        # pre-allocate the dummy frame, get the column names based on the unique elements
        values = df_exploded[column_name].unique()
        values.sort()
        columns = [f'{column_name}_{value}' for value in values]
        df_dummy = pd.DataFrame(data=0,
                                columns=columns,
                                index=range(0, df_exploded.index.nunique()))
        # make an index column for us to use as indexes since we need to use uniques when allocating
        df_exploded.reset_index(inplace=True, drop=False, names=['asq_index'])
        for val_, col_ in zip(values, columns):
            # Get the indices where the condition is true
            indices = df_exploded.loc[df_exploded[column_name] == val_, 'asq_index'].values
            df_dummy.loc[indices, col_] = 1

        return df_dummy

    dataset = df.copy()
    for multi_resp_ in multi_response_col:
        print(f'\n{multi_resp_}')
        if not multi_resp_ in dataset.columns:
            print('not in columns')
            continue
        print('processing')
        # multi_resp_ = multi_response_col[0]
        # Make a copy of the column containing lists
        df_multi_resp = pd.DataFrame(
            data=dataset[multi_resp_].copy(),
            columns=[multi_resp_]
        )
        # make all cell with sme str(list) format
        df_multi_resp[multi_resp_] = df_multi_resp[multi_resp_].apply(make_list,
                                                                      nan_replace=f'[{nan_int}]')
        # convert as a list
        df_multi_resp[multi_resp_] = df_multi_resp[multi_resp_].apply(lambda x: ast.literal_eval(x))
        # explode all the lists
        df_exploded = df_multi_resp.explode(column=multi_resp_)
        df_sparse = sparse_encoding(df_exploded=df_exploded,
                                    column_name=multi_resp_)

        df_sparse.rename(columns={f'{multi_resp_}_{nan_int}': f'{multi_resp_}_nan'},
                         inplace=True)
        if not dataset.shape[0] == df_sparse.shape[0]:
            raise ValueError(f'Unmatch dimensions in the rows for columns: '
                             f'{multi_resp_} - ({dataset.shape[0]} vs {df_sparse.shape[0]} )')

        # remove the original column
        dataset.drop(columns=multi_resp_, inplace=True)

        # append the one-hot-encoded version
        dataset = pd.concat([dataset, df_sparse], axis=1)
    return dataset




class FuzzySearch:
    def __init__(self,
                 asq_df: Union[pd.DataFrame, pathlib.Path],
                 subjects_df: Union[pd.DataFrame, pathlib.Path],
                 ):
        """
        Search for matched between the subjects_df and the asq_df
        :param asq_df:
        :param subjects_df:
        """
        self.asq_df = self._read_csv_or_dataframe(asq_df)
        self.subjects_df = self._read_csv_or_dataframe(subjects_df)
        self.dob_variations = ['dob', 'date of birth', 'date_of_birth', 'date-of-birth']
        self.col_name_standard_name = 'name'
        self.col_dob_standard_name = 'date_of_birth'

    @staticmethod
    def _read_csv_or_dataframe(data: Union[pd.DataFrame, pathlib.Path]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, (str, Path)):
            if isinstance(data, Path):
                if data.suffix == '.csv':
                    return pd.read_csv(data)
                else:
                    return pd.read_excel(data)
            else:
                return pd.read_csv(data)
        else:
            raise ValueError("Input must be a DataFrame, a path to a CSV file, or a pathlib.Path.")

    @staticmethod
    def _get_column_intersection(df_one: pd.DataFrame,
                                 df_two: pd.DataFrame) -> list:
        """
        return the intersection of two dataframes columns"
        :param df_one:
        :param df_two:
        :return:
        """
        return list(set(df_one.columns) & set(df_two.columns))

    @staticmethod
    def _check_columns_exist(dataframe: pd.DataFrame,
                             columns_to_check: Union[list, str]) -> bool:
        """Check if the columns we will work with are in the input"""
        if isinstance(columns_to_check, str):
            columns_to_check = [columns_to_check]
        missing_columns = [col for col in columns_to_check if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the dataframe: {missing_columns}")
        else:
            print("All required columns are present in the dataframe.")
            return True

    def assign_mrn_by_name_dob_matches(self, fuzzy_filter: int = 92) -> Union[pd.DataFrame, bool]:
        """
        Insert patient MRN in the ASQ from the subject dataframe
        :param fuzzy_filter:
        :return:
            pd.Dataframe with similar names
        """
        columns = [self.col_name_standard_name, self.col_dob_standard_name]
        # asq = self.asq_df.copy()
        # subjects_df = self.subjects_df.copy()
        self._pre_process_search_by_name_dob_matches(frames=[self.asq_df, self.subjects_df])

        if not self._check_columns_exist(dataframe=self.asq_df, columns_to_check=columns):
            print(F"ASQ Dataframe must have the columns {columns}")
            return False
        columns.append('mrn')
        if not self._check_columns_exist(dataframe=self.subjects_df, columns_to_check=columns):
            print(F"Subject Dataframe must have the columns {columns}")
            return False

        # implement the fuzzy search
        # asq = self.asq_df.copy()
        # subjects_df = self.subjects_df.copy()
        similar_names = []
        # fuzzy_result_df = pd.DataFrame(columns=['asq_name', 'subject_name', 'score', 'asq_dob',
        #                                         'subject_mrn', 'asq_mrn'],
        #                                # index=range(0, self.asq_df.shape[0])
        #                                )
        for idx_, subject in tqdm(self.subjects_df.iterrows(),
                                  total=len(self.subjects_df),
                                  desc="Matching Subjects Name & Dob with ASQ Records"):
            # subject is the patient we are searching in the asq database
            # subject = self.subjects_df.loc[0, :]
            # filter by  date of birth
            asq_dob_matches = self.asq_df[
                self.asq_df[self.col_dob_standard_name] == subject[self.col_dob_standard_name]]
            # in the asq rows with same dob, do a fuzzy search for the name match
            matches = process.extract(subject[self.col_name_standard_name],
                                      asq_dob_matches[self.col_name_standard_name],
                                      scorer=fuzz.token_set_ratio,
                                      limit=asq_dob_matches.shape[0])
            similar_names.extend([{'asq_name': fuzzy_[0],
                                   'subject_name': subject[self.col_name_standard_name],
                                   'score': fuzzy_[1],
                                   'asq_dob': self.asq_df.loc[fuzzy_[2], self.col_dob_standard_name],
                                   'subject_dob': subject[self.col_dob_standard_name],
                                   'subject_mrn': subject['mrn'],
                                   'asq_mrn': self.asq_df.loc[fuzzy_[2], 'mrn'],
                                   'subject_idx': idx_,
                                   'asq_survey_id': self.asq_df.loc[fuzzy_[2], 'survey_id'],
                                   # 'asq_epic_id': self.asq_df.loc[fuzzy_[2], 'epic_id'],
                                   # 'asq_completed': self.asq_df.loc[fuzzy_[2], 'completed'],
                                   } for fuzzy_ in matches])
        # Create a DataFrame with similar names and similarity scores
        similar_names_df = pd.DataFrame(similar_names)
        # filter by those higher than 92
        similar_names_df = similar_names_df[similar_names_df['score'] >= fuzzy_filter]
        # fuzzy_result_df = fuzzy_result_df.append(self._fuzzy_rule(fuzzy_matches=similar_names_df),
        #                                          ignore_index=True)
        return similar_names_df

    @staticmethod
    def _fuzzy_rule(fuzzy_matches: pd.DataFrame) -> pd.DataFrame:
        """
        We should return a dataframe with only one row, the best match
        :param fuzzy_matches:
        :return:
        """
        if fuzzy_matches.asq_name.unique().shape[0] == 1:
            # we have the same name match in all we return
            fuzzy_matches = fuzzy_matches.sort_values(by='score',
                                                      ascending=False)
            return fuzzy_matches.loc[0, :]
        elif fuzzy_matches.asq_name.unique().shape[0] > 1:
            # different names in the fuzzy match, we select the one with highet score
            top_score = fuzzy_matches[fuzzy_matches['score'] == fuzzy_matches['score'].max()]
            return top_score

    def _pre_process_search_by_name_dob_matches(self,
                                                frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Implement a pre-processing operation to the name and date of birth columns
        :param frames:
        :return: list[pd.DataFrame] list of pre-processed dataframes
        """
        for frame in frames:
            # frame = frames[1]
            if any('unnamed:' in column.lower() for column in frame.columns):
                unnamed_drop = [column for column in frame.columns if 'unnamed:' in column.lower()]
                frame.drop(columns=unnamed_drop, inplace=True)
            self._encode_names(frame=frame)
            self._encode_date_of_birth(frame=frame)
        return frames

    def _encode_names(self, frame: pd.DataFrame):
        """
        Create the name column or rename it to the standard convention.
        Creste the column name with the combination of name_fisrt, name_middle and name_last. Middle name is optional
        :param frame:
        :return:
        """
        # if any(column.casefold() == 'name' for column in frame.columns):
        #     name_drop = [column for column in frame.columns if column.casefold() == 'name'][0]
        #     frame.rename(columns={name_drop: self.col_name_standard_name}, inplace=True)

        if all(column in frame.columns for column in ['name_first', 'name_middle', 'name_last']):
            frame[['name_first', 'name_middle', 'name_last']] = frame[
                ['name_first', 'name_middle', 'name_last']].fillna('')

            frame["name"] = (frame["name_first"] + " " + frame["name_middle"] + " " + frame["name_last"]).apply(
                process_name)
            frame.drop(columns=['name_first', 'name_middle', 'name_last'], inplace=True)
            frame["name"] = frame["name"].apply(process_name)

        elif all(column in frame.columns for column in ['name_first', 'name_last']):
            frame[['name_first', 'name_last']] = frame[
                ['name_first', 'name_last']].fillna('')
            frame["name"] = (frame["name_first"] + " " + frame["name_last"]).apply(
                process_name)
            frame.drop(columns=['name_first', 'name_last'], inplace=True)
            frame["name"] = frame["name"].apply(process_name)

        else:
            name_drop = frame.filter(like='name', axis=1).columns
            name_drop = [non_name for non_name in name_drop if non_name != 'name']
            # frame.rename(columns={name_drop[0]: self.col_name_standard_name}, inplace=True)
            frame.drop(columns=name_drop,
                       inplace=True)

    def _encode_date_of_birth(self,
                              frame: pd.DataFrame):
        """
        Encode the date of birth column, it accounts for the missing values (keeps them) and set a string or datetime
        format to the date of birth. Then rename to the standard name for dob
        :param frame:
        :return:
        """
        format = 'time'
        dob_column: str = self._keep_first_occurrence(
            frame=frame,
            column_aliases=self.dob_variations)

        result = frame[dob_column].dropna().copy()
        result = pd.to_datetime(result,
                                errors='coerce').dt.strftime('%Y-%m-%d')
        if format == 'string':
            frame[dob_column] = result.combine_first(frame[dob_column])
        else:
            frame[dob_column] = result.combine_first(pd.to_datetime(frame[dob_column],
                                                                    errors='coerce'))
        frame.rename(columns={dob_column: self.col_dob_standard_name},
                     inplace=True)

    @staticmethod
    def _keep_first_occurrence(frame: pd.DataFrame, column_aliases: list) -> str:
        """
        Sme column with different aliases e.g., dob and date of birth in the same dataframe
        :param frame: datrafem to search the alises columns
        :param column_aliases: list of aliases possible for the same column
        :return:
        """
        #  TODO: test this  name_drop = frame.filter(like='name', axis=1).columns for the dob matches
        column_aliases = set(column_aliases)  # Convert to set to ensure unique variations
        columns_to_keep = []
        for alias in column_aliases:
            for column in frame.columns:
                if alias.casefold() in column.casefold():
                    columns_to_keep.append(column)
                    # break  # Break out of the inner loop after finding the first occurrence
        # keep the first one
        column = columns_to_keep[0]
        # remove all others if present
        if len(columns_to_keep) > 1:
            frame.drop(columns=columns_to_keep[1::],
                       inplace=True)
        return column

    def search_by_name_dob_matches(self, method: str = 'fuzzy', fuzzy_filter:Optional[int]=95) -> pd.DataFrame:
        """
        Wrapper function to search subjects in the main asq by name and dob
        :param method:
        :return:
        """
        self._pre_process_search_by_name_dob_matches(frames=[self.asq_df, self.subjects_df])
        if method == 'exact':
            return self._exact_search_name_dob()
        elif method == 'fuzzy':
            return self._fuzzy_search_name_dob(fuzzy_filter=fuzzy_filter)


    def _exact_search_name_dob(self) -> pd.DataFrame:
        """
        Search for the patient using the date of birth and then the exact name match
        :return:
        """
        result_exact_search = []
        for idx_, subject in tqdm(self.subjects_df.iterrows(),
                                  total=len(self.subjects_df),
                                  desc="Matching Subjects Name & Dob with ASQ Records"):
            # subject is the patient we are searching in the asq database
            # subject = self.subjects_df.loc[0, :]
            # filter by  date of birth
            asq_dob_matches = self.asq_df[
                self.asq_df[self.col_dob_standard_name] == subject[self.col_dob_standard_name]]
            asq_dob_names_matches = asq_dob_matches[asq_dob_matches[self.col_name_standard_name] ==
                                                    subject[self.col_name_standard_name]]
            result_exact_search.append(asq_dob_names_matches.to_dict(orient='records'))

        # Flatten the list of dictionaries and create the DataFrame directly
        return pd.DataFrame([item for sublist in result_exact_search for item in sublist])

    def _fuzzy_search_name_dob(self, fuzzy_filter:int):
        """
        Search for the patient using the date of birth and a fuzzy search for the name match
        :param fuzzy_filter: int, filter to implement in the fuzzy score
        :return:
        """
        similar_dob_name = []
        for idx_, subject in tqdm(self.subjects_df.iterrows(),
                                  total=len(self.subjects_df),
                                  desc="Matching Subjects Name & Dob with ASQ Records"):
            # subject is the patient we are searching in the asq database
            # subject = self.subjects_df.loc[0, :]
            # filter by  date of birth
            asq_dob_matches = self.asq_df[
                self.asq_df[self.col_dob_standard_name] == subject[self.col_dob_standard_name]]
            # in the asq rows with same dob, do a fuzzy search for the name match
            matches = process.extract(subject[self.col_name_standard_name],
                                      asq_dob_matches[self.col_name_standard_name],
                                      scorer=fuzz.token_set_ratio,
                                      limit=asq_dob_matches.shape[0])
            similar_dob_name.extend([{'asq_name': fuzzy_[0],
                                   'subject_name': subject[self.col_name_standard_name],
                                   'score': fuzzy_[1],
                                   'asq_dob': self.asq_df.loc[fuzzy_[2], self.col_dob_standard_name],
                                   'subject_dob': subject[self.col_dob_standard_name],
                                   'subject_idx': idx_,
                                   'asq_survey_id': self.asq_df.loc[fuzzy_[2], 'survey_id'],
                                   # 'asq_epic_id': self.asq_df.loc[fuzzy_[2], 'epic_id'],
                                   # 'asq_completed': self.asq_df.loc[fuzzy_[2], 'completed'],
                                   } for fuzzy_ in matches])
        # Create a DataFrame with similar names and similarity scores
        similar_names_df = pd.DataFrame(similar_dob_name)
        # filter by those higher than 92
        similar_names_df = similar_names_df[similar_names_df['score'] >= fuzzy_filter]
        # fuzzy_result_df = fuzzy_result_df.append(self._fuzzy_rule(fuzzy_matches=similar_names_df),
        #                                          ignore_index=True)
        return similar_names_df

def process_name(name:pd.Series):
    """
    Function to process 'name' column
    Remove spaces and split into first and last names
    :param name:
    :return:
    """
    # Remove spaces and split into first and last names
    if isinstance(name, str):
        # Define a regular expression pattern to match special characters
        # pattern = r'[!@#$%^&*()_+={}\[\]:;"\'<>,.?/\\|\xa0]'
        # Define a regular expression pattern to match non-alphabetic characters
        # pattern = r'[!@#$%^&*()_+={}\[\]:;"\'<>,.?/\\|\xa0℅]'
        # pattern = r'[^a-zA-Z ]'
        pattern = r'[^a-zA-Z -]'  # keep the - in the names
        # Use the re.sub() function to remove non-alphabetic characters
        name = re.sub(pattern, "", name)
        name = name.lstrip()
        name = name.strip()
        name = name.replace("  ", " ")
        name = name.lower()
        # Split the name into parts and capitalize the first letter of each part
        name_parts = name.split()
        name_parts = [part.capitalize() for part in name_parts]

        # Join the parts to form the modified name
        modified_name = " ".join(name_parts)

        return modified_name
    else:
        return name



class NameDateProcessor:
    def __init__(self):
        """
        The class has two methods:
        1. Encode the first middle and last name into single column and encode
        2. change the format of date columns

        Example usage:
        name_processor = NameProcessor()
        processed_dataframe = name_processor.encode_names(your_dataframe)
        """
        self.col_name_standard_name = "name"

    def _process_name(self, name: pd.Series) -> Union[str, pd.Series]:
        """
        Method to process 'name' column
        Remove spaces and split into first and last names
        :param name:
        :return:
        """
        # Remove spaces and split into first and last names
        if isinstance(name, str):
            # Define a regular expression pattern to match non-alphabetic characters
            pattern = r'[^a-zA-Z ]'
            # Use the re.sub() function to remove non-alphabetic characters
            name = re.sub(pattern, "", name)
            name = name.lstrip()
            name = name.strip()
            name = name.replace("  ", " ")
            name = name.lower()
            # Split the name into parts and capitalize the first letter of each part
            name_parts = name.split()
            name_parts = [part.capitalize() for part in name_parts]

            # Join the parts to form the modified name
            modified_name = " ".join(name_parts)

            return modified_name
        else:
            return name

    def encode_names(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Create the name column or rename it to the standard convention.
        Create the column name with the combination of name_first, name_middle, and name_last. Middle name is optional
        :param frame:
        :return:
        """
        if all(column in frame.columns for column in ['name_first', 'name_middle', 'name_last']):
            frame[['name_first', 'name_middle', 'name_last']] = frame[
                ['name_first', 'name_middle', 'name_last']].fillna('')

            # Concatenate columns efficiently and join with space
            frame["name"] = (frame["name_first"] + " " + frame["name_middle"] + " " + frame["name_last"]).apply(
                lambda name: " ".join(name.split())).apply(self._process_name)

            frame.drop(columns=['name_first', 'name_middle', 'name_last'], inplace=True)

            return frame

        elif all(column in frame.columns for column in ['first_name', 'middle_name', 'last_name']):
            frame[['first_name', 'middle_name', 'last_name']] = frame[
                ['first_name', 'middle_name', 'last_name']].fillna('')

            # Concatenate columns efficiently and join with space
            frame["name"] = (frame["first_name"] + " " + frame["middle_name"] + " " + frame["last_name"]).apply(
                lambda name: " ".join(name.split())).apply(self._process_name)

            frame.drop(columns=['first_name', 'middle_name', 'last_name'], inplace=True)

            return frame

        elif all(column in frame.columns for column in ['name_first', 'name_last']):
            frame[['name_first', 'name_last']] = frame[
                ['name_first', 'name_last']].fillna('')

            # Concatenate columns efficiently and join with space
            frame["name"] = (frame["name_first"] + " " + frame["name_last"]).apply(
                lambda name: " ".join(name.split())).apply(self._process_name)

            frame.drop(columns=['name_first', 'name_last'], inplace=True)

            return frame

        elif all(column in frame.columns for column in ['first_name', 'last_name']):
            frame[['first_name', 'last_name']] = frame[
                ['first_name', 'last_name']].fillna('')

            # Concatenate columns efficiently and join with space
            frame["name"] = (frame["first_name"] + " " + frame["last_name"]).apply(
                lambda name: " ".join(name.split())).apply(self._process_name)

            frame.drop(columns=['first_name', 'last_name'], inplace=True)

            return frame

    def encode_date_columns(self, frame: pd.DataFrame, dob_column:str) -> pd.DataFrame:
        """
        Encode the date of birth column, it accounts for the missing values (keeps them) and set a string or datetime
        format to the date of birth. Then rename to the standard name for dob
        :param frame:
        :return:
        """
        format = 'time'
        result = frame[dob_column].dropna().copy()
        result = pd.to_datetime(result,
                                errors='coerce').dt.strftime('%Y-%m-%d')
        if format == 'string':
            frame[dob_column] = result.combine_first(frame[dob_column])
        else:
            frame[dob_column] = result.combine_first(pd.to_datetime(frame[dob_column],
                                                                    errors='coerce'))
        return frame


def grouped_bar_chart_sleep_covid(df:pd.DataFrame,
                                  questions:list[str],
                                  style:Optional[str] = 'mpl20',
                                  cmap:Optional[str] = 'tab10',
                                  figsize:Optional[Tuple[int, int]] = (10, 6),
                                  title:str= ''
                                  ):
    """
    Bar plot with multiple bars on each axis.
    :param df: dataframe
    :param questions: list of the columns (ordinal) to plot in the same figure
    :return:
    """
    scores = range(int(df[questions].max().max()+1))  # Likert scale from 0 to 5
    # Calculate the frequency of each score for each question
    data = {question: df[question].value_counts().sort_index() for question in questions}
    # Align missing score categories with 0 count
    for question in questions:
        for score in scores:
            if score not in data[question]:
                data[question][score] = 0
        data[question] = data[question].sort_index()

    plt.style.use(style)
    cmap = plt.get_cmap(cmap)  # Use the qualitative 'tab10' colormap
    # Create a grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)
    # plt.style.use('grayscale')
    bar_width = 0.25
    index = np.arange(len(scores))
    # Plot each question as a separate set of bars
    for i, question in enumerate(questions):
        bars = ax.bar(index + i * bar_width, data[question], bar_width,
               label=question,
               color=cmap(i))

        # Add annotations for each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval),
                    va='bottom', ha='center', fontsize=10)

    # Add labels and title
    ax.set_xlabel('Score', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(scores)
    ax.set_xlim(left=df[questions].min().min()-0.5,
                right=len(scores))
    ax.legend()
    plt.grid(alpha=0.7)
    plt.tight_layout()
    plt.show()

def merge_racial_categories(df, dem_1000_col, dem_1010_col):
    """
    This function merges two racial category columns into a single column
    based on the provided dictionaries. If one column has an 'Unknown' value,
    it uses the other. If both are 'Unknown', it returns 'Unknown'.

    Args:
    df (pd.DataFrame): The DataFrame containing the two columns.
    dem_1000_col (str): The column name for the 'Main' racial category.
    dem_1010_col (str): The column name for the 'Sub' racial category.

    Returns:
    pd.DataFrame: The original DataFrame with an additional column 'combined_category'.
    """

    # Define the main and sub category mappings
    main_category = {
        0: 'blank',
        1: 'White/Caucasian',
        2: 'Black/African American',
        3: 'Amer Indian/Alaska Native',
        4: 'Asian',
        5: 'Pacific Islander',
        6: 'Two Races'
    }

    sub_category = {
        0: 'European',
        1: 'Middle Eastern',
        2: 'N. African',
        3: 'Other White/European',
        4: 'Asian Indian',
        5: 'Chinese',
        6: 'Filipino',
        7: 'Japanese',
        8: 'Korean',
        9: 'Vietnamese',
        10: 'Other Asian',
        11: 'Native Hawaiian',
        12: 'Guamanian or Chamorro',
        13: 'Samoan',
        14: 'Other Pacific Islander'
    }

    # Function to create a single category for each row
    def create_single_category(row) -> str:
        main = main_category.get(row[dem_1000_col], 'Unknown')
        sub = sub_category.get(row[dem_1010_col], 'Unknown')

        # Logic to combine categories or select the non-Unknown value
        if main != 'Unknown' and sub != 'Unknown':
            return main  # Combine both values if neither is 'Unknown'
        elif main != 'Unknown':
            return main  # Use main if it's valid
        elif sub != 'Unknown':
            return sub  # Use sub if it's valid
        else:
            return 'Unknown'  # If both are 'Unknown', return 'Unknown'

    # Apply the function to each row in the DataFrame to create the combined column
    df['race'] = df.apply(create_single_category, axis=1)

    # Return the DataFrame with the new column
    return df


def circular_plot_circadian_rhythm(df: pd.DataFrame,
                  column: str,
                  labels: Dict[int, str],
                  title: str,
                  plot: Optional[bool] = False,
                  ax: Optional[plt.Axes] = None,
                  hue: Optional[str] = None) -> None:
    """
    Plots a circular chart on the provided ax, with optional grouping by hue.

    If axes are given you can used to populate subplots in a figure e.g.,:

    schedule_dict = {
        "cir_0200": {
            5: "5-6:30am",
            4: "6:30-7:45am",
            3: "7:45-9:45am",
            2: "9:45-11am",
            1: "11am-12pm"
        },
        "cir_0300": {
            1: "Very tired",
            2: "Fairly tired",
            3: "Fairly refreshed",
            4: "Very refreshed"
        },
        }
    fig, axs = plt.subplots(nrows=2,
                        ncols=2,
                        figsize=(12, 12),
                        subplot_kw=dict(polar=True))

    # Plot each of the circular plots
    circular_plot_circadian_rhythm(ax=axs[0, 0],
                  df=df_asq,
                  column='cir_0200',
                  hue='sleep_covid_cat',
                  labels=schedule_dict.get('cir_0200'),
                  title="Ideal Waking Time")
    circular_plot_circadian_rhythm(ax=axs[0, 1],
                  df=df_asq,
                  column='cir_0300',
                  hue='sleep_covid_cat',
                  labels=schedule_dict.get('cir_0300'),
    plt.tight_layout()
    plt.show()

    :param df: DataFrame containing the data to be plotted
    :param column: Name of the column to plot
    :param labels: Dictionary mapping numeric values to category labels for the plot
    :param title: Title of the plot
    :param plot: Whether to display the plot immediately after creation
    :param ax: Matplotlib axis to plot the circular chart on
    :param hue: Optional categorical column to split the data by groups
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Define angles for the circular plot
    num_categories = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

    # If a hue is provided, split the data by hue and plot each group's distribution
    if hue:
        groups = df[hue].unique()
        colors = plt.get_cmap('tab10').colors  # Get a color palette for the groups
        for i, group in enumerate(groups):
            # Filter the dataframe for each group
            group_df = df[df[hue] == group]
            group_size = len(group_df)  # Calculate the sample size for this group
            # Get value counts for the group and ensure all labels are included
            values = group_df[column].value_counts().sort_index().reindex(labels.keys(), fill_value=0)
            values = values.tolist() + [values.iloc[0]]  # Make the plot circular
            ax.fill(angles + [angles[0]], values, color=colors[i], alpha=0.3, label=f'{group} (n={group_size})')
            ax.plot(angles + [angles[0]], values, color=colors[i], linewidth=2)
    else:
        # If no hue is provided, plot the entire dataset
        values = df[column].value_counts().sort_index().reindex(labels.keys(), fill_value=0)
        values = values.tolist() + [values.iloc[0]]  # Make the plot circular
        ax.fill(angles + [angles[0]], values, color='skyblue', alpha=0.4)
        ax.plot(angles + [angles[0]], values, color='blue', linewidth=2)

    # Add labels to the plot
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels([labels[i] for i in labels], fontsize=10)
    ax.set_title(title, fontsize=12)

    # Add legend with sample sizes
    if hue:
        ax.legend(loc='upper right')

    if plot:
        plt.tight_layout()
        plt.show()



columns_dop = [
"bthbts_0310",
"cir_0200",
"cir_0300",
"cir_0400",
"cir_0500",
"cir_0600",
"cir_0700",
"dem_0110",
"dem_0400",
"dem_0500",
"dem_0800",
"dem_0900",
"dem_1000",
"dem_1100",
"diet_0400",
"ess_0100",
"ess_0200",
"ess_0300",
"ess_0400",
"ess_0500",
"ess_0600",
"ess_0700",
"ess_0800",
"ess_0900",
"famhx_1300",
"fosq_0100",
"fosq_0200",
"fosq_0300",
"fosq_0400",
"fosq_0500",
"fosq_0600",
"fosq_0700",
"fosq_0800",
"fosq_0900",
"fosq_1000",
"fosq_1100",
"gad_0100",
"gad_0200",
"gad_0900",
"isi_0100",
"isi_0200",
"isi_0300",
"isi_0400",
"isi_0500",
"isi_0600",
"isi_0700",
"isi_score",
"map_0500",
"map_0800",
"map_1030",
"map_1120",
"map_index_4",
"map_lr",
"map_score",
"mdhx_5700",
"mdhx_5710",
"mdhx_5720",
"mdhx_5800",
"mdhx_5810",
"mdhx_5820",
"mdhx_5900",
"mdhx_5910",
"mdhx_5920",
"mdhx_5950",
"mdhx_6000",
"mdhx_6100",
"mdhx_6200",
"mdhx_6300",
"mdhx_6310",
"mdhx_6320",
"mdhx_6400",
"mdhx_6500",
"mdhx_6600",
"mdhx_6700",
"narc_0100",
"narc_0200",
"narc_2100",
"nose_0100",
"nose_0200",
"nose_0500",
"osa_0400",
"osa_0500",
"par_0210",
"par_0310",
"par_0510",
"par_0610",
"par_0710",
"par_0910",
"phq_0100",
"phq_0200",
"phq_1100",
"rls_probability",
"rls_severity",
"sched_0901",
"sched_2001",
"sched_2210",
"sched_2300",
"sched_3710",
"sched_3800",
"sched_4150",
"slpy_0150",
"slpy_0650",
"slpy_0750",
"soclhx_0110",
"soclhx_0701",
"soclhx_0710",
"soclhx_1310",
"mdhx_urinary_kidney_problem_0",
"mdhx_urinary_kidney_problem_1",
"mdhx_urinary_kidney_problem_2",
"mdhx_urinary_kidney_problem_3",
"mdhx_orthodontics_0",
"mdhx_orthodontics_1",
"mdhx_hematological_disease_0",
"mdhx_hematological_disease_1",
"mdhx_hematological_disease_2",
"mdhx_autoimmune_disease_1",
"mdhx_autoimmune_disease_2",
"mdhx_autoimmune_disease_3",
"mdhx_autoimmune_disease_4",
"mdhx_autoimmune_disease_5",
"mdhx_autoimmune_disease_6",
"mdhx_autoimmune_disease_7",
"mdhx_autoimmune_disease_8",
"mdhx_autoimmune_disease_9",
"soclhx_rec_drug_0",
"soclhx_rec_drug_1",
"soclhx_rec_drug_2",
"soclhx_rec_drug_3",
"soclhx_rec_drug_6",
"soclhx_rec_drug_7",
"soclhx_rec_drug_10",
"sched_rotating_shift_1",
"sched_rotating_shift_4",
"mdhx_anxiety_problem_0",
"mdhx_anxiety_problem_1",
"mdhx_anxiety_problem_2",
"mdhx_cancer_0",
"mdhx_cancer_1",
"mdhx_cancer_5",
"mdhx_neurology_problem_0",
"mdhx_neurology_problem_3",
"mdhx_neurology_problem_4",
"mdhx_neurology_problem_5",
"mdhx_neurology_problem_6",
"mdhx_neurology_problem_7",
"bthbts_employment_0",
"bthbts_employment_1",
"bthbts_employment_2",
"bthbts_employment_3",
"bthbts_employment_4",
"bthbts_employment_5",
"bthbts_employment_6",
"bthbts_employment_7",
"bthbts_employment_8",
"sched_alarm_clock_unused_0",
"sched_alarm_clock_unused_1",
"sched_alarm_clock_unused_2",
"mdhx_eating_disorder_0",
"mdhx_eating_disorder_1",
"mdhx_eating_disorder_2",
"mdhx_eating_disorder_3",
"mdhx_psych_problem_0",
"mdhx_psych_problem_1",
"mdhx_psych_problem_2",
"mdhx_psych_problem_3",
"mdhx_psych_problem_4",
"mdhx_psych_problem_6",
"mdhx_psych_problem_7",
"mdhx_psych_problem_8",
"mdhx_ent_surgery_0",
"mdhx_ent_surgery_1",
"mdhx_ent_surgery_2",
"mdhx_ent_surgery_7",
"soclhx_tobacco_use_0",
"soclhx_tobacco_use_1",
"soclhx_tobacco_use_2",
"soclhx_tobacco_use_3",
"soclhx_tobacco_use_4",
"famhx_anxiety_0",
"famhx_anxiety_1",
"famhx_anxiety_2",
"famhx_anxiety_3",
"famhx_anxiety_4",
"mdhx_pap_improvement_0",
"mdhx_pap_improvement_1",
"mdhx_pap_improvement_2",
"mdhx_pap_improvement_3",
"mdhx_pap_improvement_4",
"mdhx_pap_improvement_5",
"mdhx_pap_improvement_6",
"mdhx_pap_improvement_7",
"mdhx_pap_improvement_8",
"mdhx_pap_improvement_9",
"mdhx_pap_improvement_10",
"mdhx_pap_improvement_11",
"mdhx_pap_improvement_12",
"mdhx_pap_improvement_13",
"mdhx_pap_improvement_14",
"mdhx_pap_improvement_15",
"mdhx_pap_improvement_16",
"mdhx_pap_improvement_17",
"mdhx_pap_improvement_18",
"famhx_sleep_apnea_0",
"famhx_sleep_apnea_1",
"famhx_sleep_apnea_2",
"famhx_sleep_apnea_3",
"famhx_sleep_apnea_4",
"famhx_sleep_death_0",
"famhx_sleep_death_1",
"famhx_sleep_death_2",
"mdhx_other_problem_0",
"mdhx_other_problem_1",
"mdhx_other_problem_2",
"mdhx_other_problem_3",
"mdhx_other_problem_4",
"mdhx_other_problem_5",
"mdhx_other_problem_6",
"mdhx_other_problem_7",
"mdhx_other_problem_8",
"mdhx_ent_problem_0",
"mdhx_ent_problem_1",
"mdhx_ent_problem_2",
"mdhx_ent_problem_3",
"famhx_depression_0",
"famhx_depression_1",
"famhx_depression_2",
"famhx_depression_3",
"famhx_depression_4",
"famhx_rls_0",
"famhx_rls_1",
"famhx_rls_2",
"famhx_rls_4",
"famhx_narcolepsy_1",
"famhx_narcolepsy_2",
"famhx_narcolepsy_4",
"mdhx_sleep_diagnosis_0",
"mdhx_sleep_diagnosis_1",
"mdhx_sleep_diagnosis_2",
"mdhx_sleep_diagnosis_3",
"mdhx_sleep_diagnosis_4",
"mdhx_sleep_diagnosis_7",
"mdhx_sleep_diagnosis_9",
"mdhx_sleep_diagnosis_10",
"famhx_other_sleep_disorder_0",
"famhx_other_sleep_disorder_1",
"famhx_other_sleep_disorder_2",
"mdhx_cardio_problem_0",
"mdhx_cardio_problem_1",
"mdhx_cardio_problem_3",
"mdhx_cardio_problem_4",
"mdhx_cardio_problem_5",
"mdhx_cardio_problem_6",
"mdhx_cardio_problem_7",
"mdhx_cardio_problem_8",
"mdhx_cardio_problem_9",
"famhx_sleepwalk_0",
"famhx_sleepwalk_1",
"famhx_sleepwalk_2",
"famhx_sleepwalk_4",
"famhx_psych_treatment_0",
"famhx_psych_treatment_1",
"famhx_psych_treatment_2",
"famhx_psych_treatment_3",
"famhx_psych_treatment_4",
"sched_alarm_clock_use_0",
"sched_alarm_clock_use_1",
"sched_alarm_clock_use_2",
"sched_alarm_clock_use_3",
"sched_alarm_clock_use_4",
"mdhx_pap_problem_0",
"mdhx_pap_problem_1",
"mdhx_pap_problem_2",
"mdhx_pap_problem_3",
"mdhx_pap_problem_4",
"mdhx_pap_problem_5",
"mdhx_pap_problem_6",
"mdhx_pap_problem_7",
"mdhx_pap_problem_8",
"mdhx_pap_problem_9",
"mdhx_pap_problem_10",
"mdhx_pap_problem_14",
"mdhx_pap_problem_16",
"mdhx_pulmonary_problem_0",
"mdhx_pulmonary_problem_1",
"mdhx_pulmonary_problem_2",
"mdhx_pulmonary_problem_3",
"mdhx_pulmonary_problem_4",
"famhx_insomnia_0",
"famhx_insomnia_1",
"famhx_insomnia_2",
"famhx_insomnia_3",
"famhx_insomnia_4",
"mdhx_cardio_surgery_3",
"famhx_psych_illness_0",
"famhx_psych_illness_1",
"famhx_psych_illness_2",
"famhx_psych_illness_3",
"famhx_psych_illness_4",
"mdhx_pain_fatigue_problem_0",
"mdhx_pain_fatigue_problem_1",
"mdhx_pain_fatigue_problem_2",
"mdhx_pain_fatigue_problem_3",
"mdhx_pain_fatigue_problem_4",
"mdhx_pain_fatigue_problem_5",
"famhx_fibromyalgia_0",
"famhx_fibromyalgia_1",
"famhx_fibromyalgia_2",
"famhx_fibromyalgia_3",
"famhx_fibromyalgia_4",
"mdhx_headache_problem_0",
"mdhx_headache_problem_1",
"mdhx_headache_problem_2",
"mdhx_headache_problem_3",
"mdhx_dental_work_0",
"mdhx_dental_work_1",
"mdhx_dental_work_2",
"mdhx_dental_work_3",
"mdhx_dental_work_4",
"mdhx_dental_work_5",
"mdhx_dental_work_6",
"mdhx_dental_work_8",
"bthbts_sleep_disruption_0",
"bthbts_sleep_disruption_1",
"bthbts_sleep_disruption_2",
"bthbts_sleep_disruption_3",
"bthbts_sleep_disruption_4",
"bthbts_sleep_disruption_5",
"bthbts_sleep_disruption_6",
"bthbts_sleep_disruption_7",
"bthbts_sleep_disruption_8",
"bthbts_sleep_disruption_9",
"bthbts_sleep_disruption_10",
"bthbts_sleep_disruption_11",
"mdhx_sleep_problem_0",
"mdhx_sleep_problem_1",
"mdhx_sleep_problem_2",
"mdhx_sleep_problem_3",
"mdhx_sleep_problem_4",
"mdhx_sleep_problem_5",
"mdhx_sleep_problem_6",
"mdhx_sleep_problem_7",
"mdhx_sleep_problem_8",
"mdhx_sleep_problem_9",
"mdhx_sleep_problem_10",
"mdhx_sleep_problem_11",
"mdhx_sleep_problem_12",
"mdhx_sleep_problem_13",
"mdhx_sleep_problem_14",
"mdhx_metabolic_endocrine_problem_0",
"mdhx_metabolic_endocrine_problem_1",
"mdhx_metabolic_endocrine_problem_2",
"mdhx_metabolic_endocrine_problem_3",
"mdhx_sleep_treatment_0",
"mdhx_sleep_treatment_1",
"mdhx_sleep_treatment_2",
"mdhx_sleep_treatment_3",
"mdhx_sleep_treatment_4",
"mdhx_sleep_treatment_5",
"mdhx_sleep_treatment_6",
"mdhx_sleep_treatment_7",
"mdhx_sleep_treatment_8",
"mdhx_sleep_treatment_9",
"mdhx_sleep_treatment_10",
"mdhx_sleep_treatment_11",
"mdhx_sleep_treatment_12",
"mdhx_sleep_treatment_14",
"cortisol_level",
"UNREFRESHING SLEEP",
"DIFFICULTY SLEEPING",
"DAYTIME SLEEPINESS",
"hospitalized",
"covid_vaccine",
"prevalence_sleep_symptom",
"date_asq_question",
"headaches",
"nasal_congestion",
"fatigue",
"brain_fog",
"unrefreshing_sleep",
"insomnia",
"lethargy",
"post_exercial_malaise",
"change_in_smell",
"change_in_taste",
"anx_depression",
"cough",
"shortness_breath",
"lightheadedness",
"gi_symptoms",
"race",
"sched_0700_angle",
"sched_0800_angle",
"sched_0900_angle",
"sched_1000_angle",
"sched_1900_angle",
"sched_2000_angle",
"soclhx_0850_angle",
"soclhx_1000_angle",
"soclhx_1450_angle",
"soclhx_1590_angle",
"cortisol_time_angle",
"cluster",
]

columns_sparse_to_drop = [
    'bthbts_employment',
    'bthbts_sleep_disruption',
    'famhx_anxiety',
    'famhx_fibromyalgia',
    'famhx_insomnia',
    'famhx_narcolepsy',
    'famhx_other_sleep_disorder',
    'famhx_psych_treatment',
    'famhx_rls',
    'famhx_sleep_apnea',
    'famhx_sleepwalk',
    # 'mdhx_autoimmune_disease',
    # 'mdhx_cancer',
    # 'mdhx_cardio_problem',
    'mdhx_cardio_surgery',
    'mdhx_dental_work',
    # 'mdhx_eating_disorder',
    'mdhx_ent_problem',
    'mdhx_ent_surgery',
    # 'mdhx_headache_problem',
    # 'mdhx_hematological_disease',
    # 'mdhx_metabolic_endocrine_problem',
    # 'mdhx_neurology_problem',
    # 'mdhx_orthodontics',
    'mdhx_other_problem',
    # 'mdhx_pain_fatigue_problem',
    'mdhx_pap_problem',
    'mdhx_psych_problem',
    # 'mdhx_pulmonary_problem',
    # 'mdhx_sleep_diagnosis',
    # 'mdhx_sleep_problem',
    # 'mdhx_sleep_treatment',
    'mdhx_urinary_kidney_problem',
    'sched_alarm_clock_unused',
    'sched_rotating_shift',
    'soclhx_rec_drug',
    'soclhx_tobacco_use'
]