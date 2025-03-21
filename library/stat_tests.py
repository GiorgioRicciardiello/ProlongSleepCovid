import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from tabulate import tabulate
import statsmodels.formula.api as smf
import numpy as np

def stats_test_binary_symptoms(data: pd.DataFrame,
                               symptoms_mapping: Dict[str, str],
                               gender_col: str = 'gender',
                               binary_cut_off: int = 3) -> pd.DataFrame:
    """
    The test is used for binary data. For such we will convert the ordinal data into binary when needed by
    selecting the severuty trehsold of 4 as positve for the positive. If the data is already binary we keep it

    Parameters
    ----------
    data
    symptoms_mapping
    gender_col
    binary_cut_off

    Returns
    -------

    """
    # Define helper functions to compute counts and percentages.
    get_counts_rates = lambda cond: (cond.sum(), cond.mean() * 100)
    show_2x2 = True
    results = []
    for symptom, formal_name in symptoms_mapping.items():
        # Create a local copy that includes the symptom and the gender column.
        df = data[[symptom, gender_col]].dropna().copy()
        unique_vals = set(df[symptom].unique())
        if not unique_vals == {0, 1}:
            # make column as binary
            df[symptom] = df[symptom].apply(lambda x: 1 if x >= binary_cut_off else 0)

        res = {'Symptom': formal_name}

        grp0 = df[df[gender_col] == 0][symptom] == 1
        grp1 = df[df[gender_col] == 1][symptom] == 1

        female_n, female_rate = get_counts_rates(grp0)
        male_n, male_rate = get_counts_rates(grp1)
        total_n, total_rate = get_counts_rates(df[symptom] == 1)

        assert total_n == male_n + female_n
        # Construct a 2x2 contingency table:
        # - Each row corresponds to a gender group (first row: females, second row: males).
        # - The first column holds the count of positive responses (value == 1) for the group.
        # - The second column holds the count of negative responses (value == 0) for the group,
        #   computed as the total number of subjects in that gender minus the number of positives.
        table = [
            # [female_n, df[df[gender_col] == 0].shape[0] - female_n],
            [male_n, df[df[gender_col] == 1].shape[0] - male_n],
            [female_n, df[df[gender_col] == 0].shape[0] - female_n],

        ]
        if show_2x2:
            table_df = pd.DataFrame(
                table,
                # index=['Female', 'Male'],
                index=['Male', 'Female'],
                columns=['Positive', 'Negative']
            )
            print(f'\n2x2 {symptom}: \n {table_df}')

        fisher_stat = fisher_exact(table, alternative='two-sided')

        res.update({
            'Male (n)': round(male_n, 1),
            'Male (%)': round(male_rate, 1),
            'Female (n)': round(female_n, 1),
            'Female (%)': round(female_rate, 1),
            'Total (n)': round(total_n, 1),
            'Total (%)': round(total_rate, 1),
            'P-value (Fisher)': round(fisher_stat.pvalue, 4),
            'Stats (Fisher)': round(fisher_stat.statistic, 3),
            'Stat Method': "Fisher's Exact Test"
        })
        results.append(res)
    return pd.DataFrame(results)


def stats_test_ordinal_symptoms(data: pd.DataFrame,
                                symptoms_mapping: Dict[str, str],
                                gender_col: str = 'gender', ) -> pd.DataFrame:
    """
    For each symptom in symptoms_mapping assumed to be ordinal (or non-binary), perform tests
    by first dichotomizing responses using a threshold of 4 (i.e. positive if >= 4). In addition,
    perform Spearman's correlation and the Mann-Whitney U test (without dichotomizing the data)
    to capture the ordinal association.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the symptom responses and the gender column.
    symptoms_mapping : Dict[str, str]
        Dictionary mapping symptom column names to formal names.
    gender_col : str, default 'gender'
        Column name for the gender grouping.

    Returns
    -------
    pd.DataFrame
        DataFrame with counts (after dichotomizing at a threshold of 4), p-values from Spearman and
        Mann-Whitney U tests, effect size, and a column indicating the test method used.
    """
    results = []
    thresh = 4  # Threshold for dichotomizing ordinal data.

    # Define a helper lambda to compute count and percentage for a boolean Series.
    get_counts_rates = lambda cond: (cond.sum(), cond.mean() * 100)

    for symptom, formal_name in symptoms_mapping.items():
        # Create a local copy including the symptom and gender column.
        df = data[[symptom, gender_col]].dropna().copy()
        unique_vals = set(df[symptom].unique())
        if unique_vals == {0, 1}:
            continue
        # Dichotomize the symptom responses using the threshold.
        dichotomized = (df[symptom] >= thresh).astype(int)
        df['dichotomized'] = dichotomized

        res = {'Symptom': formal_name}

        # Compute counts and percentages based on the dichotomized column.
        grp0 = (df[df[gender_col] == 0]['dichotomized'] == 1)
        grp1 = (df[df[gender_col] == 1]['dichotomized'] == 1)
        female_n, female_rate = get_counts_rates(grp0)
        male_n, male_rate = get_counts_rates(grp1)
        total_n, total_rate = get_counts_rates(df['dichotomized'] == 1)

        # Compute Spearman's correlation (using the raw ordinal data)
        corr, p_spearman = spearmanr(df[gender_col], df[symptom])
        # Perform Mann-Whitney U test (using the raw ordinal data)
        group_female = df[df[gender_col] == 0][symptom]
        group_male = df[df[gender_col] == 1][symptom]
        u_stat, p_mann = mannwhitneyu(group_female, group_male, alternative='two-sided')
        n1, n2 = group_female.shape[0], group_male.shape[0]
        effect_size_r = u_stat / (n1 * n2) if n1 * n2 != 0 else 0

        res.update({
            'Male (n)': round(male_n, 1),
            'Male (%)': round(male_rate, 1),
            'Female (n)': round(female_n, 1),
            'Female (%)': round(female_rate, 1),
            'Total (n)': round(total_n, 1),
            'Total (%)': round(total_rate, 1),
            'P-value (Spearman)': round(p_spearman, 4),
            'P-value (Mann-Whitney U)': round(p_mann, 4),
            'Effect Size (r)': round(effect_size_r, 3),
            'Stat Method': "Mann-Whitney U / Spearman"
        })
        results.append(res)

    return pd.DataFrame(results)


def correct_pvalues(df: pd.DataFrame,
                    pvalue_columns: list,
                    method: Optional[str] = "fdr_bh") -> pd.DataFrame:
    """
    Corrects p-values in the specified columns of the input DataFrame using
    the specified multiple testing correction method.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame generated from your stats computations containing p-value columns.
    pvalue_columns : list
        List of column names in the DataFrame that contain the p-values to be corrected.
    method : str, default "fdr_bh"
        The multiple testing correction method to use. Options include "bonferroni", "fdr_bh", etc.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with new columns added for each corrected p-value.
        The new column name is the original p-value column name with '_corrected' appended.
    """
    df_corr = df.copy()
    for col in pvalue_columns:
        if col in df_corr.columns:
            # Convert column values to float (if necessary) and apply correction.
            pvals = df_corr[col].astype(float).values
            # multipletests returns a tuple, the second element is the corrected p-values.
            _, pvals_corr, _, _ = multipletests(pvals, method=method)
            # Add a new column with the corrected p-values.
            df_corr[f"{col}_corrected"] = pvals_corr
        else:
            print(f"Warning: Column '{col}' not found in the DataFrame.")
    return df_corr



def run_regression_models(df: pd.DataFrame, targets:List[str]):
    """
    Run logistic regression models for multiple outcomes specified in `targets`.

    This function first applies z-score normalization to continuous predictors (Duration, Age, BMI),
    then converts specified variables (Gender, Race, Hospitalized, and Vaccine Status) to categorical types.
    Logistic regression models are constructed using the statsmodels formula interface, which automatically
    dummy codes categorical variables with reference category 0 (e.g., for Gender, 0 = female; for Race, 0 = reference).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing predictors and target outcome variables.
    targets : List[str]
        List of target outcome column names in `df` for which the regression models will be run.

    Returns
    -------
    pd.DataFrame
        A summary DataFrame containing regression results, including coefficients, standard errors, odds ratios
        with 95% confidence intervals, p-values, and sample sizes.
    """
    # Helper function for z-score normalization
    def z_score_norm(series: pd.Series) -> pd.Series:
        return (series - series.mean()) / series.std()

    # Normalize continuous predictors
    for col in ['Duration', 'Age', 'BMI']:
        df[col] = z_score_norm(df[col])


    # Convert categorical columns to categorical types
    for cat in ['Gender', 'Race', 'Hospitalized',  'Vaccine Status']:
        df[cat] = df[cat].astype('category')

    # Define models as a list of tuples (outcome, list of predictors)
    # The predictors list is used to help construct the formula string.
    predictors = [var for var in df.columns if not var in targets]
    models = [(target, predictors) for target in targets]
    print(f'Number of logistic regression models to test {len(models)} ')
    # List to store regression results
    results = []
    # Run each model using the formula interface
    for outcome, predictors in models:
        # outcome = models[0][0]
        # predictors = models[0][1]
        print(f'Running regression model for outcome: {outcome}')

        # Build the formula string.
        # We wrap variables with spaces in Q(), and mark categorical predictors using C(...).
        # Gender and Race are treated as categorical with reference category 0.
        formula = (
            f"Q('{outcome}') ~ Age + Duration + "
            f"C(Gender, Treatment(reference=0)) + BMI + Hospitalized + "
            f"C(Q('Vaccine Status'), Treatment(reference=0))  +"
            f"C(Race, Treatment(reference=0))"
        )

        # Drop rows with missing values in variables used by this model
        df_current_model = df[predictors + [outcome]].dropna()

        # Fit the logistic regression model with robust covariance (HC1)
        model = smf.logit(formula=formula, data=df_current_model).fit(cov_type='HC1', disp=False)

        # Extract and store results for each predictor in the model (including the Intercept)
        for param, coef in model.params.items():
            pval = model.pvalues[param]
            se = model.bse[param]
            conf_low, conf_high = model.conf_int().loc[param]
            # Odds ratios and confidence intervals
            or_val = np.exp(coef)
            or_conf_str = f"({np.exp(conf_low):.2f}, {np.exp(conf_high):.2f})"
            results.append({
                'Outcome': outcome,
                'Variable': param,
                'Standard Error': f'{se:.3f}',
                'OR (95% CI)': f"{or_val:.3f} {or_conf_str}",
                'p-value': f"{pval:.3e}",
                'Sample': df.shape[0]
            })

    # Convert the results to a DataFrame for summary reporting
    return pd.DataFrame(results)
