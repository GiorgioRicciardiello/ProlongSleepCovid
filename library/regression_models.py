from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.api import Logit
from scipy.stats import chi2
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from statsmodels.stats.multitest import fdrcorrection

from sklearn.linear_model import LogisticRegression, LinearRegression
from statsmodels.api import add_constant

def compute_multiple_unadjusted_ordinal_model(features_list: list[str],
                                              df: pd.DataFrame,
                                              target_variable: str = 'pos_exp_better_cat',
                                              alpha: float = 0.05,
                                              correction_method: str = 'fdr') -> pd.DataFrame:
    """
    Creates a summary table containing coefficients, odds ratios (OR), confidence intervals for OR,
    intercepts, performance metrics, and sample sizes for each feature in the provided feature list,
    applying either Bonferroni or FDR correction.

    Parameters:
    - features_list (list): List of feature names to include in the analysis.
    - df (pd.DataFrame): DataFrame containing the features and the target variable.
    - target_variable (str): Name of the target variable column.
    - alpha (float): Significance level for correction.
    - correction_method (str): 'bonferroni' for Bonferroni correction or 'fdr' for FDR correction.

    Returns:
    - pd.DataFrame: A DataFrame with rows as feature names and columns for coefficients,
                    OR, confidence intervals for OR, adjusted p-values, intercepts, R², F1 score, and sample sizes.
    """
    features_list = [feat for feat in features_list if feat != target_variable]
    betas, ors, ci_lower_list, ci_upper_list, std_errors = [], [], [], [], []
    ci_lower_or, ci_upper_or, p_values, sample_sizes = [], [], [], []
    feature_names, f1_scores, pseudo_r2, log_likelihoods, aics, bics = [], [], [], [], [], []

    # Dictionary to store intercepts by threshold name for each feature
    intercepts_dict = {}

    for feature in features_list:
        # Subset the DataFrame for the current feature and the target variable
        dp = df[[feature, target_variable]].dropna()

        # Log the sample size
        sample_size = dp.shape[0]
        sample_sizes.append(sample_size)

        # Ensure target_variable is treated as an ordered categorical variable
        dp[target_variable] = pd.Categorical(dp[target_variable], ordered=True)

        # Fit the OrderedModel
        mod = OrderedModel(dp[target_variable], dp[[feature]], distr='logit')
        result = mod.fit(method='bfgs', disp=False)

        # Calculate OR and confidence intervals for OR
        beta = result.params[feature]
        ci = result.conf_int().loc[feature]
        or_val = np.exp(beta)
        ci_lower, ci_upper = np.exp(ci[0]), np.exp(ci[1])
        standard_error = result.bse[feature]
        # Store feature name, beta, OR, confidence intervals for OR, p-value
        feature_names.append(feature)
        betas.append(beta)
        ors.append(or_val)
        ci_lower_list.append(ci[0])
        ci_upper_list.append(ci[1])
        ci_lower_or.append(ci_lower)
        ci_upper_or.append(ci_upper)
        p_values.append(result.pvalues[feature])
        std_errors.append(standard_error)

        # Store intercepts for each threshold with their labels
        intercepts = {name: result.params[name] for name in result.params.index if '/' in name}
        intercepts_dict[feature] = intercepts

        # Calculate Pseudo R²
        llf = result.llf  # Log-likelihood of the model
        llnull = result.llnull  # Log-likelihood of a null model
        pseudo_r2_val = 1 - (llf / llnull)
        pseudo_r2.append(pseudo_r2_val)

        # Calculate F1 score
        predicted_classes = result.predict().argmax(axis=1)  # Classify by highest probability
        f1 = f1_score(dp[target_variable].cat.codes, predicted_classes, average='weighted')
        f1_scores.append(f1)

        # Log-Likelihood, AIC, and BIC
        log_likelihoods.append(result.llf)
        aics.append(result.aic)
        bics.append(result.bic)

    # Apply correction based on the selected method
    if correction_method == 'bonferroni':
        corrected_alpha = alpha / len(features_list)
        significance = [p < corrected_alpha for p in p_values]
    elif correction_method == 'fdr':
        _, significance = fdrcorrection(p_values, alpha=alpha, method='indep')
    else:
        raise ValueError("Invalid correction method. Choose 'bonferroni' or 'fdr'.")

    # Create a DataFrame for the summary
    summary_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': betas,
        'Standard Error': std_errors,
        'CI Lower': ci_lower_list,
        'CI Upper': ci_upper_list,
        'OR': ors,
        'CI Lower (OR)': ci_lower_or,
        'CI Upper (OR)': ci_upper_or,
        'P-value': p_values,
        f'Significant - {correction_method.capitalize()}': significance,
        'Pseudo R²': pseudo_r2,
        'F1 Score': f1_scores,
        'Log-Likelihood': log_likelihoods,
        'AIC': aics,
        'BIC': bics,
        'Sample Size': sample_sizes,
        'Model': ['OrderedModel'] * len(sample_sizes),
        'Outcome': [target_variable] * len(sample_sizes)
    })

    # Convert intercepts_dict to DataFrame for easier concatenation
    intercepts_df = pd.DataFrame(intercepts_dict).T
    intercepts_df.columns = [f"Intercept {col}" for col in intercepts_df.columns]
    # Compute the OR of the intercepts
    for col_ in intercepts_df:
        intercepts_df[col_.replace('Intercept ', 'Intercept OR ')] = intercepts_df[col_].apply(lambda x: np.exp(x))

    # Concatenate intercepts to summary_df
    summary_df = pd.concat([summary_df.set_index('Feature'), intercepts_df], axis=1).reset_index()
    summary_df.rename(columns={'index': 'Feature'}, inplace=True)
    # col_order = ['Feature'] + ['Coefficient', 'CI Lower', 'CI Upper'] + [*summary_df.columns][4:-4]
    # summary_df = summary_df[col_order]
    # Sort by OR in ascending order, then by P-value in ascending order within each OR group
    summary_df.sort_values(by=['OR', 'P-value'], ascending=[False, True], inplace=True)

    # Round the results for clarity
    third_decimal = ['CI Upper', 'CI Lower', 'P-value', 'Pseudo R²', 'F1 Score']
    second_decimal = ['Coefficient', 'OR']
    for dec in third_decimal:
        summary_df[dec] = summary_df[dec].astype(float).round(3)
    for dec in second_decimal:
        summary_df[dec] = summary_df[dec].astype(float).round(2)

    return summary_df

def compute_ordinal_model_adjusted(features_iterate: list[str],
                                   features_adjust: list[str],
                                   df: pd.DataFrame,
                                   target_variable: str = 'pos_exp_better_cat',
                                   alpha: float = 0.05,
                                   correction_method: str = 'fdr') -> pd.DataFrame:
    """
    Creates a summary table containing coefficients, odds ratios (OR), confidence intervals for OR,
    intercepts as ORs, p-values (with correction), and model performance metrics.

    Parameters:
    - features_iterate (list): List of feature names to include in the analysis.
    - features_adjust (list): List of features to adjust for in the model.
    - df (pd.DataFrame): DataFrame containing the features and the target variable.
    - target_variable (str): Name of the target variable column.
    - alpha (float): Significance level for correction.
    - correction_method (str): 'bonferroni' for Bonferroni correction or 'fdr' for FDR correction.

    Returns:
    - pd.DataFrame: A DataFrame with rows as feature names and columns for coefficients,
                    OR, confidence intervals for OR, p-values, adjusted alpha, model metrics, and sample sizes.
    """
    betas, ors, ci_lower_list, ci_upper_list, standard_errors = [], [], [], [], []
    ci_lower_or, ci_upper_or, p_values, sample_sizes = [], [], [], []
    feature_names, pseudo_r2, log_likelihoods, aics, bics = [], [], [], [], []
    corrected_alpha = alpha / len(features_iterate) if correction_method == 'bonferroni' else alpha

    # Dictionary to store intercepts OR by threshold name for each feature
    intercepts_dict = {}

    for feature in features_iterate:
        # Create a model dataframe with the current feature, target, and adjustment features
        model_features = [feature] + features_adjust
        dp = df[[target_variable] + model_features].dropna()

        # Log the sample size
        sample_size = dp.shape[0]
        sample_sizes.append(sample_size)

        # Ensure target_variable is treated as an ordered categorical variable
        dp[target_variable] = pd.Categorical(dp[target_variable], ordered=True)

        # Fit the OrderedModel with adjustments
        mod = OrderedModel(dp[target_variable], dp[model_features], distr='logit')
        result = mod.fit(method='bfgs', disp=False)

        # Extract the coefficient, confidence intervals, OR, and p-value for the primary feature
        beta = result.params[feature]
        ci = result.conf_int().loc[feature]
        or_val = np.exp(beta)
        ci_lower, ci_upper = np.exp(ci[0]), np.exp(ci[1])
        standard_error = result.bse[feature]

        # Append values to respective lists
        feature_names.append(feature)
        betas.append(beta)
        ors.append(or_val)
        ci_lower_list.append(ci[0])
        ci_upper_list.append(ci[1])
        ci_lower_or.append(ci_lower)
        ci_upper_or.append(ci_upper)
        p_values.append(result.pvalues[feature])
        standard_errors.append(standard_error)
        # Transform and store intercepts as ORs
        intercepts = {name: result.params[name] for name in result.params.index if '/' in name}
        intercepts_dict[feature] = intercepts

        # Model performance metrics
        pseudo_r2_val = 1 - (result.llf / result.llnull)  # McFadden’s Pseudo R²
        pseudo_r2.append(pseudo_r2_val)
        log_likelihoods.append(result.llf)
        aics.append(result.aic)
        bics.append(result.bic)

    # Apply correction based on the selected method
    if correction_method == 'bonferroni':
        significance = [p < corrected_alpha for p in p_values]
    elif correction_method == 'fdr':
        a, significance = fdrcorrection(p_values, alpha=alpha, method='indep')
        # signd = {a_: sig_ for a_, sig_ in zip(feature_names, a)}
    else:
        raise ValueError("Invalid correction method. Choose 'bonferroni' or 'fdr'.")

    # Create a DataFrame for the summary
    summary_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': betas,
        'Standard Error': standard_errors,
        'CI Lower': ci_lower_list,
        'CI Upper': ci_upper_list,
        'OR': ors,
        'CI Lower (OR)': ci_lower_or,
        'CI Upper (OR)': ci_upper_or,
        'P-value': p_values,
        'Corrected Alpha': [corrected_alpha] * len(sample_sizes),
        f'Significant {correction_method.capitalize()}': significance,
        'Pseudo R²': pseudo_r2,
        'Log-Likelihood': log_likelihoods,
        'AIC': aics,
        'BIC': bics,
        'Sample Size': sample_sizes,
        'Model': ['OrderedModel'] * len(sample_sizes),
        'Outcome': [target_variable] * len(sample_sizes),
        'Adjusted for': [", ".join(features_adjust)] * len(sample_sizes),
        'Adjust number of': [len(features_adjust)] * len(sample_sizes)
    })

    # Convert intercepts_dict to DataFrame for easier concatenation, including ORs
    intercepts_df = pd.DataFrame(intercepts_dict).T
    intercepts_df.columns = [f"Intercept {col}" for col in intercepts_df.columns]
    # Compute the OR of the intercepts
    for col_ in intercepts_df:
        intercepts_df[col_.replace('Intercept ', 'Intercept OR ')] = intercepts_df[col_].apply(lambda x: np.exp(x))

    # Concatenate intercepts OR to summary_df
    summary_df = pd.concat([summary_df.set_index('Feature'), intercepts_df], axis=1).reset_index()
    summary_df.rename(columns={'index': 'Feature'}, inplace=True)

    # Sort the summary first by OR in ascending order, then by P-value in ascending order within each OR
    summary_df.sort_values(by=['OR', 'P-value'], ascending=[False, True], inplace=True)

    # Round the results for clarity
    third_decimal = ['CI Upper', 'CI Lower', 'P-value', 'Pseudo R²', 'Log-Likelihood', 'AIC', 'BIC']
    second_decimal = ['Coefficient', 'OR'] + [inter for inter in [*intercepts_df.columns] if 'OR' in inter]
    for dec in third_decimal:
        summary_df[dec] = summary_df[dec].astype(float).round(3)
    for dec in second_decimal:
        summary_df[dec] = summary_df[dec].astype(float).round(2)

    return summary_df



def compute_full_model_all_covariates_ordinal(covariates: list[str],
                                      df: pd.DataFrame,
                                      target_variable: str = 'pos_exp_better_cat',
                                      alpha: float = 0.05) -> pd.DataFrame:
    """
    Creates a summary table containing coefficients, odds ratios (OR), confidence intervals for OR,
    intercepts as ORs, p-values, and sample sizes for each feature in the provided feature list,
    using a single model with all covariates and additional model performance metrics.

    Parameters:
    - covariates (list): List of features to adjust for in the model.
    - df (pd.DataFrame): DataFrame containing the features and the target variable.
    - target_variable (str): Name of the target variable column.
    - alpha (float): Significance level for determining statistical significance.

    Returns:
    - pd.DataFrame: A DataFrame with rows as feature names and columns for coefficients,
                    OR, confidence intervals for OR, p-values, intercepts, model metrics, and sample sizes.
    """
    # Remove target_variable if mistakenly included in covariates
    covariates = [cov for cov in covariates if cov != target_variable]
    dp = df[[target_variable] + covariates].dropna()

    # Ensure target_variable is treated as an ordered categorical variable
    dp[target_variable] = pd.Categorical(dp[target_variable], ordered=True)

    # Fit the single OrderedModel with all covariates
    mod = OrderedModel(endog=dp[target_variable], exog=dp[covariates], distr='logit')
    result = mod.fit(method='bfgs', disp=False)

    # Extract summary table and convert it to a DataFrame, handle missing columns, and rename columns robustly
    summary_table = result.summary().tables[1]
    summary_data = summary_table.data[1:]  # Skip header row
    summary_columns = summary_table.data[0]  # Use first row as headers

    # Create DataFrame from summary table data
    summary_df = pd.DataFrame(summary_data, columns=summary_columns)

    # Rename columns with robust handling of names
    summary_df.columns = [col.strip() for col in summary_df.columns]  # Remove extra spaces
    summary_df.rename(columns={
        '': 'Feature',
        'coef': 'Coefficient',
        'std err': 'Standard Error',
        'z': 'z-value',
        'P>|z|': 'P-value',
        '[0.025': 'CI Lower',
        '0.975]': 'CI Upper'
    }, inplace=True)

    # Convert relevant columns to numeric if needed (to handle any formatting issues)
    numeric_cols = ['Coefficient', 'Standard Error', 'z-value', 'P-value', 'CI Lower', 'CI Upper']
    summary_df[numeric_cols] = summary_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Extract intercepts as odds ratios (ORs)
    intercepts_dict = {
        name: result.params[name]
        for name in result.params.index if '/' in name
    }
    intercepts_df = pd.DataFrame.from_dict(intercepts_dict, orient='index', columns=['Coefficient'])
    intercepts_df['OR'] = intercepts_df['Coefficient'].apply(lambda x: np.exp(x))

    # Calculate model performance metrics
    pseudo_r2 = 1 - (result.llf / result.llnull)  # McFadden’s Pseudo R²
    log_likelihood = result.llf
    aic = result.aic
    bic = result.bic

    for or_ in ['Coefficient', 'CI Lower', 'CI Upper']:
        if or_ == 'Coefficient':
            col = 'OR'
        else:
            col = f'{or_ } (OR)'
        summary_df[col] = summary_df[or_].apply(lambda x: np.exp(x))

    # Add model performance metrics and other details to summary DataFrame
    alpha = 0.05  # significance level
    summary_df['Significant (alpha={})'.format(alpha)] = summary_df['P-value'] < alpha
    summary_df['Sample Size'] = dp.shape[0]
    summary_df['Model'] = 'OrderedModel'
    summary_df['Outcome'] = target_variable
    summary_df['Log-Likelihood'] = log_likelihood
    summary_df['AIC'] = aic
    summary_df['BIC'] = bic
    summary_df['Pseudo R²'] = pseudo_r2

    # Sort the summary first by OR in ascending order, then by P-value in ascending order within each OR
    summary_df.sort_values(by=['OR', 'P-value'], ascending=[False, True], inplace=True)

    # Round the results for clarity
    third_decimal = ['CI Upper', 'CI Lower', 'CI Lower (OR)', 'CI Upper (OR)', 'P-value', 'Log-Likelihood', 'AIC',
                     'BIC', 'Pseudo R²']
    second_decimal = ['Coefficient', 'OR'] + [inter for inter in [*intercepts_df.columns] if 'OR' in inter]
    for dec in third_decimal:
        summary_df[dec] = summary_df[dec].astype(float).round(3)

    for dec in second_decimal:
        summary_df[dec] = summary_df[dec].astype(float).round(2)

    return summary_df


def compute_full_model_all_covariates(covariates: list[str],
                                      df: pd.DataFrame,
                                      target_variable: str = 'pos_exp_better_cat',
                                      alpha: float = 0.05,
                                      regression_type: str = 'ordinal') -> pd.DataFrame:
    """
    Creates a summary table for different types of regression (ordinal, logistic, or linear).

    Parameters:
    - covariates (list): List of features to adjust for in the model.
    - df (pd.DataFrame): DataFrame containing the features and the target variable.
    - target_variable (str): Name of the target variable column.
    - alpha (float): Significance level for determining statistical significance.
    - regression_type (str): Type of regression ('ordinal', 'logistic', or 'linear').

    Returns:
    - pd.DataFrame: A summary DataFrame of the model coefficients, OR, confidence intervals, p-values, and metrics.
    """
    # Remove target_variable if mistakenly included in covariates
    covariates = [cov for cov in covariates if cov != target_variable]
    dp = df[[target_variable] + covariates].dropna()

    # Handle regression type
    if regression_type == 'ordinal':
        # Ensure target_variable is treated as an ordered categorical variable
        dp[target_variable] = pd.Categorical(dp[target_variable], ordered=True)

        # Fit the OrderedModel
        mod = OrderedModel(endog=dp[target_variable], exog=dp[covariates], distr='logit')
        result = mod.fit(method='bfgs', disp=False)

        # Prepare summary and metrics specific to OrderedModel
        pseudo_r2 = 1 - (result.llf / result.llnull)  # McFadden’s Pseudo R²
        aic, bic, log_likelihood = result.aic, result.bic, result.llf

    elif regression_type == 'logistic':
        # Fit Logistic Regression
        mod = LogisticRegression(max_iter=1000)
        result = mod.fit(dp[covariates], dp[target_variable])
        coefficients = mod.coef_[0]
        intercept = mod.intercept_[0]

        # Extract model metrics
        log_likelihood = None  # Not directly available
        aic, bic, pseudo_r2 = None, None, None  # Custom computation needed if required


    elif regression_type == 'linear':
        # Fit Linear Regression
        mod = LinearRegression()
        mod.fit(dp[covariates], dp[target_variable])
        coefficients = mod.coef_
        intercept = mod.intercept_

        # Extract model metrics
        log_likelihood, aic, bic, pseudo_r2 = None, None, None, None  # Not applicable for linear regression
    else:
        raise ValueError("Unsupported regression type. Use 'ordinal', 'logistic', or 'linear'.")

    # Create DataFrame for summary results
    summary_df = pd.DataFrame()
    summary_df['Feature'] = covariates
    summary_df['Coefficient'] = coefficients
    if regression_type != 'linear':  # Odds ratios applicable only to logistic and ordinal
        summary_df['OR'] = np.exp(coefficients)

    # Calculate significance
    summary_df['Significant (alpha={})'.format(alpha)] = summary_df[
                                                             'P-value'] < alpha if 'P-value' in summary_df.columns else None

    # Add model performance metrics
    summary_df['Model'] = regression_type
    summary_df['Outcome'] = target_variable
    summary_df['Log-Likelihood'] = log_likelihood
    summary_df['AIC'] = aic
    summary_df['BIC'] = bic
    summary_df['Pseudo R²'] = pseudo_r2

    return summary_df


def check_proportional_odds_assumption(df: pd.DataFrame,
                                       outcome_variable: str,
                                       predictors: list[str]) -> pd.DataFrame:
    """
    Checks the proportional odds assumption in an ordinal logistic regression model.
    - Fitting Separate Binary Logistic Models: For each cumulative threshold in the ordinal outcome, it creates a
    binary logistic regression model for each predictor.

    - Comparing Coefficients Across Thresholds: It calculates chi-squared statistics based on the differences in
    coefficients across thresholds, with a p-value indicating if the differences are significant.

    - Interpreting Results: For each predictor, if the p-value is less than 0.05, the proportional odds assumption
    is considered "Violated"; otherwise, it's "Not Violated".

    Parameters:
    - df (pd.DataFrame): DataFrame containing the outcome variable and predictors.
    - outcome_variable (str): Name of the ordinal outcome variable.
    - predictors (list): List of predictor variables to include in the model.

    Returns:
    - dict: Results for each predictor variable, indicating potential violations of
            the proportional odds assumption.
    """
    predictors = [pred for pred in predictors if pred != outcome_variable]
    # Ensure the outcome variable is ordered categorical
    df[outcome_variable] = pd.Categorical(df[outcome_variable], ordered=True)
    levels = df[outcome_variable].cat.categories

    # Dictionary to store results for each predictor
    assumption_results = {}

    # Loop over each predictor
    for predictor in predictors:
        # Create a clean model DataFrame without missing or infinite values for each predictor
        df_model = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[outcome_variable, predictor]).copy()

        # Fit binary logistic models for each threshold
        thresholds = []
        for i in range(len(levels) - 1):
            # Create a binary outcome for the cumulative threshold
            threshold_col = f"{outcome_variable}_threshold_{i}"
            df_model[threshold_col] = (df_model[outcome_variable].cat.codes <= i).astype(int)

            # Fit the logistic model
            model = Logit(df_model[threshold_col], sm.add_constant(df_model[predictor]))
            result = model.fit(disp=False)

            # Store coefficient and standard error for each threshold
            thresholds.append((result.params[predictor], result.bse[predictor]))

            # Drop the temporary threshold column after fitting
            df_model.drop(columns=[threshold_col], inplace=True)

        # Check the consistency of coefficients across thresholds
        coef_vals = [th[0] for th in thresholds]
        se_vals = [th[1] for th in thresholds]

        # Calculate a chi-squared statistic for the difference in coefficients
        chi2_stat = sum([(coef_vals[i] - coef_vals[i + 1]) ** 2 / (se_vals[i] ** 2 + se_vals[i + 1] ** 2)
                         for i in range(len(coef_vals) - 1)])
        p_value = 1 - chi2.cdf(chi2_stat, df=len(coef_vals) - 1)

        # Store results
        assumption_results[predictor] = {
            "Coefficient Difference Chi2": chi2_stat,
            "P-value": p_value,
            "Proportional Odds Assumption": "Violated" if p_value < 0.05 else "Not Violated"
        }

    df_assumption_results = pd.DataFrame(assumption_results).T
    df_assumption_results.reset_index(inplace=True, drop=False)
    return df_assumption_results
