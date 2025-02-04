import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.discrete_model import MNLogit
import pandas as pd
import numpy as np
from scipy.stats import norm

class ComputeRegression:
    def __init__(self,
                 df: pd.DataFrame,
                 covariates: list[str],
                 target_variable: str,
                 alpha: float = 0.05,
                 correction_method: str = None,
                 regression_type: str = 'ordinal'):
        """
        Compute simple regression models
        :param df:
        :param covariates:
        :param target_variable:
        :param alpha:
        :param correction_method:
        :param regression_type:
        """
        if regression_type not in ['ordinal', 'linear', 'logistic', 'multinomial']:
            raise ValueError("Unsupported regression type. Use 'ordinal', 'logistic', 'linear', or 'multinomial'.")

        if correction_method not in [None, 'bonferroni', 'fdr']:
            raise ValueError("Invalid correction method. Choose 'bonferroni' or 'fdr'.")
        self.correction_method = correction_method
        self.regression_type = regression_type
        self.covariates = covariates
        self.target_variable = target_variable
        self.alpha = alpha
        self.df = df[[self.target_variable] + self.covariates].dropna()
        self._check_covariates()

    def _check_covariates(self):
        if self.target_variable in self.covariates:
            raise ValueError(f'The output {self.target_variable} is in the covariate list: {self.covariates}.')

    def compute_regression(self):
        dp = self.df.copy()
        # Handle regression type
        if self.regression_type == 'ordinal':
            # Ensure target_variable is treated as an ordered categorical variable
            dp[self.target_variable] = pd.Categorical(dp[self.target_variable], ordered=True)
            # Fit the OrderedModel
            mod = OrderedModel(endog=dp[self.target_variable],
                               exog=dp[self.covariates],
                               distr='logit')
            result = mod.fit(method='bfgs', disp=False)

        elif self.regression_type == 'logistic':
            # Fit Logistic Regression using Logit
            dp[self.target_variable] = dp[self.target_variable].astype('category')
            mod = sm.Logit(dp[self.target_variable].cat.codes, sm.add_constant(dp[self.covariates]))
            result = mod.fit(disp=False)

        elif self.regression_type == 'linear':
            # Fit Linear Regression using OLS
            mod = sm.OLS(dp[self.target_variable], sm.add_constant(dp[self.covariates]))
            result = mod.fit()

        elif self.regression_type == 'multinomial':
            # Fit Multinomial Logistic Regression using MNLogit
            dp[self.target_variable] = dp[self.target_variable].astype('category')
            mod = MNLogit(dp[self.target_variable].cat.codes, sm.add_constant(dp[self.covariates]))
            result = mod.fit(method='newton', disp=False)

        return self._extract_model_params(result, dp)

    def _extract_model_params(self, result, dp:pd.DataFrame) -> pd.DataFrame:
        """
        Extracts a standardized table of metrics for all regression types.

        Parameters:
        - result: Fitted model result object.
        - dp: DataFrame used in regression (for sample size, etc.)

        Returns:
        - pd.DataFrame: Summary table of regression results.
        """
        df_summary = pd.DataFrame()

        # Compute z-critical value for confidence intervals
        z_critical = norm.ppf(1 - self.alpha / 2)

        if self.regression_type in ['ordinal', 'logistic', 'linear']:
            # Extract parameters
            params = result.params
            std_errors = result.bse
            p_values = result.pvalues

            # Compute confidence intervals
            ci_lower = params - z_critical * std_errors
            ci_upper = params + z_critical * std_errors

            # Compute odds ratios and their confidence intervals for logistic/ordinal
            if self.regression_type in ['ordinal', 'logistic', 'linear']:
                odds_ratios = np.exp(params)
                ci_lower_or = np.exp(ci_lower)
                ci_upper_or = np.exp(ci_upper)
            else:
                odds_ratios = [np.nan] * len(params)
                ci_lower_or = [np.nan] * len(params)
                ci_upper_or = [np.nan] * len(params)

            # Compile the summary table
            df_summary = pd.DataFrame({
                'Feature': params.index,
                'Coefficient': params.values,
                'Standard Error': std_errors.values,
                'P-Value': p_values.values,
                'CI Lower (Coefficient)': ci_lower.values,
                'CI Upper (Coefficient)': ci_upper.values,
                'Odds Ratio': odds_ratios,
                'CI Lower (OR)': ci_lower_or,
                'CI Upper (OR)': ci_upper_or,
            })

        elif self.regression_type == 'multinomial':
            # For multinomial logistic regression
            params = result.params
            std_errors = result.bse
            p_values = result.pvalues

            # Initialize list to store dataframes for each outcome category
            dfs = []

            # For each category except the base category
            categories = dp[self.target_variable].cat.categories
            for idx, category in enumerate(categories[:-1]):
                # Get parameters for this category
                category_params = params.iloc[idx]
                category_std_errors = std_errors.iloc[idx]
                category_p_values = p_values.iloc[idx]

                # Compute confidence intervals
                ci_lower = category_params - z_critical * category_std_errors
                ci_upper = category_params + z_critical * category_std_errors

                # Compute odds ratios and their confidence intervals
                odds_ratios = np.exp(category_params)
                ci_lower_or = np.exp(ci_lower)
                ci_upper_or = np.exp(ci_upper)

                # Compile the summary table for this category
                df = pd.DataFrame({
                    'Feature': category_params.index,
                    'Coefficient': category_params.values,
                    'Standard Error': category_std_errors.values,
                    'P-Value': category_p_values.values,
                    'CI Lower (Coefficient)': ci_lower.values,
                    'CI Upper (Coefficient)': ci_upper.values,
                    'Odds Ratio': odds_ratios,
                    'CI Lower (OR)': ci_lower_or,
                    'CI Upper (OR)': ci_upper_or,
                })
                df['Outcome Category'] = category
                dfs.append(df)

            # Concatenate all category dataframes
            df_summary = pd.concat(dfs, ignore_index=True)

        # Add sample size and model info
        df_summary['Sample Size'] = dp.shape[0]
        df_summary['Model'] = self.regression_type.capitalize()
        df_summary['Outcome'] = self.target_variable

        # # Adjust p-values if correction method is specified
        # if self.correction_method == 'bonferroni':
        #     df_summary['Adjusted P-Value'] = df_summary['P-Value'] * len(df_summary)
        # elif self.correction_method == 'fdr':
        #     from statsmodels.stats.multitest import fdrcorrection
        #     _, adj_pvals = fdrcorrection(df_summary['P-Value'])
        #     df_summary['Adjusted P-Value'] = adj_pvals
        # else:
        #     df_summary['Adjusted P-Value'] = df_summary['P-Value']

        # Determine significance
        # df_summary['Significant'] = df_summary['Adjusted P-Value'] < self.alpha

        # Include goodness-of-fit metrics
        if hasattr(result, 'rsquared'):
            df_summary['R-Squared'] = result.rsquared
        if hasattr(result, 'prsquared'):
            df_summary['Pseudo R-Squared'] = result.prsquared
        df_summary['AIC'] = result.aic if hasattr(result, 'aic') else np.nan
        df_summary['BIC'] = result.bic if hasattr(result, 'bic') else np.nan

        # Round the results for clarity
        numeric_cols = ['Coefficient', 'Standard Error', 'P-Value',
                        # 'Adjusted P-Value',
                        'CI Lower (Coefficient)', 'CI Upper (Coefficient)',
                        'Odds Ratio', 'CI Lower (OR)', 'CI Upper (OR)',
                        # 'R-Squared', 'Pseudo R-Squared', 'AIC', 'BIC'
                        ]

        df_summary[numeric_cols] = df_summary[numeric_cols].applymap(lambda x: round(x, 3))

        return df_summary

