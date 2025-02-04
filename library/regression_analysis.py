import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.graphics.gofplots import qqplot
import numpy as np
from typing import Optional


class RegressionAnalysis:
    def __init__(self, df: pd.DataFrame, outcome: str, base_vars: list[str], interest_vars: list[str], model_type: str):
        """
        Initialize the regression analysis class.

        Parameters:
        - df: DataFrame containing the data
        - outcome: The dependent variable
        - base_vars: List of base variables for adjustment
        - interest_vars: List of variables of interest
        - model_type: Type of regression ('linear_reg', 'logit_reg', 'ordinal_reg')
        """
        if model_type not in ['linear_reg', 'logit_reg', 'ordinal_reg']:
            raise ValueError("Model type must be 'linear_reg', 'logit_reg', or 'ordinal_reg'")

        self.df = df
        self.outcome = outcome
        self.base_vars = base_vars
        self.interest_vars = interest_vars
        self.model_type = model_type
        self.results = []

    def _fit_model(self, X, y):
        """
        Fit the appropriate model based on the model_type.
        """
        if self.model_type == 'linear_reg':
            return sm.OLS(y, X).fit(cov_type='HC0')
        elif self.model_type == 'logit_reg':
            return sm.Logit(y, X).fit(disp=False, cov_type='HC0')
        elif self.model_type == 'ordinal_reg':
            return OrderedModel(y, X, distr='logit').fit(method='bfgs', disp=False)

    def calculate_r2(self, model, X, y):
        """
        Calculate the R-squared or pseudo-R-squared depending on the model type.
        """
        if self.model_type == 'linear_reg':
            return model.rsquared  # Linear regression R^2
        else:
            # Pseudo R-squared for logistic or ordinal regression using McFadden's R^2
            null_model = sm.Logit(y, np.ones((len(y), 1))).fit(disp=False) if self.model_type != 'linear_reg' else None
            return 1 - (model.llf / null_model.llf)  # McFadden's pseudo R^2

    def extract_parameters(self, model, variables):
        """
        Extract coefficients, standard errors, and confidence intervals for the specified variables.
        """
        params = []
        for var in variables:
            coef = model.params[var]
            se = model.bse[var]
            ci_lower, ci_upper = model.conf_int().loc[var]
            params.append({
                'Variable': var,
                'Coefficient': coef,
                '95% CI Lower': ci_lower,
                '95% CI Upper': ci_upper,
                'Standard Error': se
            })
        return params

    def run_analysis(self):
        """
        Run the regression analysis for both unadjusted and adjusted models and store the results.
        """
        # Unadjusted models
        for var in self.interest_vars:
            df_model = self.df[[self.outcome, var]].dropna().copy()
            X = sm.add_constant(df_model[[var]])
            y = df_model[self.outcome]

            model = self._fit_model(X, y)
            r2 = self.calculate_r2(model, X, y)

            params = self.extract_parameters(model, ['const', var])

            for param in params:
                or_value = np.round(np.exp(param['Coefficient']), 4)
                or_ci = [round(np.exp(param['95% CI Lower']), 4), round(np.exp(param['95% CI Upper']), 4)]

                self.results.append({
                    'Variable': param['Variable'],
                    'Adjusted': 'No' if 'const' in param['Variable'] else 'Yes',
                    'Coefficient': np.round(param['Coefficient'], 4),
                    'Coef CI 95%': [round(param['95% CI Lower'], 4), round(param['95% CI Upper'], 4)],
                    'OR': or_value,
                    'OR CI 95%': or_ci,
                    'Standard Error': param['Standard Error'],
                    'R-squared': r2,
                    'Sample Size': df_model.shape[0],
                    'Model Variables': ['const', var]
                })

        # Adjusted models
        for var in self.interest_vars:
            df_model = self.df[self.base_vars + [self.outcome, var]].dropna().copy()
            X = sm.add_constant(df_model[self.base_vars + [var]])
            y = df_model[self.outcome]

            model = self._fit_model(X, y)
            r2 = self.calculate_r2(model, X, y)

            params = self.extract_parameters(model, ['const'] + self.base_vars + [var])

            for param in params:
                # Compute OR and OR CI for all models
                or_value = np.round(np.exp(param['Coefficient']), 4)
                or_ci = [round(np.exp(param['95% CI Lower']), 4), round(np.exp(param['95% CI Upper']), 4)]

                self.results.append({
                    'Variable': param['Variable'],
                    'Adjusted': 'No' if 'const' in param['Variable'] else 'Yes',
                    'Coefficient': np.round(param['Coefficient'], 4),
                    'Coef CI 95%': [round(param['95% CI Lower'], 4), round(param['95% CI Upper'], 4)],
                    'OR': or_value,
                    'OR CI 95%': or_ci,
                    'Standard Error': np.round(param['Standard Error'], 4),
                    'R-squared': np.round(r2, 4),
                    'Sample Size': df_model.shape[0],
                    'Model Variables': ['const'] + self.base_vars + [var]
                })
        return pd.DataFrame(self.results)

    def plot_relationship_and_qq(self, feature: str):
        """
        Plot QQ plot and feature vs. outcome scatter plot in a 1-row, 2-column figure.

        If the model is logit or ordinal, apply the logistic transformation to the outcome.
        """
        df_model = self.df[[self.outcome, feature]].dropna()

        # Apply logistic transformation if necessary
        if self.model_type in ['logit_reg', 'ordinal_reg']:
            y = 1 / (1 + np.exp(-df_model[self.outcome]))  # Logistic transformation
        else:
            y = df_model[self.outcome]

        X = df_model[feature]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # QQ Plot
        qqplot(y, line='s', ax=axes[0])
        axes[0].set_title('QQ Plot of Outcome')

        # Scatter plot of feature vs outcome
        axes[1].scatter(X, y, alpha=0.7, edgecolor='k')
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel('Outcome')
        axes[1].set_title(f'{feature} vs. Outcome')

        plt.tight_layout()
        plt.show()

    def get_results(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def transform_results_to_multiindex_fixed(self,
                                              orientation:Optional[str]='horizontal' ) -> pd.DataFrame:
        """
        Transform the results DataFrame into a multi-index DataFrame.
        The upper columns should be the model variables, and the second level should contain
        the metrics like OR, OR CI, R-squared, and Sample Size.

        Parameters:
        - orientation: str, if to return the results dataframe vertical (columns as metrics) or
            horizontal, which is a broad dataframe with multi columns

        Returns:
        - MultiIndex DataFrame
        """
        if not orientation in ['vertical', 'horizontal']:
            raise ValueError('Incorrect orientation parameter: valids are vertical or horizontal')
        df_results = pd.DataFrame(self.results)

        df_results['Model Variables_str'] = df_results['Model Variables'].apply(lambda x: ', '.join(x))


        model_map = {model: f'model {idx}' for idx, model in enumerate(df_results['Model Variables_str'].unique())}
        df_results['model_var_map'] = df_results['Model Variables_str'].map(model_map)

        # Metrics we want to include for each model
        metrics = ['OR', 'OR CI 95%', 'Standard Error']
        # Pivot the DataFrame
        pivoted_df = pd.DataFrame()
        for model, group in df_results.groupby('model_var_map'):
            # Reorganize each group with variables as rows and metrics as columns
            metrics_df = group.set_index('Variable')[metrics].copy()
            # Rename 'OR CI 95%' to 'OR CI' for consistency in the final output
            metrics_df.rename(columns={'OR CI 95%': 'OR CI'}, inplace=True)
            # Create multi-level columns (e.g., model 1 -> OR, OR CI, etc.)
            metrics_df.columns = pd.MultiIndex.from_product([[model], metrics_df.columns])

            # Append R-squared and Sample Size as rows under this model
            r_squared = group['R-squared'].iloc[0]
            sample_size = group['Sample Size'].iloc[0]
            metrics_df.loc['R-squared', (model, 'OR')] = r_squared
            metrics_df.loc['Sample Size', (model, 'OR')] = sample_size

            # Concatenate to the final pivoted DataFrame
            pivoted_df = pd.concat([pivoted_df, metrics_df], axis=1)

        # Move 'R-squared' and 'Sample Size' to the top of the DataFrame
        sorted_rows = ['R-squared', 'Sample Size'] + [row for row in pivoted_df.index if
                                                      row not in ['R-squared', 'Sample Size']]
        pivoted_df = pivoted_df.loc[sorted_rows]
        pivoted_df = pivoted_df.sort_index(axis=1)
        if orientation == 'vertical':
            df_vertical = pd.DataFrame()
            for idx, _ in df_results.groupby('model_var_map'):
                df_vertical_ = pivoted_df.loc[:, pd.IndexSlice[idx, :]].copy()
                df_vertical_['model'] = idx
                df_vertical_.columns =  metrics + ['model']
                df_vertical = pd.concat([df_vertical, df_vertical_], axis=0)
            # Drop rows where all columns except 'model' are NaN
            df_vertical.dropna(how='all', subset=metrics, inplace=True)
            return df_vertical

        return pivoted_df
#
# # Example usage (uncomment to run):
# if __name__ == '__main__':
#     df = pd.DataFrame({...})  # Replace with your actual dataset
#     base_vars = ['Age', 'Gender', 'BMI']
#     interest_vars = ['ACTH', 'hypertension', 'diabetes']
#     outcome = 'cortisol_level'
#
#     analysis = RegressionAnalysis(df,
#                                   outcome,
#                                   base_vars,
#                                   interest_vars,
#                                   model_type='linear_reg')
#     results_df = analysis.run_analysis()
#     print(results_df)
#
#     # Plot QQ and feature-outcome relationship for a variable
#     analysis.plot_relationship_and_qq('ACTH')