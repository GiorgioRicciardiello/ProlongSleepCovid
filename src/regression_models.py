import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import dtype, ndarray
from scipy.stats import ttest_ind, mannwhitneyu
import pathlib
import numpy as np
from config.data_paths import data_paths
from library.table_one import MakeTableOne
from tabulate import tabulate
from library.utils import compute_sparse_encoding
from library.effect_measures_plot import EffectMeasurePlot
import statsmodels.api as sm
from scipy.stats import pearsonr
from typing import Any, Optional, Tuple, List, Dict, Union
from sklearn.metrics import roc_curve, auc
from statsmodels.formula.api import mixedlm
from numpy import ndarray
from statsmodels.miscmodels.ordinal_model import OrderedModel

class RegressionModel():
    def __init__(self,
                 features: List[str],
                 outcome: str,
                 data: pd.DataFrame,
                 output_path: Optional[pathlib.Path] = None,
                 figsize: Optional[Tuple] = (10, 6)
                 ):
        """

        :param features:
        :param outcome:
        :param data:
        :param output_path:
        :param figsize:
        """
        self.features = features
        self.outcome = outcome
        self.data = data
        self.model = None
        self.y_pred = None
        self.df_comparison = None
        self.model_summary = None
        self.output_path = output_path
        self.figsize = figsize
        self.is_continuous = self._check_if_continuous()  # Determine if outcome is continuous or ordinal


    def run(self) -> pd.DataFrame:
        self._fit_model()
        self._set_summary_frame()
        self._plot()
        return self.get_summary_frame()

    def get_summary_frame(self) -> pd.DataFrame:
        """Generates a summary frame from the model results."""
        return  self.df_summary

    def _check_if_continuous(self) -> bool:
        """Check if the outcome variable is continuous based on the number of unique values."""
        return len(self.data[self.outcome].unique()) > 10

    def _prepare_data(self):
        """Prepares the data for regression by adding a constant (intercept)."""
        X = self.data[self.features]
        X = sm.add_constant(X)  # Add an intercept to the model
        y = self.data[self.outcome]
        return X, y

    def _fit_model(self):
        """Fits an OLS regression model to the data."""
        X, y = self._prepare_data()
        self.model = sm.OLS(y, X).fit()
        self.y_pred = self.model.predict(X)
        self.model_summary = self.model.summary()
        self.df_comparison = pd.DataFrame({
            'predictions': self.y_pred,
            'real_outcome': y
        })

    def _set_summary_frame(self):
        """Generates a summary frame from the model results."""
        df_metrics = self.model_summary.tables[0].as_html()
        df_metrics = pd.read_html(df_metrics, header=0)[0]
        df_metrics.to_csv(self.output_path.joinpath(r'metrics.csv'), index=True)
        self.df_summary = self.model_summary.tables[1].as_html()
        self.df_summary = pd.read_html(self.df_summary, header=0)[0]
        self.df_summary.index = ['const'] + self.features
        self.df_summary.drop(columns=['Unnamed: 0'], inplace=True)
        self.df_summary.to_csv(self.output_path.joinpath(r'summary.csv'), index=True)


    def _plot(self):
        """Decides the type of plot based on whether the model is linear or logistic."""
        if self.is_continuous:
            self._plot_predictions_vs_actual()
        else:
            self._plot_roc_curve()

    def _plot_predictions_vs_actual(self):
        """Plots the predicted vs actual outcomes along with the regression line."""
        corr_coef, _ = pearsonr(self.df_comparison['predictions'], self.df_comparison['real_outcome'])
        min_val = min(self.df_comparison['predictions'].min(), self.df_comparison['real_outcome'].min())
        max_val = max(self.df_comparison['predictions'].max(), self.df_comparison['real_outcome'].max())

        plt.figure(figsize=self.figsize)
        sns.set(style="whitegrid")

        # Scatter plot for predictions vs real outcomes
        sns.scatterplot(x='predictions',
                        y='real_outcome',
                        data=self.df_comparison,
                        alpha=0.5,
                        label='Data points')

        # Add regression line to show the trend
        sns.regplot(x='predictions',
                    y='real_outcome',
                    data=self.df_comparison,
                    scatter=False,
                    color='blue',
                    label='Fit line')

        # Plot the perfect reference line where y_pred = y_true
        plt.plot([min_val, max_val], [min_val, max_val],
                 color='red',
                 linestyle='--',
                 label='Perfect Prediction (y_pred = y_true)')

        plt.text(0.05, 0.95,
                 f'Pearson r: {corr_coef:.2f}',
                 ha='left',
                 va='center',
                 transform=plt.gca().transAxes,
                 fontsize=12)

        r_squared = self.model.rsquared
        adj_r_squared = self.model.rsquared_adj
        f_stat = self.model.fvalue
        f_pvalue = self.model.f_pvalue
        aic = self.model.aic
        bic = self.model.bic
        title = (f'Predictions vs Actual Outcomes\n'
                 f'R²: {r_squared:.2f}, Adj. R²: {adj_r_squared:.2f},\n'
                 f'F-stat: {f_stat:.2f}, p(F-stat): {f_pvalue:.2e},\n'
                 f'AIC: {aic:.2f}, BIC: {bic:.2f}')

        plt.title(title,
                  fontsize=16)
        plt.xlabel('Predicted Values',
                   fontsize=12)
        plt.ylabel('Actual Values',
                   fontsize=12)
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.legend(loc='upper right')
        plt.tight_layout()
        if self.output_path:
            plt.savefig(self.output_path.joinpath('true_vs_pred.png'), dpi=300)
            print(f"Plot of linear regression model save in {self.output_path.joinpath('true_vs_pred.png')}")
        plt.show()

    def _plot_roc_curve(self):
        """Plots the ROC curve for logistic regression."""
        fpr, tpr, _ = roc_curve(self.df_comparison['real_outcome'], self.y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=self.figsize)
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Chance line (AUC = 0.50)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.grid(alpha=0.7)
        if self.output_path:
            plt.savefig(self.output_path.joinpath('roc_curve.png'), dpi=300)
            print(f"Plot of linear regression model save in {self.output_path.joinpath('roc_curve.png')}")
        plt.show()


def recode_column(column:pd.Series,
                  ranges_dict:Dict[int, Tuple[int, int]]) -> ndarray:
    """
    General function to recode a column based on a dictionary of ranges and corresponding codes.

    Parameters:
    column (pd.Series): The column to be recoded.
    ranges_dict (dict): Dictionary where the key is the code, and the value is a tuple (start, end) for the range.

    Returns:
    pd.Series: Recoded column.
    """
    conditions = [column.between(start, end) for start, end in ranges_dict.values()]
    codes = list(ranges_dict.keys())
    return np.select(conditions, codes, default=np.nan)


if __name__ == '__main__':
    df_asq = pd.read_csv(data_paths.get('pp_data').get('asq_cov_clinic'))

    col_questions = {
        'dem_0110': 'Age',
        'dem_0500': 'Gender',
        'dem_0800': 'BMI',
        # 'dem_0900': ,
        'dem_1000': 'Race',
        # 'dem_1010': 'Race sub',
        'mdhx_sleep_problem': 'mdhx_sleep_problem',
        'mdhx_sleep_diagnosis': 'mdhx_sleep_diagnosis',
        'mdhx_sleep_treatment': 'mdhx_sleep_treatment',
        'mdhx_0900': 'Medications',
        'mdhx_1200': 'Currently pregnant',
        'mdhx_1300': 'Menopausal status',
        'mdhx_5700': 'Hypertension',
        'mdhx_5710': 'Chronic heart failure',
        # 'mdhx_cardio_problem',
        'mdhx_5800': 'Asthma',
        'mdhx_5810': 'COPD',
        'mdhx_6300': 'High cholesterol',
        'mdhx_6310': 'Diabetes Type 2',
        # 'mdhx_6340',
        'soclhx_0600': 'Exercise time of day',
        'soclhx_1550': 'Use of cannabis',
        'map_0400': 'Fall asleep at work',
        'map_score': 'MAP score',
        'ess_0900': 'ESS score',
        'slpy_0150': 'Difficulty staying awake frequency',
        'isi_0100': 'Difficulty falling asleep',
        'isi_0200': 'Difficulty staying asleep',
        'isi_0300': 'Problem waking up too early',
        'isi_score': 'ISI score',
        'cir_0200': 'What time of day would you get up',
        'cir_0300': 'How tired do you feel',
        'cir_0400': 'Evening time feel tired',
        'cir_0700': 'rMEQ Total Score',
        'fosq_0100': 'Difficulty concentrating',
        'fosq_0200': 'Difficulty remembering',
        'fosq_0700': 'Difficultly watching movies or video',
        'fosq_1100': 'FOSQ Score',
        'UNREFRESHING SLEEP': 'CV Unrefreshed Sleep',
        'DIFFICULTY SLEEPING': 'CV Difficulty Sleep',
        'DAYTIME SLEEPINESS': 'CV Daytime Sleep',
    }
    df_asq.rename(columns=col_questions, inplace=True)
    output_folder = data_paths.get('results').joinpath(f'regression_models')
    if not output_folder.exists():
        output_folder.mkdir(exist_ok=True, parents=True)

    # %% Regression
    features = ['Age', 'BMI',
                'Race', 'Gender',
                'CV Unrefreshed Sleep', 'CV Daytime Sleep', 'CV Difficulty Sleep']
    X = df_asq[features]
    X = sm.add_constant(X)  # Add an intercept to the model

    #%% Model 1: ESS_score ~ Age + BMI + Race + Gender + US + SL + DL
    y1 = 'ESS score'
    output_path_m = output_folder.joinpath(f'model_ess')
    if not output_path_m.exists():
        output_path_m.mkdir(exist_ok=True, parents=True)

    model = RegressionModel(features=features,
                            outcome=y1,
                            data=df_asq,
                            output_path=output_path_m)
    df_summary = model.run()
    forest_plot = EffectMeasurePlot(label=df_summary.index.to_list(),
                                    effect_measure=df_summary['coef'].to_list(),
                                    lcl=df_summary['[0.025'].to_list(),
                                    ucl=df_summary['0.975]'].to_list(),
                                    p_value=df_summary['P>|t|'].to_list(),
                                    alpha=0.05,
                                    )
    forest_plot.plot(figsize=(12, 8),
                     path_save=output_path_m.joinpath('forest_plot.png'),
                     )

    #%% Model 2: ISI_score ~ Age + BMI + Race + Gender + US + SL + DL
    output_path_m = output_folder.joinpath(f'model_isi')
    if not output_path_m.exists():
        output_path_m.mkdir(exist_ok=True, parents=True)
    y2 = 'ISI score'
    model = RegressionModel(features=features,
                            outcome=y2,
                            data=df_asq,
                            output_path=output_path_m
                            )

    df_summary = model.run()
    forest_plot = EffectMeasurePlot(label=df_summary.index.to_list(),
                                    effect_measure=df_summary['coef'].to_list(),
                                    lcl=df_summary['[0.025'].to_list(),
                                    ucl=df_summary['0.975]'].to_list(),
                                    p_value=df_summary['P>|t|'].to_list(),
                                    alpha=0.05,
                                    )
    forest_plot.plot(figsize=(12, 8),
                     path_save=output_path_m.joinpath(f'model_ess_odds')
                     )

    #%% Model 3: fosq ~ Age + BMI + Race + Gender + US + SL + DL
    output_path_m = output_folder.joinpath(f'model_fosq')
    if not output_path_m.exists():
        output_path_m.mkdir(exist_ok=True, parents=True)
    y3 = 'FOSQ Score'
    model = RegressionModel(features=features,
                            outcome=y3,
                            data=df_asq,
                            output_path=output_path_m
                            )

    df_summary = model.run()
    forest_plot = EffectMeasurePlot(label=df_summary.index.to_list(),
                                    effect_measure=df_summary['coef'].to_list(),
                                    lcl=df_summary['[0.025'].to_list(),
                                    ucl=df_summary['0.975]'].to_list(),
                                    p_value=df_summary['P>|t|'].to_list(),
                                    alpha=0.05,
                                    )
    forest_plot.plot(figsize=(12, 8),
                     path_save=output_path_m.joinpath(f'model_ess_odds')
                     )

    #%% Model 4: Map_score ~ Age + BMI + Race + Gender + US + SL + DL
    output_path_m = output_folder.joinpath(f'model_map')
    if not output_path_m.exists():
        output_path_m.mkdir(exist_ok=True, parents=True)
    y4 = 'MAP score'
    model = RegressionModel(features=features,
                            outcome=y4,
                            data=df_asq,
                            output_path=output_path_m
                            )

    df_summary = model.run()
    forest_plot = EffectMeasurePlot(label=df_summary.index.to_list(),
                                    effect_measure=df_summary['coef'].to_list(),
                                    lcl=df_summary['[0.025'].to_list(),
                                    ucl=df_summary['0.975]'].to_list(),
                                    p_value=df_summary['P>|t|'].to_list(),
                                    alpha=0.05,
                                    )
    forest_plot.plot(figsize=(12, 8),
                     path_save=output_path_m.joinpath(f'model_ess_odds')
                     )

    # %% Logistic Regression
    def plot_predicted_probabilities(result:OrderedModel.fit,
                                     X:pd.DataFrame,
                                     predictor_name:str,
                                     min_val:Union[int, float],
                                     max_val:Union[int, float]):
        """
        Plots predicted probabilities for each category of the ordinal outcome across a range of a predictor.

        Parameters:
        result (OrderedModel.fit() result): Fitted ordinal logistic regression model result.
        X (pd.DataFrame): Independent variables (design matrix).
        predictor_name (str): The name of the predictor to vary (e.g., 'Age').
        min_val (int/float): Minimum value of the predictor to plot.
        max_val (int/float): Maximum value of the predictor to plot.
        """
        # Generate a range of values for the predictor
        predictor_range = np.linspace(min_val, max_val, 100)

        # Copy the independent variables (X) and modify the predictor column
        X_copy = X.copy()

        # Store predicted probabilities for each category
        prob_df_list = []  # Use a list to store DataFrames

        for value in predictor_range:
            X_copy[predictor_name] = value
            # Get predicted probabilities for each category using just one row at a time
            pred_probs = result.model.predict(result.params,
                                              exog=X_copy.iloc[0:1])
            # Convert to DataFrame and add the current value of the predictor as index
            prob_df_list.append(
                pd.DataFrame(pred_probs, columns=['Category_0', 'Category_1', 'Category_2', 'Category_3'],
                             index=[value]))

        # Concatenate the list of DataFrames into a single DataFrame
        prob_df = pd.concat(prob_df_list)

        # Plot predicted probabilities across the range of the predictor
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        # Plot each category's predicted probabilities
        for category in prob_df.columns:
            plt.plot(prob_df.index, prob_df[category], label=category)

            # Calculate the slope as change in y over change in x
            slope = (prob_df[category].iloc[-1] - prob_df[category].iloc[0]) / (
                        predictor_range[-1] - predictor_range[0])

            # Intercept is the predicted probability when the predictor is at min_val
            intercept = prob_df[category].iloc[0]

            # Annotate the slope and intercept on the plot
            plt.text(x=max_val, y=prob_df[category].iloc[-1],
                     s=f"Slope: {slope:.2f}\nIntercept: {intercept:.2f}",
                     verticalalignment='bottom', horizontalalignment='right')

        plt.title(f'Predicted Probabilities Across {predictor_name}')
        plt.xlabel(predictor_name)
        plt.ylabel('Predicted Probability')
        plt.legend(title='Outcome Category')
        plt.tight_layout()
        plt.show()


    ess_ranges = {0: (0, 7), 1: (8, 9), 2: (10, 15), 3: (16, 24)}
    df_asq['ESS_cat'] = recode_column(df_asq['ESS score'], ranges_dict=ess_ranges)

    isi_ranges = {0: (0, 7), 1: (8, 14), 2: (15, 21), 3: (22, 28)}
    df_asq['ISI_cat'] = recode_column(df_asq['ISI score'], ranges_dict=isi_ranges)

    # Ensure that ESS_cat and ISI_cat are treated as ordered categorical variables
    df_asq['ESS_cat'] = pd.Categorical(df_asq['ESS_cat'], ordered=True)
    df_asq['ISI_cat'] = pd.Categorical(df_asq['ISI_cat'], ordered=True)

    # Define the model
    # Assuming we want to predict ISI_cat and ESS_cat, with random effects for some grouping (e.g., 'group_var')
    # Replace 'group_var' with the column indicating the group level for your data
    X = df_asq[features]
    y = df_asq['ISI_cat'].cat.codes

    model = OrderedModel(endog=y,
                         exog=X,
                         distr='logit')  # 'logit' for logistic regression
    result = model.fit(method='bfgs')

    plot_predicted_probabilities(result=result,
                                 X=X,
                                 predictor_name='Age',
                                 min_val=X['Age'].min(),
                                 max_val=X['Age'].max())

    # Display the model summary
    print(result.summary())

    # %%
    col_covid = [col for col in df_asq.columns if 'CV' in col]
    df_asq[col_covid].sum(axis=1)