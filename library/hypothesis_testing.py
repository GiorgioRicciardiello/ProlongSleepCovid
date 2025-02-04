import pandas as pd
from typing import List, Optional
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests


class HypothesisTesting:
    def __init__(self,
                 df: pd.DataFrame,
                 strata: str, 
                 continuous_vars: Optional[List[str]] = None,
                 discrete_vars: Optional[List[str]] = None,
                 binary_vars: Optional[List[str]] = None,
                 correction_type: str = "bonferroni",
                 alpha: float = 0.05):
        """
        Initializes the class with the dataset and variables for hypothesis testing.

        :param df: DataFrame containing the data.
        :param strata: Column name that identifies the strata/groups in df.
        :param continuous_vars: List of continuous variable names, or None.
        :param discrete_vars: List of ordinal/discrete variable names, or None.
        :param binary_vars: List of binary variable names, or None.
        :param correction_type: Type of p-value correction ('bonferroni' or 'holm').
        :param alpha: Significance level.
        """
        self.df = df
        self.continuous_vars = continuous_vars or []  # Default to empty list if None
        self.discrete_vars = discrete_vars or []
        self.binary_vars = binary_vars or []
        self.correction_type = correction_type
        self.alpha = alpha
        self.strata = strata
        self.results = []

    def _test_continuous(self, var: str):
        """
        Performs Kruskal-Wallis test for continuous variables.
        """
        groups = [group[var].dropna() for _, group in self.df.groupby(self.strata)]
        group_sizes = [len(g) for g in groups]  # Calculate sizes for each group
        if len(groups) > 1:
            stat, p = kruskal(*groups)
            return stat, p, "Kruskal-Wallis", group_sizes
        return None, None, None, group_sizes

    def _test_binary(self, var: str):
        """
        Performs Chi-square test for binary variables.
        """
        contingency_table = pd.crosstab(self.df[self.strata], self.df[var])
        group_sizes = contingency_table.sum(axis=0).to_list()  # Count for each binary category
        if contingency_table.shape[1] == 2:  # Ensure it's binary
            stat, p, _, _ = chi2_contingency(contingency_table)
            return stat, p, "Chi-square", group_sizes
        return None, None, None, group_sizes

    def _test_discrete(self, var: str):
        """
        Performs Mann-Whitney U test for ordinal/discrete variables.
        """
        groups = [group[var].dropna() for _, group in self.df.groupby(self.strata)]
        group_sizes = [len(g) for g in groups]  # Calculate sizes for each group
        if len(groups) == 2:  # Mann-Whitney requires two groups
            stat, p = mannwhitneyu(*groups,
                                   alternative='two-sided',
                                   method="auto")
            return stat, p, "Mann-Whitney U", group_sizes
        elif len(groups) > 2:
            stat, p = kruskal(*groups)  # Fall back to Kruskal-Wallis for >2 groups
            return stat, p, "Kruskal-Wallis", group_sizes
        return None, None, None, group_sizes

    def _apply_correction(self):
        """
        Applies p-value correction and updates results.
        """
        if not self.results:  # No results to correct
            return

        p_values = [result['p_value'] for result in self.results]
        corrected = multipletests(p_values, method=self.correction_type, alpha=self.alpha)
        corrected_p_values = corrected[1]
        significant = corrected[0]

        # Update results with corrected p-values
        for i, result in enumerate(self.results):
            result['corrected_p_value'] = corrected_p_values[i]
            result['significant'] = significant[i]

    def run_tests(self):
        """
        Runs hypothesis tests for all specified variables and returns a DataFrame of results.
        """
        for var in self.continuous_vars:
            stat, p, method, group_sizes = self._test_continuous(var)
            if stat is not None:
                self.results.append({
                    'feature': var,
                    'method': method,
                    'group_sizes': group_sizes,
                    'statistic': stat,
                    'p_value': p
                })

        for var in self.binary_vars:
            stat, p, method, group_sizes = self._test_binary(var)
            if stat is not None:
                self.results.append({
                    'feature': var,
                    'method': method,
                    'group_sizes': group_sizes,
                    'statistic': stat,
                    'p_value': p
                })

        for var in self.discrete_vars:
            stat, p, method, group_sizes = self._test_discrete(var)
            if stat is not None:
                self.results.append({
                    'feature': var,
                    'method': method,
                    'group_sizes': group_sizes,
                    'statistic': stat,
                    'p_value': p
                })

        # Apply correction to p-values
        self._apply_correction()

        # Return results as a DataFrame
        if self.results:
            return pd.DataFrame(self.results)
        else:
            return pd.DataFrame(
                columns=['feature', 'method', 'group_sizes', 'statistic', 'p_value', 'corrected_p_value',
                         'significant'])

if __name__ == '__main__':
    # Sample DataFrame
    data = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'C', 'C'],
        'continuous_var': [1.2, 2.3, 3.1, 4.2, 3.9, 5.1],
        'binary_var': [0, 1, 1, 0, 1, 1],
        'ordinal_var': [1, 2, 3, 4, 2, 5]
    })

    # Initialize the class
    tester = HypothesisTesting(
        df=data,
        continuous_vars=['continuous_var'],
        discrete_vars=['ordinal_var'],
        binary_vars=['binary_var'],
        correction_type="bonferroni",
        strata='group'
    )

    results = tester.run_tests()