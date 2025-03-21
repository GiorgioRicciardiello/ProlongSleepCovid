import pathlib
import pandas as pd
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt

def plot_variable(data, x: str,
                  variable_type: str = "binary",
                  hue: Optional[str] = None,
                  title: Optional[str] = None,
                  xlabel: Optional[str] = None,
                  figsize: Tuple[int, int] = (8, 6)):
    """
    Plots binary or continuous variables with custom styles using Seaborn.

    Parameters:
    - data: DataFrame containing the data to plot.
    - x: str, column name for the x-axis variable.
    - variable_type: str, "binary" or "continuous", type of variable for custom styling.
    - hue: Optional[str], column name for an optional hue variable.
    - title: Optional[str], title for the plot.
    - xlabel: Optional[str], custom label for the x-axis.
    - figsize: Tuple[int, int], figure size for the plot.
    """
    sns.set_theme(style="whitegrid")  # Apply a pleasant Seaborn theme

    # Set up the figure
    plt.figure(figsize=figsize)

    if variable_type == "binary":
        # Use countplot with unique colors for each bar by setting hue to x if hue is None
        palette = sns.color_palette("pastel", n_colors=len(data[x].unique()))
        ax = sns.countplot(data=data, x=x, hue=hue if hue else x, palette=palette, dodge=False)

        # Remove the legend if we set hue to x for coloring
        if hue is None:
            ax.legend_.remove()

        # Set tick labels for binary data if values are 0 and 1
        unique_values = data[x].unique()
        if set(unique_values).issubset({0, 1}):
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['No', 'Yes'])

        # Add annotations for each bar
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
        ax.set_ylabel("Count")
    elif variable_type == "continuous":
        ax = sns.histplot(data=data, x=x, hue=hue, kde=True, stat="density")
        ax.set_ylabel("Density")

    # Set x-axis label to provided `xlabel` or default to the column name
    ax.set_xlabel(xlabel if xlabel else x)

    # Set title if provided
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Optimize layout
    plt.tight_layout()
    plt.grid(alpha=0.1)  # Light grid for better readability
    plt.show()



def plot_mixed_correlation_heatmap(
        data: pd.DataFrame,
        binary_cols: List[str],
        cont_cols: List[str],
        title: str = "Correlation Heatmap for Mixed Data Types",
        output_path: pathlib.Path = None
) -> None:
    """
    Correlation heat map for when we have data types of type binary and continuous.
    :param data: dataframe containing the data to plot.
    :param binary_cols: list of columns of type continuous
    :param cont_cols: list of columns of type binary
    :param title: title of the heatmap
    :return:
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pointbiserialr, pearsonr
    from sklearn.metrics import matthews_corrcoef
    import seaborn as sns
    import numpy as np

    # Ensure binary columns are strictly binary
    for col in binary_cols:
        unique_vals = data[col].unique()
        if not np.array_equal(np.unique(unique_vals), [0, 1]):
            raise ValueError(f"Column {col} contains non-binary values: {unique_vals}")

    # Initialize a correlation matrix with NaN
    columns = binary_cols + cont_cols
    corr_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)

    # Fill the correlation matrix
    for i in columns:
        for j in data.columns:
            if i == j:
                # Correlation with itself is always 1
                corr_matrix.loc[i, j] = 1.0
            elif i in binary_cols and j in binary_cols:
                # Use Matthews correlation for binary-binary pairs
                corr_matrix.loc[i, j] = matthews_corrcoef(data[i], data[j])
            elif i in cont_cols and j in cont_cols:
                # Use Pearson correlation for continuous-continuous pairs
                corr_matrix.loc[i, j] = pearsonr(data[i], data[j])[0]
            elif (i in binary_cols and j in cont_cols) or (i in cont_cols and j in binary_cols):
                # Use Point-Biserial correlation for binary-continuous pairs
                corr_matrix.loc[i, j] = pointbiserialr(data[i], data[j])[0] if i in binary_cols else \
                    pointbiserialr(data[j], data[i])[0]

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        mask=corr_matrix.isna(),
        cbar_kws={'label': 'Correlation'}
    )
    plt.title(title)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()



def plot_symptom_severity(severity_df, df_data):
    """
    Plots symptom severity as a stacked bar plot and a heatmap.

    Parameters:
    - severity_df: pandas.DataFrame
        A DataFrame containing severity values for each symptom and severity level.
    - df_data: pandas.DataFrame
        A DataFrame from which the total number of patients is determined (via its number of rows).
    """
    # Convert to percentages
    # Set theme and context for the bar plot
    sns.set_theme('notebook')
    sns.set_context("talk", font_scale=1.2)

    # Plotting the stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    severity_df.T.plot(kind='bar',
                       stacked=True,
                       colormap='gist_yarg',
                       width=0.8,
                       ax=ax)

    ax.set_xlabel('Symptoms')
    ax.set_ylabel('Percentage of Patients')
    ax.legend(title='Severity Level', bbox_to_anchor=(0.5, 1.15), loc='center', ncol=6)
    plt.grid(axis='y', alpha=1)
    plt.xticks(rotation=90, ha='center')
    plt.tight_layout()
    plt.grid(alpha=0.5)
    plt.show()

    # Plotting the heatmap with enhanced aesthetics
    sns.set_context("talk", font_scale=1)
    plt.figure(figsize=(16, 6))
    ax = sns.heatmap(severity_df.iloc[::-1],
                     annot=True,
                     cmap="binary",
                     fmt=".1f",
                     annot_kws={"size": 12},
                     linewidths=0.5,
                     cbar_kws={'label': 'Percentage (%)'})

    plt.title(f"Responses Symptom Severity Percentages N={df_data.shape[0]}", fontsize=18, pad=20)
    plt.xlabel("Symptoms")
    plt.ylabel("Severity Level")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



def summarize_data(df, columns):
    summary = {}
    for col in columns:
        if set(df[col].dropna().unique()).issubset({0, 1}):  # Check if binary (0 or 1 values)
            summary[col] = df[col].value_counts().to_dict()
        else:
            summary[col] = df[col].describe().to_dict()
    return pd.DataFrame(summary)
