import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy import stats
from pathlib import Path
import pyreadstat
import sqlite3
import json
from pandas.plotting import andrews_curves, parallel_coordinates, lag_plot, autocorrelation_plot
from io import StringIO  # Added for CSV parsing

def load_data(file=None, raw_data=None):
    """Load data from uploaded file or raw CSV input."""
    if file:
        ext = Path(file.filename).suffix.lower()
        try:
            if ext == '.csv':
                return pd.read_csv(file)
            elif ext in ['.xlsx', '.xls']:
                return pd.read_excel(file)
            elif ext == '.sav':
                df, _ = pyreadstat.read_sav(file)
                return df
            elif ext == '.db':
                conn = sqlite3.connect(file.filename)
                df = pd.read_sql_query("SELECT * FROM table_name", conn)
                conn.close()
                return df
            elif ext == '.json':
                return pd.read_json(file)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")
    elif raw_data:
        try:
            # Handle CSV text input
            if not raw_data.strip():
                raise ValueError("Raw data is empty.")
            df = pd.read_csv(StringIO(raw_data))
            if df.empty:
                raise ValueError("Parsed raw data is empty.")
            return df
        except Exception as e:
            raise ValueError(f"Invalid raw data: {e}")
    else:
        raise ValueError("No input provided.")

def calculate_statistics(df):
    """Compute descriptive statistics for numerical columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.any():
        raise ValueError("No numerical columns found.")

    stats_dict = {
        col: {
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Mode': df[col].mode().iloc[0] if not df[col].mode().empty else np.nan,
            'Std Dev': df[col].std(),
            'Variance': df[col].var(),
            'Standard Error': stats.sem(df[col]),
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Skewness': df[col].skew(),
            'Kurtosis': df[col].kurtosis()
        }
        for col in numeric_cols
    }
    return pd.DataFrame(stats_dict).T.round(4)

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 string."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        base64_string = base64.b64encode(buf.read()).decode('utf-8')
        if not base64_string:
            raise ValueError("Empty base64 string generated")
        return base64_string
    except Exception as e:
        raise ValueError(f"Error converting figure to base64: {e}")

def generate_visualizations(df):
    """Generate visualizations and return them as base64 strings."""
    plots = []
    numeric_cols = df.select_dtypes(include=np.number).columns

    if not numeric_cols.any():
        raise ValueError("No numerical columns found for visualization.")

    for col in numeric_cols:
        try:
            # Histogram + KDE
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Histogram + KDE: {col}')
            plots.append(('histogram', col, fig_to_base64(fig)))

            # Boxplot
            fig, ax = plt.subplots()
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(f'Box Plot: {col}')
            plots.append(('boxplot', col, fig_to_base64(fig)))

            # Violin
            fig, ax = plt.subplots()
            sns.violinplot(y=df[col], ax=ax)
            ax.set_title(f'Violin Plot: {col}')
            plots.append(('violin', col, fig_to_base64(fig)))

            # Density
            fig, ax = plt.subplots()
            sns.kdeplot(df[col], ax=ax)
            ax.set_title(f'Density Plot: {col}')
            plots.append(('density', col, fig_to_base64(fig)))

            # Q-Q
            fig = plt.figure()
            stats.probplot(df[col].dropna(), dist="norm", plot=plt)
            plt.title(f'Q-Q Plot: {col}')
            plots.append(('qqplot', col, fig_to_base64(fig)))

            # Lag Plot
            if len(df[col].dropna()) > 1:
                fig = plt.figure()
                lag_plot(df[col].dropna())
                plt.title(f'Lag Plot: {col}')
                plots.append(('lagplot', col, fig_to_base64(fig)))

            # Autocorrelation
            if len(df[col].dropna()) > 1:
                fig = plt.figure()
                autocorrelation_plot(df[col].dropna())
                plt.title(f'Autocorrelation Plot: {col}')
                plots.append(('autocorrelation', col, fig_to_base64(fig)))

        except Exception as e:
            print(f"Error generating plot for {col}: {str(e)}")
            continue

    if len(numeric_cols) > 1:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Correlation Heatmap")
            plots.append(('correlation_heatmap', 'all', fig_to_base64(fig)))

            pair = sns.pairplot(df[numeric_cols])
            buf = io.BytesIO()
            pair.savefig(buf, format='png')
            buf.seek(0)
            plots.append(('pairplot', 'all', base64.b64encode(buf.read()).decode('utf-8')))
            plt.close()
        except Exception as e:
            print(f"Error generating multi-column plots: {str(e)}")
            pass

    return plots