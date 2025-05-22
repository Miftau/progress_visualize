import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from scipy import stats
from pathlib import Path
import pyreadstat
import sqlite3
from io import StringIO
import json
import warnings


def load_data(file=None, raw_data=None, sample_size=100000):
    """Load data with customizable sample size for large datasets."""
    if file:
        ext = Path(file.filename).suffix.lower()
        try:
            if ext == '.csv':
                df = pd.read_csv(file)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file)
            elif ext == '.sav':
                df, _ = pyreadstat.read_sav(file)
            elif ext == '.db':
                conn = sqlite3.connect(file.filename)
                df = pd.read_sql_query("SELECT * FROM table_name", conn)
                conn.close()
            elif ext == '.json':
                df = pd.read_json(file)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")
    elif raw_data:
        try:
            if not raw_data.strip():
                raise ValueError("Raw data is empty.")
            df = pd.read_csv(StringIO(raw_data))
            if df.empty:
                raise ValueError("Parsed raw data is empty.")
        except Exception as e:
            raise ValueError(f"Invalid raw data: {e}")
    else:
        raise ValueError("No input provided.")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        warnings.warn(f"Dataset too large, sampled to {sample_size} rows.")
    return df


def calculate_statistics(df):
    """Compute extended statistics for numerical and categorical columns with error handling."""
    stats_dict = {}

    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        data = df[col].dropna()
        stats_dict[col] = {
            'Type': 'Numeric',
            'Mean': data.mean(),
            'Median': data.median(),
            'Mode': data.mode().iloc[0] if not data.mode().empty else np.nan,
            'Std Dev': data.std(),
            'Variance': data.var(),
            'Standard Error': stats.sem(data) if len(data) > 1 and data.var() > 0 else np.nan,
            'Min': data.min(),
            'Max': data.max(),
            'Skewness': data.skew() if len(data) > 2 else np.nan,
            'Kurtosis': data.kurtosis() if len(data) > 2 else np.nan,
            'Q1': data.quantile(0.25),
            'Q3': data.quantile(0.75),
            'IQR': data.quantile(0.75) - data.quantile(0.25) if len(data) > 1 else np.nan,
            'Outliers': len(data[(data < (data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25)))) |
                                 (data > (data.quantile(0.75) + 1.5 * (
                                             data.quantile(0.75) - data.quantile(0.25))))]) if len(data) > 1 else 0,
            'Shapiro-Wilk p-value': stats.shapiro(data)[1] if len(data) >= 3 and len(
                data) <= 5000 and data.var() > 0 else np.nan
        }

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        data = df[col].dropna()
        stats_dict[col] = {
            'Type': 'Categorical',
            'Unique Values': data.nunique(),
            'Most Frequent': data.mode().iloc[0] if not data.mode().empty else np.nan,
            'Top Counts': data.value_counts().head(5).to_dict(),
            'Missing Values': data.isna().sum()
        }

    if len(numeric_cols) > 1:
        pearson_corr = df[numeric_cols].corr(method='pearson').to_dict()
        spearman_corr = df[numeric_cols].corr(method='spearman').to_dict()
        stats_dict['Correlations'] = {
            'Pearson': pearson_corr,
            'Spearman': spearman_corr
        }

    return pd.DataFrame(stats_dict).T.round(4)


def fig_to_png_base64(fig, width=800, height=600):
    """Convert a Plotly figure to a base64-encoded PNG string with customizable resolution."""
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height)
        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        if not base64_string:
            raise ValueError("Empty base64 string generated")
        return base64_string
    except Exception as e:
        raise ValueError(f"Error converting figure to PNG: {e}")


def generate_visualizations(df, width=800, height=600):
    """Generate interactive Plotly visualizations with PNGs for download."""
    plots = []
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in numeric_cols:
        try:
            data = df[col].dropna()

            fig = px.histogram(df, x=col, title=f'Histogram: {col}', nbins=30)
            plots.append(('histogram', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            fig = px.box(df, y=col, title=f'Box Plot: {col}')
            plots.append(('boxplot', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            fig = px.violin(df, y=col, title=f'Violin Plot: {col}')
            plots.append(('violin', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            fig = px.density_contour(df, x=col, title=f'Density Plot: {col}')
            plots.append(('density', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            qq = stats.probplot(data, dist="norm")
            fig = go.Figure()
            fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Data')
            fig.add_scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1], mode='lines', name='Fit')
            fig.update_layout(title=f'Q-Q Plot: {col}', xaxis_title='Theoretical Quantiles',
                              yaxis_title='Sample Quantiles')
            plots.append(('qqplot', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            if len(data) > 1:
                lag_data = pd.DataFrame({'t': data[:-1], 't+1': data[1:]})
                fig = px.scatter(lag_data, x='t', y='t+1', title=f'Lag Plot: {col}')
                plots.append(('lagplot', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                              fig_to_png_base64(fig, width, height)))

            if len(data) > 1:
                autocorr = [data.autocorr(lag=i) for i in range(1, min(len(data), 20))]
                fig = go.Figure()
                fig.add_bar(x=list(range(1, len(autocorr) + 1)), y=autocorr)
                fig.update_layout(title=f'Autocorrelation Plot: {col}', xaxis_title='Lag',
                                  yaxis_title='Autocorrelation')
                plots.append(('autocorrelation', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                              fig_to_png_base64(fig, width, height)))

            fig = px.density_heatmap(df, x=col, title=f'Density Heatmap: {col}')
            plots.append(('density_heatmap', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            fig = px.strip(df, y=col, title=f'Strip Plot: {col}')
            plots.append(('strip', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            fig = px.ecdf(df, x=col, title=f'ECDF Plot: {col}')
            plots.append(('ecdf', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            fig = px.line(df, y=col, title=f'Line Plot: {col}')
            plots.append(('line', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            fig = px.area(df, y=col, title=f'Area Plot: {col}')
            plots.append(('area', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            fig = px.line(x=df.index, y=df[col].cumsum(), title=f'Cumulative Sum: {col}')
            plots.append(('cumsum', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            if len(data) > 10:
                rolling_mean = data.rolling(window=10).mean()
                fig = px.line(x=df.index[:len(rolling_mean)], y=rolling_mean, title=f'Rolling Mean (window=10): {col}')
                plots.append(('rolling_mean', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                              fig_to_png_base64(fig, width, height)))

            if len(data) > 10:
                rolling_std = data.rolling(window=10).std()
                fig = px.line(x=df.index[:len(rolling_std)], y=rolling_std, title=f'Rolling Std (window=10): {col}')
                plots.append(('rolling_std', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                              fig_to_png_base64(fig, width, height)))

        except Exception as e:
            print(f"Error generating plot for {col}: {str(e)}")
            continue

    for col in categorical_cols:
        try:
            value_counts = df[col].value_counts().head(10)
            fig = px.bar(x=value_counts.index, y=value_counts.values, title=f'Bar Plot: {col}')
            fig.update_layout(xaxis_title=col, yaxis_title='Count')
            plots.append(('bar', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            fig = px.pie(values=value_counts.values, names=value_counts.index, title=f'Pie Chart: {col}')
            plots.append(('pie', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            fig = px.histogram(df, x=col, title=f'Count Plot: {col}')
            fig.update_layout(xaxis_title=col, yaxis_title='Count')
            plots.append(('count', col, fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

        except Exception as e:
            print(f"Error generating categorical plot for {col}: {str(e)}")
            continue

    if len(numeric_cols) > 1:
        try:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap', color_continuous_scale='RdBu_r')
            plots.append(('correlation_heatmap', 'all', fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            fig = px.scatter_matrix(df[numeric_cols], title='Pair Plot')
            plots.append(('pairplot', 'all', fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            # Fixed parameter name from 'columns' to 'dimensions'
            fig = px.parallel_coordinates(df, dimensions=numeric_cols, title='Parallel Coordinates')
            plots.append(('parallel_coordinates', 'all', fig.to_html(full_html=False, include_plotlyjs='cdn'),
                          fig_to_png_base64(fig, width, height)))

            if len(numeric_cols) >= 3:
                fig = px.scatter_3d(df, x=numeric_cols[0], y=numeric_cols[1], z=numeric_cols[2],
                                    title='3D Scatter Plot')
                plots.append(('scatter_3d', 'all', fig.to_html(full_html=False, include_plotlyjs='cdn'),
                              fig_to_png_base64(fig, width, height)))

            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1:]:
                    try:
                        pair_data = df[[col1, col2]].dropna()
                        if len(pair_data) < 2:
                            print(f"Skipping pairwise plot for {col1} vs {col2}: insufficient data after dropping NaN")
                            continue

                        fig = px.scatter(pair_data, x=col1, y=col2, title=f'Scatter: {col1} vs {col2}')
                        plots.append(
                            (f'scatter_{col1}_vs_{col2}', 'all', fig.to_html(full_html=False, include_plotlyjs='cdn'),
                             fig_to_png_base64(fig, width, height)))

                        fig = px.density_heatmap(pair_data, x=col1, y=col2, title=f'Density Heatmap: {col1} vs {col2}')
                        plots.append((f'density_heatmap_{col1}_vs_{col2}', 'all',
                                      fig.to_html(full_html=False, include_plotlyjs='cdn'),
                                      fig_to_png_base64(fig, width, height)))

                        fig = px.density_contour(pair_data, x=col1, y=col2, title=f'Contour Plot: {col1} vs {col2}')
                        plots.append(
                            (f'contour_{col1}_vs_{col2}', 'all', fig.to_html(full_html=False, include_plotlyjs='cdn'),
                             fig_to_png_base64(fig, width, height)))
                    except Exception as e:
                        print(f"Error generating pairwise plot for {col1} vs {col2}: {str(e)}")
                        continue

        except Exception as e:
            print(f"Error generating multi-column plots: {str(e)}")
            pass

    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
        try:
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    fig = px.box(df, x=cat_col, y=num_col, title=f'Box Plot: {num_col} by {cat_col}')
                    plots.append(
                        (f'box_{num_col}_by_{cat_col}', 'mixed', fig.to_html(full_html=False, include_plotlyjs='cdn'),
                         fig_to_png_base64(fig, width, height)))

                    fig = px.violin(df, x=cat_col, y=num_col, title=f'Violin Plot: {num_col} by {cat_col}')
                    plots.append((f'violin_{num_col}_by_{cat_col}', 'mixed',
                                  fig.to_html(full_html=False, include_plotlyjs='cdn'),
                                  fig_to_png_base64(fig, width, height)))

        except Exception as e:
            print(f"Error generating mixed plots: {str(e)}")
            pass

    return plots