"""
Credits:
    Some plot inspiration and code adapted from here:
        https://www.kaggle.com/subinium/tps-aug-simple-eda
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RED, BLUE = "#de2c37", "#3a64a3"


def indices_match(X, y):
    """Check X and y datasets have the same indices."""
    assert np.array_equal(np.array(X.index), np.array(y.index)), "Mismatched X, y indices."


def turbo_describe(X: pd.DataFrame, viz_cols: list = ("mean", "std", "50%"),
                   viz: str = "bar"):
    """Turbocharged version of pandas df.describe()

    Args:
        X: input pandas.Dataframe.
        viz_cols: the columns from pandas' df.describe() to be decorated.
        viz: the type of data visualization to decorate viz_cols with. Valid
            options are: `bar`, `background`

    Returns:

    """
    stats = X.describe().T
    if viz == "bar":
        return stats.style.bar(subset=list(viz_cols), axis=0, color=[RED, BLUE], align="mid")
    elif viz == "background":
        return stats.style.background_gradient(subset=list(viz_cols), axis=0, cmap="PuBu")
    else:
        raise ValueError(f"Invalid viz arg: {viz}")


def is_disrete(s: pd.Series) -> bool:
    """Boolean check to see if all feature values are integers.

    Args:
        s: pandas.Series of untransformed data.

    Returns: boolean.
    """
    values = s.unique()
    values.sort()
    int_values = np.array([int(i) for i in values])
    return np.array_equal(values, int_values)


def discrete_columns(X: pd.DataFrame) -> dict:
    """Dict of all columns in X which appear to be discrete.

    Args:
        X: pandas.DataFrame of untransformed data.

    Returns: dict in which keys are column names, values are number of unique
        values in that column.
    """
    keys = list(filter(lambda c: is_disrete(X[c]), X.columns))
    return {k: len(X[k].unique()) for k in keys}


def subplots_nrows_ncols(n_features) -> tuple:
    """Number of rows/columns required for plotting all features."""
    n_cols = round(n_features**0.5)
    n_rows = int(np.ceil(n_features / n_cols))
    return n_rows, n_cols


def plot_mean_med_diff_skew_analysis(X_sc: pd.DataFrame, cutoff: float = 0.5,
                                     bins: int = 100):
    """Analyze the difference between data means and medians to identify skewed
    distributions. Input data should be standard scaled.

    Args:
        X_sc: standard scaled pandas DataFrame of features.
        cutoff: the absolute difference between mean and median above which data
            will be classified as skewed.
        bins: number of bins to plot in histograms.

    Returns: Matplotlib figure plotting the least/most skewed features and the 2
        features either side of the cutoff.
    """
    stats = X_sc.describe().T
    stats["mean_med_abs_diff"] = (stats["mean"] - stats["50%"]).abs()
    stats.sort_values(by=["mean_med_abs_diff"], inplace=True)

    # Split the data around the cutoff value:
    above_cutoff = stats[stats["mean_med_abs_diff"] >= cutoff]
    below_cutoff = stats[stats["mean_med_abs_diff"] < cutoff]
    assert len(above_cutoff) and len(above_cutoff), f"Cutoff value doesn't bifurcate data: {cutoff}"
    most_skewed = above_cutoff.iloc[-1, :]
    least_skewed = below_cutoff.iloc[0, :]
    skewed_closest_to_cutoff = above_cutoff.iloc[0, :]
    unskewed_closest_to_cutoff = below_cutoff.iloc[-1, :]

    # Plot 4 features - least/most skewed and 2 features either side of cutoff:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    to_plot = {
        "Least skewed": least_skewed,
        "Unskewed closest to cutoff": unskewed_closest_to_cutoff,
        "Skewed closest to cutoff": skewed_closest_to_cutoff,
        "Most skewed": most_skewed,
    }
    for i, (label, data) in enumerate(to_plot.items()):
        ax = axes[i]
        raw_values = X_sc[data.name]
        ax.hist(raw_values, bins=bins, color=BLUE, alpha=0.75)
        ax.set_title(f"{data.name}\n{label}")
        ymin, ymax = ax.get_ylim()
        ax.vlines(data["mean"], ymin, ymax, color=RED)
        ax.vlines(data["50%"], ymin, ymax, color="black", linestyle="--")
        ax.set_ylim(ymin, ymax)

    title = f"Skew Analysis: Median (black dashed) - Mean (red) Abs Diff Cutoff = {cutoff:}\n{len(above_cutoff):} " \
            f"skewed features, {len(below_cutoff):} unskewed"
    fig.suptitle(title, y=1.1)

    return fig


def plot_disrete_dist(s: pd.Series):
    """Plot distribution of a discrete variable."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    counts = s.value_counts().sort_index()
    n = len(counts)
    alternating_colors = ([BLUE, "#e6e6e6"] * ((n + 1) // 2))[:n]
    ax.bar(counts.index, counts, color=alternating_colors, width=0.75, edgecolor="black", linewidth=0.5, alpha=0.75)
    ax.set_xlim(counts.index.min()-0.5, counts.index.max()+0.5)

    # Add percentage annotations:
    total = len(s)
    for i, ix in enumerate(counts.index):
        percent = counts[ix] / total * 100
        ax.annotate(f"{percent:.2f}%", xy=(i, counts[ix]+1000), va="bottom", ha="center", rotation=90)
    ax.set_ylim(0, max(counts) * 1.25)

    # Add title and gridlines:
    ax.set_title(f"{s.name} Distribution", weight="bold", fontsize=15)
    ax.grid(axis="y", linestyle="-", alpha=0.5)

    # Add x-labels:
    ax.set_xticks(list(range(len(counts))))
    ax.set_xticklabels(counts.index)

    return fig


def plot_hist_all_features(X: pd.DataFrame(), bins: int = 10, figsize=(20, 20)):
    """Plot histograms of all X features."""
    rows, columns = subplots_nrows_ncols(len(X.columns))
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    features = list(X.columns)
    for i in range(rows * columns):
        row = i // columns
        column = i - (row * columns)
        if axes.ndim == 2:
            ax = axes[row, column]
        else:
            ax = axes[row]
        try:
            feature = features[i]
            ax.hist(X[feature], bins=bins, color=BLUE, alpha=0.75)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(feature)
        except IndexError:  # Blank out all the empty plots.
            ax.set_yticks([])
            ax.set_xticks([])
            [ax.spines[x].set_visible(False) for x in ("top", "bottom", "left", "right")]
    return fig


def plot_scatter_all_features(X: pd.DataFrame(), y: pd.Series):
    """Plot scatterplots of all X features against y."""
    rows, columns = subplots_nrows_ncols(len(X.columns))
    fig, axes = plt.subplots(rows, columns, figsize=(20, 20))
    features = list(X.columns)
    for i in range(rows * columns):
        row = i // columns
        column = i - (row * columns)
        ax = axes[row, column]
        try:
            feature = features[i]
            ax.scatter(X[feature], y)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(feature)
        except IndexError:  # Blank out all the empty plots.
            ax.set_yticks([])
            ax.set_xticks([])
            [ax.spines[x].set_visible(False) for x in ("top", "bottom", "left", "right")]
    return fig


def plot_discrete_x_vs_discrete_y(x: pd.Series, y: pd.Series,
                                  sort_by_y: bool = True):
    """Plot counts by values in `x`, vs. the mean of `y` for each value.

    Args:
        x: discrete series of x data.
        y: discrete series of y data.
        sort_by_y: if True, plots are sorted by ascending mean value of `y` for
            each `x` value. Else sorted by `x` index.

    Returns:

    """
    indices_match(x, y)

    # Group the data:
    df = pd.concat([x, y], axis=1)

    if sort_by_y:
        y_by_x = df.groupby([x.name])[y.name].mean().sort_values()
        x_dist = df[x.name].value_counts().loc[y_by_x.index]
    else:
        x_dist = df[x.name].value_counts().sort_index()
        y_by_x = df.groupby([x.name])[y.name].mean().loc[x_dist.index]

    # Create the x-distribution plot:
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    axes[0].bar(range(len(x_dist)), x_dist, alpha=0.7, color="lightgray", label=f"{x.name} count")
    axes[0].grid(axis="y", linestyle="--", zorder=100)
    axes[0].set_title(f"{x.name} counts", loc="center", fontweight="bold")
    axes[0].set_xlim()
    axes[0].set_xticks(range(len(x_dist)))
    axes[0].set_xticklabels(x_dist.index, rotation=90)
    axes[0].set_xlim(0, len(x_dist))
    axes[0].legend()

    # Create the mean-y plot:
    axes[1].bar(range(len(y_by_x)), y_by_x, alpha=0.7, color="lightgray", label=f"{y.name} mean")
    axes[1].grid(axis="y", linestyle="--", zorder=100)
    axes[1].set_title(f"{y.name} mean by {x.name}", loc="center", fontweight="bold")
    axes[1].set_xticks(range(len(y_by_x)))
    axes[1].set_xticklabels(y_by_x.index, rotation=90)
    axes[1].set_xlim(0, len(y_by_x))
    axes[1].legend()

    return fig


def log_transform(feature: pd.Series):
    """Log transform a Pandas Series of data, making adjustments to ensure no
    values are <= 0."""
    if feature.min() > 0:
        return np.log(feature)
    elif feature.min() == 0:
        return np.log(feature + 1)
    else:
        return np.log(feature + np.abs(feature.min()) + 1)
