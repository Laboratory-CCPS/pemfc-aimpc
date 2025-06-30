import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.learning.neural_network import Scaled_Constraint_MLP, Scaled_MLP
from src.model_suh.conversions import rad2rpm


def data_as_csv(
    path, t_sim, t_calc, u_sim, x_sim, y_ft_sim, I_load, output_df=False, as_rpm=True
):
    savedict = {
        "t_sim": [],
        "t_calc": [],
        "p_O2": [],
        "p_N2": [],
        "w_cp": [],
        "p_sm": [],
        "I_load": [],
        "v_cm": [],
        "I_st": [],
        "lambda_O2": [],
    }

    savedict["t_sim"] = t_sim
    savedict["t_calc"] = t_calc
    savedict["lambda_O2"] = y_ft_sim[0]
    savedict["I_load"] = I_load

    for idx, name in enumerate(["v_cm", "I_st"]):
        savedict[name] = u_sim[idx, :]
    for idx, name in enumerate(["p_O2", "p_N2", "w_cp", "p_sm"]):
        if name == "w_cp" and as_rpm:
            savedict[name] = rad2rpm(x_sim[idx, :])
        else:
            savedict[name] = x_sim[idx, :]

    savedf = pd.DataFrame(savedict)
    savedf.to_csv(path, index=False)

    if output_df:
        return savedf


def load_scaled_NN(path):
    NN_params = torch.load(path, weights_only=False)
    load_proxy = Scaled_MLP(
        nin=NN_params["nin"],
        nout=NN_params["nout"],
        n_neurons=NN_params["n_neurons"],
        n_layers=NN_params["n_layers"],
        activation=NN_params["activation"],
        feature_mean=NN_params["feature_scaler"].mean_,
        feature_std=NN_params["feature_scaler"].scale_,
        label_mean=NN_params["label_scaler"].mean_,
        label_std=NN_params["label_scaler"].scale_,
    )
    load_proxy.load_state_dict(NN_params["model_state_dict"])
    return load_proxy


def load_scaled_constraint_NN(path):
    NN_params = torch.load(path, weights_only=False)
    load_proxy = Scaled_Constraint_MLP(
        nin=NN_params["nin"],
        nout=NN_params["nout"],
        n_neurons=NN_params["n_neurons"],
        n_layers=NN_params["n_layers"],
        activation=NN_params["activation"],
        feature_mean=NN_params["feature_scaler"].mean_,
        feature_std=NN_params["feature_scaler"].scale_,
        label_mean=NN_params["label_scaler"].mean_,
        label_std=NN_params["label_scaler"].scale_,
        out_constraints_high=NN_params["const_high"],
        out_constraints_low=NN_params["const_low"],
    )
    load_proxy.load_state_dict(NN_params["model_state_dict"])
    return load_proxy


def plot_dataframes(
    dataframes,
    x_column,
    area_columns=None,
    rescaled_columns=None,
    const_high=None,
    const_low=None,
    names=None,
    title=None,
    figsize=(16, 8),
    is_rpm=True,
    do_save=False,
):
    """
    Plots multiple DataFrames with a specified x-axis column and the remaining columns as y-values in subplots.

    Parameters:
    - dataframes (list of pd.DataFrame): The list of DataFrames containing the data to plot.
    - x_column (str): The name of the column to use for the x-axis.
    - bar_columns (list): List of columns to plot as bar plots.
    - rescaled_columns (list): List of columns to rescale before plotting.
    - names (list of str): List of names associated with each DataFrame for the legend.
    - title (str): An optional title for the overall figure.
    - figsize (tuple): The size of the figure.
    """
    if names is None:
        names = [f"DataFrame {i + 1}" for i in range(len(dataframes))]

    # Ensure the number of names matches the number of DataFrames
    if len(names) != len(dataframes):
        raise ValueError("The number of names must match the number of DataFrames.")

    # Extract columns to plot from the first DataFrame
    columns_to_plot = [col for col in dataframes[0].columns if col != x_column]

    # Format column names for LaTeX with subscripts
    def format_column_name(name):
        if "_" in name:
            base, subscript = name.split("_", 1)
            if base == "p":
                return f"${{{base}_{{\mathrm{{{subscript}}}}}}}/\mathrm{{bar}}$"
            elif base == "w":
                return f"${{\omega_{{\mathrm{{{subscript}}}}}}}/\mathrm{{krpm}}$"
            elif base == "I" and subscript != "load":
                return f"${{{base}_{{\mathrm{{{subscript}}}}}}}/\mathrm{{A}}$"
            elif base == "v":
                return f"${{{base}_{{\mathrm{{{subscript}}}}}}}/\mathrm{{V}}$"
            elif base == "lambda":
                return f"${{\{base}_{{\mathrm{{{subscript}}}}}}}$"
            elif base == "t":
                return f"${{{base}_{{\mathrm{{{subscript}}}}}}}/\mathrm{{s}}$"
        return f"${name}$"

    # Rescale values if required
    def rescale_values(df, columns_to_plot, rescaled_columns):
        rescaled_df = df.copy(deep=True)
        for col in columns_to_plot:
            if col in rescaled_columns:
                if "_" in col:
                    base, _ = col.split("_", 1)
                    if base == "p":
                        rescaled_df[col] = df[col] / 1e5
                    if base == "w":
                        if is_rpm:
                            rescaled_df[col] = df[col] / 1e3
                        else:
                            rescaled_df[col] = rad2rpm(df[col]) / 1e3
        return rescaled_df

    # Rescale all DataFrames
    rescaled_dataframes = [
        rescale_values(df, columns_to_plot, rescaled_columns or []) for df in dataframes
    ]

    # Determine subplot layout (rows and columns)
    num_plots = len(columns_to_plot)
    nrows = (num_plots + 1) // 2  # Two columns per row
    ncols = 2 if num_plots > 1 else 1

    # Create figure and axes
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, constrained_layout=True, sharex="all"
    )
    axes = axes.flatten() if num_plots > 1 else [axes]

    # Plot each column
    for i, col in enumerate(columns_to_plot):
        for idx, df_and_name in enumerate(zip(rescaled_dataframes, names)):
            df, name = df_and_name
            x = df[x_column]
            high_keys = list(const_high.keys())
            low_keys = list(const_low.keys())
            if col in high_keys:
                axes[i].axhline(
                    y=const_high[col],
                    color="black",
                    linestyle="--",
                    alpha=0.4,
                    linewidth=3,
                )
            if col in low_keys:
                axes[i].axhline(
                    y=const_low[col],
                    color="black",
                    linestyle="--",
                    alpha=0.4,
                    linewidth=3,
                )
            if col in area_columns:
                df_area = df.set_index(x_column)[col]
                df_area.plot.area(ax=axes[i], alpha=0.5, label=name, linewidth=3)
                axes[i].set_yscale("log")
            else:
                if idx > 1:
                    axes[i].plot(x, df[col], label=name, linewidth=4, linestyle="--")
                else:
                    axes[i].plot(x, df[col], label=name, linewidth=4)

        axes[i].set_xlabel(format_column_name(x_column), fontsize=30)
        axes[i].set_ylabel(format_column_name(col), fontsize=30)
        axes[i].tick_params(axis="x", labelsize=30)
        axes[i].tick_params(axis="y", labelsize=30)
        axes[i].grid(True, linestyle="--", alpha=0.7)
        if i == 0:
            axes[i].legend(fontsize=28)

    # Hide unused subplots (if any)
    for ax in axes[num_plots:]:
        ax.axis("off")

    # Add a main title if provided
    if title:
        fig.suptitle(title, fontsize=50, y=1.02)

    # Display the plot
    plt.show()
    if do_save:
        fig.savefig("Controller Comparison.svg", format="svg")
