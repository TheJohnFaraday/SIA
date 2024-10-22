import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

from dataclasses import dataclass


@dataclass
class Coordinates:
    x: float | int
    y: float | int
    z: float | int = None


@dataclass
class Rotation:
    elevation: float
    azimuth: float


CUSTOM_PALETTE = [
    "#508fbe",  # blue
    "#f37120",  # orange
    "#4baf4e",  # green
    "#f2cb31",  # yellow
    "#c178ce",  # purple
    "#cd4745",  # red
    "#9ef231",  # light green
    "#50beaa",  # green + blue
    "#8050be",  # violet
    "#cf1f51",  # magenta
]
GREY = "#6f6f6f"
LIGHT_GREY = "#bfbfbf"

PLT_THEME = {
    "axes.prop_cycle": plt.cycler(color=CUSTOM_PALETTE),  # Set palette
    "axes.spines.top": False,  # Remove spine (frame)
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": LIGHT_GREY,
    "axes.titleweight": "normal",  # Optional: ensure title weight is normal (not bold)
    "axes.titlelocation": "center",  # Center the title by default
    "axes.titlecolor": GREY,  # Set title color
    "axes.labelcolor": GREY,  # Set labels color
    "axes.labelpad": 12,
    "axes.titlesize": 10,
    "xtick.bottom": False,  # Remove ticks on the X axis
    "ytick.labelcolor": GREY,  # Set Y ticks color
    "ytick.color": GREY,  # Set Y label color
    "savefig.dpi": 128,
    "legend.frameon": False,
    "legend.labelcolor": GREY,
    "figure.titlesize": 16,  # Set suptitle size
}

plt.style.use(PLT_THEME)
sns.set_palette(CUSTOM_PALETTE)
sns.set_style(PLT_THEME)


def scatter_pca(
    title: str,
    exercise: str,
    plot_name: str,
    data_pca,
    df: pd.DataFrame,
    countries: list[str],
    scale_factor: int | float = 1,
    text_scale_factor: int | float = 1,
    arrow_text_offset: Coordinates = Coordinates(0, 0),
    subtitle: str | None = None,
):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)

    xs = data_pca[:, 0]
    ys = data_pca[:, 1]

    ax.scatter(
        xs,
        ys,
        c=[CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)] for i in range(len(xs))],
        alpha=0.8,
    )

    for i in range(len(countries)):
        ax.text(xs[i], ys[i] + 0.1, countries[i], ha="center", fontsize=8)

    for idx, variable_label in enumerate(df.index.values):
        head_x = df.iloc[idx, 0]
        head_y = df.iloc[idx, 1]
        ax.arrow(
            0,
            0,
            head_x * scale_factor,
            head_y * scale_factor,
            color=CUSTOM_PALETTE[1],
            alpha=0.5,
            head_width=0.05,
            head_length=0.09,
        )

        right_offset = Coordinates(
            x=arrow_text_offset.x,
            y=arrow_text_offset.y if head_y >= 0 else -1 * arrow_text_offset.y,
        )

        ax.text(
            head_x * text_scale_factor + right_offset.x,
            head_y * text_scale_factor + right_offset.y,
            variable_label,
            color=CUSTOM_PALETTE[1],
            ha="center",
            va="center",
            fontsize=10,
        )

    ax.set_ylabel(df.columns[1])
    ax.set_xlabel(df.columns[0])

    fig.suptitle(title)
    if subtitle:
        ax.set_title(subtitle)

    plt.tight_layout()
    plt.savefig(f"plots/{exercise}_{plot_name}_biplot.png")


def boxplot_pca(
    title: str,
    exercise: str,
    plot_name: str,
    df: pd.DataFrame,
    label_x: str,
    label_y: str,
    subtitle: str | None = None,
):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)

    ax = sns.boxplot(df, boxprops=dict(alpha=0.65))
    sns.despine(offset=10, trim=True)

    ax.set_xlabel(label_y)
    ax.set_xlabel(label_x)

    fig.suptitle(title)
    if subtitle:
        ax.set_title(subtitle)

    plt.tight_layout()
    plt.savefig(f"plots/{exercise}_{plot_name}_boxplot.png")


def pc1_pca(
    title: str,
    exercise: str,
    plot_name: str,
    xs,
    ys,
    label_x: str,
    label_y: str,
    subtitle: str | None = None,
):

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(xs, ys, alpha=0.8, color=CUSTOM_PALETTE)

    x1, x2, y1, y2 = plt.axis()
    if y1 > -1 and y2 < 1:
        plt.ylim(-1, 1)
        y1 = -1
        y2 = 1

    for i in range(len(xs)):
        if y1 < -1 and y2 > 1:
            # Big values in Y axis
            right_ys = ys[i] + 0.1 if ys[i] >= 0 else ys[i] - 0.18
        else:
            right_ys = ys[i] + 0.05 if ys[i] >= 0 else ys[i] - 0.1

        ax.text(xs[i], right_ys, f"{ys[i]:.2f}", ha="center", fontsize=10)

    ax.set_ylabel(label_y)
    ax.set_xlabel(label_x)

    plt.xticks(rotation=45, ha="right")
    if len(ys) > 10:
        plt.tick_params(bottom=True)

    fig.suptitle(title)
    if subtitle:
        ax.set_title(subtitle)

    plt.tight_layout()
    plt.savefig(f"plots/{exercise}_{plot_name}_pc1.png")


def three_dimension_pca(
    title: str,
    exercise: str,
    plot_name: str,
    data_pca,
    df: pd.DataFrame,
    countries: list[str],
    scale_factor: int | float = 1,
    text_scale_factor: int | float = 1,
    arrow_text_offset: Coordinates = Coordinates(0, 0, 0),
    subtitle: str | None = None,
    rotation: Rotation = Rotation(elevation=10, azimuth=-25),
):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    xs = data_pca[:, 0]
    ys = data_pca[:, 1]
    zs = data_pca[:, 2]

    ax.scatter(
        xs,
        ys,
        zs,
        c=[CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)] for i in range(len(xs))],
        alpha=0.8,
    )

    for i in range(len(countries)):
        ax.text(xs[i], ys[i], zs[i] + 0.1, countries[i], ha="center", fontsize=8)

    for idx, variable_label in enumerate(df.index.values):
        head_x = df.iloc[idx, 0]
        head_y = df.iloc[idx, 1]
        head_z = df.iloc[idx, 2]
        ax.quiver(
            0,
            0,
            0,
            head_x * scale_factor,
            head_y * scale_factor,
            head_z * scale_factor,
            color=CUSTOM_PALETTE[1],
            alpha=0.5,
        )

        right_offset = Coordinates(
            x=arrow_text_offset.x,
            y=arrow_text_offset.y if head_y >= 0 else -1 * arrow_text_offset.y,
            z=arrow_text_offset.z if head_z >= 0 else -1 * arrow_text_offset.z,
        )

        ax.text(
            head_x * text_scale_factor + right_offset.x,
            head_y * text_scale_factor + right_offset.y,
            head_z * text_scale_factor + right_offset.z,
            variable_label,
            color=CUSTOM_PALETTE[1],
            ha="center",
            va="center",
            fontsize=10,
        )

    # Plot rotation
    ax.view_init(elev=rotation.elevation, azim=rotation.azimuth)

    ax.set_zlabel(df.columns[2])
    ax.set_ylabel(df.columns[1])
    ax.set_xlabel(df.columns[0])

    fig.suptitle(title)
    if subtitle:
        ax.set_title(subtitle)
    else:
        ax.set_title(f"Elevation: {rotation.elevation} | Azimuth: {rotation.azimuth}")

    plt.tight_layout()
    plt.savefig(f"plots/{exercise}_{plot_name}_biplot_3d_rot-{rotation.elevation}-{rotation.azimuth}.png")
