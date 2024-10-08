import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from exercises.NetworkOutput import NetworkOutput
from src.configuration import Configuration, TrainingStyle, Optimizer

CUSTOM_PALETTE = [
    "#508fbe",
    "#f37120",
    "#4baf4e",
    "#f2cb31",
    "#c178ce",
    "#cd4745",
]
GREY = "#6f6f6f"
LIGHT_GREY = "#bfbfbf"

PLT_THEME = {
    "axes.prop_cycle": plt.cycler(color=CUSTOM_PALETTE),  # Set palette
    "axes.spines.top": False,  # Remove spine (frame)
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": True,
    "axes.edgecolor": LIGHT_GREY,
    "axes.titleweight": "normal",  # Optional: ensure title weight is normal (not bold)
    "axes.titlelocation": "center",  # Center the title by default
    "axes.titlecolor": GREY,  # Set title color
    "axes.labelcolor": GREY,  # Set labels color
    "axes.labelpad": 10,
    "axes.titlesize": 8,
    "xtick.bottom": False,  # Remove ticks on the X axis
    "ytick.labelcolor": GREY,  # Set Y ticks color
    "ytick.color": GREY,  # Set Y label color
    "savefig.dpi": 128,
    "legend.frameon": False,
    "legend.labelcolor": GREY,
}

plt.style.use(PLT_THEME)
sns.set_palette(CUSTOM_PALETTE)
sns.set_style(PLT_THEME)


def graph_error_by_epoch(
    config: Configuration, outputs: list[NetworkOutput], errors_by_epoch: list[float]
):
    map_training = {
        TrainingStyle.BATCH: "Batch",
        TrainingStyle.MINIBATCH: "MiniBatch",
        TrainingStyle.ONLINE: "Online",
    }

    map_optimizer = {
        Optimizer.GRADIENT_DESCENT: "Gradient Descent",
        Optimizer.ADAM: "Adam",
        Optimizer.MOMENTUM: "Momentum",
    }

    def print_epoch_error(df: pd.DataFrame):
        fig, ax = plt.subplots()

        ax.plot(df.index, df["Error"], color=CUSTOM_PALETTE[1])

        ax.set_ylabel("Error")
        ax.set_xlabel("Epoch")
        fig.suptitle("Error by Epoch")
        ax.set_title(
            f"Learning Rate = {config.learning_rate}"
            " | "
            f"Training = {map_training[config.multilayer.training_style]}"
            " | "
            f"Optimizer = {map_optimizer[config.multilayer.optimizer]}"
        )
        plt.savefig("plots/error_by_epoch.png")

    def print_output_matrix(df: pd.DataFrame):
        confusion_matrix = {
            "Not Predicted": [0, 0],
            "Predicted": [0, 0],
        }
        for _, row in df.iterrows():
            expected = row["Expected Output"]
            value = row["Rounded Output"]

            if expected == value:
                confusion_matrix["Predicted"][expected] += 1
            else:
                confusion_matrix["Not Predicted"][expected] += 1

        matrix_df = pd.DataFrame(confusion_matrix, index=["Expected 0", "Expected 1"])

        fig, ax = plt.subplots(figsize=(6, 4))

        fig.suptitle("Predictions by Expected Value")
        ax.set_title(
            f"Learning Rate = {config.learning_rate}"
            " | "
            f"Training = {map_training[config.multilayer.training_style]}"
            " | "
            f"Optimizer = {map_optimizer[config.multilayer.optimizer]}"
        )

        sns.heatmap(
            matrix_df,
            annot=True,
            cmap=LinearSegmentedColormap.from_list(
                "CUSTOM_GRADIENT", ["#eeeeee", CUSTOM_PALETTE[0]]
            ),
            cbar=True,
            fmt="d",
        )
        plt.savefig("plots/output_matrix.png")

    errors_df = pd.DataFrame(
        [float(e) for e in errors_by_epoch],
        columns=["Error"],
    )

    outputs_df = pd.DataFrame(
        {
            "Expected Output": [output.expected for output in outputs],
            "Output": [output.output for output in outputs],
            "Error": [output.error for output in outputs],
        }
    )
    outputs_df["Rounded Output"] = outputs_df["Output"].apply(lambda x: round(x[0][0]))

    print(errors_df)
    print(outputs_df)
    print_epoch_error(errors_df)
    print_output_matrix(outputs_df)
