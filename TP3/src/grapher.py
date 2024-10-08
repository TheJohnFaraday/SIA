import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from exercises.NetworkOutput import NetworkOutput
from src.configuration import (
    Configuration,
    TrainingStyle,
    Optimizer,
    ActivationFunction,
)

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

map_activation = {
    ActivationFunction.TANH: "tanh",
    ActivationFunction.LOGISTIC: "Logistic",
}


def graph_error_by_epoch(
    exercise: str,
    config: Configuration,
    errors_by_epoch: list[float],
    proportion: float,
):
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
            " | "
            f"Activation = {map_activation[config.multilayer.digits_discrimination_activation_function]}"
            " | "
            f"Proportion = {proportion}"
        )
        plt.savefig(
            f"plots/{exercise}_error_by_epoch"
            f"_lr-{config.learning_rate}"
            f"_style-{config.multilayer.training_style}"
            f"_opt-{config.multilayer.optimizer}"
            f"_activation-{config.multilayer.digits_discrimination_activation_function}"
            f"_proportion-{round(proportion, 1)}"
            f".png"
        )

    errors_df = pd.DataFrame(
        [float(e) for e in errors_by_epoch],
        columns=["Error"],
    )

    print_epoch_error(errors_df)


def graph_is_odd_matrix(
    exercise: str,
    config: Configuration,
    outputs: list[NetworkOutput],
    proportion: float,
):
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
            " | "
            f"Activation = {map_activation[config.multilayer.digits_discrimination_activation_function]}"
            " | "
            f"Proportion = {proportion}"
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
        plt.savefig(
            f"plots/{exercise}_output_matrix"
            f"_lr-{config.learning_rate}"
            f"_style-{config.multilayer.training_style}"
            f"_opt-{config.multilayer.optimizer}"
            f"_activation-{config.multilayer.digits_discrimination_activation_function}"
            f"_proportion-{round(proportion, 1)}"
            f".png"
        )

    outputs_df = pd.DataFrame(
        {
            "Expected Output": [output.expected for output in outputs],
            "Output": [output.output for output in outputs],
            "Error": [output.error for output in outputs],
        }
    )
    outputs_df["Rounded Output"] = outputs_df["Output"].apply(lambda x: round(x[0][0]))
    print_output_matrix(outputs_df)


def graph_accuracy_vs_dataset(
    exercise: str, config: Configuration, dataset_accuracy: list[tuple[float, float]]
):
    def print_accuracy_vs_proportion(df: pd.DataFrame):
        fig, ax = plt.subplots()

        ax.plot(df["Proportion"], df["Accuracy"], color=CUSTOM_PALETTE[0])

        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Proportion")

        plt.ylim(min(0.0, df["Accuracy"].min()), max(1.0, df["Accuracy"].max()))

        fig.suptitle("Accuracy vs Proportion of the Dataset")
        ax.set_title(
            f"Learning Rate = {config.learning_rate}"
            " | "
            f"Training = {map_training[config.multilayer.training_style]}"
            " | "
            f"Optimizer = {map_optimizer[config.multilayer.optimizer]}"
        )
        plt.savefig(
            f"plots/{exercise}_accuracy_vs_proportion"
            f"_lr-{config.learning_rate}"
            f"_style-{config.multilayer.training_style}"
            f"_opt-{config.multilayer.optimizer}"
            f"_activation-{config.multilayer.digits_discrimination_activation_function}"
            f".png"
        )

    df = pd.DataFrame(dataset_accuracy, columns=["Proportion", "Accuracy"])
    print_accuracy_vs_proportion(df)


def graph_which_number_matrix(
    exercise: str,
    config: Configuration,
    outputs: list[NetworkOutput],
    proportion: float,
):
    def print_output_matrix(df: pd.DataFrame):
        confusion_matrix = [[0] * 10 for _ in range(10)]
        for _, row in df.iterrows():
            expected = np.argmax(row["Expected Output"])
            predicted = np.argmax(row["Output"])

            confusion_matrix[expected][predicted] += 1

        # for i in range(len(confusion_matrix)):
        #     row_expected = confusion_matrix[i]
        #     total = sum(row_expected)
        #     for j in range(len(row_expected)):
        #         confusion_matrix[i][j] = confusion_matrix[i][j] / total

        matrix_df = pd.DataFrame(confusion_matrix, index=[i for i in range(10)], columns=[i for i in range(10)])

        fig, ax = plt.subplots(figsize=(8, 6))

        fig.suptitle("Predictions by Expected Value")
        ax = sns.heatmap(
            matrix_df,
            annot=True,
            cmap=LinearSegmentedColormap.from_list(
                "CUSTOM_GRADIENT", ["#eeeeee", CUSTOM_PALETTE[0]]
            ),
            cbar=True,
            fmt="d",
        )

        ax.set_title(
            f"Learning Rate = {config.learning_rate}"
            " | "
            f"Training = {map_training[config.multilayer.training_style]}"
            " | "
            f"Optimizer = {map_optimizer[config.multilayer.optimizer]}"
            " | "
            f"Activation = {map_activation[config.multilayer.digits_discrimination_activation_function]}"
            " | "
            f"Proportion = {proportion}"
        )

        ax.set_ylabel("Expected")
        ax.set_xlabel("Predicted")

        plt.savefig(
            f"plots/{exercise}_output_matrix"
            f"_lr-{config.learning_rate}"
            f"_style-{config.multilayer.training_style}"
            f"_opt-{config.multilayer.optimizer}"
            f"_activation-{config.multilayer.digits_discrimination_activation_function}"
            f"_proportion-{round(proportion, 1)}"
            f".png"
        )

    outputs_df = pd.DataFrame(
        {
            "Expected Output": [output.expected for output in outputs],
            "Output": [output.output.flatten() for output in outputs],
            "Error": [output.error for output in outputs],
        }
    )
    outputs_df["Rounded Output"] = outputs_df["Output"].apply(lambda x: round(x[0]))
    print(outputs_df)
    print_output_matrix(outputs_df)
