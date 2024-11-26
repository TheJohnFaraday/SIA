import random
from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dataset.font_data import Font3
from src.Autoencoder import Autoencoder
from src.Configuration import Configuration
from src.Layer import Layer
from src.Letters import get_letters, get_letters_labels, convert_fonts_to_binary_matrix
from src.errors import MSE
from src.parse_configuration import read_configuration


@dataclass
class TrainedAutoencoder:
    autoencoder: Autoencoder
    network: Layer
    errors: Union[Layer, np.array, list[float]]
    trained_input: np.array
    trained_output: np.array
    binary_letters_matrix: np.array


def add_noise_to_letters(letters_matrix, intensity: float, spread: int):
    def add_noise_to_single_letter(letter):
        noise_matrix = np.array(letter).astype(float).reshape(7, 5)

        def add_noise_around(row: int, column: int):
            for y_offset in range(-spread, spread + 1):
                for x_offset in range(-spread, spread + 1):
                    if noise_matrix.shape[0] > y_offset + row >= 0 \
                            and noise_matrix.shape[1] > x_offset + column >= 0:
                        noise_matrix[row + y_offset][column + x_offset] += np.random.normal(1, 1) * intensity / 2

        for i in range(noise_matrix.shape[0]):
            for j in range(noise_matrix.shape[1]):
                if noise_matrix[i][j] != 1:
                    continue

                add_noise_around(i, j)

        maximum_cell_value = max(cell for row in noise_matrix for cell in row)
        return [cell / maximum_cell_value for row in noise_matrix for cell in row]

    return np.array([add_noise_to_single_letter(letter) for letter in letters_matrix])


def plot_latent_space(autoencoder, input_data, labels, suffix_filename: str):

    latent_points = []

    for i, pattern in enumerate(input_data):
        encoded = autoencoder.encode(pattern)
        encoded = encoded.flatten()
        latent_points.append(encoded)

    latent_points = np.array(latent_points)

    plt.figure(figsize=(12, 10))
    for i, (x, y) in enumerate(latent_points):
        plt.scatter(x, y, label=labels[i])
        plt.text(x+ 0.001, y+0.03, labels[i], fontsize=18, weight='bold', color='darkblue')

    plt.title("Espacio Latente del Autoencoder", fontsize=16, weight='bold')
    plt.xlabel("Nodo 1", fontsize=14)
    plt.ylabel("Nodo 2", fontsize=14)
    plt.grid(True)
    # plt.show()
    plt.savefig(f"./plots/latent_space-{suffix_filename}.png")


def display_comparison_heatmaps(input_matrix, autoencoder_output, suffix_filename: str,  middle_row=None):
    num_chars = input_matrix.shape[0]
    half = num_chars // 2
    fig, axes = plt.subplots(
        4 if middle_row is None else 6,
        half, figsize=(16, 8)
    )  # 2 filas por mitad: Input y Output

    fig.suptitle("Input vs Autoencoder Output Heatmaps", fontsize=20)

    input_cmap = "gray_r"  # Monochrome
    middle_cmap = "Reds"
    output_cmap = "Blues"  # Continuous colormap para Output

    for i in range(num_chars):
        row, col = divmod(i, half)

        # Primera fila
        ax_input = axes[row * 2 if middle_row is None else row * 3, col]
        sns.heatmap(
            input_matrix[i],
            linewidths=0.2,
            cbar=False,
            square=True,
            cmap=input_cmap,
            linecolor="k",
            ax=ax_input,
        )
        ax_input.axis("off")

        if middle_row is not None:
            # Middle row
            ax_output = axes[row * 3 + 1, col]
            reshaped_output = middle_row[i].reshape(7, 5)  # Ajuste a (7,5)
            sns.heatmap(
                reshaped_output,
                linewidths=0.2,
                cbar=False,
                square=True,
                cmap=middle_cmap,
                linecolor="k",
                ax=ax_output,
            )
            ax_output.axis("off")

        # Segunda fila
        ax_output = axes[row * 2 + 1 if middle_row is None else row * 3 + 2, col]
        reshaped_output = autoencoder_output[i].reshape(7, 5)  # Ajuste a (7,5)
        sns.heatmap(
            reshaped_output,
            linewidths=0.2,
            cbar=False,
            square=True,
            cmap=output_cmap,
            linecolor="k",
            ax=ax_output,
        )
        ax_output.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"./plots/comparison-heatmaps-{suffix_filename}.png")
    # plt.show()



def display_single_character_heatmap(binary_matrix, index, suffix_filename: str):
    fig, ax = plt.subplots(figsize=(2, 3))

    monochromatic_cmap = plt.cm.colors.ListedColormap(["white", "black"])

    sns.heatmap(
        binary_matrix[index],
        linewidths=0.2,
        cbar=False,
        square=True,
        cmap=monochromatic_cmap,
        linecolor="k",
        ax=ax,
    )
    ax.axis("off")
    plt.title(f"Character {index}")
    plt.savefig(f"./plots/single-character-comparison-heatmap-{index}-{suffix_filename}.png")


def plot_training_error(errors, suffix_filename: str):
    epochs = range(1, len(errors) + 1)  # Crear un rango para las épocas (comienza en 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, errors, label="Training Error", color="blue")
    plt.title("Error durante el entrenamiento del Autoencoder")
    plt.xlabel("Época")
    plt.ylabel("Error")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./plots/training-error-{suffix_filename}.png")


def train_predictor(configuration: Configuration):
    letters_matrix = get_letters()
    # layers = [60, 50, 30, 10, 5]
    layers = [60, 50, 40, 30, 20, 10, 5]
    # layers = [70, 60, 50, 40, 30, 20, 15, 10, 5]
    latent_space_dim = 2
    autoencoder = Autoencoder(
        letters_matrix.shape[1] * letters_matrix.shape[2],  # 35 (flattened)
        letters_matrix.shape[0],
        layers,
        latent_space_dim,
        MSE(),
        configuration.epochs,
        configuration.beta,
        configuration.adam.beta1,
        configuration.adam.beta2,
        configuration.epsilon,
        configuration.learning_rate,
        )

    new_network, errors = autoencoder.train(letters_matrix, letters_matrix)

    # Generate predictions for each input
    reconstructed_output = np.array(
        [
            autoencoder.predict(x).reshape(
                7, 5
            )  # Reshape each output to 7x5 for visualization
            for x in letters_matrix
        ]
    )

    # Reshape input for the display function (to match reconstructed_output)
    reshaped_input = np.array([x.reshape(7, 5) for x in letters_matrix])

    return TrainedAutoencoder(autoencoder=autoencoder, network=new_network, errors=errors, trained_input=reshaped_input,
                              trained_output=reconstructed_output, binary_letters_matrix=letters_matrix)



def ej_1_a(configuration: Configuration, trained: TrainedAutoencoder):
    # Seleccionar un subconjunto de dos caracteres (e.g., índices 0 y 1)
    subset_indices = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]  # Cambia los índices según los caracteres que quieras usar
    subset_binary_matrix = get_letters()[subset_indices]

    # print("INPUT")
    # print(reshaped_input)
    # print("OUTPUT")
    # print(reconstructed_output)

    a_encoded_value = trained.autoencoder.encode(trained.binary_letters_matrix[1])
    b_encoded_value = trained.autoencoder.encode(trained.binary_letters_matrix[2])
    print(a_encoded_value)
    print(b_encoded_value)
    delta_x = abs(b_encoded_value[0] - a_encoded_value[0]) / 5
    delta_y = abs(b_encoded_value[1] - a_encoded_value[1]) / 5
    new_x = a_encoded_value[0]
    new_y = a_encoded_value[1]
    new_letters = []
    for i in range(6):
        new_letters.append(trained.autoencoder.decode([new_x, new_y]))
        new_x += delta_x
        new_y += delta_y
    new_letters = np.array(new_letters).reshape(len(range(6)), 7, 5)
    print(new_letters)

    if configuration.plot:
        display_comparison_heatmaps(trained.trained_input, trained.trained_output, suffix_filename="ej_1a")
        for i in range(6):
            display_single_character_heatmap(new_letters, i, suffix_filename="ej_1a")
        plot_training_error(trained.errors, suffix_filename="ej_1a")
        plot_latent_space(trained.autoencoder, trained.binary_letters_matrix, get_letters_labels(), suffix_filename="ej_1a")


def ej_1_b(configuration: Configuration, trained: TrainedAutoencoder):
    if not configuration.noise:
        raise RuntimeError("'noise' configuration is needed for this item to be executed")

    letters_with_noise = add_noise_to_letters(convert_fonts_to_binary_matrix(Font3), configuration.noise.intensity,
                                              configuration.noise.spread)

    letters_with_noise_for_autoencoder = np.reshape(
        letters_with_noise, (32, 35, 1)
    )

    predicted = []
    for letter in letters_with_noise_for_autoencoder:
        predicted_letter = trained.autoencoder.predict(letter).reshape(7, 5)
        predicted.append(predicted_letter)

    if configuration.plot:
        display_comparison_heatmaps(trained.trained_output, predicted, suffix_filename="ej_1b",
                                    middle_row=letters_with_noise)
        plot_training_error(trained.errors, suffix_filename="ej_1b")


if __name__ == "__main__":
    configuration: Configuration = read_configuration("config.toml")
    if configuration.seed:
        random.seed(configuration.seed)
        np.random.seed(configuration.seed)

    trained = train_predictor(configuration)

    ej_1_a(configuration, trained)
    ej_1_b(configuration, trained)
