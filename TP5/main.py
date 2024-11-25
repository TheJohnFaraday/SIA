import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset.font_data import Font3
from src.Configuration import Configuration
from src.parse_configuration import read_configuration
from src.Dense import Dense
from src.activation_functions import ReLU, Logistic, Tanh, Sigmoid, Linear
from src.MultiLayerPerceptron import MultiLayerPerceptron
from src.errors import MSE
from src.Optimizer import GradientDescent, Adam
from src.Training import MiniBatch, Batch
from src.Autoencoder import Autoencoder


def convert_fonts_to_binary_matrix(font_array):
    binary_matrix = []
    for character in font_array:
        char_matrix = []
        for hex_value in character:
            binary_value = bin(hex_value)[2:].zfill(8)[
                -5:
            ]  # Convert to binary and get last 5 bits
            char_matrix.append([int(bit) for bit in binary_value])
        binary_matrix.append(char_matrix)
    return np.array(binary_matrix)

def plot_latent_space(autoencoder, input_data, labels):

    latent_points = []

    for i, pattern in enumerate(input_data):
        encoded = autoencoder.encode(pattern)
        encoded = encoded.flatten()
        latent_points.append(encoded)

    latent_points = np.array(latent_points)

    plt.figure(figsize=(12, 10))
    for i, (x, y) in enumerate(latent_points):
        plt.scatter(x, y, label=labels[i])
        plt.text(x+ 0.001, y+0.03, labels[i], fontsize=18, weight='bold', color='darkblue')  # Letras más grandes y resaltadas

    plt.title("Espacio Latente del Autoencoder", fontsize=16, weight='bold')
    plt.xlabel("Nodo 1", fontsize=14)
    plt.ylabel("Nodo 2", fontsize=14)
    plt.grid(True)
    plt.show()




def display_comparison_heatmaps(input_matrix, autoencoder_output, rows=4, cols=8):
    num_chars = input_matrix.shape[0]
    fig, axes = plt.subplots(
        2, rows * cols, figsize=(32, 16)
    )  # Two rows: one for input, one for output
    fig.suptitle("Input vs Autoencoder Output Heatmaps", fontsize=40)

    # Use 'gray_r' for binary input visualization
    input_cmap = "gray_r"  # Monochrome inverted colormap
    output_cmap = "coolwarm"  # Continuous colormap for autoencoder output

    for i in range(num_chars):
        row, col = divmod(i, cols)

        # Display input character
        ax_input = axes[0, row * cols + col]
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

        # Display autoencoder output character
        ax_output = axes[1, row * cols + col]
        reshaped_output = autoencoder_output[i].reshape(7, 5)  # Reshape output to (7,5)
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
    plt.savefig("./plots/comparison-heatmaps.png")


def display_single_character_heatmap(binary_matrix, index):
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
    plt.savefig(f"./plots/single-character-comparison-heatmap-{index}.png")


def plot_training_error(errors):
    epochs = range(1, len(errors) + 1)  # Crear un rango para las épocas (comienza en 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, errors, label="Training Error", color="blue")
    plt.title("Error durante el entrenamiento del Autoencoder")
    plt.xlabel("Época")
    plt.ylabel("Error")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/training-error.png")


if __name__ == "__main__":
    configuration: Configuration = read_configuration("config.toml")
    if configuration.seed:
        random.seed(configuration.seed)
        np.random.seed(configuration.seed)

    # Convert font data to binary matrix and reshape
    binary_matrix = convert_fonts_to_binary_matrix(Font3)
    binary_matrix = np.reshape(
        binary_matrix, (32, 35, 1)
    )  # Reshape input to (32, 35, 1) for compatibility

    labels = [
        '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
        'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
        't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'DEL'
    ]

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
    subset_binary_matrix = binary_matrix[subset_indices]

    input_size = binary_matrix.shape[1] * binary_matrix.shape[2]  # 35 (flattened)

    autoencoder = Autoencoder(
        input_size,
        MSE(),
        configuration.epochs,
        configuration.beta,
        configuration.adam.beta1,
        configuration.adam.beta2,
        configuration.epsilon,
        configuration.learning_rate,
    )

    new_network, errors = autoencoder.train(binary_matrix, binary_matrix)

    # Generate predictions for each input
    reconstructed_output = np.array(
        [
            autoencoder.predict(x).reshape(
                7, 5
            )  # Reshape each output to 7x5 for visualization
            for x in binary_matrix
        ]
    )

    # Reshape input for the display function (to match reconstructed_output)
    reshaped_input = np.array([x.reshape(7, 5) for x in binary_matrix])

    # print("INPUT")
    # print(reshaped_input)
    # print("OUTPUT")
    # print(reconstructed_output)

    a_encoded_value = autoencoder.encode(binary_matrix[1])
    b_encoded_value = autoencoder.encode(binary_matrix[2])
    print(a_encoded_value)
    print(b_encoded_value)
    delta_x = abs(b_encoded_value[0] - a_encoded_value[0]) / 5
    delta_y = abs(b_encoded_value[1] - a_encoded_value[1]) / 5
    new_x = a_encoded_value[0]
    new_y = a_encoded_value[1]
    new_letters = []
    for i in range(6):
        new_letters.append(autoencoder.decode([new_x, new_y]))
        new_x += delta_x
        new_y += delta_y
    new_letters = np.array(new_letters).reshape(len(range(6)), 7, 5)
    print(new_letters)

    if configuration.plot:
        display_comparison_heatmaps(reshaped_input, reconstructed_output)
        for i in range(6):
            display_single_character_heatmap(new_letters, i)
        plot_training_error(errors)

    plot_latent_space(autoencoder, binary_matrix, labels)