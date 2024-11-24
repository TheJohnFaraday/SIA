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
from src.Optimizer import GradientDescent
from src.Training import MiniBatch, Batch


def convert_fonts_to_binary_matrix(font_array):
    binary_matrix = []
    for character in font_array:
        char_matrix = []
        for hex_value in character:
            binary_value = bin(hex_value)[2:].zfill(8)[-5:]  # Convert to binary and get last 5 bits
            char_matrix.append([int(bit) for bit in binary_value])
        binary_matrix.append(char_matrix)
    return np.array(binary_matrix)


def display_comparison_heatmaps(input_matrix, autoencoder_output, rows=4, cols=8):
    num_chars = input_matrix.shape[0]
    fig, axes = plt.subplots(2, rows * cols, figsize=(16, 8))  # Two rows: one for input, one for output
    fig.suptitle("Input vs Autoencoder Output Heatmaps", fontsize=16)

    # Use 'gray_r' for binary input visualization
    input_cmap = 'gray_r'  # Monochrome inverted colormap
    output_cmap = 'coolwarm'  # Continuous colormap for autoencoder output

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
            linecolor='k',
            ax=ax_input
        )
        ax_input.set_title(f"Input {i + 1}")
        ax_input.axis('off')

        # Display autoencoder output character
        ax_output = axes[1, row * cols + col]
        reshaped_output = autoencoder_output[i].reshape(7, 5)  # Reshape output to (7,5)
        sns.heatmap(
            reshaped_output,
            linewidths=0.2,
            cbar=False,
            square=True,
            cmap=output_cmap,
            linecolor='k',
            ax=ax_output
        )
        ax_output.set_title(f"Output {i + 1}")
        ax_output.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def display_single_character_heatmap(binary_matrix, index):
    fig, ax = plt.subplots(figsize=(2, 3))

    monochromatic_cmap = plt.cm.colors.ListedColormap(['white', 'black'])

    sns.heatmap(
        binary_matrix[index],
        linewidths=0.2,
        cbar=False,
        square=True,
        cmap=monochromatic_cmap,
        linecolor='k',
        ax=ax
    )
    ax.axis('off')
    plt.title(f"Character {index}")
    plt.show()


if __name__ == '__main__':
    configuration: Configuration = read_configuration("config.toml")
    if configuration.seed:
        random.seed(configuration.seed)
        np.random.seed(configuration.seed)

    # Convert font data to binary matrix and reshape
    binary_matrix = convert_fonts_to_binary_matrix(Font3)
    binary_matrix = np.reshape(binary_matrix, (32, 35, 1))  # Reshape input to (32, 35, 1) for compatibility

    # Seleccionar un subconjunto de dos caracteres (e.g., índices 0 y 1)
    subset_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Cambia los índices según los caracteres que quieras usar
    subset_binary_matrix = binary_matrix[subset_indices]

    input_size = binary_matrix.shape[1] * binary_matrix.shape[2]  # 35 (flattened)

    encoder_layers = [
        Dense(input_size=input_size, output_size=20, optimizer=GradientDescent(learning_rate=configuration.learning_rate)),
        Tanh(beta=configuration.beta),
        Dense(input_size=20, output_size=15, optimizer=GradientDescent(learning_rate=configuration.learning_rate)),
        Tanh(beta=configuration.beta),
        Dense(input_size=15, output_size=2, optimizer=GradientDescent(learning_rate=configuration.learning_rate)),  # Espacio latente de dimensión 2
        Tanh(beta=configuration.beta),
    ]

    decoder_layers = [
        Dense(input_size=2, output_size=15, optimizer=GradientDescent(learning_rate=configuration.learning_rate)),
        Tanh(beta=configuration.beta),
        Dense(input_size=15, output_size=20, optimizer=GradientDescent(learning_rate=configuration.learning_rate)),
        Tanh(beta=configuration.beta),
        Dense(input_size=20, output_size=input_size, optimizer=GradientDescent(learning_rate=configuration.learning_rate)),
        Tanh(beta=configuration.beta)
    ]

    error_function = MSE()

    # Define the autoencoder model
    autoencoder = MultiLayerPerceptron(training_method=Batch(
        MultiLayerPerceptron.predict,
        batch_size=len(subset_indices),  # Batch size igual al número de caracteres del subset
        epsilon=configuration.epsilon,
    ), neural_network=encoder_layers + decoder_layers, error=error_function, epochs=configuration.epochs, learning_rate=configuration.learning_rate)

    # Normalización opcional
    normalized_input = 2 * subset_binary_matrix - 1

    # Train the autoencoder
    new_network, _ = autoencoder.train(binary_matrix, binary_matrix)

    # Generate predictions for each input
    reconstructed_output = np.array([
        MultiLayerPerceptron.predict(new_network, x).reshape(7, 5)  # Reshape each output to 7x5 for visualization
        for x in binary_matrix
    ])

    # Reshape input for the display function (to match reconstructed_output)
    reshaped_input = np.array([x.reshape(7, 5) for x in binary_matrix])

    print("INPUT")
    print(reshaped_input)
    print("OUTPUT")
    print(reconstructed_output)

    if configuration.plot:
        display_comparison_heatmaps(reshaped_input, reconstructed_output)
