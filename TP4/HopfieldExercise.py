import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import itertools
import pandas as pd

from src.Hopfield import Hopfield

letters = {
    "A": [
            [-1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
        ],
    "B":  [
            [1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, -1],
        ],
    "C": [
            [-1, 1, 1, 1, 1],
            [1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1],
            [-1, 1, 1, 1, 1],
        ],
    "D": [
            [1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, -1],
        ],
    "E": [
            [1, 1, 1, 1, 1],
            [1, -1, -1, -1, -1],
            [1, 1, 1, 1, -1],
            [1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1],
        ],
    "F": [
            [1, 1, 1, 1, 1],
            [1, -1, -1, -1, -1],
            [1, 1, 1, -1, -1],
            [1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1],
        ],
    "G": [
            [1, 1, 1, 1, 1],
            [1, -1, -1, -1, -1],
            [1, -1, 1, 1, 1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1],
        ],
    "H": [
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
        ],
    "I": [
            [-1, 1, 1, 1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, 1, 1, 1, -1],
        ],
    "J": [
            [-1, 1, 1, 1, -1],
            [-1, -1, -1, 1, -1],
            [-1, -1, -1, 1, -1],
            [-1, 1, -1, 1, -1],
            [-1, 1, 1, 1, -1],
        ],
    "K": [
            [1, -1, -1, 1, -1],
            [1, -1, 1, -1, -1],
            [1, 1, -1, -1, -1],
            [1, -1, 1, -1, -1],
            [1, -1, -1, 1, -1],
        ],
    "L": [
            [1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1],
            [1, 1, 1, 1, -1],
        ],
    "M": [
            [1, -1, -1, -1, 1],
            [1, 1, -1, 1, 1],
            [1, -1, 1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
        ],
    "N": [
            [1, -1, -1, -1, 1],
            [1, 1, -1, -1, 1],
            [1, -1, 1, -1, 1],
            [1, -1, -1, 1, 1],
            [1, -1, -1, -1, 1],
        ],
    "O": [
            [-1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [-1, 1, 1, 1, -1],
        ],
    "P": [
            [1, 1, 1, 1, -1],
            [1, -1, -1, 1, -1],
            [1, 1, 1, 1, -1],
            [1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1],
        ],
    "Q": [
            [-1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, -1, 1, -1, 1],
            [1, -1, -1, 1, -1],
            [-1, 1, 1, -1, 1],
        ],
    "R": [
            [1, 1, 1, 1, -1],
            [1, -1, -1, 1, -1],
            [1, 1, 1, 1, -1],
            [1, -1, 1, -1, -1],
            [1, -1, -1, 1, -1],
        ],
    "S": [
            [-1, 1, 1, 1, 1],
            [1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1],
            [-1, -1, -1, -1, 1],
            [1, 1, 1, 1, -1],
        ],
    "T": [
            [1, 1, 1, 1, 1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
        ],
    "U": [
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [-1, 1, 1, 1, -1],
        ],
    "V": [
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [-1, 1, -1, 1, -1],
            [-1, -1, 1, -1, -1],
        ],
    "W": [
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, 1, -1, 1],
            [1, 1, -1, 1, 1],
            [1, -1, -1, -1, 1],
        ],
    "X": [
            [1, -1, -1, -1, 1],
            [-1, 1, -1, 1, -1],
            [-1, -1, 1, -1, -1],
            [-1, 1, -1, 1, -1],
            [1, -1, -1, -1, 1],
        ],
    "Y": [
            [1, -1, -1, -1, 1],
            [-1, 1, -1, 1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
        ],
    "Z": [
            [1, 1, 1, 1, 1],
            [-1, -1, -1, 1, -1],
            [-1, -1, 1, -1, -1],
            [-1, 1, -1, -1, -1],
            [1, 1, 1, 1, 1],
        ],
}

def plot_letters (letters):
    colors = ['lightblue', 'black']
    cmap = ListedColormap(colors)

    plt.figure(figsize=(12, 8))
    num_letras = len(letters)
    grid_size = 5
    padding = 1

    for i, (letra, matriz) in enumerate(letters.items()):
        plt.subplot(5, 7, i + 1)
        plt.imshow(matriz, cmap=cmap, aspect='equal', vmin=-1, vmax=1)
        plt.title(letra)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def best_groups (letters):
    results = []
    options = itertools.combinations(letters.keys(), 4)

    for o in options:
        group = np.array([np.array(letters[k]).flatten() for k in o])
        matrix = group.dot(group.T)
        #saco consigo mismo:
        np.fill_diagonal(matrix, 0)

        mean = np.abs(matrix).sum() / (matrix.size - len(group))
        max = np.abs(matrix).max()
        results.append({
            "group": o,
            "mean": mean,
            "max": max
        })

    df = pd.DataFrame(results)

    return df


FONT = [
    [0x7C, 0x44, 0x44, 0x7C, 0x44],
    [0x7C, 0x44, 0x7C, 0x44, 0x7C],
    [0x7C, 0x40, 0x40, 0x40, 0x7C],
    [0x78, 0x44, 0x44, 0x44, 0x78],
    [0x7C, 0x40, 0x78, 0x40, 0x7C],
    [0x7C, 0x40, 0x70, 0x40, 0x40],
    [0x7C, 0x40, 0x4C, 0x44, 0x7C],
    [0x44, 0x44, 0x7C, 0x44, 0x44],
    [0x7C, 0x10, 0x10, 0x10, 0x7C],
    [0x0C, 0x04, 0x04, 0x44, 0x7C],
    [0x44, 0x48, 0x70, 0x48, 0x44],
    [0x40, 0x40, 0x40, 0x40, 0x7C],
    [0x44, 0x6C, 0x54, 0x44, 0x44],
    [0x44, 0x64, 0x54, 0x4C, 0x44],
    [0x38, 0x44, 0x44, 0x44, 0x38],
    [0x78, 0x44, 0x78, 0x40, 0x40],
    [0x7C, 0x44, 0x44, 0x7C, 0x10],
    [0x78, 0x44, 0x78, 0x44, 0x44],
    [0x7C, 0x40, 0x7C, 0x04, 0x7C],
    [0x7C, 0x10, 0x10, 0x10, 0x10],
    [0x44, 0x44, 0x44, 0x44, 0x7C],
    [0x44, 0x44, 0x28, 0x28, 0x10],
    [0x44, 0x44, 0x54, 0x54, 0x28],
    [0x44, 0x28, 0x10, 0x28, 0x44],
    [0x44, 0x44, 0x28, 0x10, 0x10],
    [0x7C, 0x08, 0x10, 0x20, 0x7C],
]


def create_pattern(noise_proportion=0.4, seed=None):
    # mask = [0, 0, 4, 8, 16, 32, 64, 0, 0]
    # patterns = []
    # for f in FONT[0:4]:
    #     pattern = []
    #     for row in f:
    #         for i in range(2, 7):
    #             if row & mask[i] > 0:
    #                 pattern.append(1)
    #             else:
    #                 pattern.append(-1)
    #     patterns.append(np.array(pattern))
    patterns = [
        letters["J"],
        letters["L"],
        letters["T"],
        letters["X"],
    ]
    patterns = np.array(patterns)
    patterns = np.reshape(patterns, (4, 25))

    noisy_patterns = np.copy(patterns)

    total_bits = 25 #cambiar si cambiamos tamaño de patrones
    num_bits_to_change = int(total_bits * noise_proportion)

    for p in noisy_patterns:
        if seed is not None:
            np.random.seed(seed)
        indexes_to_change = np.random.choice(total_bits, num_bits_to_change, replace=False)
        p[indexes_to_change] *= -1

    return (patterns, noisy_patterns)


if __name__ == "__main__":
    # np.random.seed(101)
    # Configuración de colores
    #plot_letters(letters)
    df = best_groups(letters)
    top_avg = df.nsmallest(5, 'mean')
    bottom_avg = df.nlargest(5, 'mean')
    print(top_avg)
    print(bottom_avg)
    patterns, noisy_patterns = create_pattern()
    hopfield = Hopfield(len(patterns[0]))
    print(f"Train Patterns: \n{patterns[:4]}")
    for p in patterns[:4]:
        hopfield.set_patterns(p)
    print(f"Original Pattern:\n{patterns[3]}")
    print(f"Noisy Pattern to train:\n{noisy_patterns[3]}")
    hopfield.set_initial_state(noisy_patterns[3])
    final_result, trace = hopfield.train()
    print(f"Final Result:\n{final_result}")
    print(f"Number of steps:{len(trace)}")

    plt.subplot(211)
    plt.imshow(np.reshape(patterns[3], (5, 5)))
    plt.subplot(212)
    plt.imshow(np.reshape(patterns[3], (5, 5)), cmap="Greys", interpolation="nearest")
    plt.savefig("./plots/original-Hopfield.png")
    plt.subplot(211)
    plt.imshow(np.reshape(noisy_patterns[3], (5, 5)))
    plt.subplot(212)
    plt.imshow(
        np.reshape(noisy_patterns[3], (5, 5)), cmap="Greys", interpolation="nearest"
    )
    plt.savefig("./plots/noisy_test-Hopfield.png")
    plt.subplot(211)
    plt.imshow(np.reshape(final_result, (5, 5)))
    plt.subplot(212)
    plt.imshow(np.reshape(final_result, (5, 5)), cmap="Greys", interpolation="nearest")
    plt.savefig("./plots/final_result-Hopfield.png")
    i = 0
    for t in trace:
        plt.subplot(211)
        plt.imshow(np.reshape(t, (5, 5)))
        plt.subplot(212)
        plt.imshow(
            np.reshape(final_result, (5, 5)),  cmap=ListedColormap(['lightblue', 'blue']), interpolation="nearest"
        )
        plt.savefig(f"./plots/trace-{i}-Hopfield.png")
        i += 1
