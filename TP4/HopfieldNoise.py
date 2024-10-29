import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.Hopfield import Hopfield
from HopfieldExercise import letters

def create_patterns(selected_letters):
    patterns = [letters[letter] for letter in selected_letters]
    patterns = np.array(patterns)
    patterns = np.reshape(patterns, (len(patterns), 25))

    return patterns

def add_noise(pattern, noise_proportion, seed, num):
    total_bits = 25
    num_bits_to_change = int(total_bits * noise_proportion)
    noisy_patterns = []

    if seed is not None:
        np.random.seed(seed)

    for _ in range(num):
        noisy_pattern = pattern.copy()
        indexes_to_change = np.random.choice(total_bits, num_bits_to_change, replace=False)
        noisy_pattern[indexes_to_change] *= -1
        noisy_patterns.append(noisy_pattern)

    return noisy_patterns

def hopfield_noise(patterns, labels, noise_proportions, num_runs=100):
    results = {label: [] for label in labels}
    spurious_count = {label: 0 for label in labels}

    hopfield = Hopfield()

    for pattern in patterns:
        hopfield.set_patterns(pattern)

    for noise in noise_proportions:
        for idx, pattern in enumerate(patterns):
            noisy_patterns = add_noise(pattern, noise, seed=None, num=num_runs)
            correct_predictions = 0

            for noisy_pattern in noisy_patterns:
                hopfield.set_initial_state(noisy_pattern)
                prediction, trace = hopfield.train()
                if np.array_equal(prediction, pattern):
                    correct_predictions += 1

                if not np.array_equal(prediction, pattern):
                    spurious_count[labels[idx]] += 1

            results[labels[idx]].append(correct_predictions)

    return results, spurious_count

def plot_results(results, noise_proportions):
    data = {label: results[label] for label in results}
    df = pd.DataFrame(data, index=noise_proportions)

    df.plot(kind='bar', figsize=(10, 6))
    plt.title('Aciertos por Letra vs. Porcentaje de Ruido')
    plt.xlabel('Porcentaje de Ruido')
    plt.ylabel('Cantidad de Aciertos')
    plt.xticks(rotation=0)
    plt.legend(title='Letras')
    plt.show()

if __name__ == "__main__":
    noise_proportions = [0.1, 0.2, 0.3, 0.4, 0.5]
    selected_letters = ["O", "K", "T", "V"] 
    patterns = create_patterns(selected_letters)
    results, spurious_counts = hopfield_noise(patterns, selected_letters, noise_proportions, num_runs=1000)
    plot_results(results, noise_proportions)