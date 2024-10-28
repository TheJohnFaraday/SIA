import math

import numpy as np
import copy
import random
import matplotlib.pyplot as plt

from src.utils import get_euclidean_distance, get_exponential_distance
from enum import Enum


class DistanceType(Enum):
    EUCLIDEAN = "euclidean"
    EXPONENTIAL = "exponential"


class Neuron:
    def __init__(self, dataset, use_dataset=False):
        if use_dataset:
            self.init_weights(dataset)
        else:
            self.init_weights_random(dataset.shape[1])
        self.hit_count = 0

    def init_weights(self, dataset):
        idx = np.random.randint(0, dataset.shape[0])
        entry = dataset[idx, :]
        self.weights = copy.copy(entry)

    def init_weights_random(self, length):
        self.weights = np.random.uniform(low=-1, high=1, size=(length,))


def init_output_neuron_matrix(k, entries, dataset):
    output_neuron_mtx = []
    for i in range(0, k):
        output_neuron_mtx.append([])
        for j in range(0, k):
            neuron = Neuron(entries, dataset)
            output_neuron_mtx[i].append(neuron)
    return output_neuron_mtx


def calculate_distance(distance_type: DistanceType, arr1, arr2):
    if distance_type == DistanceType.EUCLIDEAN:
        return get_euclidean_distance(arr1, arr2)
    elif distance_type == DistanceType.EXPONENTIAL:
        return get_exponential_distance(arr1, arr2)
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")


def find_bmu(output_neuron_mtx, entry, distance_type: DistanceType):
    min_i = 0
    min_j = 0
    min_dist = math.inf
    for i in range(0, len(output_neuron_mtx)):
        for j in range(0, len(output_neuron_mtx[0])):
            d = calculate_distance(distance_type, output_neuron_mtx[i][j].weights, entry)
            if d < min_dist:
                min_i = i
                min_j = j
                min_dist = d
    return min_i, min_j


def get_neighbours(output_neuron_mtx, idx, radius):
    output_mtx_row_limit = len(output_neuron_mtx)
    row_bottom_limit = idx[0] - radius if idx[0] - radius >= 0 else 0
    row_bottom_limit = int(row_bottom_limit)
    row_upper_limit = idx[0] + radius if idx[0] + radius < output_mtx_row_limit else output_mtx_row_limit
    row_upper_limit = int(row_upper_limit)

    output_mtx_col_limit = len(output_neuron_mtx[0])
    col_bottom_limit = idx[1] - radius if idx[1] - radius >= 0 else 0
    col_bottom_limit = int(col_bottom_limit)
    col_upper_limit = idx[1] + radius if idx[1] + radius < output_mtx_col_limit else output_mtx_col_limit
    col_upper_limit = int(col_upper_limit)

    ne = []
    for i in range(row_bottom_limit, row_upper_limit):
        for j in range(col_bottom_limit, col_upper_limit):
            dist = math.sqrt((idx[0] - i)**2 + (idx[1] - j)**2)
            # print(dist)
            if dist <= radius:
                ne.append((i, j))
    return ne


def update_neuron(output_neuron_mtx, pos, entry, eta_f, it):
    (i, j) = pos
    neuron = output_neuron_mtx[i][j]
    eta = eta_f(it)
    neuron.weights += eta * (entry - neuron.weights)


def update_neighbours(output_neuron_mtx, best_match_idx, entry, radius, eta_f, it):
    ne = get_neighbours(output_neuron_mtx, best_match_idx, radius)
    for (i, j) in ne:
        update_neuron(output_neuron_mtx, (i, j), entry, eta_f, it)


def update_radius(initial_radius, min_radius, current_epoch, total_epochs, stable_radius):
    # Si el radio es menor que el radio estable, mantenlo constante
    if initial_radius - (initial_radius - min_radius) * (current_epoch / total_epochs) <= stable_radius:
        return stable_radius
    else:
        # Disminuye linealmente hasta el radio estable
        return initial_radius - (initial_radius - min_radius) * (current_epoch / total_epochs)


def process_input(output_neuron_mtx, entry, radius, eta_f, it, distance_type: DistanceType):
    (best_i, best_j) = find_bmu(output_neuron_mtx, entry, distance_type)
    output_neuron_mtx[best_i][best_j].hit_count += 1
    update_neighbours(output_neuron_mtx, (best_i, best_j), entry, radius, eta_f, it)


def display_results(output_neuron_mtx):
    k = len(output_neuron_mtx)
    a = np.zeros((k, k))
    for i in range(0, k):
        for j in range(0, k):
            a[i, j] = output_neuron_mtx[i][j].hit_count

    fig, ax = plt.subplots()
    k = len(output_neuron_mtx)
    ax.set_title(f'Entries per node with k={k}')
    im = plt.imshow(a, cmap='hot', interpolation='nearest')
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(range(k))
    ax.set_yticklabels(range(k))

    # Loop over data dimensions and create text annotations.
    max_val = np.amax(np.array(a))

    for i in range(k):
        for j in range(k):
            if a[i][j] > max_val/2:
                color = "k"
            else:
                color = "w"
            text = ax.text(j, i, f'{int(a[i][j])}', ha="center", va="center", color=color)

    plt.colorbar(im)
    plt.show()


def display_u_matrix(output_neuron_mtx, radius, distance_type: DistanceType):  # grey matrix
    k = len(output_neuron_mtx)
    a = np.zeros((k, k))
    for i in range(0, k):
        for j in range(0, k):
            ne = get_neighbours(output_neuron_mtx, (i, j), radius)
            # print(ne)
            d = 0.0
            cant = len(ne)
            for (ne_i, ne_j) in ne:
                d += calculate_distance(distance_type, output_neuron_mtx[i][j].weights, output_neuron_mtx[ne_i][ne_j].weights)
            a[i, j] = d/cant

    fig, ax = plt.subplots()
    k = len(output_neuron_mtx)
    ax.set_title(f'U Matrix with k={k}')
    im = plt.imshow(a, cmap='Greys', interpolation='nearest')
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(range(k))
    ax.set_yticklabels(range(k))

    # Loop over data dimensions and create text annotations.
    max_val = np.amax(np.array(a))

    for i in range(k):
        for j in range(k):
            if a[i][j] > max_val/2:
                color = "w"
            else:
                color = "k"
            text = ax.text(j, i, f'{a[i][j]:.3f}', ha="center", va="center", color=color)

    plt.colorbar(im)
    plt.show()


def kohonen(epochs_mlt, entries, k, initial_radius, dataset, eta_f, distance_type: DistanceType):
    epochs = epochs_mlt * entries.shape[0]
    radius = initial_radius

    # Inicializamos los weights (k * inputs)
    output_neuron_mtx = init_output_neuron_matrix(k, entries, dataset)

    # Iteramos por todos los inputs
    for epoch in range(epochs):
        aux_entries = copy.copy(entries)
        random.shuffle(aux_entries)
        for i in range(0, aux_entries.shape[0]):
            entry = aux_entries[i, :]
            process_input(output_neuron_mtx, entry, radius, eta_f, epoch+1, distance_type)
            # if (radius - 1) > 1:
            #    radius -= 1

        radius = update_radius(initial_radius, 1, epoch, epochs, 1)
        print("epoch: {} radius: {}".format(epoch, radius))

    display_results(output_neuron_mtx)
    display_u_matrix(output_neuron_mtx, radius, distance_type)

    return output_neuron_mtx


def display_final_assignments(data, std_data, output_neuron_mtx, distance_type: DistanceType):
    k = len(output_neuron_mtx)
    names = [[[] for j in range(0, k)] for i in range(0, k)]
    a = np.zeros((k, k))
    countries_list = data['Country'].values.tolist()

    for i in range(0, std_data.shape[0]):
        entry = std_data[i]
        (x, y) = find_bmu(output_neuron_mtx, entry, distance_type)
        a[x, y] += 1
        names[x][y].append(countries_list[i])

    fig, ax = plt.subplots(figsize=(10, 10))  # Ajustar el tamaño de la figura aquí
    ax.set_title(f'Final entries per node with k={k}')
    im = plt.imshow(a, cmap='hot', interpolation='nearest')
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(range(k))
    ax.set_yticklabels(range(k))

    # Loop over data dimensions and create text annotations.
    max_val = np.amax(np.array(a))

    for l in range(0, len(names)):
        for t in range(0, len(names[l])):
            countries = names[l][t]

            s = ''
            for country in countries:
                s += country + '\n'

            names[l][t] = s

    for i in range(k):
        for j in range(k):
            if a[i][j] > max_val/2:
                color = "k"
            else:
                color = "w"
            text = ax.text(j, i, f'{names[i][j]}', ha="center", va="center", color=color, fontsize=10)  # Ajustar el tamaño de la fuente aquí

    plt.colorbar(im)
    plt.tight_layout()  # Asegura que los elementos no se solapen
    plt.show()


