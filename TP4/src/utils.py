from sklearn.preprocessing import StandardScaler
import math


def get_euclidean_distance(arr1, arr2):
    dist = 0
    for i in range(0, arr1.shape[0]):
        dist += (arr1[i] - arr2[i])**2
    return math.sqrt(dist)


def get_exponential_distance(arr1, arr2):
    dist = 0
    for i in range(0, arr1.shape[0]):
        dist += (arr1[i] - arr2[i])**2
    return math.exp(-dist)


def standardize_data(data):
    return StandardScaler().fit_transform(data)