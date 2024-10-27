from sklearn.preprocessing import StandardScaler


def standardize_data(data):
    return StandardScaler().fit_transform(data)