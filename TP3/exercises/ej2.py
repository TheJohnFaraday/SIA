import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from src.configuration import Configuration
from src.LinearPerceptron import LinearPerceptron, ActivationFunction
from src.errors import MSE
from src.utils import unnormalize


def train_and_evaluate_perceptron(train_input, train_output, test_input, test_output, test_unnom_output, config, activation_function):
    """Entrena y evalúa un perceptrón (lineal o no lineal)"""
    perceptron = LinearPerceptron(
        len(train_input[0]),
        config.learning_rate,
        activation_function,
        config.beta,
        MSE(),
    )
    perceptron.train(train_input, train_output, config.epoch)

    test_predictions, test_errors = perceptron.test(test_input, test_output, test_unnom_output)

    return perceptron, test_errors


def plot_mse_comparison(folds, mse_linear, mse_nonlinear, title, ylabel):
    """Genera una gráfica de barras para comparar el MSE entre lineal y no lineal"""
    plt.figure(figsize=(10, 5))
    plt.bar(folds - 0.15, mse_linear, width=0.3, label='Lineal')
    plt.bar(folds + 0.15, mse_nonlinear, width=0.3, label='No Lineal')
    plt.xlabel('Fold')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def train_with_kfold(inputs, outputs, unnom_outputs, config):
    """Entrena y evalúa el modelo utilizando K-Fold Cross-Validation"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_linear, mse_nonlinear = [], []
    fold = 1

    for train_index, test_index in kf.split(inputs):
        print(f"\n===== Fold {fold} =====")

        train_input, test_input = inputs[train_index], inputs[test_index]
        train_output, test_output = outputs[train_index], outputs[test_index]
        train_unnom_output, test_unnom_output = unnom_outputs[train_index], unnom_outputs[test_index]

        # Entrenamos y evaluamos el perceptrón lineal
        linear_perceptron, test_lineal_errors = train_and_evaluate_perceptron(
            train_input, train_output, test_input, test_output, test_unnom_output, config, ActivationFunction.LINEAR
        )
        mse_linear.append(linear_perceptron.final_error[-1])

        # Entrenamos y evaluamos el perceptrón no lineal
        non_linear_perceptron, test_non_lineal_errors = train_and_evaluate_perceptron(
            train_input, train_output, test_input, test_output, test_unnom_output, config, config.linear_non_linear_activation_function
        )
        mse_nonlinear.append(non_linear_perceptron.final_error[-1])

        fold += 1

    # Gráfica de comparación del MSE para el entrenamiento y la evaluación
    folds = np.arange(1, len(mse_linear) + 1)
    plot_mse_comparison(folds, mse_linear, mse_nonlinear, 'Comparación del MSE de entrenamiento', 'MSE (entrenamiento)')
    plot_mse_comparison(folds, test_lineal_errors, test_non_lineal_errors, 'Comparación del MSE de evaluación', 'MSE (evaluación)')


def train_without_kfold(inputs, outputs, config):
    """Entrena el modelo sin K-Fold usando train_test_split"""
    train_input, test_input, train_output, test_output = train_test_split(
        inputs, outputs, train_size=config.train_proportion, random_state=42
    )

    # Perceptrón Lineal
    linear_perceptron = LinearPerceptron(
        len(train_input[0]),
        config.learning_rate,
        ActivationFunction.LINEAR,
        config.beta,
        MSE(),
    )
    linear_perceptron.train(train_input, train_output, config.epoch)

    # Perceptrón No Lineal
    non_linear_perceptron = LinearPerceptron(
        len(train_input[0]),
        config.learning_rate,
        config.linear_non_linear_activation_function,
        config.beta,
        MSE(),
    )
    non_linear_perceptron.train(train_input, train_output, config.epoch)

    # Normalización de los errores de entrenamiento para graficar
    min_train_error = min(min(linear_perceptron.train_errors), min(non_linear_perceptron.train_errors))
    max_train_error = max(max(linear_perceptron.train_errors), max(non_linear_perceptron.train_errors))

    linear_perceptron.train_errors = [(error - min_train_error) / (max_train_error - min_train_error) for error in linear_perceptron.train_errors]
    non_linear_perceptron.train_errors = [(error - min_train_error) / (max_train_error - min_train_error) for error in non_linear_perceptron.train_errors]

    # Determinar el mínimo de los final_epochs entre ambos perceptrones
    min_epochs = min(len(linear_perceptron.final_epochs), len(non_linear_perceptron.final_epochs))

    # Gráfica de evolución del error de entrenamiento
    plt.figure(figsize=(10, 5))
    plt.plot(linear_perceptron.final_epochs[:min_epochs], linear_perceptron.train_errors[:min_epochs], label='Lineal (train_proportion)')
    plt.plot(non_linear_perceptron.final_epochs[:min_epochs], non_linear_perceptron.train_errors[:min_epochs], label='No Lineal (train_proportion)')
    plt.xlabel('Época')
    plt.ylabel('Error (MSE)')
    plt.title('Evolución del MSE durante el entrenamiento (sin K-Fold)')
    plt.legend()
    plt.xlim(0, min_epochs)
    plt.ylim(0, 1)  # Normalizado
    plt.show()


def ej2(config: Configuration):
    inputs = config.linear_non_linear_input
    outputs = config.linear_non_linear_output_norm
    unnom_outputs = np.array(config.linear_non_linear_output)

    # Entrenamiento y evaluación usando K-Fold
    train_with_kfold(inputs, outputs, unnom_outputs, config)

    # Entrenamiento sin K-Fold usando train_proportion
    train_without_kfold(inputs, outputs, config)


