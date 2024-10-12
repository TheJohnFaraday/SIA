import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from src.configuration import Configuration
from src.LinearPerceptron import (
    LinearPerceptron,
    TanhPerceptron,
    LogisticPerceptron,
    ActivationFunction,
)
from src.errors import MSE


def create_perceptron(activation_function, train_input, learning_rate, beta):
    match activation_function:
        case ActivationFunction.LINEAR:
            return LinearPerceptron(
                len(train_input[0]),
                learning_rate,
                beta,
                MSE(),
            )
        case ActivationFunction.LOGISTIC:
            return LogisticPerceptron(
                len(train_input[0]),
                learning_rate,
                beta,
                MSE(),
            )
        case ActivationFunction.TANH:
            return TanhPerceptron(
                len(train_input[0]),
                learning_rate,
                beta,
                MSE(),
            )
        case _:
            raise RuntimeError("Non valid Activation Function entered")


def train_and_evaluate_perceptron(
    train_input,
    train_output,
    test_input,
    test_output,
    config,
    activation_function,
):
    """Entrena y evalúa un perceptrón (lineal o no lineal)"""
    perceptron = create_perceptron(
        activation_function, train_input, config.learning_rate, config.beta
    )

    perceptron.train(train_input, train_output, config.epoch)

    test_predictions, test_errors = perceptron.test(test_input, test_output)

    return perceptron, test_errors


def plot_mse_comparison(folds, mse_linear, mse_nonlinear, title, ylabel, config):
    """Genera una gráfica de barras para comparar el MSE entre lineal y no lineal"""
    plt.figure(figsize=(10, 5))
    plt.bar(folds - 0.15, mse_linear, width=0.3, label="Lineal")
    plt.bar(folds + 0.15, mse_nonlinear, width=0.3, label="No Lineal")
    plt.xlabel("Fold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(
        f"plots/ej2_mse_comparison"
        f"_lr-{config.learning_rate}"
        f"_activation-{config.linear_non_linear_activation_function}"
        f".png"
    )


def plot_epochs_comparison(
    folds, epochs_linear, epochs_nonlinear, title, ylabel, config
):
    """Genera una gráfica de barras para comparar los epochs alcanzados en cada fold por tipo de perceptrón"""
    plt.figure(figsize=(10, 5))
    bar_width = 0.35

    # Gráfico para el perceptrón lineal
    plt.bar(
        folds - bar_width / 2,
        epochs_linear,
        width=bar_width,
        color="skyblue",
        label="Linear Perceptron",
    )

    # Gráfico para el perceptrón no lineal
    plt.bar(
        folds + bar_width / 2,
        epochs_nonlinear,
        width=bar_width,
        color="lightgreen",
        label="Non-Linear Perceptron",
    )

    plt.xlabel("Fold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(folds)
    plt.legend()
    plt.savefig(
        f"plots/ej2_epoch_comparison"
        f"_lr-{config.learning_rate}"
        f"_activation-{config.linear_non_linear_activation_function}"
        f".png"
    )


def train_with_kfold(inputs, outputs, config):
    """Entrena y evalúa el modelo utilizando K-Fold Cross-Validation"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_linear, mse_nonlinear = [], []
    epochs_linear, epochs_nonlinear = [], []
    fold = 1

    print("Train with kfold")
    for train_index, test_index in kf.split(inputs):
        print(f"\n===== Fold {fold} =====")

        train_input, test_input = inputs[train_index], inputs[test_index]
        train_output, test_output = outputs[train_index], outputs[test_index]

        # Entrenamos y evaluamos el perceptrón lineal
        print("Linear Perceptron")
        linear_perceptron, test_lineal_errors = train_and_evaluate_perceptron(
            train_input,
            train_output,
            test_input,
            test_output,
            config,
            ActivationFunction.LINEAR,
        )
        mse_linear.append(linear_perceptron.final_error[-1])
        epochs_linear.append(
            len(linear_perceptron.final_epochs)
        )  # Guardamos el número de epochs lineales

        # Entrenamos y evaluamos el perceptrón no lineal
        print("No Linear Perceptron")
        non_linear_perceptron, test_non_lineal_errors = train_and_evaluate_perceptron(
            train_input,
            train_output,
            test_input,
            test_output,
            config,
            config.linear_non_linear_activation_function,
        )
        mse_nonlinear.append(non_linear_perceptron.final_error[-1])
        epochs_nonlinear.append(
            len(non_linear_perceptron.final_epochs)
        )  # Guardamos el número de epochs no lineales

        fold += 1

    # Gráfica de comparación del MSE para el entrenamiento y la evaluación
    folds = np.arange(1, len(mse_linear) + 1)
    plot_mse_comparison(
        folds,
        mse_linear,
        mse_nonlinear,
        "Comparison of training MSE",
        "Training MSE",
        config,
    )
    plot_mse_comparison(
        folds,
        test_lineal_errors,
        test_non_lineal_errors,
        "Comparison of testing MSE",
        "Testing MSE",
        config,
    )
    plot_epochs_comparison(
        folds,
        epochs_linear,
        epochs_nonlinear,
        "Epochs Reached per Fold",
        "Epochs",
        config,
    )


def train_without_kfold(inputs, outputs, config):
    print("\nTrain without kfold\n")
    """Entrena el modelo sin K-Fold usando train_test_split"""
    train_input, test_input, train_output, test_output = train_test_split(
        inputs, outputs, train_size=config.train_proportion, random_state=42
    )

    # Perceptrón Lineal
    linear_perceptron = create_perceptron(
        ActivationFunction.LINEAR, train_input, config.learning_rate, config.beta
    )

    print("Linear Perceptron")
    linear_perceptron.train(train_input, train_output, config.epoch)

    # Perceptrón No Lineal
    non_linear_perceptron = create_perceptron(
        config.linear_non_linear_activation_function,
        train_input,
        config.learning_rate,
        config.beta,
    )

    print("No Linear Perceptron")
    non_linear_perceptron.train(train_input, train_output, config.epoch)

    # Normalización de los errores de entrenamiento para graficar
    min_train_error = min(
        min(linear_perceptron.train_errors), min(non_linear_perceptron.train_errors)
    )
    max_train_error = max(
        max(linear_perceptron.train_errors), max(non_linear_perceptron.train_errors)
    )

    linear_perceptron.train_errors = [
        (error - min_train_error) / (max_train_error - min_train_error)
        for error in linear_perceptron.train_errors
    ]
    non_linear_perceptron.train_errors = [
        (error - min_train_error) / (max_train_error - min_train_error)
        for error in non_linear_perceptron.train_errors
    ]

    # Determinar el mínimo de los final_epochs entre ambos perceptrones
    min_epochs = min(
        len(linear_perceptron.final_epochs), len(non_linear_perceptron.final_epochs)
    )

    # Gráfica de evolución del error de entrenamiento
    plt.figure(figsize=(10, 5))
    plt.plot(
        linear_perceptron.final_epochs[:min_epochs],
        linear_perceptron.train_errors[:min_epochs],
        label="Lineal (train_proportion)",
    )
    plt.plot(
        non_linear_perceptron.final_epochs[:min_epochs],
        non_linear_perceptron.train_errors[:min_epochs],
        label="No Lineal (train_proportion)",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Error (MSE)")
    plt.title("Evolution of MSE during training (without K-Fold)")
    plt.legend()
    plt.xlim(0, min_epochs if min_epochs < 200 else 200)
    plt.ylim(0, 1)  # Normalizado
    plt.savefig(
        f"plots/ej2_mse_comparison_without_kfold"
        f"_lr-{config.learning_rate}"
        f"_activation-{config.linear_non_linear_activation_function}"
        f".png"
    )


def ej2(config: Configuration):
    inputs = config.linear_non_linear_input
    outputs = np.array(config.linear_non_linear_output)

    # Entrenamiento y evaluación usando K-Fold
    train_with_kfold(inputs, outputs, config)

    # Entrenamiento sin K-Fold usando train_proportion
    train_without_kfold(inputs, outputs, config)
