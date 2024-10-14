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


def train_with_kfold(inputs, outputs, config):
    """Entrena el modelo utilizando K-Fold Cross-Validation y guarda la información necesaria para testing."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1

    linear_perceptrons_data = []
    non_linear_perceptrons_data = []

    print("Train with K-Fold Cross-Validation")
    for train_index, test_index in kf.split(inputs):
        print(f"\n===== Fold {fold} =====")

        train_input, test_input = inputs[train_index], inputs[test_index]
        train_output, test_output = outputs[train_index], outputs[test_index]

        # Entrenamos el perceptrón lineal
        print("Linear Perceptron")
        linear_perceptron = create_perceptron(
            ActivationFunction.LINEAR, train_input, config.learning_rate, config.beta
        )
        linear_perceptron.train(train_input, train_output, config.epoch)

        # Guardamos el perceptrón y los datos necesarios para testeo
        linear_perceptrons_data.append({
            "perceptron": linear_perceptron,
            "test_input": test_input,
            "test_output": test_output
        })

        # Entrenamos el perceptrón no lineal
        print("Non-Linear Perceptron")
        non_linear_perceptron = create_perceptron(
            config.linear_non_linear_activation_function,
            train_input,
            config.learning_rate,
            config.beta,
        )
        non_linear_perceptron.train(train_input, train_output, config.epoch)

        # Guardamos el perceptrón y los datos necesarios para testeo
        non_linear_perceptrons_data.append({
            "perceptron": non_linear_perceptron,
            "test_input": test_input,
            "test_output": test_output
        })

        fold += 1

    return linear_perceptrons_data, non_linear_perceptrons_data


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

    # Gráfica de evolución del error de entrenamiento
    plt.figure(figsize=(10, 5))
    plt.plot(
        linear_perceptron.final_epochs,
        linear_perceptron.train_errors,
        label="Lineal (train_proportion)",
    )
    plt.plot(
        non_linear_perceptron.final_epochs,
        non_linear_perceptron.train_errors,
        label="No Lineal (train_proportion)",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Error (MSE)")
    plt.title("Evolution of MSE during training (without K-Fold)")
    plt.legend()
    plt.xlim(0, 5000)
    plt.ylim(0, 5000)  # Normalizado
    plt.savefig(
        f"plots/ej2_mse_comparison_without_kfold"
        f"_lr-{config.learning_rate}"
        f"_activation-{config.linear_non_linear_activation_function}"
        f".png"
    )


def plot_errors_by_fold(perceptrons, fold, config):
    """Genera un gráfico que muestre el error de entrenamiento en función de las épocas por fold"""
    plt.figure(figsize=(10, 5))

    plt.plot(
        perceptrons.final_epochs,
        perceptrons.train_errors,
        label=f"Fold {fold}",
    )

    plt.xlabel("Epochs")
    plt.ylabel("Training Error (MSE)")
    plt.xlim(0, 500)
    plt.ylim(0, 5000)
    plt.title("Training Errors per Epoch for Each Fold")
    plt.legend()

    # Guardar el gráfico en un archivo
    plt.savefig(
        f"plots/ej2_training_errors_fold-{fold}"
        f"_lr-{config.learning_rate}"
        f"_activation-{config.linear_non_linear_activation_function}"
        f".png"
    )
    plt.show()


def plot_best_and_worst_folds(best_perceptron, worst_perceptron, config):
    """Genera un gráfico comparando el mejor y peor fold basado en el error final"""
    plt.figure(figsize=(10, 5))

    # Graficamos el mejor fold
    plt.plot(
        best_perceptron.final_epochs,
        best_perceptron.train_errors,
        label="Best Fold",
        color="green",
    )

    # Graficamos el peor fold
    plt.plot(
        worst_perceptron.final_epochs,
        worst_perceptron.train_errors,
        label="Worst Fold",
        color="red",
    )

    plt.xlabel("Epochs")
    plt.ylabel("Training Error (MSE)")
    plt.xlim(0, 500)
    plt.ylim(0, 5000)
    plt.title("Comparison of Best and Worst Folds")
    plt.legend()

    # Guardar el gráfico
    plt.savefig(
        f"plots/ej2_best_worst_folds_comparison"
        f"_lr-{config.learning_rate}"
        f"_activation-{config.linear_non_linear_activation_function}"
        f".png"
    )
    plt.show()


def plot_best_and_worst_folds_test_errors(perceptron_best, perceptron_worst, inputs, outputs):
    """Compara los errores de prueba entre el mejor y peor perceptrón"""

    # Obtener los errores de test para el mejor perceptrón
    _, best_errors = perceptron_best.test(inputs, outputs)

    # Obtener los errores de test para el peor perceptrón
    _, worst_errors = perceptron_worst.test(inputs, outputs)

    # Graficar los errores
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(best_errors)), best_errors, label="Best Fold", color='green')
    plt.plot(range(len(worst_errors)), worst_errors, label="Worst Fold", color='red')

    # Añadir detalles al gráfico
    plt.xlabel("Test Example")
    plt.ylabel("Test Error (MSE)")
    plt.title("Test Error per Example: Best vs Worst Fold")
    plt.legend()
    plt.show()


def ej2(config: Configuration):
    inputs = config.linear_non_linear_input
    outputs = np.array(config.linear_non_linear_output)

    # Entrenamiento y evaluación usando K-Fold
    linear_perceptrons_data, non_linear_perceptrons_data = train_with_kfold(inputs, outputs, config)

    best_perceptron = min(non_linear_perceptrons_data, key=lambda p: p["perceptron"].final_error[-1])["perceptron"]
    worst_perceptron = max(non_linear_perceptrons_data, key=lambda p: p["perceptron"].final_error[-1])["perceptron"]

    plot_best_and_worst_folds(best_perceptron, worst_perceptron, config)

    # Entrenamiento sin K-Fold usando train_proportion
    train_without_kfold(inputs, outputs, config)
