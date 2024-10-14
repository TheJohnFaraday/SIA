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


# === Perceptron Creation ===
def create_perceptron(activation_function, train_input, learning_rate, beta):
    """Crea un perceptrón según la función de activación especificada."""
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


# === Training Functions ===
def train_with_kfold(inputs, outputs, config):
    """Entrena el modelo utilizando K-Fold Cross-Validation y guarda la información necesaria para testing."""
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
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
    """Entrena el modelo sin K-Fold usando train_test_split"""
    print("\nTrain without kfold\n")
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


# === Testing Functions ===
def test_with_kfold(perceptrons):
    """Evalúa los perceptrones entrenados con K-Fold."""
    avg_errors = []
    fold_data = []

    for i, fold_data_dict in enumerate(perceptrons):
        perceptron_obj = fold_data_dict["perceptron"]
        test_input = fold_data_dict["test_input"]
        test_output = fold_data_dict["test_output"]

        _, fold_errors = perceptron_obj.test(test_input, test_output)
        avg_error = np.mean(fold_errors)
        avg_errors.append(avg_error)

        fold_data.append({
            "fold": i + 1,
            "avg_error": avg_error,
            "errors": fold_errors,
            "input": test_input,
            "output": test_output,
            "perceptron": perceptron_obj
        })
        print(f"Fold {i + 1} - Error promedio: {avg_error}")

    return fold_data


# === Plot Functions ===
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


def plot_errors_by_fold(perceptrons, fold, config):
    """Genera un gráfico que muestre el error de entrenamiento en función de las épocas por fold"""
    plt.figure(figsize=(10, 5))
    plt.plot(perceptrons.final_epochs, perceptrons.train_errors, label=f"Fold {fold}")
    plt.xlabel("Epochs")
    plt.ylabel("Training Error (MSE)")
    plt.xlim(0, 500)
    plt.ylim(0, 5000)
    plt.title("Training Errors per Epoch for Each Fold")
    plt.legend()
    plt.savefig(
        f"plots/ej2_training_errors_fold-{fold}"
        f"_lr-{config.learning_rate}"
        f"_activation-{config.linear_non_linear_activation_function}"
        f".png"
    )
    plt.show()


def plot_best_and_worst_folds(best_perceptron, worst_perceptron, best_fold, worst_fold, config):
    """Genera un gráfico comparando el mejor y peor fold basado en el error final"""
    plt.figure(figsize=(10, 5))
    plt.plot(best_perceptron.final_epochs, best_perceptron.train_errors, label=f"Best Fold ({best_fold})", color="green")
    plt.plot(worst_perceptron.final_epochs, worst_perceptron.train_errors, label=f"Worst Fold ({worst_fold})", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Training Error (MSE)")
    plt.xlim(0, 500)
    plt.ylim(0, 5000)
    plt.title(f"Comparison of Best and Worst Folds (Best: {best_fold}, Worst: {worst_fold})")
    plt.legend()
    plt.savefig(
        f"plots/ej2_best_worst_folds_comparison"
        f"_lr-{config.learning_rate}"
        f"_activation-{config.linear_non_linear_activation_function}"
        f".png"
    )
    plt.show()


def plot_fold_test_errors_comparison(perceptron_fold1, perceptron_fold2, inputs_fold1, inputs_fold2,
                                     outputs_fold1, outputs_fold2, fold1, fold2):
    """Compara los errores de prueba entre dos folds especificados"""
    _, fold1_errors = perceptron_fold1.test(inputs_fold1, outputs_fold1)
    _, fold2_errors = perceptron_fold2.test(inputs_fold2, outputs_fold2)

    plt.figure(figsize=(10, 5))
    index = np.arange(len(fold1_errors))
    bar_width = 0.35
    plt.bar(index, fold1_errors, width=bar_width, label=f"Fold {fold1}")
    plt.bar(index + bar_width, fold2_errors, width=bar_width, label=f"Fold {fold2}")
    plt.xlabel("Sample")
    plt.ylabel("Test Error")
    plt.title(f"Test Error Comparison Between Fold {fold1} and Fold {fold2}")
    plt.legend()
    plt.savefig(
        f"plots/ej2_test_error_comparison_folds_{fold1}_vs_{fold2}.png"
    )
    plt.show()


def ej2(config):
    inputs = config.linear_non_linear_input
    outputs = np.array(config.linear_non_linear_output)

    linear_perceptrons_data, non_linear_perceptrons_data = train_with_kfold(inputs, outputs, config)

    best_perceptron_data = min(non_linear_perceptrons_data, key=lambda p: p["perceptron"].final_error[-1])
    worst_perceptron_data = max(non_linear_perceptrons_data, key=lambda p: p["perceptron"].final_error[-1])

    best_perceptron = best_perceptron_data["perceptron"]
    worst_perceptron = worst_perceptron_data["perceptron"]

    best_fold = non_linear_perceptrons_data.index(best_perceptron_data) + 1
    worst_fold = non_linear_perceptrons_data.index(worst_perceptron_data) + 1

    plot_best_and_worst_folds(best_perceptron, worst_perceptron, best_fold, worst_fold, config)

    plot_fold_test_errors_comparison(
        best_perceptron, worst_perceptron,
        best_perceptron_data["test_input"], worst_perceptron_data["test_input"],
        best_perceptron_data["test_output"], worst_perceptron_data["test_output"],
        best_fold, worst_fold
    )

    train_without_kfold(inputs, outputs, config)
    fold_data = test_with_kfold(non_linear_perceptrons_data)

    best_fold = min(fold_data, key=lambda x: x["avg_error"])
    worst_fold = max(fold_data, key=lambda x: x["avg_error"])

    plot_fold_test_errors_comparison(
        perceptron_fold1=best_fold["perceptron"],
        perceptron_fold2=worst_fold["perceptron"],
        inputs_fold1=best_fold["input"],
        inputs_fold2=worst_fold["input"],
        outputs_fold1=best_fold["output"],
        outputs_fold2=worst_fold["output"],
        fold1=best_fold["fold"],
        fold2=worst_fold["fold"]
    )

