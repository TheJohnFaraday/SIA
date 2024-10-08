import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from src.configuration import Configuration
from src.LinearPerceptron import LinearPerceptron, ActivationFunction
from src.errors import MSE
from src.utils import unnormalize


def ej2(config: Configuration):
    # KFold configuration
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Usaremos 5-fold cross-validation

    inputs = config.linear_non_linear_input
    outputs = config.linear_non_linear_output_norm
    unnom_outputs = np.array(config.linear_non_linear_output)

    mse_linear = []
    mse_nonlinear = []

    # Entrenamiento y evaluación con K-Fold Cross-Validation
    fold = 1
    for train_index, test_index in kf.split(inputs):
        print(f"\n===== Fold {fold} =====")

        # Dividimos los datos en entrenamiento y test para este fold
        train_input, test_input = inputs[train_index], inputs[test_index]
        train_output, test_output = outputs[train_index], outputs[test_index]
        train_unnom_output, test_unnom_output = unnom_outputs[train_index], unnom_outputs[test_index]

        # LINEAR PERCEPTRON
        linear_perceptron = LinearPerceptron(
            len(train_input[0]),
            config.learning_rate,
            ActivationFunction.LINEAR,
            config.beta,
            MSE(),
        )
        linear_perceptron.train(train_input, train_output, config.epoch)
        mse_linear.append(linear_perceptron.final_error[-1])

        print(f"Evaluando perceptrón lineal en Fold {fold}:")
        test_lineal_predictions, test_lineal_errors = linear_perceptron.test(test_input, test_output, test_unnom_output)

        # NON-LINEAR PERCEPTRON
        non_linear_perceptron = LinearPerceptron(
            len(train_input[0]),
            config.learning_rate,
            config.linear_non_linear_activation_function,
            config.beta,
            MSE(),
        )
        non_linear_perceptron.train(train_input, train_output, config.epoch)
        mse_nonlinear.append(non_linear_perceptron.final_error[-1])

        print(f"Evaluando perceptrón no lineal en Fold {fold}:")
        test_non_lineal_predictions, test_non_lineal_error = non_linear_perceptron.test(test_input, test_output, test_unnom_output)

        fold += 1

    plt.figure(figsize=(10, 5))
    folds = np.arange(1, len(mse_linear) + 1)
    plt.bar(folds - 0.15, mse_linear, width=0.3, label='Lineal')
    plt.bar(folds + 0.15, mse_nonlinear, width=0.3, label='No Lineal')
    plt.xlabel('Fold')
    plt.ylabel('MSE')
    plt.title('Comparación del MSE entre el Perceptrón Lineal y No Lineal por Fold')
    plt.legend()
    plt.show()


    # Entrenamiento sin K-Fold (usando train_proportion)
    train_input, test_input, train_output, test_output = train_test_split(
        inputs, outputs, train_size=config.train_proportion, random_state=42
    )

    # LINEAR PERCEPTRON con train_proportion
    linear_perceptron_proportion = LinearPerceptron(
        len(train_input[0]),
        config.learning_rate,
        ActivationFunction.LINEAR,
        config.beta,
        MSE(),
    )
    linear_perceptron_proportion.train(train_input, train_output, config.epoch)
    print(linear_perceptron_proportion.final_error)

    # NON-LINEAR PERCEPTRON con train_proportion
    non_linear_perceptron_proportion = LinearPerceptron(
        len(train_input[0]),
        config.learning_rate,
        config.linear_non_linear_activation_function,
        config.beta,
        MSE(),
    )
    non_linear_perceptron_proportion.train(train_input, train_output, config.epoch)
    min_train_error = min(min(linear_perceptron_proportion.train_errors), min(non_linear_perceptron_proportion.train_errors))
    max_train_error = max(max(linear_perceptron_proportion.train_errors), max(non_linear_perceptron_proportion.train_errors))

    linear_perceptron_proportion.train_errors = [(error - min_train_error) / (max_train_error - min_train_error) for error in linear_perceptron_proportion.train_errors]
    non_linear_perceptron_proportion.train_errors = [(error - min_train_error) / (max_train_error - min_train_error) for error in non_linear_perceptron_proportion.train_errors]

    max_train_error = max(max(linear_perceptron_proportion.train_errors), max(non_linear_perceptron_proportion.train_errors))

    plt.figure(figsize=(10, 5))
    plt.plot(linear_perceptron_proportion.final_epochs, linear_perceptron_proportion.train_errors, label='Lineal (train_proportion)')
    plt.plot(non_linear_perceptron_proportion.final_epochs, non_linear_perceptron_proportion.train_errors, label='No Lineal (train_proportion)')
    plt.xlabel('Época')
    plt.ylabel('Error (MSE)')
    plt.title('Evolución del MSE durante el entrenamiento (sin K-Fold)')
    plt.legend()
    plt.xlim(0, config.epoch)
    plt.ylim(0, max_train_error)
    plt.show()

