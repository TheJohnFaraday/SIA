import numpy as np

from src.configuration import Configuration
from src.LinearPerceptron import LinearPerceptron, ActivationFunction
from src.errors import MSE
from src.utils import unnormalize


def ej2(config: Configuration):
    linear_non_linear_separation = int(
        len(config.linear_non_linear_input) * config.train_proportion
    )
    print(f"LINEAR SEPARATION: {linear_non_linear_separation}")

    linear_non_linear_training_input = config.linear_non_linear_input[
        :linear_non_linear_separation
    ]
    linear_non_linear_test_input = config.linear_non_linear_input[
        linear_non_linear_separation:
    ]

    linear_non_linear_training_output = config.linear_non_linear_output[
        :linear_non_linear_separation
    ]
    linear_non_linear_training_output_norm = config.linear_non_linear_output_norm[
        linear_non_linear_separation:
    ]

    print(f"LINEAR TRAINING INPUT: {linear_non_linear_training_input}")
    print(f"LINEAR TRAINING OUTPUT: {linear_non_linear_training_output}")
    linear_non_linear_test_output = config.linear_non_linear_output[
        linear_non_linear_separation:
    ]
    linear_non_linear_test_output_norm = config.linear_non_linear_output_norm[
        linear_non_linear_separation:
    ]

    linear_perceptron = LinearPerceptron(
        len(linear_non_linear_training_input[0]),
        config.learning_rate,
        ActivationFunction.LINEAR,
        config.beta,
        MSE(),
    )
    linear_perceptron.train(
        linear_non_linear_training_input,
        linear_non_linear_training_output_norm,
        config.epoch,
    )

    # TODO evaluar con los datos de prueba
    for x, y in zip(linear_non_linear_test_input, linear_non_linear_test_output):
        pred, _ = linear_perceptron.predict(x)
        print(
            f"Lineal: Entrada: {x}, Predicción: {unnormalize(pred, np.min(linear_non_linear_test_output), np.max(linear_non_linear_test_output))}, Valor Esperado: {y}"
        )

    non_linear_perceptron = LinearPerceptron(
        len(linear_non_linear_training_input[0]),
        config.learning_rate,
        config.linear_non_linear_activation_function,
        config.beta,
        MSE(),
    )
    non_linear_perceptron.train(
        linear_non_linear_training_input,
        linear_non_linear_training_output_norm,
        config.epoch,
    )

    for x, y in zip(linear_non_linear_test_input, linear_non_linear_test_output):
        pred, _ = non_linear_perceptron.predict(x)
        print(
            f"No Lineal: Entrada: {x}, Predicción: {unnormalize(pred, np.min(linear_non_linear_test_output), np.max(linear_non_linear_test_output))}, Valor Esperado: {y}"
        )
