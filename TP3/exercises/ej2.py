import numpy as np
from sklearn.model_selection import KFold
from src.configuration import Configuration
from src.LinearPerceptron import LinearPerceptron, ActivationFunction
from src.errors import MSE
from src.utils import unnormalize


def ej2(config: Configuration):
    # KFold configuration
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Usaremos 5-fold cross-validation

    inputs = config.linear_non_linear_input
    outputs = config.linear_non_linear_output_norm

    # Entrenamiento y evaluación con K-Fold Cross-Validation
    fold = 1
    for train_index, test_index in kf.split(inputs):
        print(f"\n===== Fold {fold} =====")

        # Dividimos los datos en entrenamiento y test para este fold
        train_input, test_input = inputs[train_index], inputs[test_index]
        train_output, test_output = outputs[train_index], outputs[test_index]

        # LINEAR PERCEPTRON
        linear_perceptron = LinearPerceptron(
            len(train_input[0]),
            config.learning_rate,
            ActivationFunction.LINEAR,
            config.beta,
            MSE(),
        )
        linear_perceptron.train(train_input, train_output, config.epoch)

        print(f"Evaluando perceptrón lineal en Fold {fold}:")
        for x, y in zip(test_input, test_output):
            pred, _ = linear_perceptron.predict(x)
            print(
                f"Lineal: Entrada: {x}, Predicción: {unnormalize(pred, np.min(test_output), np.max(test_output))}, Valor Esperado: {y}"
            )

        # NON-LINEAR PERCEPTRON
        non_linear_perceptron = LinearPerceptron(
            len(train_input[0]),
            config.learning_rate,
            config.linear_non_linear_activation_function,
            config.beta,
            MSE(),
        )
        non_linear_perceptron.train(train_input, train_output, config.epoch)

        print(f"Evaluando perceptrón no lineal en Fold {fold}:")
        for x, y in zip(test_input, test_output):
            pred, _ = non_linear_perceptron.predict(x)
            print(
                f"No Lineal: Entrada: {x}, Predicción: {unnormalize(pred, np.min(test_output), np.max(test_output))}, Valor Esperado: {y}"
            )

        fold += 1

