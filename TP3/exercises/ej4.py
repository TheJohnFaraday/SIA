import numpy as np
import pandas as pd
import keras

from src.configuration import (
    Configuration,
    TrainingStyle,
    Optimizer,
    ActivationFunction,
)

from src.Dense import Dense
from src.errors import MSE

from src.MultiLayerPerceptron import MultiLayerPerceptron
from src.Optimizer import GradientDescent, Momentum, Adam
from src.Training import Online, MiniBatch, Batch
from src.activation_functions import Tanh, Logistic
from src.utils import normalize_0_1

from exercises.NetworkOutput import NetworkOutput
from src.grapher import (
    graph_error_by_epoch,
    graph_which_number_matrix,
)


def get_optimizer_instance(config: Configuration):
    match config.multilayer.optimizer:
        case Optimizer.GRADIENT_DESCENT:
            return GradientDescent(config.learning_rate)
        case Optimizer.MOMENTUM:
            return Momentum(config.learning_rate, config.multilayer.momentum)
        case Optimizer.ADAM:
            return Adam(
                learning_rate=config.learning_rate,
                beta1=config.multilayer.beta1,
                beta2=config.multilayer.beta2,
                epsilon=config.multilayer.epsilon,
            )
        case _:
            raise RuntimeError("Invalid optimizer selected")


def get_confusion_matrix(df: pd.DataFrame):
    confusion_matrix = {
        "Not Predicted": [0, 0],
        "Predicted": [0, 0],
    }
    for _, row in df.iterrows():
        expected = row["Expected Output"]
        value = row["Rounded Output"]

        if expected == value:
            confusion_matrix["Predicted"][expected] += 1
        else:
            confusion_matrix["Not Predicted"][expected] += 1

    return confusion_matrix


def mnist_digit_clasification(config: Configuration):
    # x_train.shape = (60000, 28, 28)
    # y_train.shape = (60000,)
    # x_test.shape = (60000, 28, 28)
    # y_test.shape = (60000,)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(
        path=config.multilayer.mnist_path
    )

    X = np.reshape(x_train, (60000, 784, 1))
    Y = []
    for y in y_train:
        row = np.zeros(10)
        row[y] = 1
        Y.append(row)
    Y = np.reshape(Y, (60000, 10, 1))

    X_test = np.reshape(x_test, (10000, 784, 1))
    Y_test = []
    for y in y_test:
        row = np.zeros(10)
        row[y] = 1
        Y_test.append(row)
    Y_test = np.reshape(Y_test, (10000, 10, 1))

    X_norm = np.array(
        list(
            map(
                lambda row: np.array(
                    list(map(lambda x: normalize_0_1(x, 0, 255), row))
                ),
                X,
            )
        )
    )

    X_test_norm = np.array(
        list(
            map(
                lambda row: np.array(
                    list(map(lambda x: normalize_0_1(x, 0, 255), row))
                ),
                X_test,
            )
        )
    )

    match config.multilayer.parity_discrimination_activation_function:
        case ActivationFunction.TANH:
            layer1 = Tanh(config.beta)
            layer2 = Tanh(config.beta)
            layer3 = Tanh(config.beta)
            layer4 = Tanh(config.beta)
        case ActivationFunction.LOGISTIC:
            layer1 = Logistic(config.beta)
            layer2 = Logistic(config.beta)
            layer3 = Logistic(config.beta)
            layer4 = Logistic(config.beta)

    mse = MSE()
    network = [
        # layer1,
        Dense(784, 10, get_optimizer_instance(config)),
        layer2,
        # Dense(196, 28, GradientDescent(config.learning_rate)),
        # layer3,
        # Dense(28, 10, GradientDescent(config.learning_rate)),
        # layer4,
    ]

    match config.multilayer.training_style:
        case TrainingStyle.ONLINE:
            training_style = Online(
                MultiLayerPerceptron.predict,
                epsilon=config.multilayer.acceptable_error_epsilon,
            )
        case TrainingStyle.MINIBATCH:
            training_style = MiniBatch(
                MultiLayerPerceptron.predict,
                batch_size=config.multilayer.batch_size,
                epsilon=config.multilayer.acceptable_error_epsilon,
            )
        case TrainingStyle.BATCH:
            training_style = Batch(
                MultiLayerPerceptron.predict,
                batch_size=config.multilayer.batch_size,
                epsilon=config.multilayer.acceptable_error_epsilon,
            )
        case _:
            raise RuntimeError("Invalid TrainingStyle")

    mlp = MultiLayerPerceptron(
        training_style,
        network,
        MSE(),
        config.epoch,
        config.learning_rate,
    )

    new_network, errors_by_epoch = mlp.train(X_norm, Y)

    outputs_with_error: list[NetworkOutput] = []
    for x, y in zip(X_test_norm, Y_test):
        output = MultiLayerPerceptron.predict(new_network, x)
        loss = mse.error(y, output)
        outputs_with_error.append(
            NetworkOutput(expected=y, output=output, error=loss)
        )

        print(f"Number Expected Output: {y}")
        print(f"Number Output:\n{output}")
        print(f"Number Output - Rounded:\n{[round(n[0]) for n in output]}")

    graph_error_by_epoch("which_number_mnist", config, errors_by_epoch, proportion=1.0)
    graph_which_number_matrix(
        "which_number_mnist", config, outputs_with_error, proportion=1.0
    )
