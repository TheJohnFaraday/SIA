import numpy as np
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

    X_norm = np.array(
        list(map(lambda row: np.array(list(map(lambda x: normalize_0_1(x, 0, 255), row))), X))
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

    network = [
        layer1,
        Dense(784, 196, GradientDescent(config.learning_rate)),
        layer2,
        Dense(196, 28, GradientDescent(config.learning_rate)),
        layer3,
        Dense(28, 10, GradientDescent(config.learning_rate)),
        layer4,
    ]

    match config.multilayer.training_style:
        case TrainingStyle.ONLINE:
            training_style = Online(MultiLayerPerceptron.predict)
        case TrainingStyle.MINIBATCH:
            training_style = MiniBatch(
                MultiLayerPerceptron.predict, config.multilayer.batch_size
            )
        case TrainingStyle.BATCH:
            training_style = Batch(
                MultiLayerPerceptron.predict, config.multilayer.batch_size
            )

    mlp = MultiLayerPerceptron(
        training_style,
        network,
        MSE(),
        config.epoch,
        config.learning_rate,
    )

    new_network = mlp.train(X_norm, Y)

    '''
    X_with_noise = list(
        map(
            lambda block: list(
                map(
                    lambda row: list(
                        map(lambda x: x + np.random.normal(0, config.noise_val), row)
                    ),
                    block,
                )
            ),
            config.multilayer.digits_input,
        ),
    )

    X_with_noise = np.reshape(
        list(
            map(
                lambda block: list(
                    map(
                        lambda row: list(
                            map(lambda x: 0 if x < 0 else (1 if x > 1 else x), row)
                        ),
                        block,
                    )
                ),
                X_with_noise,
            )
        ),
        (10, 35, 1),
    )

    for x, y in zip(X_with_noise, is_odd_output):
        output = MultiLayerPerceptron.predict(new_network, x)
        print(f"Is Odd Expected Output: {y}")
        print(f"Is Odd Output:\n{output}")
        '''
