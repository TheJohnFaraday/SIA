import numpy as np

from src.configuration import Configuration, TrainingStyle

from src.Dense import Dense
from src.errors import MSE

from src.MultiLayerPerceptron import MultiLayerPerceptron
from src.Optimizer import GradientDescent
from src.Training import Online, MiniBatch, Batch

from exercises.NetworkOutput import NetworkOutput
from src.grapher import graph_error_by_epoch


def is_odd(config: Configuration):
    is_odd_output = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X = np.reshape(config.multilayer.digits_input, (10, 35, 1))
    Y = np.reshape(is_odd_output, (10, 1, 1))
    network = [
        Dense(35, 70, GradientDescent(config.learning_rate)),
        config.multilayer.parity_discrimination_activation_function,
        Dense(70, 1, GradientDescent(config.learning_rate)),
        config.multilayer.parity_discrimination_activation_function,
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
        case _:
            raise RuntimeError("Invalid TrainingStyle")

    mse = MSE()
    mlp = MultiLayerPerceptron(
        training_style,
        network,
        mse,
        config.epoch,
        config.learning_rate,
    )

    new_network, errors_by_epoch = mlp.train(X, Y)

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
        np.array(
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
            )
        ),
        (10, 35, 1),
    )

    outputs_with_error: list[NetworkOutput] = []
    for x, y in zip(X_with_noise, is_odd_output):
        output = MultiLayerPerceptron.predict(new_network, x)
        loss = mse.error(y, output)
        outputs_with_error.append(NetworkOutput(expected=y, output=output, error=loss))

        print(f"Is Odd Expected Output: {y}")
        print(f"Is Odd Output:\n{output}")

    graph_error_by_epoch(outputs_with_error, errors_by_epoch)


def which_number(config: Configuration):
    X = np.reshape(config.multilayer.digits_input, (10, 35, 1))
    Y = np.reshape(config.multilayer.digits_output, (10, 10, 1))
    network = [
        Dense(35, 10, GradientDescent(config.learning_rate)),
        config.multilayer.digits_discrimination_activation_function,
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
        case _:
            raise RuntimeError("Invalid TrainingStyle")

    mlp = MultiLayerPerceptron(
        training_style,
        network,
        MSE(),
        config.epoch,
        config.learning_rate,
    )

    new_network, errors = mlp.train(X, Y)

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
        np.array(
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
            )
        ),
        (10, 35, 1),
    )

    for x, y in zip(X_with_noise, config.multilayer.digits_output):
        output = MultiLayerPerceptron.predict(new_network, x)
        print(f"Number Expected Output: {y}")
        print(f"Number Output:\n{output}")
