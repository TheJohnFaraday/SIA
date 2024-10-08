import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.configuration import Configuration, TrainingStyle, Optimizer

from src.Dense import Dense
from src.errors import MSE

from src.MultiLayerPerceptron import MultiLayerPerceptron
from src.Optimizer import GradientDescent, Momentum, Adam
from src.Training import Online, MiniBatch, Batch

from exercises.NetworkOutput import NetworkOutput
from src.grapher import graph_error_by_epoch, graph_accuracy_vs_dataset


@dataclass
class TrainSet:
    train_input: np.array
    test_input: np.array
    train_output: np.array
    test_output: np.array


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


def get_train_set(config: Configuration, output, proportion: float):
    train_input, test_input, train_output, test_output = train_test_split(
        config.multilayer.digits_input,
        output,
        train_size=proportion,
        random_state=config.random_seed,
    )
    return TrainSet(
        train_input=train_input,
        test_input=test_input,
        train_output=train_output,
        test_output=test_output,
    )


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


def is_odd(config: Configuration):
    is_odd_output = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    dataset_accuracy = []
    if config.train_proportion < 1:
        proportion = 0.2
    else:
        proportion = 1

    while proportion <= 1.0:
        train_set = get_train_set(config, is_odd_output, proportion)

        X = np.reshape(train_set.train_input, (train_set.train_input.shape[0], 35, 1))
        Y = np.reshape(train_set.train_output, (train_set.train_input.shape[0], 1, 1))

        network = [
            Dense(35, 70, get_optimizer_instance(config)),
            config.multilayer.parity_discrimination_activation_function,
            Dense(70, 1, get_optimizer_instance(config)),
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
                            map(
                                lambda x: x + np.random.normal(0, config.noise_val), row
                            )
                        ),
                        block,
                    )
                ),
                train_set.train_input,
            ),
        )

        X_with_noise = np.reshape(
            np.array(
                list(
                    map(
                        lambda block: list(
                            map(
                                lambda row: list(
                                    map(
                                        lambda x: 0 if x < 0 else (1 if x > 1 else x),
                                        row,
                                    )
                                ),
                                block,
                            )
                        ),
                        X_with_noise,
                    )
                )
            ),
            (train_set.train_input.shape[0], 35, 1),
        )

        outputs_with_error: list[NetworkOutput] = []
        for x, y in zip(X_with_noise, is_odd_output):
            output = MultiLayerPerceptron.predict(new_network, x)
            loss = mse.error(y, output)
            outputs_with_error.append(
                NetworkOutput(expected=y, output=output, error=loss)
            )

            print(f"Is Odd Expected Output: {y}")
            print(f"Is Odd Output:\n{output}")

        outputs_df = pd.DataFrame(
            {
                "Expected Output": [output.expected for output in outputs_with_error],
                "Output": [output.output for output in outputs_with_error],
                "Error": [output.error for output in outputs_with_error],
            }
        )
        outputs_df["Rounded Output"] = outputs_df["Output"].apply(
            lambda x: round(x[0][0])
        )

        matrix = get_confusion_matrix(outputs_df)
        accuracy = sum(matrix["Predicted"]) / (
            sum(matrix["Predicted"]) + sum(matrix["Not Predicted"])
        )
        dataset_accuracy.append((proportion, accuracy))

        graph_error_by_epoch(config, outputs_with_error, errors_by_epoch)
        proportion += 0.1

    graph_accuracy_vs_dataset(config, dataset_accuracy)


def which_number(config: Configuration):
    X = np.reshape(config.multilayer.digits_input, (10, 35, 1))
    Y = np.reshape(config.multilayer.digits_output, (10, 10, 1))
    network = [
        Dense(35, 10, get_optimizer_instance(config)),
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
