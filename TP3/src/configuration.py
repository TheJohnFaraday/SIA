import numpy as np
import pandas as pd
import tomllib
from dataclasses import dataclass
from .utils import (
    key_from_enum_value_with_fallback,
    normalize,
    normalize2,
    normalize_0_1,
)
from .LinearPerceptron import ActivationFunction as LinearNonLinearActivationFunction
from .Activation import Activation
from .activation_functions import Tanh, Logistic
from .Training import Training, Batch, MiniBatch, Online
from enum import Enum


class TrainingStyle(Enum):
    ONLINE = "online"
    MINIBATCH = "minibatch"
    BATCH = "batch"


class Optimizer(Enum):
    GRADIENT_DESCENT = "gradient_descent"
    MOMENTUM = "momentum"
    ADAM = "adam"


class ActivationFunction(Enum):
    TANH = "tanh"
    LOGISTIC = "logistic"


@dataclass(frozen=True)
class MultiLayer:
    optimizer: Optimizer
    training_style: TrainingStyle
    digits_input: np.ndarray
    digits_output: np.ndarray
    parity_discrimination_activation_function: ActivationFunction
    digits_discrimination_activation_function: ActivationFunction
    noise_val: float
    mnist_path: str
    momentum: float
    beta1: float
    beta2: float
    epsilon: float
    acceptable_error_epsilon: float
    batch_size: int = 0


@dataclass(frozen=True)
class Configuration:
    plot: bool
    random_seed: int | None
    learning_rate: float
    beta: float
    epoch: int
    train_proportion: float
    and_input: np.ndarray
    and_output: np.ndarray
    xor_input: np.ndarray
    xor_output: np.ndarray
    linear_non_linear_input: np.ndarray
    linear_non_linear_output: np.ndarray
    linear_non_linear_output_norm: np.ndarray
    linear_non_linear_activation_function: LinearNonLinearActivationFunction
    multilayer: MultiLayer
    mnist_path: str
    noise_val: float
    random_seed: int | None


def read_configuration():
    with open("config.toml", "rb") as f:
        data = tomllib.load(f, parse_float=float)

        beta = float(data.get("beta", 0.4))
        and_input = np.array(data["and"]["input"])
        and_output = np.array(data["and"]["output"])

        if len(and_input) != len(and_output):
            print("Unmatched input and output size for single layer simple AND")
            raise RuntimeError

        xor_input = np.array(data["xor"]["input"])
        xor_output = np.array(data["xor"]["output"])

        if len(xor_input) != len(xor_output):
            print("Unmatched input and output size for single layer simple XOR")
            raise RuntimeError

        linear_non_linear_path = data["single_layer"]["linear_non_linear"]["path"]

        df = pd.read_csv(linear_non_linear_path, header=0)
        linear_non_linear_input = np.array(
            list(
                map(
                    lambda row: [row[1]["x1"], row[1]["x2"], row[1]["x3"]],
                    df.iterrows(),
                )
            )  # (index, row)
        )

        linear_non_linear_activation_function = key_from_enum_value_with_fallback(
            LinearNonLinearActivationFunction,
            data["single_layer"]["linear_non_linear"]["activation_function"],
            LinearNonLinearActivationFunction.TANH,
        )

        linear_non_linear_output = list(map(lambda row: row[1]["y"], df.iterrows()))
        if (
            linear_non_linear_activation_function
            == LinearNonLinearActivationFunction.TANH
        ):
            normalize_fn = normalize
        else:
            normalize_fn = normalize_0_1

        linear_non_linear_output_norm = np.array(
            list(
                map(
                    lambda y: normalize_fn(
                        y,
                        np.min(linear_non_linear_output),
                        np.max(linear_non_linear_output),
                    ),
                    linear_non_linear_output,
                )
            )
        )

        if not linear_non_linear_path or linear_non_linear_path == "":
            print(
                "Invalid or unexistent dataset path provided for Linear and Non Linear"
            )
            raise RuntimeError

        digits_path = data["multi_layer"]["parity_discrimination"]["path"]

        if not digits_path or digits_path == "":
            print(
                "Invalid or unexistent dataset path provided for Multi Layer Parity Discrimination"
            )
            raise RuntimeError

        digits_input = []
        with open(digits_path, "r") as file:
            matrix = []
            for i, line in enumerate(file):
                if i != 0 and i % 7 == 0:
                    digits_input.append(matrix)
                    matrix = []
                    row = list(map(int, line.split()))
                    matrix.append(row)
                else:
                    row = list(map(int, line.split()))
                    matrix.append(row)
            digits_input.append(matrix)

        digits_input = np.array(digits_input)
        digits_output = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        parity_discrimination_activation_function = data["multi_layer"][
            "parity_discrimination"
        ].get("activation_function", "tanh")

        training_style = key_from_enum_value_with_fallback(
            TrainingStyle,
            data["multi_layer"].get("training_style"),
            TrainingStyle.ONLINE,
        )

        parity_discrimination_activation_function = key_from_enum_value_with_fallback(
            ActivationFunction,
            data["multi_layer"]["parity_discrimination"]["activation_function"],
            ActivationFunction.TANH,
        )

        digits_discrimination_activation_function = key_from_enum_value_with_fallback(
            ActivationFunction,
            data["multi_layer"]["digit_discrimination"]["activation_function"],
            ActivationFunction.TANH,
        )

        noise_val = data.get("noise_val", 0.0)

        mnist_path = data["multi_layer"]["mnist"].get("path", "./datasets/mnist.npz")

        optimizer = key_from_enum_value_with_fallback(
            Optimizer, data["multi_layer"].get("optimizer"), Optimizer.GRADIENT_DESCENT
        )

        multilayer_configuration = MultiLayer(
            optimizer=optimizer,
            training_style=training_style,
            digits_input=digits_input,
            digits_output=digits_output,
            parity_discrimination_activation_function=parity_discrimination_activation_function,
            digits_discrimination_activation_function=digits_discrimination_activation_function,
            mnist_path=mnist_path,
            noise_val=noise_val,
            momentum=data["multi_layer"]["momentum"].get("alpha", 0.9),
            beta1=data["multi_layer"]["adam"].get("beta1", 0.9),
            beta2=data["multi_layer"]["adam"].get("beta2", 0.99),
            epsilon=data["multi_layer"]["adam"].get("epsilon", 1e-8),
            batch_size=data["multi_layer"].get("batch_size", 0),
            acceptable_error_epsilon=data["multi_layer"].get("acceptable_error_epsilon", 1e-16)
        )

        configuration = Configuration(
            plot=bool(data["plot"]),
            learning_rate=float(data.get("learning_rate", 0.01)),
            beta=beta,
            epoch=int(data.get("epoch", 10000)),
            train_proportion=float(data.get("train_proportion", 0.7)),
            and_input=and_input,
            and_output=and_output,
            xor_input=xor_input,
            xor_output=xor_output,
            linear_non_linear_input=linear_non_linear_input,
            linear_non_linear_output=linear_non_linear_output,
            linear_non_linear_output_norm=linear_non_linear_output_norm,
            linear_non_linear_activation_function=linear_non_linear_activation_function,
            mnist_path=mnist_path,
            noise_val=noise_val,
            random_seed=data.get("seed", None),
            multilayer=multilayer_configuration,
        )

        return configuration
