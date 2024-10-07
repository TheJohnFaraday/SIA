import numpy as np
import pandas as pd
import tomllib
from dataclasses import dataclass
from .utils import key_from_enum_value_with_fallback, normalize
from .LinearPerceptron import ActivationFunction as LinearNonLinearActivationFunction


@dataclass(frozen=True)
class MultiLayer:
    parity_discrimination_path: str
    noise_val: float
    mnist_path: str
    momentum: float
    beta1: float
    beta2: float
    epsilon: float


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


def read_configuration():
    with open("config.toml", "rb") as f:
        data = tomllib.load(f, parse_float=float)

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
        linear_non_linear_output = list(map(lambda row: row[1]["y"], df.iterrows()))
        linear_non_linear_output_norm = np.array(
            list(
                map(
                    lambda y: normalize(y, np.min(linear_non_linear_output), np.max(linear_non_linear_output)), linear_non_linear_output
                )
            )
        )
        linear_non_linear_activation_function = key_from_enum_value_with_fallback(
            LinearNonLinearActivationFunction,
            data["single_layer"]["linear_non_linear"]["activation_function"],
            LinearNonLinearActivationFunction.TANH,
        )

        if not linear_non_linear_path or linear_non_linear_path == "":
            print(
                "Invalid or unexistent dataset path provided for Linear and Non Linear"
            )
            raise RuntimeError

        parity_discrimination_path = data["multi_layer"]["parity_discrimination"][
            "path"
        ]

        if not parity_discrimination_path or parity_discrimination_path == "":
            print(
                "Invalid or unexistent dataset path provided for Multi Layer Parity Discrimination"
            )
            raise RuntimeError

        noise_val = data["multi_layer"]["digit_discrimination"].get("noise_val", 0.0)

        mnist_path = data["multi_layer"].get("mnist", "./datasets/mnist.npz")

        multilayer_configuration = MultiLayer(
            parity_discrimination_path=parity_discrimination_path,
            mnist_path=mnist_path,
            noise_val=noise_val,
            momentum=data["multi_layer"]["momentum"].get("alpha", 0.9),
            beta1=data["multi_layer"]["adam"].get("beta1", 0.9),
            beta2=data["multi_layer"]["adam"].get("beta2", 0.99),
            epsilon=data["multi_layer"]["adam"].get("epsilon", 1e-8),
        )

        configuration = Configuration(
            plot=bool(data["plot"]),
            learning_rate=float(data.get('learning_rate', 0.01)),
            beta=float(data.get('beta', 0.4)),
            epoch=int(data.get('epoch', 10000)),
            train_proportion=float(data.get('train_proportion', 0.7)),
            and_input=and_input,
            and_output=and_output,
            xor_input=xor_input,
            xor_output=xor_output,
            linear_non_linear_input=linear_non_linear_input,
            linear_non_linear_output=linear_non_linear_output,
            linear_non_linear_output_norm=linear_non_linear_output_norm,
            linear_non_linear_activation_function=linear_non_linear_activation_function,
            random_seed=data.get("seed", None),
            multilayer=multilayer_configuration
        )

        return configuration
