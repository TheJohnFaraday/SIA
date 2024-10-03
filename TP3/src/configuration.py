import tomllib
from dataclasses import dataclass

@dataclass(frozen=True)
class And:
    input: (int, int)
    output: int


@dataclass(frozen=True)
class Xor:
    input: (int, int)
    output: int


@dataclass(frozen=True)
class Configuration:
    plot: bool
    _and: [And]
    _xor: [Xor]
    linear_non_linear_path: str
    parity_discrimination_path: str
    mnist_path: str
    noise_val: float
    random_seed: int | None


def read_configuration():
    with open("config.toml", "rb") as f:
        data = tomllib.load(f, parse_float=float)

        and_input = data["and"]["input"]
        and_output = data["and"]["output"]

        if len(and_input) != len(and_output):
            print("Unmatched input and output size for single layer simple AND")
            raise RuntimeError
        _and = [And]
        for i in range(len(and_input)):
            _and.append(And(tuple(and_input[i]), and_output[i]))

        xor_input = data["xor"]["input"]
        xor_output = data["xor"]["output"]

        if len(xor_input) != len(xor_output):
            print("Unmatched input and output size for single layer simple XOR")
            raise RuntimeError
        _xor = [Xor]
        for i in range(len(xor_input)):
            _xor.append(And(tuple(xor_input[i]), xor_output[i]))

        linear_non_linear_path = data["single_layer"]["linear_non_linear"]["path"]

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

        configuration = Configuration(
            plot=bool(data["plot"]),
            _and=_and,
            _xor=_xor,
            linear_non_linear_path=linear_non_linear_path,
            parity_discrimination_path=parity_discrimination_path,
            mnist_path=mnist_path,
            noise_val=noise_val,
            random_seed=data.get("seed", None),
        )

        return configuration
