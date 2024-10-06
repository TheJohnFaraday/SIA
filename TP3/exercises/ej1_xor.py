from src.configuration import Configuration
from src.SimplePerceptron import SimplePerceptron


def ej1_xor(and_expected, config: Configuration):
    # XOR:
    xor_input = config.xor_input
    xor_expected = config.xor_output
    xor_perceptron = SimplePerceptron(len(xor_input[0]), config.learning_rate)
    xor_perceptron.train(xor_input, and_expected, config.epoch)

    for x, y in zip(xor_input, xor_expected):
        pred = xor_perceptron.predict(x)
        print(f"Entrada: {x}, Predicción: {pred}, Valor Esperado: {y}")
