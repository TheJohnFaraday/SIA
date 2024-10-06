from src.configuration import Configuration
from src.SimplePerceptron import SimplePerceptron


def ej1_and(config: Configuration):
    # AND:
    and_input = config.and_input
    and_expected = config.and_output
    and_perceptron = SimplePerceptron(len(and_input[0]), config.learning_rate)
    and_perceptron.train(and_input, and_expected, config.epoch)

    for x, y in zip(and_input, and_expected):
        pred = and_perceptron.predict(x)
        print(f"Entrada: {x}, Predicción: {pred}, Valor Esperado: {y}")

    return and_expected
