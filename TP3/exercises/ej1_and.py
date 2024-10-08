from src.configuration import Configuration
from src.SimplePerceptron import SimplePerceptron
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(perceptron, input_data, expected_output):
    # Graficar puntos de entrada
    for i, point in enumerate(input_data):
        if expected_output[i] == 1:
            plt.scatter(point[0], point[1], color='blue')
        else:
            plt.scatter(point[0], point[1], color='red')

    # Obtener pesos y bias
    weights = perceptron.weights
    bias = perceptron.bias

    # Graficar la recta de decisión
    x = np.linspace(-1, 2, 100)  # Rango de valores en el eje x
    y = -(weights[0] * x + bias) / weights[1]  # Ecuación de la recta: w1*x + w2*y + b = 0
    plt.plot(x, y, '-g')

    plt.title("Decisión Final del Perceptrón")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

def ej1_and(config: Configuration):
    # AND:
    and_input = config.and_input
    and_expected = config.and_output
    and_perceptron = SimplePerceptron(len(and_input[0]), config.learning_rate)
    and_perceptron.train(and_input, and_expected, config.epoch)

    for x, y in zip(and_input, and_expected):
        pred = and_perceptron.predict(x)
        print(f"Entrada: {x}, Predicción: {pred}, Valor Esperado: {y}")

    plot_decision_boundary(and_perceptron, and_input, and_expected)


    return and_expected
