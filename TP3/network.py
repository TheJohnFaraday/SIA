def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, error, error_prime, input, expected_output, epochs = 1000, learning_rate = 0.01, print = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(input, expected_output):
            # forward
            output = predict(network, x)

            # error
            error += error(expected_output, output)

            # backward
            grad = error_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(input)
        if print:
            print(f"{e + 1}/{epochs}, error={error}")