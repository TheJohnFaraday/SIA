

class MultiLayerPerceptron:

    def __init__(self, layers, error_function, optimizer, training_method, learning_rate, epochs):
        self.layers = layers
        self.error_function = error_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.training_method = training_method
    
    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def train(self, X, Y):
        self.training_method.train(self.layers, self.error_function, X, Y, self.epochs, self.learning_rate)