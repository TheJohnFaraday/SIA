#
#   X(input)   ---------------------      Y (output) 
#    ----->    |                   |   ----->
#              |       LAYER       |
#    <------   |                   |   <------ 
#    dE/dX     ---------------------       dE/dY
#
#Forward: agarra input y te da output
#Backward: agarra la derivada del error respecto del output (output_gradient) => update de los param y devuelve deriv input
#lo del learning rate --> podría ser optimizer (creo que lo piden)

class Layer:
    def __init__(self):
        self.input = None
        self.output = None


    def forward(self, input):
        #TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        #TODO: update parameters and return input_gradient
        pass