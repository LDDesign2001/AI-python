import numpy as np

x_entrer = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1], [4, 1.5]),
                    dtype=float)  # input data
y = np.array(([1], [0], [1], [0], [1], [0], [1], [0]), dtype=float)  # output data /  1 = red /  0 = blue

# Change the scale of our values to be between 0 and 1
x_entrer = x_entrer / np.amax(x_entrer, axis=0)  # I divide each input by the maximum value of the inputs

# We get what we are interested in
X = np.split(x_entrer, [8])[0]  # Data on which we will practice, the first 8 of our matrix
xPrediction = np.split(x_entrer, [8])[1]  # Value we want to find


# Our class of neural network
class Neural_Network(object):
    def __init__(self):

        # Our parameters
        self.inputSize = 2  # Number of neurons to enter
        self.outputSize = 1  # Number of output neurons
        self.hiddenSize = 3  # Number of hidden neurons

        # Our weights
        self.W1 = np.random.randn(self.inputSize,
                                  self.hiddenSize)  # (2x3) Weight matrix between entering and hidden neurons
        self.W2 = np.random.randn(self.hiddenSize,
                                  self.outputSize)  # (3x1) Weight matrix between hidden neurons and output

    # Forward propagation function
    def forward(self, X):

        self.z = np.dot(X, self.W1)  # Matrix multiplication between input values and W1 weights
        self.z2 = self.sigmoid(self.z)  # Application of activation function (Sigmoid)
        self.z3 = np.dot(self.z2, self.W2)  # Matrix multiplication between hidden values and W2 weights
        o = self.sigmoid(
            self.z3)  # Apply the activation function, and get our final output value
        return o

    # Activation function
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    # Derived from the activation function
    def sigmoidPrime(self, s):
        return s * (1 - s)

    # Backpropagation function
    def backward(self, X, y, o):

        self.o_error = y - o  # Calculation of the error
        self.o_delta = self.o_error * self.sigmoidPrime(o)  # Apply the derivative of the sigmoid to this error

        self.z2_error = self.o_delta.dot(self.W2.T)  # Calculation of the error of our hidden neurons
        self.z2_delta = self.z2_error * self.sigmoidPrime(
            self.z2) # Apply the derivative of the sigmoid to this error

        self.W1 += X.T.dot(self.z2_delta)  # We adjust our weights W1
        self.W2 += self.z2.T.dot(self.o_delta)  # We adjust our weights W2

    # Training function
    def train(self, X, y):

        o = self.forward(X)
        self.backward(X, y, o)

    #Prediction function
    def predict(self):

        print("Predicted data after training: ")
        print("Input : \n" + str(xPrediction))
        print("Output : \n" + str(self.forward(xPrediction)))

        if (self.forward(xPrediction) < 0.5):
            print("The flower is BLUE! \n")
        else:
            print("The flower is RED! \n")


NN = Neural_Network()

for i in range(1000):  # Choose a number of iterations, be careful too many can create overfitting!
    print("# " + str(i) + "\n")
    print("Entry value: \n" + str(X))
    print("Actual value: \n" + str(y))
    print("Predicted output: \n" + str(np.matrix.round(NN.forward(X), 2)))
    print("\n")
    NN.train(X, y)

NN.predict()