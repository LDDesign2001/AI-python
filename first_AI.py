import numpy as np

x_enter = np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[5.5,1],[1,1],[4,1.5]),dtype=float)
#entry of numbers
y = np.array(([1], [0], [1],[0],[1],[0],[1],[0]), dtype=float)
#1 is red and 0 is blue

x_enter = x_enter/np.amax(x_enter,axis=0)#devide every entry with the highest entry

X = np.split(x_enter,[8])[0]
XPrediction = np.split(x_enter,[8])[1]


#neural network class
class Neural_network(object):
    def __init__(self):
        #parameters
        self.inputSize = 2 #neuronal entries
        self.outputSize = 1 #neuronal outputs
        self.hiddenSize = 3 #hidden neurons
        #our Weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #2X3 Weight matrix between neurons entering and hidden
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) #3X1 Weight matrix between hidden neurons and output
    #fonction of forward propagation
    def forward(self,X):
        self.z = np.dot(X,self.W1)# Matrix multiplication between input values and W1 weights
        self.z2 = self.sigmoid(self.z) # Applying of the activation function (Sigmoid)
        self.z3 = np.dot(self.z2, self.W2)# Matrix multiplication between hidden values and W2 weights
        o = self.sigmoid(self.z)# Applying the activation function, and getting our final output value
        return o
